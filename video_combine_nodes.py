import os
import sys
import json
import subprocess
import numpy as np
import re
import datetime
from typing import List
import torch
from PIL import Image, ExifTags
from PIL.PngImagePlugin import PngInfo
from pathlib import Path
from string import Template
import itertools
import functools
import time
import shutil
import logging
import copy

import folder_paths
from comfy.utils import ProgressBar
import server

# Logger setup
class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[0;36m",  # CYAN
        "INFO": "\033[0;32m",  # GREEN
        "WARNING": "\033[0;33m",  # YELLOW
        "ERROR": "\033[0;31m",  # RED
        "CRITICAL": "\033[0;37;41m",  # WHITE ON RED
        "RESET": "\033[0m",  # RESET COLOR
    }

    def format(self, record):
        colored_record = copy.copy(record)
        levelname = colored_record.levelname
        seq = self.COLORS.get(levelname, self.COLORS["RESET"])
        colored_record.levelname = f"{seq}{levelname}{self.COLORS['RESET']}"
        return super().format(colored_record)

# Create logger
logger = logging.getLogger("ComfyUI_EQX_Video")
logger.propagate = False

# Add handler if we don't have one
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColoredFormatter("[%(name)s] - %(levelname)s - %(message)s"))
    logger.addHandler(handler)

# Configure logger
loglevel = logging.INFO
logger.setLevel(loglevel)

# Constants
BIGMIN = -(2**53-1)
BIGMAX = (2**53-1)
DIMMAX = 8192
ENCODE_ARGS = ("utf-8", 'backslashreplace')

# Utility functions
def ffmpeg_suitability(path):
    try:
        version = subprocess.run([path, "-version"], check=True,
                                 capture_output=True).stdout.decode(*ENCODE_ARGS)
    except:
        return 0
    score = 0
    #rough layout of the importance of various features
    simple_criterion = [("libvpx", 20),("264",10), ("265",3),
                        ("svtav1",5),("libopus", 1)]
    for criterion in simple_criterion:
        if version.find(criterion[0]) >= 0:
            score += criterion[1]
    #obtain rough compile year from copyright information
    copyright_index = version.find('2000-2')
    if copyright_index >= 0:
        copyright_year = version[copyright_index+6:copyright_index+9]
        if copyright_year.isnumeric():
            score += int(copyright_year)
    return score

class MultiInput(str):
    def __new__(cls, string, allowed_types="*"):
        res = super().__new__(cls, string)
        res.allowed_types=allowed_types
        return res
    def __ne__(self, other):
        if self.allowed_types == "*" or other == "*":
            return False
        return other not in self.allowed_types

imageOrLatent = MultiInput("IMAGE", ["IMAGE", "LATENT"])
floatOrInt = MultiInput("FLOAT", ["FLOAT", "INT"])

class ContainsAll(dict):
    def __contains__(self, other):
        return True
    def __getitem__(self, key):
        return super().get(key, (None, {}))

# FFmpeg path detection
if "VHS_FORCE_FFMPEG_PATH" in os.environ:
    ffmpeg_path = os.environ.get("VHS_FORCE_FFMPEG_PATH")
else:
    ffmpeg_paths = []
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        imageio_ffmpeg_path = get_ffmpeg_exe()
        ffmpeg_paths.append(imageio_ffmpeg_path)
    except:
        if "VHS_USE_IMAGEIO_FFMPEG" in os.environ:
            raise
        logger.warn("Failed to import imageio_ffmpeg")
    if "VHS_USE_IMAGEIO_FFMPEG" in os.environ:
        ffmpeg_path = imageio_ffmpeg_path
    else:
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg is not None:
            ffmpeg_paths.append(system_ffmpeg)
        if os.path.isfile("ffmpeg"):
            ffmpeg_paths.append(os.path.abspath("ffmpeg"))
        if os.path.isfile("ffmpeg.exe"):
            ffmpeg_paths.append(os.path.abspath("ffmpeg.exe"))
        if len(ffmpeg_paths) == 0:
            logger.error("No valid ffmpeg found.")
            ffmpeg_path = None
        elif len(ffmpeg_paths) == 1:
            ffmpeg_path = ffmpeg_paths[0]
        else:
            ffmpeg_path = max(ffmpeg_paths, key=ffmpeg_suitability)

gifski_path = os.environ.get("VHS_GIFSKI", None)
if gifski_path is None:
    gifski_path = os.environ.get("JOV_GIFSKI", None)
    if gifski_path is None:
        gifski_path = shutil.which("gifski")

def merge_filter_args(args, ftype="-vf"):
    try:
        start_index = args.index(ftype)+1
        index = start_index
        while True:
            index = args.index(ftype, index)
            args[start_index] += ',' + args[index+1]
            args.pop(index)
            args.pop(index)
    except ValueError:
        pass

def cached(duration):
    def dec(f):
        cached_ret = None
        cache_time = 0
        def cached_func():
            nonlocal cache_time, cached_ret
            if time.time() > cache_time + duration or cached_ret is None:
                cache_time = time.time()
                cached_ret = f()
            return cached_ret
        return cached_func
    return dec

# Stub for requeue_workflow - not needed in this integration
prompt_queue = server.PromptServer.instance.prompt_queue if hasattr(server, 'PromptServer') else None
requeue_guard = [None, 0, 0, {}]

def requeue_workflow(requeue_required=(-1,True)):
    """Minimal implementation - just returns without doing anything"""
    return

# Setup video formats folder
if 'VHS_video_formats' not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["VHS_video_formats"] = ((),{".json"})
if len(folder_paths.folder_names_and_paths['VHS_video_formats'][1]) == 0:
    folder_paths.folder_names_and_paths["VHS_video_formats"][1].add(".json")

def flatten_list(l):
    ret = []
    for e in l:
        if isinstance(e, list):
            ret.extend(e)
        else:
            ret.append(e)
    return ret

def iterate_format(video_format, for_widgets=True):
    """Provides an iterator over widgets, or arguments"""
    def indirector(cont, index):
        if isinstance(cont[index], list) and (not for_widgets
          or len(cont[index])> 1 and not isinstance(cont[index][1], dict)):
            inp = yield cont[index]
            if inp is not None:
                cont[index] = inp
                yield
    for k in video_format:
        if k == "extra_widgets":
            if for_widgets:
                yield from video_format["extra_widgets"]
        elif k.endswith("_pass"):
            for i in range(len(video_format[k])):
                yield from indirector(video_format[k], i)
            if not for_widgets:
                video_format[k] = flatten_list(video_format[k])
        else:
            yield from indirector(video_format, k)

base_formats_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "video_formats")

@cached(5)
def get_video_formats():
    format_files = {}
    for format_name in folder_paths.get_filename_list("VHS_video_formats"):
        format_files[format_name] = folder_paths.get_full_path("VHS_video_formats", format_name)
    for item in os.scandir(base_formats_dir):
        if not item.is_file() or not item.name.endswith('.json'):
            continue
        format_files[item.name[:-5]] = item.path
    formats = []
    format_widgets = {}
    for format_name, path in format_files.items():
        with open(path, 'r') as stream:
            video_format = json.load(stream)
        if "gifski_pass" in video_format and gifski_path is None:
            #Skip format
            continue
        widgets = list(iterate_format(video_format))
        formats.append("video/" + format_name)
        if (len(widgets) > 0):
            format_widgets["video/"+ format_name] = widgets
    return formats, format_widgets

def apply_format_widgets(format_name, kwargs):
    if os.path.exists(os.path.join(base_formats_dir, format_name + ".json")):
        video_format_path = os.path.join(base_formats_dir, format_name + ".json")
    else:
        video_format_path = folder_paths.get_full_path("VHS_video_formats", format_name)
    with open(video_format_path, 'r') as stream:
        video_format = json.load(stream)
    for w in iterate_format(video_format):
        if w[0] not in kwargs:
            if len(w) > 2 and 'default' in w[2]:
                default = w[2]['default']
            else:
                if type(w[1]) is list:
                    default = w[1][0]
                else:
                    default = {"BOOLEAN": False, "INT": 0, "FLOAT": 0, "STRING": ""}[w[1]]
            kwargs[w[0]] = default
            logger.warn(f"Missing input for {w[0]} has been set to {default}")
    wit = iterate_format(video_format, False)
    for w in wit:
        while isinstance(w, list):
            if len(w) == 1:
                w = [Template(x).substitute(**kwargs) for x in w[0]]
                break
            elif isinstance(w[1], dict):
                w = w[1][str(kwargs[w[0]])]
            elif len(w) > 3:
                w = Template(w[3]).substitute(val=kwargs[w[0]])
            else:
                w = str(kwargs[w[0]])
        wit.send(w)
    return video_format

def tensor_to_int(tensor, bits):
    tensor = tensor.cpu().numpy() * (2**bits-1) + 0.5
    return np.clip(tensor, 0, (2**bits-1))
def tensor_to_shorts(tensor):
    return tensor_to_int(tensor, 16).astype(np.uint16)
def tensor_to_bytes(tensor):
    return tensor_to_int(tensor, 8).astype(np.uint8)

def ffmpeg_process(args, video_format, video_metadata, file_path, env):
    res = None
    frame_data = yield
    total_frames_output = 0
    if video_format.get('save_metadata', 'False') != 'False':
        os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
        metadata = json.dumps(video_metadata)
        metadata_path = os.path.join(folder_paths.get_temp_directory(), "metadata.txt")
        metadata = metadata.replace("\\","\\\\")
        metadata = metadata.replace(";","\\;")
        metadata = metadata.replace("#","\\#")
        metadata = metadata.replace("=","\\=")
        metadata = metadata.replace("\n","\\\n")
        metadata = "comment=" + metadata
        with open(metadata_path, "w") as f:
            f.write(";FFMETADATA1\n")
            f.write(metadata)
        m_args = args[:1] + ["-i", metadata_path] + args[1:] + ["-metadata", "creation_time=now"]
        with subprocess.Popen(m_args + [file_path], stderr=subprocess.PIPE,
                              stdin=subprocess.PIPE, env=env) as proc:
            try:
                while frame_data is not None:
                    proc.stdin.write(frame_data)
                    frame_data = yield
                    total_frames_output+=1
                proc.stdin.flush()
                proc.stdin.close()
                res = proc.stderr.read()
            except BrokenPipeError as e:
                err = proc.stderr.read()
                if os.path.exists(file_path):
                    raise Exception("An error occurred in the ffmpeg subprocess:\n" \
                            + err.decode(*ENCODE_ARGS))
                print(err.decode(*ENCODE_ARGS), end="", file=sys.stderr)
                logger.warn("An error occurred when saving with metadata")
    if res != b'':
        with subprocess.Popen(args + [file_path], stderr=subprocess.PIPE,
                              stdin=subprocess.PIPE, env=env) as proc:
            try:
                while frame_data is not None:
                    proc.stdin.write(frame_data)
                    frame_data = yield
                    total_frames_output+=1
                proc.stdin.flush()
                proc.stdin.close()
                res = proc.stderr.read()
            except BrokenPipeError as e:
                res = proc.stderr.read()
                raise Exception("An error occurred in the ffmpeg subprocess:\n" \
                        + res.decode(*ENCODE_ARGS))
    yield total_frames_output
    if len(res) > 0:
        print(res.decode(*ENCODE_ARGS), end="", file=sys.stderr)

def gifski_process(args, dimensions, video_format, file_path, env):
    frame_data = yield
    with subprocess.Popen(args + video_format['main_pass'] + ['-f', 'yuv4mpegpipe', '-'],
                          stderr=subprocess.PIPE, stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, env=env) as procff:
        with subprocess.Popen([gifski_path] + video_format['gifski_pass']
                              + ['-W', f'{dimensions[0]}', '-H', f'{dimensions[1]}']
                              + ['-q', '-o', file_path, '-'], stderr=subprocess.PIPE,
                              stdin=procff.stdout, stdout=subprocess.PIPE,
                              env=env) as procgs:
            try:
                while frame_data is not None:
                    procff.stdin.write(frame_data)
                    frame_data = yield
                procff.stdin.flush()
                procff.stdin.close()
                resff = procff.stderr.read()
                resgs = procgs.stderr.read()
                outgs = procgs.stdout.read()
            except BrokenPipeError as e:
                procff.stdin.close()
                resff = procff.stderr.read()
                resgs = procgs.stderr.read()
                raise Exception("An error occurred while creating gifski output\n" \
                        + "Make sure you are using gifski --version >=1.32.0\nffmpeg: " \
                        + resff.decode(*ENCODE_ARGS) + '\ngifski: ' + resgs.decode(*ENCODE_ARGS))
    if len(resff) > 0:
        print(resff.decode(*ENCODE_ARGS), end="", file=sys.stderr)
    if len(resgs) > 0:
        print(resgs.decode(*ENCODE_ARGS), end="", file=sys.stderr)
    if len(outgs) > 0:
        print(outgs.decode(*ENCODE_ARGS))

def to_pingpong(inp):
    if not hasattr(inp, "__getitem__"):
        inp = list(inp)
    yield from inp
    for i in range(len(inp)-2,0,-1):
        yield inp[i]

class VideoCombineEQX:
    @classmethod
    def INPUT_TYPES(s):
        ffmpeg_formats, format_widgets = get_video_formats()
        format_widgets["image/webp"] = [['lossless', "BOOLEAN", {'default': True}]]
        return {
            "required": {
                "images": (imageOrLatent,),
                "frame_rate": (
                    floatOrInt,
                    {"default": 8, "min": 1, "step": 1},
                ),
                "loop_count": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "filename_prefix": ("STRING", {"default": "AnimateDiff"}),
                "filename_suffix": ("STRING", {"default": ""}),
                "save_path": ("STRING", {"default": "", "multiline": False}),
                "format": (["image/gif", "image/webp"] + ffmpeg_formats, {'formats': format_widgets}),
                "pingpong": ("BOOLEAN", {"default": False}),
                "save_output": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "meta_batch": ("VHS_BatchManager",),
                "vae": ("VAE",),
            },
            "hidden": ContainsAll({
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID"
            }),
        }

    RETURN_TYPES = ("VHS_FILENAMES",)
    RETURN_NAMES = ("Filenames",)
    OUTPUT_NODE = True
    CATEGORY = "EQX/Video"
    FUNCTION = "combine_video"

    def combine_video(
        self,
        frame_rate: int,
        loop_count: int,
        images=None,
        latents=None,
        filename_prefix="AnimateDiff",
        filename_suffix="",
        save_path="",
        format="image/gif",
        pingpong=False,
        save_output=True,
        prompt=None,
        extra_pnginfo=None,
        audio=None,
        unique_id=None,
        manual_format_widgets=None,
        meta_batch=None,
        vae=None,
        **kwargs
    ):
        if latents is not None:
            images = latents
        if images is None:
            return ((save_output, []),)
        if vae is not None:
            if isinstance(images, dict):
                images = images['samples']
            else:
                vae = None

        if isinstance(images, torch.Tensor) and images.size(0) == 0:
            return ((save_output, []),)
        num_frames = len(images)
        pbar = ProgressBar(num_frames)
        if vae is not None:
            downscale_ratio = getattr(vae, "downscale_ratio", 8)
            width = images.size(-1)*downscale_ratio
            height = images.size(-2)*downscale_ratio
            frames_per_batch = (1920 * 1080 * 16) // (width * height) or 1
            def batched(it, n):
                while batch := tuple(itertools.islice(it, n)):
                    yield batch
            def batched_encode(images, vae, frames_per_batch):
                for batch in batched(iter(images), frames_per_batch):
                    image_batch = torch.from_numpy(np.array(batch))
                    yield from vae.decode(image_batch)
            images = batched_encode(images, vae, frames_per_batch)
            first_image = next(images)
            images = itertools.chain([first_image], images)
            while len(first_image.shape) > 3:
                first_image = first_image[0]
        else:
            first_image = images[0]
            images = iter(images)
        # get output information
        logger.info("💾 Saving video...")
        if save_path and save_path.strip():
            save_path_clean = save_path.strip()
            logger.info(f"📁 Path: {save_path_clean}")
            save_path_clean = os.path.expanduser(save_path_clean)
            save_path_clean = os.path.expandvars(save_path_clean)
            if not os.path.isabs(save_path_clean):
                save_path_clean = os.path.abspath(save_path_clean)
            os.makedirs(save_path_clean, exist_ok=True)
            output_dir = save_path_clean
            full_output_folder = output_dir
            filename = filename_prefix
            subfolder = ""
            import time
            timestamp = int(time.time())
            filename = f"{filename}_{timestamp:08d}"
        else:
            output_dir = (
                folder_paths.get_output_directory()
                if save_output
                else folder_paths.get_temp_directory()
            )
            logger.info(f"📁 Path: {output_dir}")
            (
                full_output_folder,
                filename,
                _,
                subfolder,
                _,
            ) = folder_paths.get_save_image_path(filename_prefix, output_dir)
        output_files = []

        metadata = PngInfo()
        video_metadata = {}
        if prompt is not None:
            metadata.add_text("prompt", json.dumps(prompt))
            video_metadata["prompt"] = json.dumps(prompt)
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                metadata.add_text(x, json.dumps(extra_pnginfo[x]))
                video_metadata[x] = extra_pnginfo[x]
            extra_options = extra_pnginfo.get('workflow', {}).get('extra', {})
        else:
            extra_options = {}
        metadata.add_text("CreationTime", datetime.datetime.now().isoformat(" ")[:19])

        if meta_batch is not None and unique_id in meta_batch.outputs:
            (counter, output_process) = meta_batch.outputs[unique_id]
        else:
            max_counter = 0
            if save_path and save_path.strip():
                if filename_suffix:
                    matcher = re.compile(f"{re.escape(filename_prefix)}_\\d{{8,10}}_(\\d{{5}})_{re.escape(filename_suffix)}\\..+", re.IGNORECASE)
                else:
                    matcher = re.compile(f"{re.escape(filename_prefix)}_\\d{{8,10}}_(\\d{{5}})\\..+", re.IGNORECASE)
            else:
                matcher = re.compile(f"{re.escape(filename)}_(\\d+)\\D*\\..+", re.IGNORECASE)

            for existing_file in os.listdir(full_output_folder):
                match = matcher.fullmatch(existing_file)
                if match:
                    file_counter = int(match.group(1))
                    if file_counter > max_counter:
                        max_counter = file_counter

            counter = max_counter + 1
            output_process = None

        format_type, format_ext = format.split("/")
        if format_type == "image":
            if meta_batch is not None:
                raise Exception("Pillow('image/') formats are not compatible with batched output")
            image_kwargs = {}
            if format_ext == "gif":
                image_kwargs['disposal'] = 2
            if format_ext == "webp":
                exif = Image.Exif()
                exif[ExifTags.IFD.Exif] = {36867: datetime.datetime.now().isoformat(" ")[:19]}
                image_kwargs['exif'] = exif
                image_kwargs['lossless'] = kwargs.get("lossless", True)
            if filename_suffix:
                file = f"{filename}_{counter:05}_{filename_suffix}.{format_ext}"
            else:
                file = f"{filename}_{counter:05}.{format_ext}"
            file_path = os.path.join(full_output_folder, file)
            if pingpong:
                images = to_pingpong(images)
            def frames_gen(images):
                for i in images:
                    pbar.update(1)
                    yield Image.fromarray(tensor_to_bytes(i))
            frames = frames_gen(images)
            next(frames).save(
                file_path,
                format=format_ext.upper(),
                save_all=True,
                append_images=frames,
                duration=round(1000 / frame_rate),
                loop=loop_count,
                compress_level=4,
                **image_kwargs
            )
            output_files.append(file_path)
            logger.info(f"📝 File: {file}")
            logger.info("✅ Video saved successfully!")
        else:
            if ffmpeg_path is None:
                raise ProcessLookupError(f"ffmpeg is required for video outputs and could not be found.\nIn order to use video outputs, you must either:\n- Install imageio-ffmpeg with pip,\n- Place a ffmpeg executable in {os.path.abspath('')}, or\n- Install ffmpeg and add it to the system path.")

            if manual_format_widgets is not None:
                logger.warn("Format args can now be passed directly. The manual_format_widgets argument is now deprecated")
                kwargs.update(manual_format_widgets)

            has_alpha = first_image.shape[-1] == 4
            kwargs["has_alpha"] = has_alpha
            video_format = apply_format_widgets(format_ext, kwargs)
            dim_alignment = video_format.get("dim_alignment", 2)
            if (first_image.shape[1] % dim_alignment) or (first_image.shape[0] % dim_alignment):
                to_pad = (-first_image.shape[1] % dim_alignment,
                          -first_image.shape[0] % dim_alignment)
                padding = (to_pad[0]//2, to_pad[0] - to_pad[0]//2,
                           to_pad[1]//2, to_pad[1] - to_pad[1]//2)
                padfunc = torch.nn.ReplicationPad2d(padding)
                def pad(image):
                    image = image.permute((2,0,1))
                    padded = padfunc(image.to(dtype=torch.float32))
                    return padded.permute((1,2,0))
                images = map(pad, images)
                dimensions = (-first_image.shape[1] % dim_alignment + first_image.shape[1],
                              -first_image.shape[0] % dim_alignment + first_image.shape[0])
                logger.warn("Output images were not of valid resolution and have had padding applied")
            else:
                dimensions = (first_image.shape[1], first_image.shape[0])
            if pingpong:
                if meta_batch is not None:
                    logger.error("pingpong is incompatible with batched output")
                images = to_pingpong(images)
                if num_frames > 2:
                    num_frames += num_frames -2
                    pbar.total = num_frames
            if loop_count > 0:
                loop_args = ["-vf", "loop=loop=" + str(loop_count)+":size=" + str(num_frames)]
            else:
                loop_args = []
            if video_format.get('input_color_depth', '8bit') == '16bit':
                images = map(tensor_to_shorts, images)
                if has_alpha:
                    i_pix_fmt = 'rgba64'
                else:
                    i_pix_fmt = 'rgb48'
            else:
                images = map(tensor_to_bytes, images)
                if has_alpha:
                    i_pix_fmt = 'rgba'
                else:
                    i_pix_fmt = 'rgb24'
            has_audio = False
            if audio is not None:
                try:
                    if audio.get('waveform') is not None:
                        has_audio = True
                except:
                    pass

            if has_audio:
                file = f"{filename}_{counter:05}_temp.{video_format['extension']}"
            else:
                if filename_suffix:
                    file = f"{filename}_{counter:05}_{filename_suffix}.{video_format['extension']}"
                else:
                    file = f"{filename}_{counter:05}.{video_format['extension']}"
            file_path = os.path.join(full_output_folder, file)
            bitrate_arg = []
            bitrate = video_format.get('bitrate')
            if bitrate is not None:
                bitrate_arg = ["-b:v", str(bitrate) + "M" if video_format.get('megabit') == 'True' else str(bitrate) + "K"]
            args = [ffmpeg_path, "-v", "error", "-f", "rawvideo", "-pix_fmt", i_pix_fmt,
                    "-color_range", "pc", "-colorspace", "rgb", "-color_primaries", "bt709",
                    "-color_trc", video_format.get("fake_trc", "iec61966-2-1"),
                    "-s", f"{dimensions[0]}x{dimensions[1]}", "-r", str(frame_rate), "-i", "-"] \
                    + loop_args

            images = map(lambda x: x.tobytes(), images)
            env=os.environ.copy()
            if  "environment" in video_format:
                env.update(video_format["environment"])

            if "pre_pass" in video_format:
                if meta_batch is not None:
                    raise Exception("Formats which require a pre_pass are incompatible with Batch Manager.")
                images = [b''.join(images)]
                os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
                in_args_len = args.index("-i") + 2
                pre_pass_args = args[:in_args_len] + video_format['pre_pass']
                merge_filter_args(pre_pass_args)
                try:
                    subprocess.run(pre_pass_args, input=images[0], env=env,
                                   capture_output=True, check=True)
                except subprocess.CalledProcessError as e:
                    raise Exception("An error occurred in the ffmpeg prepass:\n" \
                            + e.stderr.decode(*ENCODE_ARGS))
            if "inputs_main_pass" in video_format:
                in_args_len = args.index("-i") + 2
                args = args[:in_args_len] + video_format['inputs_main_pass'] + args[in_args_len:]

            if output_process is None:
                if 'gifski_pass' in video_format:
                    format = 'image/gif'
                    output_process = gifski_process(args, dimensions, video_format, file_path, env)
                    audio = None
                else:
                    args += video_format['main_pass'] + bitrate_arg
                    merge_filter_args(args)
                    output_process = ffmpeg_process(args, video_format, video_metadata, file_path, env)
                output_process.send(None)
                if meta_batch is not None:
                    meta_batch.outputs[unique_id] = (counter, output_process)

            for image in images:
                pbar.update(1)
                output_process.send(image)
            if meta_batch is not None:
                requeue_workflow((meta_batch.unique_id, not meta_batch.has_closed_inputs))
            if meta_batch is None or meta_batch.has_closed_inputs:
                try:
                    total_frames_output = output_process.send(None)
                    output_process.send(None)
                except StopIteration:
                    pass
                if meta_batch is not None:
                    meta_batch.outputs.pop(unique_id)
                    if len(meta_batch.outputs) == 0:
                        meta_batch.reset()
            else:
                return {"ui": {"unfinished_batch": [True]}, "result": ((save_output, []),)}

            a_waveform = None
            if audio is not None:
                try:
                    a_waveform = audio['waveform']
                except:
                    pass

            if a_waveform is not None:
                if filename_suffix:
                    output_file_with_audio = f"{filename}_{counter:05}_{filename_suffix}.{video_format['extension']}"
                else:
                    output_file_with_audio = f"{filename}_{counter:05}.{video_format['extension']}"
                output_file_with_audio_path = os.path.join(full_output_folder, output_file_with_audio)
                if "audio_pass" not in video_format:
                    logger.warn("Selected video format does not have explicit audio support")
                    video_format["audio_pass"] = ["-c:a", "libopus"]

                channels = audio['waveform'].size(1)
                min_audio_dur = total_frames_output / frame_rate + 1
                if video_format.get('trim_to_audio', 'False') != 'False':
                    apad = []
                else:
                    apad = ["-af", "apad=whole_dur="+str(min_audio_dur)]
                mux_args = [ffmpeg_path, "-v", "error", "-y", "-i", file_path,
                            "-ar", str(audio['sample_rate']), "-ac", str(channels),
                            "-f", "f32le", "-i", "-", "-c:v", "copy"] \
                            + video_format["audio_pass"] \
                            + apad + ["-shortest", output_file_with_audio_path]

                audio_data = audio['waveform'].squeeze(0).transpose(0,1) \
                        .numpy().tobytes()
                merge_filter_args(mux_args, '-af')
                try:
                    res = subprocess.run(mux_args, input=audio_data,
                                         env=env, capture_output=True, check=True)
                except subprocess.CalledProcessError as e:
                    raise Exception("An error occured in the ffmpeg subprocess:\n" \
                            + e.stderr.decode(*ENCODE_ARGS))
                if res.stderr:
                    print(res.stderr.decode(*ENCODE_ARGS), end="", file=sys.stderr)
                output_files.append(output_file_with_audio_path)
                file = output_file_with_audio
                logger.info(f"📝 File: {output_file_with_audio}")
                logger.info("✅ Video saved successfully!")
                if os.path.exists(file_path):
                    os.remove(file_path)
            else:
                output_files.append(file_path)
                logger.info(f"📝 File: {file}")
                logger.info("✅ Video saved successfully!")
        if extra_options.get('VHS_KeepIntermediate', True) == False:
            for intermediate in output_files[1:-1]:
                if os.path.exists(intermediate):
                    os.remove(intermediate)
        preview = {
                "filename": file,
                "subfolder": subfolder,
                "type": "output" if save_output else "temp",
                "format": format,
                "frame_rate": frame_rate,
                "fullpath": output_files[-1] if output_files else "",
            }
        if num_frames == 1 and 'png' in format and '%03d' in file:
            preview['format'] = 'image/png'
            preview['filename'] = file.replace('%03d', '001')
        return {"ui": {"gifs": [preview]}, "result": ((save_output, output_files),)}


NODE_CLASS_MAPPINGS = {
    "SaveVideoEQX": VideoCombineEQX,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveVideoEQX": "Save Video EQX",
}