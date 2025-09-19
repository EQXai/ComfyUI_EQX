# ComfyUI EQX

## Node overview

| Node (display name) | Category | Brief description |
|---------------------|----------|-------------------|
| **SaveImage _EQX** | Save | Saves images with dynamic file-names, DPI/quality settings, embeds the workflow as PNG metadata and keeps a history log. |
| **File Image Selector** | Load | Picks an image from a folder in *random*, *incremental* or fixed-index mode and returns the tensor + filename. |
| **Load Prompt From File – EQX** | Load | Reads a text file containing prompts and outputs `id`, `prompt`, `negative_prompt` and a rolling seed. |
| **Lora Stack EQX (Random)** | LORA | Randomly selects several LoRA files from (sub)folders and builds a stack with the given strengths—perfect for style mixing. |
| **Extract Filename EQX** | Text | Extracts the plain filename (no path) from a string. |
| **Extract LORA name EQX** | Text | Finds `.safetensors` filenames inside a string and returns them in three formats: dash-separated, underscore-separated and list. |
| **NSFW Detector EQX** | Detection | Uses CLIP Safety Checker + NudeNet to flag NSFW content. Returns the original image and a boolean `nsfw`. |
| **NSFW Detector Advanced EQX** | Detection | Advanced NudeNet classification: can process folders, filter allowed categories and move/save images based on the result. |
| **Load RetinaFace EQX** | Detection | Downloads/loads the RetinaFace model (FaceX Lib) for downstream face-detection nodes. |
| **Count Faces EQX** | Detection | Counts faces in an image with RetinaFace and optionally saves images into sub-folders by face-count. |
| **Face Crop & Mask EQX** | Detection | Detects faces and creates cropped face images with masks, supporting padding, square crops and max resolution output. |
| **Body Crop & Mask EQX** | Detection | Detects human bodies using NudeNet and creates cropped body images with masks, supporting padding and square crops. |
| **Resolution Selector EQX** | Image | Provides preset resolutions with automatic width/height swap based on orientation, perfect for SDXL and other models. |
| **Aspect Ratio Crop EQX** | Image | Crops images to specific aspect ratios (16:9, 4:3, 1:1, etc.) with multiple alignment options (center, top, bottom, etc.). |
| **Uncrop by Mask EQX** | Image | Expands cropped images back to original dimensions using mask positioning information. |
| **Image Duplicate Remover EQX** | Image | Removes duplicate images from batches using perceptual hashing and similarity thresholds. |
| **Batch Image Trimmer EQX** | Image | Trims transparent/white/black borders from batches of images with configurable thresholds. |
| **Batch Image Trimmer Multi EQX** | Image | Advanced batch trimmer with multiple trim modes, edge detection options and tolerance settings. |
| **Video Combine Nodes** | Video | Combines image sequences into video files with configurable codecs, framerates and quality settings. |
| **Video Fragments Node** | Video | Splits videos into fragments for processing, supporting overlap and various extraction modes. |
| **Video Fragments Splitter** | Video | Splits video batches into smaller segments for parallel processing or memory management. |
| **WorkFlow Check** | Logic | Validates workflow integrity and checks for missing nodes, connections or configuration issues. |

---
