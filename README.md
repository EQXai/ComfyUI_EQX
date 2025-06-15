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

---

Enjoy creating with **ComfyUI EQX** 🎨 
