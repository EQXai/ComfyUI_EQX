import os
import itertools
import re
import hashlib
import random

class PromptConcatenateUnified:
    """Single node that builds positive and negative prompts from text files.

    The *base_dir* directory must contain two subfolders named **positive**
    and **negative**.  Each `.txt` file inside these subfolders is treated as
    a group: one random non-empty line is sampled from every file and all the
    selected lines are concatenated.

    A single *seed* drives the process.  For each file we derive an
    independent sub-seed based on the file name and the base seed; this keeps
    the overall result deterministic while avoiding the "same index for every
    file" problem.

    Parameters
    ----------
    base_dir : str
        Folder that contains the `positive/` and `negative/` subfolders.
    positive_joiner : str
        String used to join positive parts.
    negative_joiner : str
        String used to join negative parts.
    seed : int
        Seed used for deterministic randomness.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_dir": ("STRING", {"multiline": False}),
                "positive_joiner": ("STRING", {"default": " ", "multiline": False}),
                "negative_joiner": ("STRING", {"default": " ", "multiline": False}),
                "seed": ("INT", {"default": 42, "min": 0, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = (
        "positive_prompt",
        "negative_prompt",
        "positive_indices",
        "negative_indices",
        "seed_used",
    )
    FUNCTION = "assemble"
    CATEGORY = "Prompt/Modular"
    NODE_NAME = "Prompt Concatenate Unified - EQX"

    def _assemble_from_dir(self, dir_path: str, joiner: str, base_seed: int):
        """Return *(prompt_str, indices_list)*.

        *prompt_str* is built by sampling one random non-empty line from each
        `.txt` file inside *dir_path* and joining them with *joiner*.

        *indices_list* contains the 0-based index of the chosen line for each
        file, following the same alphabetical order used to sort *txt_files*.
        """
        path = os.path.expanduser(dir_path)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"No directory: {path}")

        txt_files = sorted([
            f for f in os.listdir(path)
            if f.lower().endswith(".txt")
        ])
        if not txt_files:
            raise ValueError(f"Directory '{path}' has no .txt files")

        parts = []
        sel_indices = []
        for fname in txt_files:
            fpath = os.path.join(path, fname)
            with open(fpath, encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            if not lines:
                continue  # skip empty file

            # Derive a per-file sub-seed from the file name and the base seed
            digest = hashlib.sha256((fname + str(base_seed)).encode()).digest()
            sub_seed = int.from_bytes(digest[:8], "big")
            rng = random.Random(sub_seed)

            idx = rng.randrange(len(lines))
            parts.append(lines[idx])
            sel_indices.append(idx)

        return joiner.join(parts), sel_indices

    def assemble(self, base_dir: str, positive_joiner: str, negative_joiner: str, seed: int):
        import os

        positive_dir = os.path.join(os.path.expanduser(base_dir), "positive")
        negative_dir = os.path.join(os.path.expanduser(base_dir), "negative")

        if not os.path.isdir(positive_dir):
            raise FileNotFoundError(f"Missing subfolder 'positive' inside {base_dir}")
        if not os.path.isdir(negative_dir):
            raise FileNotFoundError(f"Missing subfolder 'negative' inside {base_dir}")

        pos_prompt, pos_idx = self._assemble_from_dir(positive_dir, positive_joiner, seed)
        neg_prompt, neg_idx = self._assemble_from_dir(negative_dir, negative_joiner, seed)

        pos_idx_str = ",".join(map(str, pos_idx))
        neg_idx_str = ",".join(map(str, neg_idx))

        return (
            pos_prompt,
            neg_prompt,
            pos_idx_str,
            neg_idx_str,
            seed,
        )


# ----------------------------------------------------------------------
# Node registration is handled in __init__.py so that the implementation
# can live in its own module without boilerplate duplication.
# ---------------------------------------------------------------------- 