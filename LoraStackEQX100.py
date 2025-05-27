# This file is intentionally left blank. The actual content is in the original file. 

import folder_paths
import random
import time

class EQX_LoraStack100:
    FIXED_SELECTION_100 = None
    INTERNAL_SEED_100 = int(time.time() * 1000) # Keep this multiplication factor for seed generation

    UsedLorasMap100 = {}
    StridesMap100 = {}
    LastHashMap100 = {}

    @classmethod
    def INPUT_TYPES(cls):
        loras = ["None"] + folder_paths.get_filename_list("loras")

        required_inputs = {
            "selection_fix": ("BOOLEAN", {"default": False}),
            "exclusive_mode": (["Off", "On"],),
            "stride": (("INT", {"default": 1, "min": 1, "max": 100})), # Max stride changed to 100
            "force_randomize_after_stride": (["Off", "On"],),
        }
        for i in range(1, 101): # Range changed to 101
            required_inputs[f"lora_name_{i}"] = (loras,)
            required_inputs[f"switch_{i}"] = (["Off", "On"],)

        return {
            "required": required_inputs,
            "optional": {"lora_stack": ("LORA_STACK",)}
        }

    RETURN_TYPES = ("LORA_STACK",)
    FUNCTION = "random_lora_stacker_100" # Function name changed
    CATEGORY = "Comfyroll/LoRA - 100 Slots" # Category name changed

    DEFAULT_CHANCE = 1.0
    DEFAULT_MODEL_WEIGHT = 0.3
    DEFAULT_CLIP_WEIGHT = 1.0
    
    @staticmethod
    def getIdHash100(*lora_names) -> int: # Method name changed
        return hash(frozenset(set(lora_names)))

    @staticmethod
    def deduplicateLoraNames100(*lora_names): # Method name changed
        lora_names = list(lora_names)
        name_counts = {}
        for i, name in enumerate(lora_names):
            if name != "None":
                count = name_counts.get(name, 0)
                if count > 0:
                    lora_names[i] = f"{name}EQX_LoraStack100_{count+1}" # Suffix changed
                name_counts[name] = count + 1
        return lora_names

    @staticmethod
    def cleanLoraName100(lora_name) -> str: # Method name changed
        import re
        return re.sub(r'EQX_LoraStack100_\d+', '', lora_name) # Suffix changed in regex

    @classmethod
    def IS_CHANGED(cls, selection_fix, exclusive_mode, stride, force_randomize_after_stride, **kwargs):
        cls.INTERNAL_SEED_100 = (cls.INTERNAL_SEED_100 * 1664525 + 1013904223) % 2**32
        if not selection_fix or cls.FIXED_SELECTION_100 is None:
            new_selection = cls._generate_new_selection_100(exclusive_mode, stride, force_randomize_after_stride, **kwargs) # Method call changed
            cls.FIXED_SELECTION_100 = new_selection
            return new_selection
        return cls.FIXED_SELECTION_100

    @classmethod
    def _generate_new_selection_100(cls, exclusive_mode, stride, force_randomize_after_stride, **kwargs): # Method name changed
        import random
        random.seed(cls.INTERNAL_SEED_100)
        
        num_loras = 100 # num_loras changed
        lora_names = []
        switches = []
        chances = []

        for i in range(1, num_loras + 1):
            lora_name = kwargs.get(f"lora_name_{i}")
            switch = kwargs.get(f"switch_{i}")
            lora_names.append(lora_name)
            switches.append(switch)
            chances.append(cls.DEFAULT_CHANCE)

        total_on = sum(
            1 for name, sw, chance in zip(lora_names, switches, chances)
            if name != "None" and sw == "On" and chance > 0.0
        )

        # Desduplicar
        lora_names = cls.deduplicateLoraNames100(*lora_names) # Method call changed
        id_hash = cls.getIdHash100(*lora_names) # Method call changed

        if id_hash not in cls.StridesMap100:
            cls.StridesMap100[id_hash] = 0
        cls.StridesMap100[id_hash] += 1

        if stride > 1 and cls.StridesMap100[id_hash] < stride and id_hash in cls.LastHashMap100:
            return cls.LastHashMap100[id_hash]
        else:
            cls.StridesMap100[id_hash] = 0

        def perform_randomization() -> set:
            _lora_set = set()
            random_values = [random.random() for _ in range(num_loras)]
            applies = [(random_values[i] <= chances[i]) and (switches[i] == "On") for i in range(num_loras)]
            
            indices = [i for i, apply in enumerate(applies) if apply]

            if exclusive_mode == "On" and len(indices) > 1:
                min_index = min(indices, key=lambda idx: random_values[idx])
                applies = [False] * num_loras
                applies[min_index] = True
            else:
                if len(indices) > 5:
                    selected_indices = sorted(indices, key=lambda i: random_values[i])[:5]
                    applies = [i in selected_indices for i in range(num_loras)]

            for i in range(num_loras):
                if lora_names[i] != "None" and applies[i]:
                    _lora_set.add(lora_names[i])

            return _lora_set

        last_lora_set = cls.UsedLorasMap100.get(id_hash, set())
        lora_set = perform_randomization()

        if force_randomize_after_stride == "On" and len(last_lora_set) > 0 and total_on > 1:
            while lora_set == last_lora_set:
                lora_set = perform_randomization()

        cls.UsedLorasMap100[id_hash] = lora_set
        hash_str = str(hash(frozenset(lora_set)))
        cls.LastHashMap100[id_hash] = hash_str
        
        return hash_str

    def random_lora_stacker_100( # Method name changed
        self,
        selection_fix,
        exclusive_mode,
        stride,
        force_randomize_after_stride,
        **kwargs
    ):
        lora_list = []

        lora_stack = kwargs.get('lora_stack', None)
        if lora_stack is not None:
            lora_list.extend([l for l in lora_stack if l[0] != "None"])

        num_loras = 100 # num_loras changed
        lora_names = []
        switches = []

        for i in range(1, num_loras + 1):
            lora_name = kwargs.get(f"lora_name_{i}")
            switch = kwargs.get(f"switch_{i}")
            lora_names.append(lora_name)
            switches.append(switch)

        lora_names = self.deduplicateLoraNames100(*lora_names) # Method call changed
        id_hash = self.getIdHash100(*lora_names) # Method call changed

        used_loras = self.UsedLorasMap100.get(id_hash, set())

        for i in range(num_loras):
            if (
                lora_names[i] != "None"
                and switches[i] == "On"
                and lora_names[i] in used_loras
            ):
                lora_list.append((
                    self.cleanLoraName100(lora_names[i]), # Method call changed
                    self.DEFAULT_MODEL_WEIGHT,
                    self.DEFAULT_CLIP_WEIGHT
                ))

        return (lora_list,)

NODE_CLASS_MAPPINGS = {
    "EQX_LoraStack100": EQX_LoraStack100 # Class name in mapping changed
} 