"""Describe the desired appearance of semantic classes with a small local LLM.

JSON constraints enforce structure, not visual fidelity or absence of hallucinations.
Heavy dependencies and model weights are loaded only when describe() is called.
"""
import gc
import json

if __package__:
    from .prompt_builder import STYLE_PROMPTS, normalize_style
else:
    from prompt_builder import STYLE_PROMPTS, normalize_style

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


STYLE_DIRECTIONS = {
    "cyberpunk": (
        "Futuristic cyberpunk rendering: high tech, low life, neon glow, "
        "cyan and magenta accents, deep violet shadows, "
        "emissive edge lighting and high contrast. Apply these to existing surfaces."
    ),
    "gta v": (
        "GTA V in-game open-world rendering, not illustrated loading-screen artwork: "
        "sunlit natural colors, warm golden light, crisp game textures, realistic "
        "materials and atmospheric haze. Keep natural surfaces non-emissive."
    ),
    "pixel art": (
        "Pixel art rendering: 8-bit"
        "or 16-bit video game style, limited palette,"
        "blocky shapes, dithering, and pixel-level detail. Apply these to existing surfaces."
    ),
}


def build_style_messages(names, style):
    """Condition on only the selected artistic target; avoid cross-style examples."""
    style_key = normalize_style(style)
    return [
        {"role": "system", "content": (
            "You art-direct landscape images. Return only the requested JSON. "
            "Write short English visual phrases: global describes the shared style, palette "
            "and lighting in 4-8 words; each class gets 3-6 words describing its desired "
            "colors, surface or illumination. Apply the supplied art_direction to EVERY "
            "class, not just global. Make material-specific descriptions consistent with "
            "that visual target. Use one coherent palette. Do not repeat class names "
            "inside descriptions. Preserve each object's identity. Do not introduce "
            "objects, buildings, vehicles, people, counts or spatial relations not supplied "
            "by the input. These are desired appearances, not observations."
        )},
        {"role": "user", "content": json.dumps({
            "style": STYLE_PROMPTS[style_key],
            "art_direction": STYLE_DIRECTIONS[style_key],
            "classes": names,
            "output_schema": style_schema(names),
        })},
    ]


def style_schema(names):
    phrase = {"type": "string", "minLength": 1, "maxLength": 100}
    return {
        "type": "object",
        "properties": {
            "global": phrase,
            "classes": {
                "type": "object",
                "properties": {name: phrase for name in names},
                "required": list(names),
                "additionalProperties": False,
            },
        },
        "required": ["global", "classes"],
        "additionalProperties": False,
    }


def validate_description(value, names):
    if not isinstance(value, dict) or set(value) != {"global", "classes"}:
        raise ValueError("Expected only global and classes fields.")
    classes = value["classes"]
    if not isinstance(classes, dict) or set(classes) != set(names):
        raise ValueError("The JSON class keys must exactly match the selected classes.")
    for phrase in [value["global"], *classes.values()]:
        if not isinstance(phrase, str) or not phrase.strip() or len(phrase) > 100:
            raise ValueError("Descriptions must be nonempty strings of at most 100 characters.")
    return {
        "global": " ".join(value["global"].split()),
        "classes": {name: " ".join(classes[name].split()) for name in names},
    }


class ClassStyleDescriber:
    def __init__(self, model_id=MODEL_ID, device=None):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.tokenizer = None

    def describe(self, features, style):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

        style_key = normalize_style(style)
        names = [feature.name for feature in features]
        if len(names) != len(set(names)):
            raise ValueError("Selected class names must be unique.")
        if not names:
            return {"global": STYLE_PROMPTS[style_key], "classes": {}}
        if self.model is None:
            self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                dtype=torch.float16 if self.device.startswith("cuda") else torch.float32,
            ).to(self.device).eval()

        schema = style_schema(names)
        messages = build_style_messages(names, style_key)
        chat = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(chat, return_tensors="pt").to(self.device)
        prefix = build_transformers_prefix_allowed_tokens_fn(self.tokenizer, JsonSchemaParser(schema))
        # A fresh config avoids inherited sampling flags during greedy decoding.
        config = GenerationConfig(
            do_sample=False, max_new_tokens=128 + 64 * len(names),
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, generation_config=config, prefix_allowed_tokens_fn=prefix,
            )
        raw = self.tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        try:
            return validate_description(json.loads(raw), names)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Invalid or incomplete style JSON (possibly token limit). {exc} Output: {raw[:800]!r}"
            ) from exc

    def close(self):
        """Release weights before loading Stable Diffusion on Kaggle."""
        import torch
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def compose_prompt(description, features, style, tokenizer):
    """Fit CLIP's actual token budget, retaining every selected class name.

    Trim trailing description words starting with the least-covered class.
    Return the effective descriptions too, so reductions are inspectable.
    """
    names = [feature.name for feature in features]
    value = validate_description(description, names)
    style_text = STYLE_PROMPTS[normalize_style(style)]
    global_words = value["global"].split()
    words = {name: value["classes"][name].split() for name in names}

    def render():
        clauses = [style_text, " ".join(global_words)]
        clauses.extend(name + (" with " + " ".join(words[name]) if words[name] else "") for name in names)
        return ", ".join(clause for clause in clauses if clause)

    def count(prompt):
        return len(tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"])

    limit = tokenizer.model_max_length
    prompt = render()
    while count(prompt) > limit:
        target = next((name for name in reversed(names) if words[name]), None)
        if target is not None:
            words[target].pop()
        elif global_words:
            global_words.pop()
        else:
            raise ValueError("Class names alone exceed SD's token budget; reduce max_classes.")
        prompt = render()
    used = {"global": " ".join(global_words), "classes": {name: " ".join(words[name]) for name in names}}
    return prompt, {"used_descriptions": used, "token_count": count(prompt),
                    "token_limit": limit, "shortened": used != value}
