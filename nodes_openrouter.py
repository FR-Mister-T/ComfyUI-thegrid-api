import base64
import json
import time

import requests

from ._utils import retry_request, tensor_to_base64, bytes_to_tensor, url_to_tensor, extract_chat_text

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

MODALITY_OPTIONS = ["all", "text", "image", "video", "audio"]

_FALLBACK_LIMIT = 10  # models returned from unfiltered list when a filter yields nothing

_MODELS_CACHE: dict = {"data": None, "ts": 0.0}
_CACHE_TTL = 300.0


# ---------------------------------------------------------------------------
# Model fetching / filtering
# ---------------------------------------------------------------------------

def _fetch_models() -> list:
    now = time.time()
    if _MODELS_CACHE["data"] is not None and (now - _MODELS_CACHE["ts"]) < _CACHE_TTL:
        return _MODELS_CACHE["data"]
    try:
        resp = requests.get(f"{OPENROUTER_BASE}/models", timeout=10)
        if resp.status_code == 200:
            models = resp.json().get("data") or []
            _MODELS_CACHE["data"] = models
            _MODELS_CACHE["ts"] = now
            return models
    except Exception:
        pass
    return _MODELS_CACHE["data"] or []


def _is_free(model: dict) -> bool:
    pricing = model.get("pricing") or {}
    return str(pricing.get("prompt", "1")).strip() in ("0", "0.0", "0.00")


def _output_modality(model: dict) -> str:
    arch = model.get("architecture") or {}
    # Prefer the explicit output_modalities array when present
    out_mods = arch.get("output_modalities") or []
    if out_mods:
        return " ".join(str(m) for m in out_mods).lower()
    # Fall back to the "input->output" modality string
    modality = arch.get("modality") or ""
    if "->" in modality:
        return modality.split("->", 1)[1].strip().lower()
    return modality.lower()


def _model_ids(
    free_only: bool = False,
    modality_filter: str = "all",
    filter_text: str = "",
    limit: int = 200,
) -> list:
    all_models = _fetch_models()
    models = all_models

    if free_only:
        models = [m for m in models if _is_free(m)]

    if modality_filter == "text":
        models = [m for m in models if "text" in _output_modality(m)]
    elif modality_filter == "image":
        models = [m for m in models if "image" in _output_modality(m)]
    elif modality_filter == "video":
        models = [m for m in models if "video" in _output_modality(m)]
    elif modality_filter == "audio":
        models = [m for m in models if "audio" in _output_modality(m)]

    if filter_text:
        ft = filter_text.lower()
        models = [m for m in models if ft in (m.get("id") or "").lower()]

    ids = [m["id"] for m in models if m.get("id")]
    if limit:
        ids = ids[:limit]

    if not ids:
        # Filters matched nothing — return first N real models from the live list
        fallback = [m["id"] for m in all_models if m.get("id")]
        return fallback[:_FALLBACK_LIMIT]

    return ids


# ---------------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------------

def _raise_openrouter_error(resp):
    try:
        err = resp.json().get("error") or {}
        message  = err.get("message") or resp.text
        meta     = err.get("metadata") or {}
        provider = meta.get("provider_name", "")
        provider_tag = f" [{provider}]" if provider else ""
        raise RuntimeError(f"OpenRouter error {resp.status_code}{provider_tag}: {message}")
    except (ValueError, KeyError):
        raise RuntimeError(f"OpenRouter error {resp.status_code}: {resp.text}")


# ---------------------------------------------------------------------------
# Image response helpers
# ---------------------------------------------------------------------------

def _load_image(url_or_data_uri: str):
    if url_or_data_uri.startswith("data:"):
        _, encoded = url_or_data_uri.split(",", 1)
        return bytes_to_tensor(base64.b64decode(encoded))
    return url_to_tensor(url_or_data_uri)


def _extract_image_from_response(data: dict):
    choices = data.get("choices") or []
    if choices:
        content = choices[0].get("message", {}).get("content")
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "image_url":
                    url = (item.get("image_url") or {}).get("url", "")
                    if url:
                        return _load_image(url)
                # Some providers embed base64 directly in a content block
                if item.get("type") == "image" and item.get("source"):
                    src = item["source"]
                    if src.get("type") == "base64":
                        return bytes_to_tensor(base64.b64decode(src["data"]))
                    if src.get("type") == "url":
                        return url_to_tensor(src["url"])
        elif isinstance(content, str):
            if content.startswith("data:"):
                return _load_image(content)
            # Plain URL string (many image-gen models return just a URL)
            stripped = content.strip()
            if stripped.startswith("http://") or stripped.startswith("https://"):
                return url_to_tensor(stripped)

    # DALL-E / OpenAI images endpoint style
    images = data.get("data") or []
    if images:
        entry = images[0]
        if "b64_json" in entry:
            return bytes_to_tensor(base64.b64decode(entry["b64_json"]))
        if "url" in entry:
            return url_to_tensor(entry["url"])

    # Build a diagnostic that reveals the actual content shape
    diag_parts = [f"Top-level keys: {list(data.keys())}"]
    if choices:
        msg = choices[0].get("message", {})
        c = msg.get("content")
        if isinstance(c, list):
            diag_parts.append(f"choices[0].message.content is a list of {len(c)} block(s) with types: {[i.get('type') for i in c]}")
        elif isinstance(c, str):
            diag_parts.append(f"choices[0].message.content is a string starting with: {c[:120]!r}")
        else:
            diag_parts.append(f"choices[0].message.content type: {type(c).__name__}")
        diag_parts.append(f"finish_reason: {choices[0].get('finish_reason')!r}")

    raise RuntimeError(
        "Could not extract an image from the API response. "
        + " | ".join(diag_parts)
        + " — enable debug=True to print the full response."
    )


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

class OpenRouterModelListNode:
    CATEGORY = "OpenRouter"

    @classmethod
    def INPUT_TYPES(cls):
        all_models = _model_ids(limit=500)
        return {
            "required": {
                "model_select": (all_models, {
                    "tooltip": (
                        "Available models. "
                        "Adjust the filters below — the list updates automatically."
                    ),
                }),
                "free_only": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Show only zero-cost models",
                }),
                "modality_filter": (MODALITY_OPTIONS, {
                    "default": "all",
                    "tooltip": "Filter by output modality",
                }),
                "filter_text": ("STRING", {
                    "default": "",
                    "tooltip": "Substring filter on model ID (e.g. 'claude', 'flux', 'gpt-4o')",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("model_name", "all_matches")
    FUNCTION = "execute"

    def execute(self, model_select, free_only, modality_filter, filter_text):
        filtered = _model_ids(
            free_only=free_only,
            modality_filter=modality_filter,
            filter_text=filter_text,
            limit=500,
        )
        if not filtered:
            raise RuntimeError(
                "No models matched the current filters. "
                "Try relaxing free_only or modality_filter."
            )

        # Use the COMBO selection if it passes the active filters; otherwise fall back to first match
        if model_select in filtered:
            model_name = model_select
        else:
            model_name = filtered[0]
            print(
                f"[OpenRouterModelList] '{model_select}' is not in the filtered list "
                f"— using '{model_name}' instead."
            )

        return (model_name, "\n".join(filtered))


class OpenRouterChatNode:
    CATEGORY = "OpenRouter"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "Your OpenRouter API key — get one at openrouter.ai/keys",
                }),
                "model": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Connect the model_name output of an OpenRouter Model List node here",
                }),
                "user_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                }),
                "max_tokens": ("INT", {
                    "default": 4096,
                    "min": 1,
                    "max": 32768,
                    "step": 1,
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": "Seed value used by ComfyUI to detect changes. Only sent to the API when send_seed is enabled.",
                }),
            },
            "optional": {
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
                "image": ("IMAGE", {
                    "tooltip": "Optional image for vision-capable models",
                }),
                "send_seed": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Forward the seed to the API for reproducibility. "
                        "Leave OFF for models that do not support per-request seeds "
                        "(JAX-based providers such as Gemini will error if a seed is sent)."
                    ),
                }),
                "json_mode": ("BOOLEAN", {"default": False}),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("response_text", "json_object")
    FUNCTION = "execute"

    def execute(
        self,
        api_key,
        model,
        user_prompt,
        temperature,
        max_tokens,
        seed,
        system_prompt="",
        image=None,
        send_seed=False,
        json_mode=False,
        debug=False,
    ):
        if not api_key.strip():
            raise ValueError("OpenRouter API key is required. Get one at openrouter.ai/keys")
        if not model or not model.strip():
            raise ValueError("Model is required. Connect an OpenRouter Model List node to the model input.")

        if image is not None:
            b64 = tensor_to_base64(image)
            user_content = [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]
        else:
            user_content = user_prompt

        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": user_content})

        payload = {
            "model": model.strip(),
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if send_seed and seed != 0:
            payload["seed"] = seed
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json",
        }

        if debug:
            debug_payload = json.loads(json.dumps(payload))
            if image is not None:
                debug_payload["messages"][-1]["content"][1]["image_url"]["url"] = "[base64 truncated]"
            print(f"[OpenRouterChat] POST {OPENROUTER_BASE}/chat/completions")
            print(f"[OpenRouterChat] Model: {model.strip()}")
            print(f"[OpenRouterChat] Payload: {json.dumps(debug_payload, indent=2)}")

        resp = retry_request(
            lambda: requests.post(
                f"{OPENROUTER_BASE}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120,
            )
        )

        if resp.status_code != 200:
            _raise_openrouter_error(resp)

        data = resp.json()
        if debug:
            print(f"[OpenRouterChat] Response: {json.dumps(data, indent=2)}")

        content = extract_chat_text(data, "OpenRouterChat")
        return (content, json.dumps(data))


class OpenRouterImageGenNode:
    CATEGORY = "OpenRouter"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "Your OpenRouter API key — get one at openrouter.ai/keys",
                }),
                "model": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Connect the model_name output of an OpenRouter Model List node here",
                }),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Text prompt describing the image to generate",
                }),
                "width": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 4096,
                    "step": 64,
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 4096,
                    "step": 64,
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": "Seed value used by ComfyUI to detect changes. Only sent to the API when send_seed is enabled.",
                }),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Negative prompt (supported by some models)",
                }),
                "send_seed": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Forward the seed to the API for reproducibility. "
                        "Leave OFF for models that do not support per-request seeds "
                        "(JAX-based providers such as Gemini will error if a seed is sent)."
                    ),
                }),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "generation_data")
    FUNCTION = "execute"

    def execute(
        self,
        api_key,
        model,
        prompt,
        width,
        height,
        seed,
        negative_prompt="",
        send_seed=False,
        debug=False,
    ):
        if not api_key.strip():
            raise ValueError("OpenRouter API key is required. Get one at openrouter.ai/keys")
        if not model or not model.strip():
            raise ValueError("Model is required. Connect an OpenRouter Model List node to the model input.")

        user_content = prompt
        if negative_prompt.strip():
            user_content += f"\n\nNegative prompt: {negative_prompt.strip()}"

        payload = {
            "model": model.strip(),
            "messages": [{"role": "user", "content": user_content}],
            "modalities": ["image"],
            "max_width": width,
            "max_height": height,
        }
        if send_seed and seed != 0:
            payload["seed"] = seed

        headers = {
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json",
        }

        if debug:
            print(f"[OpenRouterImageGen] POST {OPENROUTER_BASE}/chat/completions")
            print(f"[OpenRouterImageGen] Model: {model.strip()}")
            print(f"[OpenRouterImageGen] Payload: {json.dumps(payload, indent=2)}")

        resp = retry_request(
            lambda: requests.post(
                f"{OPENROUTER_BASE}/chat/completions",
                headers=headers,
                json=payload,
                timeout=180,
            )
        )

        if resp.status_code != 200:
            _raise_openrouter_error(resp)

        data = resp.json()
        if debug:
            print(f"[OpenRouterImageGen] Response keys: {list(data.keys())}")
            print(f"[OpenRouterImageGen] Response: {json.dumps(data, indent=2)}")

        img_tensor = _extract_image_from_response(data)
        return (img_tensor, json.dumps(data))


NODE_CLASS_MAPPINGS = {
    "OpenRouterChat": OpenRouterChatNode,
    "OpenRouterImageGen": OpenRouterImageGenNode,
    "OpenRouterModelList": OpenRouterModelListNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenRouterChat": "OpenRouter Chat",
    "OpenRouterImageGen": "OpenRouter Image Gen",
    "OpenRouterModelList": "OpenRouter Model List",
}
