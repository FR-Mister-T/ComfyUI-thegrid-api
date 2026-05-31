import json
import requests
from ._utils import retry_request, extract_chat_text, extract_anthropic_text

THEGRID_OPENAI_URL = "https://api.thegrid.ai/v1/chat/completions"
THEGRID_ANTHROPIC_URL = "https://messages-beta.api.thegrid.ai/v1/messages"

INSTRUMENTS = [
    "text-prime",   "text-standard",   "text-max",
    "code-prime",   "code-standard",   "code-max",
    "agent-prime",  "agent-standard",  "agent-max",
]


class TheGridChatNode:
    CATEGORY = "TheGrid"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "Your TheGrid API key — get one at thegrid.ai",
                }),
                "instrument": (INSTRUMENTS, {
                    "default": "text-prime",
                    "tooltip": (
                        "TheGrid instrument: task type × quality tier.\n"
                        "text / code / agent  ×  standard / prime / max"
                    ),
                }),
                "user_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "User message",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Sampling temperature (0 = deterministic)",
                }),
                "max_tokens": ("INT", {
                    "default": 4096,
                    "min": 1,
                    "max": 32768,
                    "step": 1,
                    "tooltip": (
                        "Maximum tokens to generate. "
                        "Reasoning models consume tokens for internal thinking before producing output — "
                        "use 4096+ to avoid empty responses."
                    ),
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                }),
            },
            "optional": {
                "system_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "System prompt (optional)",
                }),
                "json_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Request JSON-formatted output (response_format: json_object)",
                }),
                "debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print full request and response to console",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("response_text", "json_object")
    FUNCTION = "execute"

    def execute(
        self,
        api_key,
        instrument,
        user_prompt,
        temperature,
        max_tokens,
        seed,
        system_prompt="",
        json_mode=False,
        debug=False,
    ):
        if not api_key.strip():
            raise ValueError("TheGrid API key is required. Get one at thegrid.ai")

        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": instrument,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json",
        }

        if debug:
            print(f"[TheGridChat] POST {THEGRID_OPENAI_URL}")
            print(f"[TheGridChat] Payload: {json.dumps(payload, indent=2)}")

        resp = retry_request(
            lambda: requests.post(
                THEGRID_OPENAI_URL, headers=headers, json=payload, timeout=120
            )
        )

        if resp.status_code != 200:
            raise RuntimeError(f"TheGrid API error {resp.status_code}: {resp.text}")

        data = resp.json()
        if debug:
            print(f"[TheGridChat] Response: {json.dumps(data, indent=2)}")

        text = extract_chat_text(data, "TheGridChat")
        return (text, json.dumps(data))


class TheGridAnthropicChatNode:
    CATEGORY = "TheGrid"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "Your TheGrid API key — get one at thegrid.ai",
                }),
                "instrument": (INSTRUMENTS, {
                    "default": "text-prime",
                    "tooltip": "TheGrid instrument: text / code / agent  ×  standard / prime / max",
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
                    "default": 1024,
                    "min": 1,
                    "max": 32768,
                    "step": 1,
                }),
            },
            "optional": {
                "system_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("response_text", "json_object")
    FUNCTION = "execute"

    def execute(
        self,
        api_key,
        instrument,
        user_prompt,
        temperature,
        max_tokens,
        system_prompt="",
        debug=False,
    ):
        if not api_key.strip():
            raise ValueError("TheGrid API key is required. Get one at thegrid.ai")

        payload = {
            "model": instrument,
            "messages": [{"role": "user", "content": user_prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system_prompt.strip():
            payload["system"] = system_prompt.strip()

        headers = {
            "x-api-key": api_key.strip(),
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        if debug:
            print(f"[TheGridAnthropicChat] POST {THEGRID_ANTHROPIC_URL}")
            print(f"[TheGridAnthropicChat] Payload: {json.dumps(payload, indent=2)}")

        resp = retry_request(
            lambda: requests.post(
                THEGRID_ANTHROPIC_URL, headers=headers, json=payload, timeout=120
            )
        )

        if resp.status_code != 200:
            raise RuntimeError(
                f"TheGrid Anthropic API error {resp.status_code}: {resp.text}"
            )

        data = resp.json()
        if debug:
            print(f"[TheGridAnthropicChat] Response: {json.dumps(data, indent=2)}")

        text = extract_anthropic_text(data, "TheGridAnthropicChat")
        return (text, json.dumps(data))


NODE_CLASS_MAPPINGS = {
    "TheGridChat": TheGridChatNode,
    "TheGridAnthropicChat": TheGridAnthropicChatNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TheGridChat": "TheGrid Chat",
    "TheGridAnthropicChat": "TheGrid Anthropic Chat (Beta)",
}
