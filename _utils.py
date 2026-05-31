import io as _io
import base64
import time


def retry_request(fn, max_retries=3, backoff_base=2):
    import requests as _requests

    last_exc = None
    last_resp = None
    for attempt in range(max_retries + 1):
        try:
            resp = fn()
            last_resp = resp
            if resp.status_code in (429, 500, 502, 503, 504):
                if attempt < max_retries:
                    time.sleep(backoff_base ** attempt)
                    continue
            return resp
        except _requests.RequestException as exc:
            last_exc = exc
            if attempt < max_retries:
                time.sleep(backoff_base ** attempt)
    if last_exc:
        raise last_exc
    return last_resp


def tensor_to_base64(tensor):
    from PIL import Image
    import numpy as np

    if tensor.ndim == 4:
        tensor = tensor[0]
    np_img = (tensor.cpu().numpy() * 255).clip(0, 255).astype("uint8")
    pil_img = Image.fromarray(np_img)
    max_dim = 2048
    if pil_img.width > max_dim or pil_img.height > max_dim:
        pil_img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    buf = _io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def bytes_to_tensor(image_bytes):
    import numpy as np
    import torch
    from PIL import Image

    pil_img = Image.open(_io.BytesIO(image_bytes)).convert("RGB")
    np_img = np.array(pil_img).astype("float32") / 255.0
    return torch.from_numpy(np_img).unsqueeze(0)  # (1, H, W, C)


def url_to_tensor(url):
    import requests

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return bytes_to_tensor(resp.content)


def extract_chat_text(data: dict, node_name: str = "") -> str:
    """
    Extract text from an OpenAI-style chat completion response.
    Raises RuntimeError with actionable messages on all common failure modes,
    including reasoning models that exhaust max_tokens before producing output.
    """
    import json as _json
    prefix = f"[{node_name}] " if node_name else ""

    choices = data.get("choices") or []
    if not choices:
        # Some providers return 200 with an error body instead of a proper HTTP error code
        error = data.get("error") or {}
        if error:
            message = error.get("message") or str(error)
            code = error.get("code", "")
            code_tag = f" (code {code})" if code else ""
            raise RuntimeError(f"{prefix}API error{code_tag}: {message}")
        raise RuntimeError(
            f"{prefix}API returned no choices. "
            f"Full response: {_json.dumps(data)}"
        )

    choice = choices[0]
    message = choice.get("message") or {}
    finish_reason = choice.get("finish_reason") or "unknown"

    content = message.get("content") or ""
    if isinstance(content, list):
        content = " ".join(
            item.get("text", "") for item in content if item.get("type") == "text"
        )

    if content.strip():
        return content

    # Content is empty — diagnose why and give an actionable message
    reasoning = (
        message.get("reasoning_content")
        or (message.get("provider_specific_fields") or {}).get("reasoning_content")
        or ""
    )

    if finish_reason == "length":
        if reasoning.strip():
            raise RuntimeError(
                f"{prefix}max_tokens exhausted before any output was produced. "
                f"This model used all available tokens for internal reasoning "
                f"and never reached its final answer. "
                f"Increase max_tokens (try 4096 or higher) and retry. "
                f"The full reasoning chain is in json_object under "
                f"choices[0].message.reasoning_content."
            )
        raise RuntimeError(
            f"{prefix}Response is empty (finish_reason=length). "
            f"Increase max_tokens and retry."
        )

    if finish_reason == "content_filter":
        raise RuntimeError(
            f"{prefix}Request refused by content filter (finish_reason=content_filter)."
        )

    raise RuntimeError(
        f"{prefix}Empty response content (finish_reason={finish_reason!r}). "
        f"Enable debug=True to inspect the full response."
    )


def extract_anthropic_text(data: dict, node_name: str = "") -> str:
    """
    Extract text from an Anthropic Messages-style response.
    Raises RuntimeError with actionable messages on common failure modes.
    """
    prefix = f"[{node_name}] " if node_name else ""

    content_blocks = data.get("content") or []
    stop_reason = data.get("stop_reason") or "unknown"

    if not content_blocks:
        if stop_reason == "max_tokens":
            raise RuntimeError(
                f"{prefix}max_tokens limit reached before any response was produced. "
                f"Increase max_tokens and retry."
            )
        raise RuntimeError(
            f"{prefix}API returned no content blocks (stop_reason={stop_reason!r}). "
            f"Enable debug=True to inspect the full response."
        )

    text = "".join(
        block.get("text", "")
        for block in content_blocks
        if block.get("type") == "text"
    )

    if text.strip():
        return text

    if stop_reason == "max_tokens":
        raise RuntimeError(
            f"{prefix}max_tokens limit reached. Response content is empty. "
            f"Increase max_tokens and retry."
        )

    raise RuntimeError(
        f"{prefix}Empty response text (stop_reason={stop_reason!r}). "
        f"Enable debug=True to inspect the full response."
    )


def poll_async_job(api_key, job_id, base_url):
    # Reserved for future video/audio async node support
    raise NotImplementedError(
        "Async job polling is reserved for future video/audio nodes."
    )
