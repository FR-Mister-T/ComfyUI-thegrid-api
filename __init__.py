from .nodes_thegrid import (
    NODE_CLASS_MAPPINGS as _TG_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _TG_NAMES,
)
from .nodes_openrouter import (
    NODE_CLASS_MAPPINGS as _OR_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _OR_NAMES,
)

NODE_CLASS_MAPPINGS = {**_TG_CLASSES, **_OR_CLASSES}
NODE_DISPLAY_NAME_MAPPINGS = {**_TG_NAMES, **_OR_NAMES}

# ---------------------------------------------------------------------------
# ComfyUI v3 registration (optional — v1 fallback is NODE_CLASS_MAPPINGS above)
# ---------------------------------------------------------------------------
# Future node modules to add here when ready:
#   from .nodes_openrouter_video import NODE_CLASS_MAPPINGS as _VID_CLASSES, ...
#   from .nodes_openrouter_audio import NODE_CLASS_MAPPINGS as _AUD_CLASSES, ...

try:
    try:
        from comfy.nodes import io as _comfy_io
        _ComfyExtension = _comfy_io.ComfyExtension
    except (ImportError, AttributeError):
        from comfy.nodes.package_typing import ComfyExtension as _ComfyExtension  # type: ignore

    class _TheGridAPIExtension(_ComfyExtension):
        name = "ComfyUI-thegrid-api"

        async def get_node_list(self):
            return list(NODE_CLASS_MAPPINGS.values())

    async def comfy_entrypoint():
        return _TheGridAPIExtension

except (ImportError, AttributeError):
    pass  # ComfyUI v1 — NODE_CLASS_MAPPINGS handles registration

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
