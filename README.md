# ComfyUI-TheGrid-API

ComfyUI custom nodes for interacting with **TheGrid.ai** and **OpenRouter** cloud inference services. Supports text/LLM chat, vision input, and image generation. Compatible with ComfyUI v3 spec and cloud-hosted environments (MimicPC, etc.) where local system access may be limited.

---

## Nodes

### TheGrid

| Node | Display Name | Description |
|------|-------------|-------------|
| `TheGridChat` | TheGrid Chat | Text/code/agent chat via TheGrid's OpenAI-compatible endpoint |
| `TheGridAnthropicChat` | TheGrid Anthropic Chat (Beta) | Chat via TheGrid's Anthropic Messages-compatible endpoint |

### OpenRouter

| Node | Display Name | Description |
|------|-------------|-------------|
| `OpenRouterChat` | OpenRouter Chat | LLM chat with optional vision input across hundreds of models |
| `OpenRouterImageGen` | OpenRouter Image Gen | Text-to-image generation, returns a ComfyUI IMAGE tensor |
| `OpenRouterModelList` | OpenRouter Model List | Query and filter available models; output chains into other nodes |

---

## Installation

### Via ComfyUI Manager
Search for **ComfyUI-TheGrid-API** and install.

### Manual
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/user/ComfyUI-thegrid-api
cd ComfyUI-thegrid-api
pip install -r requirements.txt
```

Restart ComfyUI after installation.

---

## API Keys

| Service | Where to get your key |
|---------|----------------------|
| TheGrid | [thegrid.ai](https://thegrid.ai) — account settings |
| OpenRouter | [openrouter.ai/keys](https://openrouter.ai/keys) |

API keys are entered directly in each node's `api_key` input. They are never stored on disk.

---

## Node Reference

### TheGrid Chat

Text, code, and agent completions via TheGrid's OpenAI-compatible endpoint.

**Inputs**

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `api_key` | STRING | — | TheGrid API key |
| `instrument` | COMBO | `text-prime` | Task type × quality tier (see table below) |
| `user_prompt` | STRING | — | User message |
| `temperature` | FLOAT | 0.7 | Sampling temperature (0 = deterministic) |
| `max_tokens` | INT | 1024 | Maximum tokens to generate |
| `seed` | INT | 0 | Reproducibility seed |
| `system_prompt` *(opt)* | STRING | — | System prompt |
| `json_mode` *(opt)* | BOOLEAN | false | Force JSON-formatted output |
| `debug` *(opt)* | BOOLEAN | false | Print request/response to console |

**Outputs:** `response_text` (STRING), `json_object` (STRING — full API response)

**Instruments**

| | standard | prime | max |
|---|---|---|---|
| **text** | `text-standard` | `text-prime` | `text-max` |
| **code** | `code-standard` | `code-prime` | `code-max` |
| **agent** | `agent-standard` | `agent-prime` | `agent-max` |

`prime` variants are the production defaults. `max` routes to the highest-quality suppliers available on the market.

---

### TheGrid Anthropic Chat *(Beta)*

Same instruments as TheGrid Chat but routed through TheGrid's Anthropic Messages-compatible endpoint. Useful if your downstream tooling expects Anthropic-formatted responses.

> **Note:** This endpoint is marked beta by TheGrid. Behaviour may change without notice.

**Inputs**

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `api_key` | STRING | — | TheGrid API key |
| `instrument` | COMBO | `text-prime` | Same 9 instruments as TheGrid Chat |
| `user_prompt` | STRING | — | User message |
| `temperature` | FLOAT | 0.7 | Sampling temperature |
| `max_tokens` | INT | 1024 | Maximum tokens |
| `system_prompt` *(opt)* | STRING | — | System prompt (sent as top-level `system` field) |
| `debug` *(opt)* | BOOLEAN | false | Print request/response to console |

**Outputs:** `response_text` (STRING), `json_object` (STRING)

---

### OpenRouter Chat

LLM chat supporting text and vision input across OpenRouter's full model catalog. The model dropdown is populated automatically from the OpenRouter API on startup and cached for 5 minutes.

**Inputs**

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `api_key` | STRING | — | OpenRouter API key |
| `model` | COMBO | Manual Input | Model selected from the live catalog |
| `user_prompt` | STRING | — | User message |
| `temperature` | FLOAT | 0.7 | Sampling temperature |
| `max_tokens` | INT | 1024 | Maximum tokens |
| `seed` | INT | 0 | Seed (honoured by OpenAI-based models) |
| `model_override` *(opt)* | STRING | — | Exact model ID — overrides the dropdown when non-empty. Wire an **OpenRouter Model List** output here. |
| `system_prompt` *(opt)* | STRING | — | System prompt |
| `image` *(opt)* | IMAGE | — | Vision input for multimodal models |
| `json_mode` *(opt)* | BOOLEAN | false | Force JSON-formatted output |
| `debug` *(opt)* | BOOLEAN | false | Print request details to console |

**Outputs:** `response_text` (STRING), `json_object` (STRING)

---

### OpenRouter Image Gen

Text-to-image generation via OpenRouter. Returns a native ComfyUI IMAGE tensor that can be wired directly into any downstream image node. The model dropdown is pre-filtered to image-generating models.

**Inputs**

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `api_key` | STRING | — | OpenRouter API key |
| `model` | COMBO | Manual Input | Image-generating model from the catalog |
| `prompt` | STRING | — | Text prompt describing the image |
| `width` | INT | 1024 | Target width in pixels (64–4096, step 64) |
| `height` | INT | 1024 | Target height in pixels (64–4096, step 64) |
| `seed` | INT | 0 | Generation seed |
| `model_override` *(opt)* | STRING | — | Exact model ID override |
| `negative_prompt` *(opt)* | STRING | — | Negative prompt (supported by some models) |
| `debug` *(opt)* | BOOLEAN | false | Print full response to console |

**Outputs:** `image` (IMAGE), `generation_data` (STRING — full API response)

> **Tip:** If generation fails with an unclear error, enable `debug=True`. The node handles several response formats (OpenAI content-block, data URI, DALL-E `b64_json`/URL) but the exact format depends on the model.

---

### OpenRouter Model List

Queries the OpenRouter models endpoint, applies filters, and returns the first matching model name as a string. Wire `model_name` into the `model_override` input of **OpenRouter Chat** or **OpenRouter Image Gen** for dynamic model selection.

**Inputs**

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `free_only` | BOOLEAN | false | Return only zero-cost models |
| `modality_filter` | COMBO | `all` | Filter by output modality: `all` / `text` / `image` / `video` / `audio` |
| `filter_text` | STRING | — | Substring match on model ID (e.g. `claude`, `flux`, `gpt-4o`) |

**Outputs**

| Output | Type | Description |
|--------|------|-------------|
| `model_name` | STRING | First model matching all filters — wire into `model_override` |
| `all_matches` | STRING | Newline-separated list of all matching model IDs |

> `video` and `audio` modality filters are available now for browsing and future node support.

---

## Workflow Examples

### Basic TheGrid text generation
```
TheGrid Chat
  api_key  → your key
  instrument → text-prime
  user_prompt → "Explain diffusion models in one paragraph"
```

### OpenRouter chat with free models only
```
OpenRouter Model List          OpenRouter Chat
  free_only → true      →      model_override ←─ model_name
  modality_filter → text       api_key → your key
  filter_text → "gemma"        user_prompt → "..."
```

### Image generation
```
OpenRouter Image Gen
  api_key → your key
  model → black-forest-labs/flux-1.1-pro
  prompt → "A cyberpunk cityscape at dusk"
  width → 1024
  height → 768
```

---

## Compatibility

- ComfyUI v3 (primary) with v1 backward-compatible registration
- Cloud-hosted ComfyUI environments (MimicPC, etc.)
- AMD ROCm and NVIDIA CUDA
- No dependency on the `openai` or `anthropic` Python SDKs — uses `requests` only

---

## Dependencies

- `requests`
- `pillow`

`torch` and `torchvision` are required by ComfyUI itself and do not need to be declared separately.

---

## Acknowledgements

This project drew inspiration from **[ComfyUI-EACloudNodes](https://github.com/EnragedAntelope/ComfyUI-EACloudNodes)** by EnragedAntelope. Their clean implementation of ComfyUI v3-compatible cloud LLM nodes (Groq, OpenRouter) served as a reference for the node architecture, v3/v1 dual registration pattern, and retry/vision handling approach.

---

## License

MIT
