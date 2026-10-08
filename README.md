# ComfyUI-Illust-Similarity

How alike two illustrations are, scored 0-100 on the local GPU. Five measures,
each a node, plus one node that runs them all and a workflow for
[desktop-comfyui-server](https://github.com/mintani/desktop-comfyui-server)
that returns the result as JSON to a job server.

| Measure | Model | What it answers |
| --- | --- | --- |
| CCIP | deepghs/imgutils (ONNX) | the same character? |
| PixAI | PixAI Tagger v1.0 | the same tags? (and which) |
| SigLIP 2 | OpenCLIP ViT-B-16-SigLIP2-256 | the same content? |
| DINOv2 | facebook/dinov2-small | the same visual features? |
| Depth | Depth Anything V2 Small | the same composition? |

Nothing calls a remote inference API. Every model is downloaded once from
Hugging Face into `ComfyUI/models/illust-similarity` and runs on the device
ComfyUI picked; CCIP runs through onnxruntime.

PixAI Tagger's model code lives in its Hugging Face repository and is loaded
with `trust_remote_code`. `PIXAI_REVISION` in `metrics.py` pins the commit, so
the code that runs is the code at that commit.

The scoring — raw value to 0-100 through measured anchors — is ported from
[Tesixki/anime-illust-similarity](https://github.com/Tesixki/anime-illust-similarity)
(MIT). Roughly: an unrelated pair scores 0, a different character of the same
work 30, the same character drawn again 70, a near-duplicate 100.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/mintani/comfyui-illust-similarity
cd comfyui-illust-similarity
pip install -r requirements.txt      # portable: ..\..\..\python_embeded\python.exe -m pip install -r requirements.txt
```

Restart ComfyUI. The first run downloads about 4 GB of models (PixAI Tagger
1.9 GB, SigLIP 2 1.5 GB); nothing needs downloading by hand.

PixAI Tagger reads images at 1008 px and is the heavy one: run ComfyUI on the
GPU. On a 10-thread CPU it took about 10 seconds per image, against under a
second for each of the other measures.

For CCIP on the GPU, install `onnxruntime-gpu` in place of `onnxruntime`. On
the CPU it still runs in under a second per image.

## The workflow

`workflows/illust-similarity.json` is an API-format workflow: two `LoadImage`
nodes and **Illust Similarity (All)**, titled `similarity`. Upload it on the
job server's page as a preset (the `__INPUT_IMAGE__` / `__INPUT_IMAGE_2__`
placeholders take the two uploaded images), or drop it into the host's
workflows folder; the ComfyUI editor also opens API-format files.

The run records one value, under the node's title:

```json
{
  "total": 65.95, "band": "fairly similar",
  "ccip": { "raw": 0.171, "score": 55.0, "threshold": 0.178, "same_character": true },
  "pixai": { "raw": 0.78, "score": 74.8, "tags_a": [...], "tags_b": [...], "shared_tags": [...], ... },
  "siglip2": { "raw": 0.91, "score": 71.0 },
  "dinov2": { "raw": 0.63, "score": 71.0 },
  "depth": { "raw": 0.78, "score": 67.0, "mirrored": false }
}
```

A metric that fails (a download that did not finish, say) reports `error` and
the total is taken over the rest. CCIP is skipped when PixAI Tagger finds no
person in one of the images, since it only knows characters. A flat depth map
(a blank image, say) has no correlation to measure, so depth counts it as 0.
The JSON is strict: a number that is not finite is written as `null`, never
`NaN`.

`workflows/illust-similarity-separate.json` does the same with one node per
metric feeding **Illust Similarity: Report**, for when you want to look at the
depth maps or drop a metric.

## Nodes

Each takes `image_a` and `image_b` and returns the 0-100 `score`, the raw
value, and a `json` string.

- **Illust Similarity (All)** — every metric; output node. `use_*` switches
  drop a metric, and its weight, from the total.
- **CCIP** — `difference` (lower is the same character) and `same_character`
  against the model's threshold.
- **PixAI tags** — `cosine` of the embeddings the tag head reads, plus
  `tags_a`, `tags_b` and `shared_tags` as comma-separated strings.
- **SigLIP 2**, **DINOv2** — `cosine` of the embeddings.
- **Depth** — `correlation` of the normalised depth maps, and both maps as
  IMAGE outputs.
- **Report** — weighted total of whatever scores are wired in; output node.

Weights: CCIP 0.35, PixAI 0.25, SigLIP 2 0.20, DINOv2 0.10, Depth 0.10,
renormalised over the metrics present.

## Check it without ComfyUI

```bash
uv venv && uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install -r requirements.txt
uv run python tests/smoke.py                 # two drawn images
uv run python tests/smoke.py a.png b.png     # your own
```
