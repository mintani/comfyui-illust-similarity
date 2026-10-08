"""Similarity metrics between two illustrations, on the local GPU.

Five measures, each turned into a 0-100 score, and a weighted total:

- CCIP      character identity (deepghs/imgutils, ONNX)
- PixAI     tag-embedding similarity (PixAI Tagger v1.0)
- SigLIP 2  semantic similarity (OpenCLIP ViT-B-16-SigLIP2-256)
- DINOv2    self-supervised visual features (facebook/dinov2-small, CLS)
- Depth     composition: correlation of Depth Anything V2 depth maps

Nothing here talks to a remote inference API. Every model is downloaded once
into the cache directory and run locally; the torch models on the device
ComfyUI picks, the CCIP model through onnxruntime.

The raw-value-to-score mapping and its anchors are ported from
Tesixki/anime-illust-similarity (MIT): a piecewise-linear map through the
measured medians of four kinds of pair — unrelated 0, a different character
of the same work 30, the same character drawn again 70, a near-duplicate 100
(for composition: unrelated, another cut, flipped or cropped, same framing).
"""

from __future__ import annotations

import json
import math
import os
import threading
from typing import Any, Callable

import numpy as np
from PIL import Image

CATEGORY_SCORE = (0.0, 30.0, 70.0, 100.0)

# Raw-value anchors in the order (unrelated, different character, same
# character, near-duplicate). "cosine" rises with similarity, "distance" falls.
CALIBRATION: dict[str, dict[str, Any]] = {
    "siglip2": {"kind": "cosine", "anchors": (0.566, 0.800, 0.904, 0.988)},
    "dinov2": {"kind": "cosine", "anchors": (0.306, 0.546, 0.620, 0.977)},
    # CCIP tells "different character" from "unrelated" poorly, so the 0 anchor
    # is the 95th percentile of different-character pairs. The model's own
    # threshold (0.178) lands near 54 on this scale.
    "ccip": {"kind": "distance", "anchors": (0.461, 0.327, 0.074, 0.004)},
    "pixai": {"kind": "cosine", "anchors": (0.282, 0.508, 0.742, 0.978)},
    "depth": {"kind": "cosine", "anchors": (0.285, 0.451, 0.805, 0.999)},
}

WEIGHTS = {
    "ccip": 0.35,
    "pixai": 0.25,
    "siglip2": 0.20,
    "dinov2": 0.10,
    "depth": 0.10,
}

METRIC_KEYS = ("ccip", "pixai", "siglip2", "dinov2", "depth")

# The model's code runs from its repository, so the revision pins what runs.
PIXAI_MODEL = "pixai-labs/pixai-tagger-v1.0"
PIXAI_REVISION = "9fe10addf9326e292da8a85a98ea74cd91b41771"
DINOV2_MODELS = ("facebook/dinov2-small", "facebook/dinov2-base")
SIGLIP2_MODEL = ("ViT-B-16-SigLIP2-256", "webli")
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
DEPTH_GRID = 64

# Danbooru tags that say a person is in the picture, and that more than one is.
PERSON_TAGS = frozenset({"1girl", "1boy", "1other", "solo", "2girls", "2boys", "multiple_girls",
                         "multiple_boys", "3girls", "3boys", "4girls", "4boys", "5girls", "6+girls",
                         "6+boys", "2others", "3others", "multiple_others"})
MULTI_TAGS = frozenset({"2girls", "2boys", "multiple_girls", "multiple_boys", "3girls", "3boys",
                        "4girls", "4boys", "5girls", "6+girls", "6+boys", "2others", "3others",
                        "multiple_others"})


class Settings:
    """Where models are cached and which device the torch models run on."""

    cache_root: str = os.path.join(os.path.expanduser("~"), ".cache", "illust-similarity")
    device: str = "cpu"


settings = Settings()


def configure(cache_root: str | None = None, device: str | None = None) -> None:
    if cache_root:
        settings.cache_root = cache_root
    if device:
        settings.device = device


def _hf_cache() -> str:
    path = os.path.join(settings.cache_root, "hf")
    os.makedirs(path, exist_ok=True)
    return path


_LOCK = threading.Lock()
_CACHE: dict[str, Any] = {}


def _cached(key: str, build: Callable[[], Any]) -> Any:
    """Build a model once per process, whichever node asks first."""
    if key not in _CACHE:
        with _LOCK:
            if key not in _CACHE:
                _CACHE[key] = build()
    return _CACHE[key]


def piecewise_score(x: float, key: str) -> float:
    """Map a raw value onto 0-100 through the metric's anchors, clipped outside."""
    spec = CALIBRATION[key]
    xs = np.asarray(spec["anchors"], dtype=np.float64)
    ys = np.asarray(CATEGORY_SCORE, dtype=np.float64)
    if spec["kind"] == "distance":
        xs, ys = xs[::-1], ys[::-1]
    return float(np.clip(np.interp(x, xs, ys), 0.0, 100.0))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation, 0 when either side is flat and the coefficient is undefined."""
    with np.errstate(invalid="ignore", divide="ignore"):
        value = float(np.corrcoef(np.ravel(a), np.ravel(b))[0, 1])
    return value if math.isfinite(value) else 0.0


def band(score: float) -> str:
    if score >= 80:
        return "very similar"
    if score >= 60:
        return "fairly similar"
    if score >= 40:
        return "partly similar"
    return "not similar"


# ---------------------------------------------------------------------------
# CCIP — character identity
# ---------------------------------------------------------------------------


def ccip(a: Image.Image, b: Image.Image) -> dict[str, Any]:
    from imgutils.metrics import ccip_batch_differences, ccip_default_threshold, ccip_extract_feature

    threshold = _cached("ccip_threshold", lambda: float(ccip_default_threshold()))
    features = [ccip_extract_feature(a), ccip_extract_feature(b)]
    difference = float(ccip_batch_differences(features)[0, 1])
    return {
        "raw": difference,
        "score": piecewise_score(difference, "ccip"),
        "threshold": threshold,
        "same_character": difference < threshold,
    }


# ---------------------------------------------------------------------------
# PixAI Tagger — tag-head embedding and shared tags
# ---------------------------------------------------------------------------


def _pixai():
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoImageProcessor, AutoModel

    options = {"revision": PIXAI_REVISION, "trust_remote_code": True, "cache_dir": _hf_cache()}
    processor = AutoImageProcessor.from_pretrained(PIXAI_MODEL, **options)
    # `from_pretrained` builds the model on the meta device, and the tagger's
    # own `__init__` calls `.item()` on a tensor it makes there, which raises.
    # `from_config` builds on a real device, so load the weights by hand after.
    config = AutoConfig.from_pretrained(PIXAI_MODEL, **options)
    model = AutoModel.from_config(config, trust_remote_code=True)
    weights = hf_hub_download(PIXAI_MODEL, "model.safetensors", revision=PIXAI_REVISION, cache_dir=_hf_cache())
    model.load_state_dict(load_file(weights))
    model.eval().to(settings.device)
    # The tag list is the concatenation of the categories, in this order.
    spans, start = {}, 0
    for category, count in model.config.tags_split:
        spans[category] = (start, start + count, float(model.config.category_best_threshold[category]))
        start += count
    return model, processor, torch, list(model.config.tags), spans


def pixai_tags(image: Image.Image) -> dict[str, Any]:
    """The input of the tag head (the embedding) and the tags over the model's thresholds."""
    model, processor, torch, tags, spans = _cached("pixai", _pixai)
    with torch.no_grad():
        pixels = processor(image, return_tensors="pt")["pixel_values"].to(settings.device, model.dtype)
        features = model.forward_feature(pixels)[-1]  # [1, C, h, w]
        pooled = model.head_pool(features.view(features.shape[0], features.shape[1], -1).permute(0, 2, 1))
        probs = torch.sigmoid(model.head(pooled))[0].float().cpu().numpy()
    result: dict[str, Any] = {"embedding": pooled[0].float().cpu().numpy()}
    for category in ("general", "character"):
        first, last, threshold = spans[category]
        picked = np.nonzero(probs[first:last] > threshold)[0]
        result[category] = {tags[first + i]: float(probs[first + i]) for i in picked}
    return result


def pixai(a: Image.Image, b: Image.Image) -> dict[str, Any]:
    ta, tb = pixai_tags(a), pixai_tags(b)
    cos = cosine(ta["embedding"], tb["embedding"])
    tags_a = set(ta["general"]) | set(ta["character"])
    tags_b = set(tb["general"]) | set(tb["character"])
    shared = sorted(
        tags_a & tags_b,
        key=lambda tag: -(ta["general"].get(tag, 0.0) + tb["general"].get(tag, 0.0)),
    )
    return {
        "raw": cos,
        "score": piecewise_score(cos, "pixai"),
        "tags_a": sorted(tags_a),
        "tags_b": sorted(tags_b),
        "character_a": sorted(ta["character"]),
        "character_b": sorted(tb["character"]),
        "shared_tags": shared,
        # CCIP assumes one character in frame; these say whether that holds.
        "humans_a": _has_humans(ta["general"]),
        "humans_b": _has_humans(tb["general"]),
        "multiple_a": bool(set(ta["general"]) & MULTI_TAGS),
        "multiple_b": bool(set(tb["general"]) & MULTI_TAGS),
    }


def _has_humans(general: dict[str, float]) -> bool:
    tags = set(general)
    return not ("no_humans" in tags and not (tags & PERSON_TAGS))


# ---------------------------------------------------------------------------
# SigLIP 2 — semantic image embedding
# ---------------------------------------------------------------------------


def _siglip2():
    import open_clip
    import torch

    name, pretrained = SIGLIP2_MODEL
    model, _, preprocess = open_clip.create_model_and_transforms(
        name, pretrained=pretrained, cache_dir=_hf_cache()
    )
    model.eval().to(settings.device)
    return model, preprocess, torch


def siglip2_embed(image: Image.Image) -> np.ndarray:
    model, preprocess, torch = _cached("siglip2", _siglip2)
    with torch.no_grad():
        features = model.encode_image(preprocess(image).unsqueeze(0).to(settings.device))
        features = features / features.norm(dim=-1, keepdim=True)
    return features[0].float().cpu().numpy()


def siglip2(a: Image.Image, b: Image.Image) -> dict[str, Any]:
    cos = cosine(siglip2_embed(a), siglip2_embed(b))
    return {"raw": cos, "score": piecewise_score(cos, "siglip2")}


# ---------------------------------------------------------------------------
# DINOv2 — CLS embedding
# ---------------------------------------------------------------------------


def _dinov2(model_name: str):
    def build():
        import torch
        from transformers import AutoImageProcessor, AutoModel

        processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=_hf_cache())
        model = AutoModel.from_pretrained(model_name, cache_dir=_hf_cache())
        model.eval().to(settings.device)
        return model, processor, torch

    return _cached(f"dinov2:{model_name}", build)


def dinov2_embed(image: Image.Image, model_name: str = DINOV2_MODELS[0]) -> np.ndarray:
    model, processor, torch = _dinov2(model_name)
    with torch.no_grad():
        inputs = {k: v.to(settings.device) for k, v in processor(images=image, return_tensors="pt").items()}
        return model(**inputs).pooler_output[0].float().cpu().numpy()


def dinov2(a: Image.Image, b: Image.Image, model_name: str = DINOV2_MODELS[0]) -> dict[str, Any]:
    cos = cosine(dinov2_embed(a, model_name), dinov2_embed(b, model_name))
    return {"raw": cos, "score": piecewise_score(cos, "dinov2"), "model": model_name}


# ---------------------------------------------------------------------------
# Depth — composition, from Depth Anything V2 relative depth
# ---------------------------------------------------------------------------


def _depth():
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL, cache_dir=_hf_cache())
    model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL, cache_dir=_hf_cache())
    model.eval().to(settings.device)
    return model, processor, torch


def depth_map(image: Image.Image) -> dict[str, np.ndarray]:
    """The relative depth at the image's size, and a normalised 64x64 grid of it.

    Normalising by median and standard deviation removes the scale and offset
    the model is free to choose, so two maps compare on shape alone.
    """
    model, processor, torch = _cached("depth", _depth)
    with torch.no_grad():
        inputs = {k: v.to(settings.device) for k, v in processor(images=image, return_tensors="pt").items()}
        predicted = model(**inputs).predicted_depth  # [1, h, w]
        full = torch.nn.functional.interpolate(
            predicted.unsqueeze(1), size=image.size[::-1], mode="bilinear", align_corners=False
        )
        grid = torch.nn.functional.interpolate(
            predicted.unsqueeze(1), size=(DEPTH_GRID, DEPTH_GRID), mode="area"
        )
    full = full[0, 0].float().cpu().numpy()
    grid = grid[0, 0].float().cpu().numpy()
    grid = (grid - np.median(grid)) / (grid.std() + 1e-6)
    return {"full": full, "grid": grid.astype(np.float32)}


def depth(a: Image.Image, b: Image.Image) -> dict[str, Any]:
    da, db = depth_map(a), depth_map(b)
    corr = correlation(da["grid"], db["grid"])
    corr_flipped = correlation(da["grid"], db["grid"][:, ::-1])
    return {
        "raw": corr,
        "score": piecewise_score(corr, "depth"),
        "correlation_flipped": corr_flipped,
        "mirrored": corr_flipped > corr + 0.15,
        "_maps": (da["full"], db["full"]),
    }


# ---------------------------------------------------------------------------
# Everything at once
# ---------------------------------------------------------------------------


def total_score(scores: dict[str, float]) -> float | None:
    """Weighted mean of the scores present, weights renormalised over them."""
    numerator = denominator = 0.0
    for key, weight in WEIGHTS.items():
        score = scores.get(key)
        if score is None:
            continue
        numerator += score * weight
        denominator += weight
    return round(numerator / denominator, 3) if denominator else None


def compute_all(
    a: Image.Image,
    b: Image.Image,
    dinov2_model: str = DINOV2_MODELS[0],
    enabled: tuple[str, ...] = METRIC_KEYS,
) -> dict[str, Any]:
    """Run every enabled metric, keeping one failure from taking the rest down."""
    results: dict[str, Any] = {}
    runners: dict[str, Callable[[], dict[str, Any]]] = {
        "ccip": lambda: ccip(a, b),
        "pixai": lambda: pixai(a, b),
        "siglip2": lambda: siglip2(a, b),
        "dinov2": lambda: dinov2(a, b, dinov2_model),
        "depth": lambda: depth(a, b),
    }
    for key in METRIC_KEYS:
        if key not in enabled:
            continue
        try:
            results[key] = runners[key]()
        except Exception as err:  # a missing download is one metric's problem, not the run's
            results[key] = {"error": f"{type(err).__name__}: {err}"}

    # CCIP assumes a single character; a landscape gets no character score.
    tags = results.get("pixai", {})
    if "score" in results.get("ccip", {}) and "score" in tags:
        if not (tags["humans_a"] and tags["humans_b"]):
            results["ccip"] = {"skipped": "no person in one of the images", "raw": results["ccip"]["raw"]}
        elif tags["multiple_a"] or tags["multiple_b"]:
            results["ccip"]["note"] = "more than one character; CCIP is less reliable"

    scores = {key: value["score"] for key, value in results.items() if "score" in value}
    total = total_score(scores)
    results["total"] = total
    results["band"] = band(total) if total is not None else None
    return results


def public(results: dict[str, Any]) -> dict[str, Any]:
    """The results without the arrays (keys starting with `_`), ready for JSON."""
    return _strip(results)


def _strip(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip(v) for k, v in value.items() if not k.startswith("_")}
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def to_json(results: dict[str, Any]) -> str:
    """The results as strict JSON; a number that is not finite is written as null."""
    return json.dumps(public(results), ensure_ascii=False, allow_nan=False)
