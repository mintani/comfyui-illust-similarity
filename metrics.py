"""Similarity metrics between two illustrations, on the local GPU.

Six measures, each turned into a 0-100 score, and a weighted total:

- CCIP      character identity (deepghs/imgutils, ONNX)
- WD14      tag-embedding similarity (WD SwinV2 tagger v3, ONNX)
- DreamSim  perceptual similarity (CLIP + DINO + OpenCLIP ensemble)
- SigLIP 2  semantic similarity (OpenCLIP ViT-B-16-SigLIP2-256)
- DINOv2    self-supervised visual features (facebook/dinov2-small, CLS)
- Depth     composition: correlation of Depth Anything V2 depth maps

Nothing here talks to a remote inference API. Every model is downloaded once
into the cache directory and run locally; the torch models on the device
ComfyUI picks, the ONNX models through onnxruntime.

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
    "dreamsim": {"kind": "distance", "anchors": (0.758, 0.532, 0.314, 0.023)},
    # CCIP tells "different character" from "unrelated" poorly, so the 0 anchor
    # is the 95th percentile of different-character pairs. The model's own
    # threshold (0.178) lands near 54 on this scale.
    "ccip": {"kind": "distance", "anchors": (0.461, 0.327, 0.074, 0.004)},
    "wd14": {"kind": "cosine", "anchors": (0.451, 0.530, 0.748, 0.991)},
    "depth": {"kind": "cosine", "anchors": (0.285, 0.451, 0.805, 0.999)},
}

WEIGHTS = {
    "ccip": 0.25,
    "wd14": 0.10,
    "siglip2": 0.10,
    "dreamsim": 0.10,
    "dinov2": 0.05,
    "depth": 0.10,
}

METRIC_KEYS = ("ccip", "wd14", "dreamsim", "siglip2", "dinov2", "depth")

WD14_MODELS = ("SwinV2_v3", "ConvNext_v3", "ViT_v3", "EVA02_Large")
DINOV2_MODELS = ("facebook/dinov2-small", "facebook/dinov2-base")
SIGLIP2_MODEL = ("ViT-B-16-SigLIP2-256", "webli")
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
DEPTH_GRID = 64

# WD14 tags that say a person is in the picture, and that more than one is.
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
# WD14 tagger — tag embedding and shared tags
# ---------------------------------------------------------------------------


def wd14_tags(image: Image.Image, model_name: str = "SwinV2_v3") -> dict[str, Any]:
    from imgutils.tagging import get_wd14_tags

    general, character, embedding = get_wd14_tags(
        image, model_name=model_name, fmt=("general", "character", "embedding")
    )
    return {
        "embedding": np.asarray(embedding, dtype=np.float32),
        "general": dict(general),
        "character": dict(character),
    }


def wd14(a: Image.Image, b: Image.Image, model_name: str = "SwinV2_v3") -> dict[str, Any]:
    ta, tb = wd14_tags(a, model_name), wd14_tags(b, model_name)
    cos = cosine(ta["embedding"], tb["embedding"])
    tags_a = set(ta["general"]) | set(ta["character"])
    tags_b = set(tb["general"]) | set(tb["character"])
    shared = sorted(
        tags_a & tags_b,
        key=lambda tag: -(ta["general"].get(tag, 0.0) + tb["general"].get(tag, 0.0)),
    )
    return {
        "raw": cos,
        "score": piecewise_score(cos, "wd14"),
        "model": model_name,
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
# DreamSim — perceptual distance, 1 - cos of the ensemble embedding
# ---------------------------------------------------------------------------


def _dreamsim():
    import sys

    import torch
    from dreamsim import dreamsim as load

    cache = os.path.join(settings.cache_root, "dreamsim")
    os.makedirs(cache, exist_ok=True)

    # DreamSim fetches facebookresearch/dino through torch.hub, and that code
    # does `from utils import trunc_normal_` — a top-level `utils` that
    # ComfyUI's own `utils` package shadows once ComfyUI has imported it. Take
    # ComfyUI's out of sys.modules while the hub loads, so the name resolves
    # from the hub's directory (which torch.hub puts first on sys.path), then
    # put ComfyUI's back.
    shadowed = {name: module for name, module in sys.modules.items()
                if name == "utils" or name.startswith("utils.")}
    for name in shadowed:
        del sys.modules[name]
    try:
        model, preprocess = load(pretrained=True, device=settings.device, cache_dir=cache)
    finally:
        for name in [n for n in sys.modules if n == "utils" or n.startswith("utils.")]:
            del sys.modules[name]
        sys.modules.update(shadowed)
    model.eval()
    return model, preprocess, torch


def dreamsim_embed(image: Image.Image) -> np.ndarray:
    model, preprocess, torch = _cached("dreamsim", _dreamsim)
    with torch.no_grad():
        return model.embed(preprocess(image).to(settings.device))[0].float().cpu().numpy()


def dreamsim(a: Image.Image, b: Image.Image) -> dict[str, Any]:
    distance = 1.0 - cosine(dreamsim_embed(a), dreamsim_embed(b))
    return {"raw": distance, "score": piecewise_score(distance, "dreamsim")}


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
    wd14_model: str = "SwinV2_v3",
    dinov2_model: str = DINOV2_MODELS[0],
    enabled: tuple[str, ...] = METRIC_KEYS,
) -> dict[str, Any]:
    """Run every enabled metric, keeping one failure from taking the rest down."""
    results: dict[str, Any] = {}
    runners: dict[str, Callable[[], dict[str, Any]]] = {
        "ccip": lambda: ccip(a, b),
        "wd14": lambda: wd14(a, b, wd14_model),
        "dreamsim": lambda: dreamsim(a, b),
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
    tags = results.get("wd14", {})
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
