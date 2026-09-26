"""The ComfyUI nodes: thin wrappers around `metrics`.

Every node takes two IMAGE inputs and returns the metric's 0-100 score, its
raw value and a JSON string. `Illust Similarity (All)` runs the six at once
and is an output node, so the JSON lands in the run's history — which is how
a job server reads it back.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image

from . import metrics


def _configure() -> None:
    """Point the metrics at ComfyUI's models folder and device, when inside ComfyUI."""
    cache_root = None
    device = None
    try:
        import folder_paths

        cache_root = os.path.join(folder_paths.models_dir, "illust-similarity")
    except ImportError:
        pass
    try:
        import comfy.model_management as model_management

        device = str(model_management.get_torch_device())
    except ImportError:
        pass
    metrics.configure(cache_root, device)


_configure()


def to_pil(image: torch.Tensor) -> Image.Image:
    """The first image of a ComfyUI batch as RGB."""
    array = (image[0].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def to_image(gray: np.ndarray) -> torch.Tensor:
    """A single-channel float map as a ComfyUI IMAGE, normalised to 0-1, near is bright."""
    lo, hi = float(gray.min()), float(gray.max())
    scaled = (gray - lo) / (hi - lo) if hi > lo else np.zeros_like(gray)
    tensor = torch.from_numpy(scaled.astype(np.float32))
    return tensor.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3)


def pair():
    return {"required": {"image_a": ("IMAGE",), "image_b": ("IMAGE",)}}


CATEGORY = "illust-similarity"


class IllustSimilarityCCIP:
    @classmethod
    def INPUT_TYPES(cls):
        return pair()

    RETURN_TYPES = ("FLOAT", "FLOAT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("score", "difference", "same_character", "json")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Character identity (CCIP). Lower difference means the same character."

    def run(self, image_a, image_b):
        result = metrics.ccip(to_pil(image_a), to_pil(image_b))
        return (result["score"], result["raw"], result["same_character"], metrics.to_json(result))


class IllustSimilarityWD14:
    @classmethod
    def INPUT_TYPES(cls):
        spec = pair()
        spec["required"]["model"] = (list(metrics.WD14_MODELS), {"default": "SwinV2_v3"})
        return spec

    RETURN_TYPES = ("FLOAT", "FLOAT", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("score", "cosine", "tags_a", "tags_b", "shared_tags", "json")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Tag-embedding similarity from a WD14 tagger, with the tags themselves."

    def run(self, image_a, image_b, model):
        result = metrics.wd14(to_pil(image_a), to_pil(image_b), model)
        return (
            result["score"],
            result["raw"],
            ", ".join(result["tags_a"]),
            ", ".join(result["tags_b"]),
            ", ".join(result["shared_tags"]),
            metrics.to_json(result),
        )


class IllustSimilarityDreamSim:
    @classmethod
    def INPUT_TYPES(cls):
        return pair()

    RETURN_TYPES = ("FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("score", "distance", "json")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Perceptual distance (DreamSim ensemble). 0 is identical."

    def run(self, image_a, image_b):
        result = metrics.dreamsim(to_pil(image_a), to_pil(image_b))
        return (result["score"], result["raw"], metrics.to_json(result))


class IllustSimilaritySigLIP2:
    @classmethod
    def INPUT_TYPES(cls):
        return pair()

    RETURN_TYPES = ("FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("score", "cosine", "json")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Semantic similarity of SigLIP 2 image embeddings."

    def run(self, image_a, image_b):
        result = metrics.siglip2(to_pil(image_a), to_pil(image_b))
        return (result["score"], result["raw"], metrics.to_json(result))


class IllustSimilarityDINOv2:
    @classmethod
    def INPUT_TYPES(cls):
        spec = pair()
        spec["required"]["model"] = (list(metrics.DINOV2_MODELS), {"default": metrics.DINOV2_MODELS[0]})
        return spec

    RETURN_TYPES = ("FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("score", "cosine", "json")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Similarity of DINOv2 CLS embeddings."

    def run(self, image_a, image_b, model):
        result = metrics.dinov2(to_pil(image_a), to_pil(image_b), model)
        return (result["score"], result["raw"], metrics.to_json(result))


class IllustSimilarityDepth:
    @classmethod
    def INPUT_TYPES(cls):
        return pair()

    RETURN_TYPES = ("FLOAT", "FLOAT", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("score", "correlation", "depth_a", "depth_b", "json")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Composition: correlation of Depth Anything V2 depth maps, with the maps to look at."

    def run(self, image_a, image_b):
        result = metrics.depth(to_pil(image_a), to_pil(image_b))
        map_a, map_b = result["_maps"]
        return (result["score"], result["raw"], to_image(map_a), to_image(map_b), metrics.to_json(result))


class IllustSimilarityReport:
    """Weighted total of whichever scores are wired in. A score below 0 counts as skipped."""

    @classmethod
    def INPUT_TYPES(cls):
        optional = {key: ("FLOAT", {"forceInput": True}) for key in metrics.METRIC_KEYS}
        return {"required": {}, "optional": optional}

    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("total", "json")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = "Weighted mean of the connected scores; the JSON is recorded in the run's history."

    def run(self, **scores):
        present = {key: float(value) for key, value in scores.items() if value is not None and value >= 0}
        total = metrics.total_score(present)
        report = {
            "total": total,
            "band": metrics.band(total) if total is not None else None,
            "scores": present,
            "weights": {key: metrics.WEIGHTS[key] for key in present},
        }
        text = metrics.to_json(report)
        return {"ui": {"text": [text]}, "result": (total if total is not None else -1.0, text)}


class IllustSimilarityAll:
    """All six metrics in one node, as the job-server workflow uses it."""

    @classmethod
    def INPUT_TYPES(cls):
        spec = pair()
        spec["required"]["wd14_model"] = (list(metrics.WD14_MODELS), {"default": "SwinV2_v3"})
        spec["required"]["dinov2_model"] = (list(metrics.DINOV2_MODELS), {"default": metrics.DINOV2_MODELS[0]})
        for key in metrics.METRIC_KEYS:
            spec["required"][f"use_{key}"] = ("BOOLEAN", {"default": True})
        return spec

    RETURN_TYPES = ("FLOAT", "STRING", "IMAGE", "IMAGE")
    RETURN_NAMES = ("total", "json", "depth_a", "depth_b")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = "Every metric at once. The JSON (per-metric raw value, score, tags, total) is recorded in the run's history."

    def run(self, image_a, image_b, wd14_model, dinov2_model, **flags):
        enabled = tuple(key for key in metrics.METRIC_KEYS if flags.get(f"use_{key}", True))
        results = metrics.compute_all(to_pil(image_a), to_pil(image_b), wd14_model, dinov2_model, enabled)
        maps = results.get("depth", {}).get("_maps")
        blank = torch.zeros((1, 8, 8, 3))
        depth_a = to_image(maps[0]) if maps else blank
        depth_b = to_image(maps[1]) if maps else blank
        text = metrics.to_json(results)
        total = results["total"] if results["total"] is not None else -1.0
        return {"ui": {"text": [text]}, "result": (total, text, depth_a, depth_b)}


NODE_CLASS_MAPPINGS = {
    "IllustSimilarityAll": IllustSimilarityAll,
    "IllustSimilarityCCIP": IllustSimilarityCCIP,
    "IllustSimilarityWD14": IllustSimilarityWD14,
    "IllustSimilarityDreamSim": IllustSimilarityDreamSim,
    "IllustSimilaritySigLIP2": IllustSimilaritySigLIP2,
    "IllustSimilarityDINOv2": IllustSimilarityDINOv2,
    "IllustSimilarityDepth": IllustSimilarityDepth,
    "IllustSimilarityReport": IllustSimilarityReport,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IllustSimilarityAll": "Illust Similarity (All)",
    "IllustSimilarityCCIP": "Illust Similarity: CCIP",
    "IllustSimilarityWD14": "Illust Similarity: WD14 tags",
    "IllustSimilarityDreamSim": "Illust Similarity: DreamSim",
    "IllustSimilaritySigLIP2": "Illust Similarity: SigLIP 2",
    "IllustSimilarityDINOv2": "Illust Similarity: DINOv2",
    "IllustSimilarityDepth": "Illust Similarity: Depth",
    "IllustSimilarityReport": "Illust Similarity: Report",
}
