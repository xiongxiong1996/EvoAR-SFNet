import argparse
import csv
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import h5py
import numpy as np
import scipy.io as sio
import torch
from matplotlib import pyplot as plt

try:
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
except Exception:
    SamAutomaticMaskGenerator = None
    sam_model_registry = None


COLLECTION_SPECS = {
    "WV3_Reduced": {
        "h5_relpath": os.path.join("WV3", "test_wv3_multiExm1.h5"),
        "has_gt": True,
    },
    "QB_Reduced": {
        "h5_relpath": os.path.join("QB", "test_qb_multiExm1.h5"),
        "has_gt": True,
    },
    "GF2_Reduced": {
        "h5_relpath": os.path.join("GF2", "test_gf2_multiExm1.h5"),
        "has_gt": True,
    },
    "WV3_Full": {
        "h5_relpath": os.path.join("WV3", "test_wv3_OrigScale_multiExm1.h5"),
        "has_gt": False,
    },
    "QB_Full": {
        "h5_relpath": os.path.join("QB", "test_qb_OrigScale_multiExm1.h5"),
        "has_gt": False,
    },
    "GF2_Full": {
        "h5_relpath": os.path.join("GF2", "test_gf2_OrigScale_multiExm1.h5"),
        "has_gt": False,
    },
}

DEFAULT_REDUCED_COLLECTIONS = ["WV3_Reduced", "QB_Reduced", "GF2_Reduced"]
MAT_FILE_PATTERN = re.compile(r"^output_mulExm_(\d+)\.mat$")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run SAM automatic segmentation on pansharpened .mat results under PanCollection. "
            "The default scope is reduced-resolution collections because they have GT and are "
            "more suitable for a downstream-oriented qualitative supplement."
        )
    )
    parser.add_argument(
        "--pan_root",
        type=str,
        default="./2_DL_Result/PanCollection",
        help="Root directory that contains WV3_Reduced/QB_Reduced/... result folders.",
    )
    parser.add_argument(
        "--h5_root",
        type=str,
        default="./pansharpening/test_data",
        help="Root directory that contains WV3/QB/GF2 H5 test files.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory used to save SAM outputs and ranking files.",
    )
    parser.add_argument(
        "--collections",
        nargs="*",
        default=None,
        help=(
            "Collections to process, e.g. WV3_Reduced QB_Reduced. "
            "If omitted, the script processes reduced-resolution collections only."
        ),
    )
    parser.add_argument(
        "--include_full",
        action="store_true",
        help="When --collections is omitted, include *_Full collections in addition to reduced ones.",
    )
    parser.add_argument(
        "--indices",
        nargs="*",
        type=int,
        default=None,
        help="Specific sample indices to process. If omitted, the script uses all discovered .mat files.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=9999,
        help="Maximum number of samples processed per collection when --indices is omitted.",
    )
    parser.add_argument(
        "--tol_low",
        default=0.01,
        type=float,
        help="Lower percentile used for pseudo-RGB visualization stretch.",
    )
    parser.add_argument(
        "--tol_high",
        default=0.99,
        type=float,
        help="Upper percentile used for pseudo-RGB visualization stretch.",
    )
    parser.add_argument(
        "--boundary_tol",
        default=2,
        type=int,
        help="Pixel tolerance used when matching SAM-derived boundaries against GT-derived boundaries.",
    )
    parser.add_argument(
        "--boundary_thickness",
        default=1,
        type=int,
        help="Boundary thickness used in saved overlay figures.",
    )
    parser.add_argument(
        "--topk_candidates",
        default=8,
        type=int,
        help="Number of top-ranked samples saved to the per-collection and global ranking files.",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Random seed used for deterministic visualization colors.",
    )

    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        required=True,
        help="Path to the SAM checkpoint file, e.g. sam_vit_h_4b8939.pth.",
    )
    parser.add_argument(
        "--sam_model_type",
        default="vit_h",
        type=str,
        choices=["vit_h", "vit_l", "vit_b"],
        help="SAM backbone type.",
    )
    parser.add_argument(
        "--sam_device",
        default="auto",
        type=str,
        help="Device for SAM inference: auto, cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--sam_points_per_side",
        default=32,
        type=int,
        help="SAM automatic mask generator parameter.",
    )
    parser.add_argument(
        "--sam_points_per_batch",
        default=64,
        type=int,
        help="SAM automatic mask generator parameter.",
    )
    parser.add_argument(
        "--sam_pred_iou_thresh",
        default=0.88,
        type=float,
        help="SAM automatic mask generator parameter.",
    )
    parser.add_argument(
        "--sam_stability_score_thresh",
        default=0.92,
        type=float,
        help="SAM automatic mask generator parameter.",
    )
    parser.add_argument(
        "--sam_box_nms_thresh",
        default=0.70,
        type=float,
        help="SAM automatic mask generator parameter.",
    )
    parser.add_argument(
        "--sam_crop_n_layers",
        default=1,
        type=int,
        help="SAM automatic mask generator parameter.",
    )
    parser.add_argument(
        "--sam_crop_n_points_downscale_factor",
        default=2,
        type=int,
        help="SAM automatic mask generator parameter.",
    )
    parser.add_argument(
        "--sam_min_mask_region_area",
        default=100,
        type=int,
        help="Minimum connected region area kept by SAM post-processing.",
    )
    parser.add_argument(
        "--sam_max_side",
        default=1024,
        type=int,
        help="Resize RGB input so that its long side is at most this value before SAM inference.",
    )
    return parser.parse_args()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def normalize_from_sensor(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return arr
    if float(arr.max()) > 1.5:
        arr = arr / 2047.0
    return arr.astype(np.float32)


def load_h5_data(file_path: str) -> Dict[str, np.ndarray]:
    with h5py.File(file_path, "r") as data:
        result = {
            "lms": normalize_from_sensor(data["lms"][...]),
            "pan": normalize_from_sensor(data["pan"][...]),
        }
        if "gt" in data:
            result["gt"] = normalize_from_sensor(data["gt"][...])
    return result


def to_hwc(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D array after squeeze, got shape={arr.shape}.")
    if arr.shape[0] in (1, 3, 4, 8) and arr.shape[-1] not in (1, 3, 4, 8):
        arr = np.transpose(arr, (1, 2, 0))
    return arr.astype(np.float32)


def load_sr_from_mat(mat_path: str) -> np.ndarray:
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"SR mat file not found: {mat_path}")
    mat = sio.loadmat(mat_path)
    if "sr" not in mat:
        valid_keys = [key for key in mat.keys() if not key.startswith("__")]
        raise KeyError(f"'sr' not found in {mat_path}. Available keys: {valid_keys}")
    sr = normalize_from_sensor(mat["sr"])
    return np.clip(to_hwc(sr), 0.0, 1.0)


def pick_rgb_channels(x_hwc: np.ndarray) -> np.ndarray:
    channels = x_hwc.shape[-1]
    if channels == 1:
        return np.repeat(x_hwc, 3, axis=-1)
    if channels == 3:
        return x_hwc
    if channels == 4:
        return x_hwc[..., [2, 1, 0]]
    if channels == 8:
        return x_hwc[..., [4, 2, 1]]
    raise ValueError(f"Unsupported channel count for RGB visualization: {channels}")


def percentile_stretch(rgb: np.ndarray, tol_low: float, tol_high: float) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)
    out = np.zeros_like(rgb, dtype=np.float32)
    for channel_idx in range(rgb.shape[-1]):
        channel = rgb[..., channel_idx]
        low = float(np.quantile(channel, tol_low))
        high = float(np.quantile(channel, tol_high))
        if high - low < 1e-8:
            out[..., channel_idx] = 0.0
        else:
            out[..., channel_idx] = np.clip((channel - low) / (high - low), 0.0, 1.0)
    return out


def to_display_rgb(x_hwc: np.ndarray, tol_low: float, tol_high: float) -> np.ndarray:
    rgb = pick_rgb_channels(np.clip(x_hwc, 0.0, 1.0))
    return percentile_stretch(rgb, tol_low, tol_high)


def resolve_collections(args) -> List[str]:
    if args.collections:
        unknown = [name for name in args.collections if name not in COLLECTION_SPECS]
        if unknown:
            raise ValueError(
                f"Unknown collection names: {unknown}. "
                f"Available choices: {list(COLLECTION_SPECS.keys())}"
            )
        return list(args.collections)
    if args.include_full:
        return list(COLLECTION_SPECS.keys())
    return list(DEFAULT_REDUCED_COLLECTIONS)


def resolve_h5_path(h5_root: str, collection_name: str) -> str:
    if collection_name not in COLLECTION_SPECS:
        raise KeyError(f"Unknown collection: {collection_name}")
    return os.path.join(h5_root, COLLECTION_SPECS[collection_name]["h5_relpath"])


def find_results_dir(collection_dir: str) -> str:
    preferred = os.path.join(collection_dir, "EvoARFS", "results")
    if os.path.isdir(preferred):
        return preferred

    direct = os.path.join(collection_dir, "results")
    if os.path.isdir(direct):
        return direct

    if not os.path.isdir(collection_dir):
        raise FileNotFoundError(f"Collection directory not found: {collection_dir}")

    child_names = sorted(os.listdir(collection_dir))
    for child_name in child_names:
        candidate = os.path.join(collection_dir, child_name, "results")
        if os.path.isdir(candidate):
            return candidate

    raise FileNotFoundError(f"No results directory found under: {collection_dir}")


def list_mat_indices(results_dir: str) -> List[int]:
    indices: List[int] = []
    for filename in sorted(os.listdir(results_dir)):
        match = MAT_FILE_PATTERN.match(filename)
        if match is not None:
            indices.append(int(match.group(1)))
    return sorted(indices)


def select_sample_indices(
    requested_indices: Optional[Sequence[int]],
    discovered_indices: Sequence[int],
    sample_count: int,
    max_samples: int,
) -> List[int]:
    valid_discovered = [idx for idx in discovered_indices if 0 <= idx < sample_count]
    if requested_indices is not None:
        requested = [idx for idx in requested_indices if idx in valid_discovered]
        return sorted(requested)
    return sorted(valid_discovered[:max_samples])


def resolve_torch_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_sam_generator(args) -> SamAutomaticMaskGenerator:
    if SamAutomaticMaskGenerator is None or sam_model_registry is None:
        raise ImportError(
            "segment_anything is not available. Install the official SAM package before running "
            "this script."
        )
    if not os.path.exists(args.sam_checkpoint):
        raise FileNotFoundError(f"SAM checkpoint not found: {args.sam_checkpoint}")

    device = resolve_torch_device(args.sam_device)
    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint)
    sam.to(device=device)
    sam.eval()

    generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=int(args.sam_points_per_side),
        points_per_batch=int(args.sam_points_per_batch),
        pred_iou_thresh=float(args.sam_pred_iou_thresh),
        stability_score_thresh=float(args.sam_stability_score_thresh),
        box_nms_thresh=float(args.sam_box_nms_thresh),
        crop_n_layers=int(args.sam_crop_n_layers),
        crop_n_points_downscale_factor=int(args.sam_crop_n_points_downscale_factor),
        min_mask_region_area=int(args.sam_min_mask_region_area),
    )
    return generator


def resize_rgb_for_sam(rgb: np.ndarray, max_side: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    rgb_uint8 = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    h, w = rgb_uint8.shape[:2]
    if max_side <= 0 or max(h, w) <= max_side:
        return rgb_uint8, (h, w)

    scale = float(max_side) / float(max(h, w))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    resized = cv2.resize(rgb_uint8, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized, (h, w)


def generate_sam_masks(
    rgb: np.ndarray,
    sam_generator: SamAutomaticMaskGenerator,
    max_side: int,
) -> List[Dict[str, object]]:
    resized_rgb, original_size = resize_rgb_for_sam(rgb, max_side)
    orig_h, orig_w = original_size

    raw_masks = sam_generator.generate(resized_rgb)
    raw_masks = sorted(raw_masks, key=lambda item: int(item["area"]), reverse=True)

    processed_masks: List[Dict[str, object]] = []
    for idx, mask_dict in enumerate(raw_masks, start=1):
        mask = np.asarray(mask_dict["segmentation"], dtype=np.uint8)
        if mask.shape[0] != orig_h or mask.shape[1] != orig_w:
            mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(bool)
        area = int(mask.sum())
        if area <= 0:
            continue
        processed_masks.append(
            {
                "id": idx,
                "segmentation": mask,
                "area": area,
                "bbox": mask_dict.get("bbox"),
                "predicted_iou": float(mask_dict.get("predicted_iou", 0.0)),
                "stability_score": float(mask_dict.get("stability_score", 0.0)),
            }
        )
    return processed_masks


def masks_to_label_map(masks: Sequence[Dict[str, object]], shape: Tuple[int, int]) -> np.ndarray:
    label_map = np.zeros(shape, dtype=np.int32)
    for label_idx, mask_info in enumerate(masks, start=1):
        label_map[np.asarray(mask_info["segmentation"], dtype=bool)] = label_idx
    return label_map


def label_boundaries(label_map: np.ndarray) -> np.ndarray:
    label_map = np.asarray(label_map, dtype=np.int32)
    boundary = np.zeros(label_map.shape, dtype=np.uint8)
    boundary[:-1, :] |= label_map[:-1, :] != label_map[1:, :]
    boundary[:, :-1] |= label_map[:, :-1] != label_map[:, 1:]
    return boundary


def dilate_binary(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return (mask > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate((mask > 0).astype(np.uint8), kernel)


def boundary_metrics(pred: np.ndarray, ref: np.ndarray, tol: int) -> Dict[str, float]:
    pred = (pred > 0).astype(np.uint8)
    ref = (ref > 0).astype(np.uint8)
    pred_match = pred & dilate_binary(ref, tol)
    ref_match = ref & dilate_binary(pred, tol)

    pred_sum = int(pred.sum())
    ref_sum = int(ref.sum())
    precision = float(pred_match.sum() / max(pred_sum, 1))
    recall = float(ref_match.sum() / max(ref_sum, 1))
    f1 = 0.0 if precision + recall < 1e-8 else float(2.0 * precision * recall / (precision + recall))
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "boundary_density": float(pred.mean()),
    }


def mask_summary_stats(masks: Sequence[Dict[str, object]], image_area: int, prefix: str) -> Dict[str, float]:
    if not masks:
        return {
            f"{prefix}_mask_count": 0,
            f"{prefix}_mean_area_ratio": 0.0,
            f"{prefix}_median_area_ratio": 0.0,
            f"{prefix}_union_coverage_ratio": 0.0,
            f"{prefix}_mean_predicted_iou": 0.0,
            f"{prefix}_mean_stability_score": 0.0,
        }

    areas = np.array([float(mask_info["area"]) for mask_info in masks], dtype=np.float32)
    predicted_ious = np.array([float(mask_info["predicted_iou"]) for mask_info in masks], dtype=np.float32)
    stability_scores = np.array([float(mask_info["stability_score"]) for mask_info in masks], dtype=np.float32)

    union_mask = np.zeros((image_area,), dtype=np.uint8)
    for mask_info in masks:
        union_mask |= np.asarray(mask_info["segmentation"], dtype=np.uint8).reshape(-1)

    return {
        f"{prefix}_mask_count": int(len(masks)),
        f"{prefix}_mean_area_ratio": float(areas.mean() / float(image_area)),
        f"{prefix}_median_area_ratio": float(np.median(areas) / float(image_area)),
        f"{prefix}_union_coverage_ratio": float(union_mask.mean()),
        f"{prefix}_mean_predicted_iou": float(predicted_ious.mean()),
        f"{prefix}_mean_stability_score": float(stability_scores.mean()),
    }


def make_palette(num_colors: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    palette = rng.uniform(0.15, 0.95, size=(num_colors, 3)).astype(np.float32)
    return palette


def render_mask_overlay(
    rgb: np.ndarray,
    masks: Sequence[Dict[str, object]],
    alpha: float = 0.45,
    seed: int = 0,
) -> np.ndarray:
    overlay = np.clip(rgb.copy(), 0.0, 1.0)
    palette = make_palette(max(len(masks), 1) + 1, seed)
    for mask_idx, mask_info in enumerate(masks, start=1):
        mask = np.asarray(mask_info["segmentation"], dtype=bool)
        color = palette[mask_idx % len(palette)]
        overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * color
    return np.clip(overlay, 0.0, 1.0)


def render_mask_flat(
    masks: Sequence[Dict[str, object]],
    shape: Tuple[int, int],
    seed: int = 0,
) -> np.ndarray:
    canvas = np.ones((shape[0], shape[1], 3), dtype=np.float32)
    palette = make_palette(max(len(masks), 1) + 1, seed)
    for mask_idx, mask_info in enumerate(masks, start=1):
        mask = np.asarray(mask_info["segmentation"], dtype=bool)
        color = palette[mask_idx % len(palette)]
        canvas[mask] = color
    return np.clip(canvas, 0.0, 1.0)


def overlay_boundaries(rgb: np.ndarray, boundary: np.ndarray, thickness: int) -> np.ndarray:
    overlay = np.clip(rgb.copy(), 0.0, 1.0)
    if thickness > 1:
        draw_mask = dilate_binary(boundary, max(thickness - 1, 0))
    else:
        draw_mask = (boundary > 0).astype(np.uint8)
    overlay[draw_mask > 0] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return overlay


def save_image(path: str, image: np.ndarray):
    plt.imsave(path, np.clip(image, 0.0, 1.0))


def save_binary_mask(path: str, mask: np.ndarray):
    plt.imsave(path, (mask > 0).astype(np.float32), cmap="gray", vmin=0.0, vmax=1.0)


def make_horizontal_strip(images: Sequence[np.ndarray], gap: int = 8) -> np.ndarray:
    if not images:
        raise ValueError("No images provided for strip creation.")
    h = images[0].shape[0]
    gap_block = np.ones((h, gap, 3), dtype=np.float32)
    parts: List[np.ndarray] = []
    for idx, image in enumerate(images):
        parts.append(np.clip(image.astype(np.float32), 0.0, 1.0))
        if idx != len(images) - 1:
            parts.append(gap_block)
    return np.concatenate(parts, axis=1)


def make_vertical_stack(images: Sequence[np.ndarray], gap: int = 8) -> np.ndarray:
    if not images:
        raise ValueError("No images provided for stack creation.")
    w = images[0].shape[1]
    gap_block = np.ones((gap, w, 3), dtype=np.float32)
    parts: List[np.ndarray] = []
    for idx, image in enumerate(images):
        parts.append(np.clip(image.astype(np.float32), 0.0, 1.0))
        if idx != len(images) - 1:
            parts.append(gap_block)
    return np.concatenate(parts, axis=0)


def save_visual_assets(
    sample_dir: str,
    lms_rgb: np.ndarray,
    sr_rgb: np.ndarray,
    gt_rgb: Optional[np.ndarray],
    lms_mask_overlay: np.ndarray,
    sr_mask_overlay: np.ndarray,
    gt_mask_overlay: Optional[np.ndarray],
    lms_mask_flat: np.ndarray,
    sr_mask_flat: np.ndarray,
    gt_mask_flat: Optional[np.ndarray],
    lms_boundary_overlay: np.ndarray,
    sr_boundary_overlay: np.ndarray,
    gt_boundary_overlay: Optional[np.ndarray],
    lms_boundary: np.ndarray,
    sr_boundary: np.ndarray,
    gt_boundary: Optional[np.ndarray],
):
    ensure_dir(sample_dir)

    save_image(os.path.join(sample_dir, "lms_rgb.png"), lms_rgb)
    save_image(os.path.join(sample_dir, "sr_rgb.png"), sr_rgb)
    save_image(os.path.join(sample_dir, "lms_sam_overlay.png"), lms_mask_overlay)
    save_image(os.path.join(sample_dir, "sr_sam_overlay.png"), sr_mask_overlay)
    save_image(os.path.join(sample_dir, "lms_sam_masks.png"), lms_mask_flat)
    save_image(os.path.join(sample_dir, "sr_sam_masks.png"), sr_mask_flat)
    save_image(os.path.join(sample_dir, "lms_boundary_overlay.png"), lms_boundary_overlay)
    save_image(os.path.join(sample_dir, "sr_boundary_overlay.png"), sr_boundary_overlay)
    save_binary_mask(os.path.join(sample_dir, "lms_boundary_mask.png"), lms_boundary)
    save_binary_mask(os.path.join(sample_dir, "sr_boundary_mask.png"), sr_boundary)

    rgb_strip_images = [lms_rgb, sr_rgb]
    mask_overlay_strip_images = [lms_mask_overlay, sr_mask_overlay]
    boundary_strip_images = [lms_boundary_overlay, sr_boundary_overlay]
    mask_flat_strip_images = [lms_mask_flat, sr_mask_flat]

    if (
        gt_rgb is not None
        and gt_mask_overlay is not None
        and gt_mask_flat is not None
        and gt_boundary_overlay is not None
        and gt_boundary is not None
    ):
        save_image(os.path.join(sample_dir, "gt_rgb.png"), gt_rgb)
        save_image(os.path.join(sample_dir, "gt_sam_overlay.png"), gt_mask_overlay)
        save_image(os.path.join(sample_dir, "gt_sam_masks.png"), gt_mask_flat)
        save_image(os.path.join(sample_dir, "gt_boundary_overlay.png"), gt_boundary_overlay)
        save_binary_mask(os.path.join(sample_dir, "gt_boundary_mask.png"), gt_boundary)

        rgb_strip_images.append(gt_rgb)
        mask_overlay_strip_images.append(gt_mask_overlay)
        boundary_strip_images.append(gt_boundary_overlay)
        mask_flat_strip_images.append(gt_mask_flat)

    rgb_strip = make_horizontal_strip(rgb_strip_images)
    mask_overlay_strip = make_horizontal_strip(mask_overlay_strip_images)
    boundary_strip = make_horizontal_strip(boundary_strip_images)
    mask_flat_strip = make_horizontal_strip(mask_flat_strip_images)
    overview = make_vertical_stack([rgb_strip, mask_overlay_strip, boundary_strip, mask_flat_strip])

    save_image(os.path.join(sample_dir, "rgb_triplet.png"), rgb_strip)
    save_image(os.path.join(sample_dir, "sam_overlay_triplet.png"), mask_overlay_strip)
    save_image(os.path.join(sample_dir, "boundary_triplet.png"), boundary_strip)
    save_image(os.path.join(sample_dir, "sam_masks_triplet.png"), mask_flat_strip)
    save_image(os.path.join(sample_dir, "overview.png"), overview)


def compute_candidate_score(row: Dict[str, object], has_gt: bool) -> float:
    if has_gt:
        improvement = float(row.get("delta_f1_sr_minus_lms", 0.0))
        sr_f1 = float(row.get("sr_f1_vs_gt", 0.0))
        scene_complexity = min(float(row.get("gt_mask_count", 0.0)) / 200.0, 1.0)
        mask_count_alignment = 0.0
        gt_count = int(row.get("gt_mask_count", 0))
        sr_count = int(row.get("sr_mask_count", 0))
        if gt_count > 0:
            mask_count_alignment = 1.0 - min(abs(sr_count - gt_count) / float(gt_count), 1.0)
        return 0.60 * improvement + 0.25 * sr_f1 + 0.10 * scene_complexity + 0.05 * mask_count_alignment

    density_gain = float(row.get("delta_boundary_density_sr_minus_lms", 0.0))
    count_gain = float(row.get("sr_mask_count", 0.0)) - float(row.get("lms_mask_count", 0.0))
    return density_gain + 0.001 * count_gain


def ordered_fieldnames(rows: List[Dict[str, object]]) -> List[str]:
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames


def save_rows_csv(rows: List[Dict[str, object]], path: str):
    if not rows:
        return
    fieldnames = ordered_fieldnames(rows)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rank_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    def score_value(row: Dict[str, object]) -> float:
        value = row.get("candidate_score", -1e9)
        if value is None:
            return -1e9
        try:
            value = float(value)
        except (TypeError, ValueError):
            return -1e9
        if np.isnan(value):
            return -1e9
        return value

    return sorted(rows, key=score_value, reverse=True)


def save_ranking_csv(rows: List[Dict[str, object]], path: str, topk: Optional[int] = None):
    ranked_rows = rank_rows(rows)
    if topk is not None:
        ranked_rows = ranked_rows[:topk]
    save_rows_csv(ranked_rows, path)


def save_summary_txt(rows: List[Dict[str, object]], path: str, has_gt: bool):
    if not rows:
        return
    ranked_rows = rank_rows(rows)
    with open(path, "w") as f:
        f.write("SAM-based downstream-oriented qualitative segmentation summary\n")
        if has_gt:
            f.write(
                "Ranking favors SAM-derived boundary F1 improvement over LMS-up, together with "
                "absolute SR boundary agreement and scene complexity.\n\n"
            )
        else:
            f.write(
                "Ranking is heuristic only because no GT is available. It mainly reflects "
                "whether the SR result yields richer and denser object boundaries than LMS-up.\n\n"
            )
        keys = [
            "candidate_score",
            "lms_f1_vs_gt",
            "sr_f1_vs_gt",
            "delta_f1_sr_minus_lms",
            "lms_precision_vs_gt",
            "sr_precision_vs_gt",
            "lms_recall_vs_gt",
            "sr_recall_vs_gt",
            "lms_boundary_density",
            "sr_boundary_density",
            "lms_mask_count",
            "sr_mask_count",
            "gt_mask_count",
        ]
        available_keys = [key for key in keys if key in rows[0]]
        for key in available_keys:
            values = [float(row[key]) for row in rows if key in row]
            if values:
                f.write(f"{key}: mean={np.mean(values):.6f}, std={np.std(values):.6f}\n")
        f.write("\nTop candidates\n")
        for row in ranked_rows[: min(10, len(ranked_rows))]:
            sample_idx = int(row["sample_idx"])
            score = float(row["candidate_score"])
            overview_path = str(row.get("overview_path", ""))
            f.write(f"sample_{sample_idx:03d}: candidate_score={score:.6f} | {overview_path}\n")


def process_sample(
    collection_name: str,
    sample_idx: int,
    args,
    data_dict: Dict[str, np.ndarray],
    results_dir: str,
    collection_save_dir: str,
    h5_path: str,
    has_gt: bool,
    sam_generator: SamAutomaticMaskGenerator,
) -> Dict[str, object]:
    sample_dir = os.path.join(collection_save_dir, f"sample_{sample_idx:03d}")
    ensure_dir(sample_dir)

    lms = data_dict["lms"]
    gt = data_dict.get("gt")

    lms_hwc = to_hwc(lms[sample_idx])
    gt_hwc = to_hwc(gt[sample_idx]) if gt is not None else None

    mat_path = os.path.join(results_dir, f"output_mulExm_{sample_idx}.mat")
    sr_hwc = load_sr_from_mat(mat_path)

    lms_rgb = to_display_rgb(lms_hwc, args.tol_low, args.tol_high)
    sr_rgb = to_display_rgb(sr_hwc, args.tol_low, args.tol_high)
    gt_rgb = to_display_rgb(gt_hwc, args.tol_low, args.tol_high) if gt_hwc is not None else None

    lms_masks = generate_sam_masks(lms_rgb, sam_generator, args.sam_max_side)
    sr_masks = generate_sam_masks(sr_rgb, sam_generator, args.sam_max_side)
    gt_masks = generate_sam_masks(gt_rgb, sam_generator, args.sam_max_side) if gt_rgb is not None else []

    shape = lms_rgb.shape[:2]
    lms_label_map = masks_to_label_map(lms_masks, shape)
    sr_label_map = masks_to_label_map(sr_masks, shape)
    gt_label_map = masks_to_label_map(gt_masks, shape) if gt_masks else None

    lms_boundary = label_boundaries(lms_label_map)
    sr_boundary = label_boundaries(sr_label_map)
    gt_boundary = label_boundaries(gt_label_map) if gt_label_map is not None else None

    lms_mask_overlay = render_mask_overlay(lms_rgb, lms_masks, seed=args.seed + 11)
    sr_mask_overlay = render_mask_overlay(sr_rgb, sr_masks, seed=args.seed + 23)
    gt_mask_overlay = render_mask_overlay(gt_rgb, gt_masks, seed=args.seed + 37) if gt_rgb is not None else None

    lms_mask_flat = render_mask_flat(lms_masks, shape, seed=args.seed + 11)
    sr_mask_flat = render_mask_flat(sr_masks, shape, seed=args.seed + 23)
    gt_mask_flat = render_mask_flat(gt_masks, shape, seed=args.seed + 37) if gt_masks else None

    lms_boundary_overlay = overlay_boundaries(lms_rgb, lms_boundary, args.boundary_thickness)
    sr_boundary_overlay = overlay_boundaries(sr_rgb, sr_boundary, args.boundary_thickness)
    gt_boundary_overlay = (
        overlay_boundaries(gt_rgb, gt_boundary, args.boundary_thickness)
        if gt_rgb is not None and gt_boundary is not None
        else None
    )

    save_visual_assets(
        sample_dir=sample_dir,
        lms_rgb=lms_rgb,
        sr_rgb=sr_rgb,
        gt_rgb=gt_rgb,
        lms_mask_overlay=lms_mask_overlay,
        sr_mask_overlay=sr_mask_overlay,
        gt_mask_overlay=gt_mask_overlay,
        lms_mask_flat=lms_mask_flat,
        sr_mask_flat=sr_mask_flat,
        gt_mask_flat=gt_mask_flat,
        lms_boundary_overlay=lms_boundary_overlay,
        sr_boundary_overlay=sr_boundary_overlay,
        gt_boundary_overlay=gt_boundary_overlay,
        lms_boundary=lms_boundary,
        sr_boundary=sr_boundary,
        gt_boundary=gt_boundary,
    )

    np.save(os.path.join(sample_dir, "lms_label_map.npy"), lms_label_map)
    np.save(os.path.join(sample_dir, "sr_label_map.npy"), sr_label_map)
    np.save(os.path.join(sample_dir, "lms_boundary.npy"), lms_boundary)
    np.save(os.path.join(sample_dir, "sr_boundary.npy"), sr_boundary)
    if gt_label_map is not None and gt_boundary is not None:
        np.save(os.path.join(sample_dir, "gt_label_map.npy"), gt_label_map)
        np.save(os.path.join(sample_dir, "gt_boundary.npy"), gt_boundary)

    image_area = int(shape[0] * shape[1])
    row: Dict[str, object] = {
        "collection": collection_name,
        "sample_idx": int(sample_idx),
        "has_gt": bool(gt_boundary is not None),
        "sam_model_type": args.sam_model_type,
        "sam_points_per_side": int(args.sam_points_per_side),
        "sam_pred_iou_thresh": float(args.sam_pred_iou_thresh),
        "sam_stability_score_thresh": float(args.sam_stability_score_thresh),
        "mat_path": mat_path,
        "h5_path": h5_path,
        "overview_path": os.path.join(sample_dir, "overview.png"),
        "boundary_triplet_path": os.path.join(sample_dir, "boundary_triplet.png"),
        "sam_overlay_triplet_path": os.path.join(sample_dir, "sam_overlay_triplet.png"),
        "lms_boundary_density": float(lms_boundary.mean()),
        "sr_boundary_density": float(sr_boundary.mean()),
        "delta_boundary_density_sr_minus_lms": float(sr_boundary.mean() - lms_boundary.mean()),
    }

    row.update(mask_summary_stats(lms_masks, image_area, prefix="lms"))
    row.update(mask_summary_stats(sr_masks, image_area, prefix="sr"))
    if gt_masks:
        row.update(mask_summary_stats(gt_masks, image_area, prefix="gt"))

    if gt_boundary is not None:
        lms_stats = boundary_metrics(lms_boundary, gt_boundary, args.boundary_tol)
        sr_stats = boundary_metrics(sr_boundary, gt_boundary, args.boundary_tol)
        row.update(
            {
                "lms_precision_vs_gt": lms_stats["precision"],
                "lms_recall_vs_gt": lms_stats["recall"],
                "lms_f1_vs_gt": lms_stats["f1"],
                "sr_precision_vs_gt": sr_stats["precision"],
                "sr_recall_vs_gt": sr_stats["recall"],
                "sr_f1_vs_gt": sr_stats["f1"],
                "delta_f1_sr_minus_lms": float(sr_stats["f1"] - lms_stats["f1"]),
            }
        )

    row["candidate_score"] = compute_candidate_score(row, has_gt=bool(gt_boundary is not None))
    return row


def print_top_candidates(rows: List[Dict[str, object]], topk: int):
    ranked_rows = rank_rows(rows)[:topk]
    if not ranked_rows:
        return
    print("Top candidates:")
    for row in ranked_rows:
        print(
            f"  sample_{int(row['sample_idx']):03d} | "
            f"score={float(row['candidate_score']):.6f} | "
            f"{row.get('overview_path', '')}"
        )


def main():
    args = parse_args()
    ensure_dir(args.save_dir)
    sam_generator = build_sam_generator(args)

    collections = resolve_collections(args)
    global_rows: List[Dict[str, object]] = []
    manifest_rows: List[Dict[str, object]] = []

    for collection_name in collections:
        collection_dir = os.path.join(args.pan_root, collection_name)
        results_dir = find_results_dir(collection_dir)
        h5_path = resolve_h5_path(args.h5_root, collection_name)
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"H5 file not found for {collection_name}: {h5_path}")

        data_dict = load_h5_data(h5_path)
        discovered_indices = list_mat_indices(results_dir)
        selected_indices = select_sample_indices(
            requested_indices=args.indices,
            discovered_indices=discovered_indices,
            sample_count=int(data_dict["lms"].shape[0]),
            max_samples=int(args.max_samples),
        )
        if not selected_indices:
            raise RuntimeError(f"No valid sample indices found for {collection_name} in {results_dir}")

        collection_save_dir = os.path.join(args.save_dir, collection_name)
        ensure_dir(collection_save_dir)

        has_gt = bool(COLLECTION_SPECS[collection_name]["has_gt"]) and ("gt" in data_dict)
        collection_rows: List[Dict[str, object]] = []
        for sample_idx in selected_indices:
            row = process_sample(
                collection_name=collection_name,
                sample_idx=sample_idx,
                args=args,
                data_dict=data_dict,
                results_dir=results_dir,
                collection_save_dir=collection_save_dir,
                h5_path=h5_path,
                has_gt=has_gt,
                sam_generator=sam_generator,
            )
            collection_rows.append(row)
            global_rows.append(row)

        save_rows_csv(collection_rows, os.path.join(collection_save_dir, "sam_metrics.csv"))
        save_ranking_csv(collection_rows, os.path.join(collection_save_dir, "candidate_ranking.csv"))
        save_ranking_csv(
            collection_rows,
            os.path.join(collection_save_dir, f"candidate_ranking_top{int(args.topk_candidates)}.csv"),
            topk=int(args.topk_candidates),
        )
        save_summary_txt(collection_rows, os.path.join(collection_save_dir, "sam_summary.txt"), has_gt=has_gt)

        manifest_rows.append(
            {
                "collection": collection_name,
                "results_dir": results_dir,
                "h5_path": h5_path,
                "num_discovered_mat": len(discovered_indices),
                "num_processed_samples": len(collection_rows),
                "has_gt": has_gt,
            }
        )

        print(f"[{collection_name}] processed {len(collection_rows)} samples")
        print_top_candidates(collection_rows, int(args.topk_candidates))

    save_rows_csv(global_rows, os.path.join(args.save_dir, "global_sam_metrics.csv"))
    save_ranking_csv(global_rows, os.path.join(args.save_dir, "global_candidate_ranking.csv"))
    save_ranking_csv(
        global_rows,
        os.path.join(args.save_dir, f"global_candidate_ranking_top{int(args.topk_candidates)}.csv"),
        topk=int(args.topk_candidates),
    )
    save_rows_csv(manifest_rows, os.path.join(args.save_dir, "run_manifest.csv"))
    print(f"Saved outputs to: {args.save_dir}")


if __name__ == "__main__":
    main()
