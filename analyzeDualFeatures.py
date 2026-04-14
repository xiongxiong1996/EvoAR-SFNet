import argparse
import csv
import os
from typing import Dict, List, Tuple

import h5py
import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt

from models import EvoARFSNet

matplotlib.use("Agg")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze intermediate features from the spatial and frequency branches."
    )
    parser.add_argument("--ckpath", type=str, required=True, help="Path to checkpoint file.")
    parser.add_argument(
        "--test_data_path",
        type=str,
        required=True,
        help="Path to reduced/full-resolution H5 test file.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory to save feature visualizations and statistics.",
    )
    parser.add_argument(
        "--task",
        default="wv3",
        type=str,
        choices=["wv3", "qb", "gf2"],
        help="Dataset type.",
    )
    parser.add_argument(
        "--hw_range",
        nargs=2,
        type=int,
        default=[1, 18],
        help="Range used by ARConv.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=1000,
        help="Epoch argument passed to the model during inference.",
    )
    parser.add_argument(
        "--indices",
        nargs="*",
        type=int,
        default=None,
        help="Specific sample indices to analyze. If omitted, use the first max_samples images.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=8,
        help="Maximum number of samples to analyze when indices are not specified.",
    )
    parser.add_argument(
        "--topk_channels",
        type=int,
        default=6,
        help="Number of strongest channels to visualize for each branch.",
    )
    parser.add_argument(
        "--fft_quantile",
        type=float,
        default=0.99,
        help="Upper quantile for clipping FFT heatmaps.",
    )
    return parser.parse_args()


def load_set(file_path: str):
    data = h5py.File(file_path, "r")
    lms = torch.from_numpy(np.array(data["lms"][...], dtype=np.float32) / 2047.0)
    pan = torch.from_numpy(np.array(data["pan"][...], dtype=np.float32) / 2047.0)
    gt = None
    if "gt" in data:
        gt = torch.from_numpy(np.array(data["gt"][...], dtype=np.float32) / 2047.0)
    return lms.float(), pan.float(), gt.float() if gt is not None else None


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def minmax_norm(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    min_v = float(x.min())
    max_v = float(x.max())
    if max_v - min_v < 1e-8:
        return np.zeros_like(x)
    return (x - min_v) / (max_v - min_v)


def to_numpy_map(feat: torch.Tensor) -> np.ndarray:
    return feat.detach().float().cpu().numpy()


def get_channel_energy(feat: torch.Tensor) -> np.ndarray:
    energy = feat.detach().pow(2).mean(dim=(-1, -2)).squeeze(0).cpu().numpy()
    return energy


def topk_channel_indices(feat: torch.Tensor, topk: int) -> List[int]:
    energy = get_channel_energy(feat)
    topk = min(topk, len(energy))
    order = np.argsort(-energy)
    return order[:topk].tolist()


def feature_mean_map(feat: torch.Tensor) -> np.ndarray:
    fmap = feat.detach().abs().mean(dim=1).squeeze(0).cpu().numpy()
    return minmax_norm(fmap)


def feature_fft_map(feat: torch.Tensor, quantile: float) -> np.ndarray:
    x = feat.detach().mean(dim=1).squeeze(0)
    fft = torch.fft.fftshift(torch.fft.fft2(x))
    mag = torch.log1p(torch.abs(fft)).cpu().numpy()
    high = float(np.quantile(mag, quantile))
    mag = np.clip(mag, 0.0, high if high > 1e-8 else None)
    return minmax_norm(mag)


def flatten_feature(feat: torch.Tensor) -> torch.Tensor:
    return feat.detach().reshape(feat.shape[0], -1).float()


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a = flatten_feature(a)
    b = flatten_feature(b)
    val = torch.nn.functional.cosine_similarity(a, b, dim=1).mean()
    return float(val.item())


def pearson_correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    a = flatten_feature(a)
    b = flatten_feature(b)
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    num = (a * b).sum(dim=1)
    den = torch.sqrt((a.pow(2).sum(dim=1) + 1e-8) * (b.pow(2).sum(dim=1) + 1e-8))
    return float((num / den).mean().item())


def linear_cka(a: torch.Tensor, b: torch.Tensor) -> float:
    a = flatten_feature(a)
    b = flatten_feature(b)
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    gram_a = a @ a.t()
    gram_b = b @ b.t()
    hsic = (gram_a * gram_b).sum()
    norm_a = torch.linalg.norm(gram_a)
    norm_b = torch.linalg.norm(gram_b)
    val = hsic / (norm_a * norm_b + 1e-8)
    return float(val.item())


def spatial_sparsity(feat: torch.Tensor) -> float:
    fmap = feat.detach().abs().mean(dim=1).squeeze(0)
    threshold = fmap.mean() + fmap.std()
    return float((fmap > threshold).float().mean().item())


def high_frequency_ratio(feat: torch.Tensor) -> float:
    x = feat.detach().mean(dim=1).squeeze(0)
    fft = torch.fft.fftshift(torch.fft.fft2(x))
    mag = torch.abs(fft)
    h, w = mag.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=mag.device),
        torch.arange(w, device=mag.device),
        indexing="ij",
    )
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    dist = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    radius = 0.25 * min(h, w)
    high = mag[dist >= radius].sum()
    total = mag.sum() + 1e-8
    return float((high / total).item())


def pca_projection(feat: torch.Tensor, max_points: int = 2048) -> np.ndarray:
    x = feat.detach().squeeze(0).cpu().numpy()
    c, h, w = x.shape
    points = x.reshape(c, h * w).T
    if points.shape[0] > max_points:
        choice = np.linspace(0, points.shape[0] - 1, max_points).astype(np.int64)
        points = points[choice]
    points = points - points.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(points, full_matrices=False)
    proj = u[:, :2] * s[:2]
    return proj.astype(np.float32)


def save_heatmap(array: np.ndarray, title: str, path: str, cmap: str = "viridis"):
    plt.figure(figsize=(4.2, 4.0))
    plt.imshow(array, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def save_channel_grid(feat: torch.Tensor, title_prefix: str, path: str, topk: int):
    x = to_numpy_map(feat.squeeze(0))
    indices = topk_channel_indices(feat, topk)
    cols = min(3, len(indices))
    rows = int(np.ceil(len(indices) / cols))
    plt.figure(figsize=(4.5 * cols, 3.8 * rows))
    for plot_idx, ch_idx in enumerate(indices, start=1):
        plt.subplot(rows, cols, plot_idx)
        plt.imshow(minmax_norm(np.abs(x[ch_idx])), cmap="coolwarm")
        plt.title(f"{title_prefix} ch={ch_idx}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def save_pca_scatter(ar_feat: torch.Tensor, fd_feat: torch.Tensor, path: str):
    ar_proj = pca_projection(ar_feat)
    fd_proj = pca_projection(fd_feat)
    plt.figure(figsize=(5.2, 4.5))
    plt.scatter(ar_proj[:, 0], ar_proj[:, 1], s=10, alpha=0.55, label="Spatial branch")
    plt.scatter(fd_proj[:, 0], fd_proj[:, 1], s=10, alpha=0.55, label="Frequency branch")
    plt.legend()
    plt.title("Feature Distribution PCA")
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def save_similarity_bar(stats: Dict[str, float], path: str):
    names = ["cosine", "pearson", "cka"]
    values = [stats[k] for k in names]
    plt.figure(figsize=(4.8, 3.8))
    plt.bar(names, values, color=["#4C72B0", "#55A868", "#C44E52"])
    plt.ylim(min(-0.1, min(values) - 0.05), 1.0)
    plt.title("Cross-Branch Similarity")
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def save_csv(rows: List[Dict[str, float]], path: str):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: List[Dict[str, float]]) -> Dict[str, float]:
    summary = {}
    numeric_keys = [k for k in rows[0].keys() if k != "index"]
    for key in numeric_keys:
        summary[key] = float(np.mean([float(r[key]) for r in rows]))
    return summary


def write_summary_text(summary: Dict[str, float], path: str):
    lines = [
        "Dual-stream feature analysis summary",
        f"cosine_similarity_mean: {summary['cosine']:.6f}",
        f"pearson_correlation_mean: {summary['pearson']:.6f}",
        f"linear_cka_mean: {summary['cka']:.6f}",
        f"spatial_branch_high_freq_ratio_mean: {summary['ar_high_freq_ratio']:.6f}",
        f"frequency_branch_high_freq_ratio_mean: {summary['fd_high_freq_ratio']:.6f}",
        f"spatial_branch_sparsity_mean: {summary['ar_sparsity']:.6f}",
        f"frequency_branch_sparsity_mean: {summary['fd_sparsity']:.6f}",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def get_model(task: str, ckpath: str, device: torch.device):
    if task == "wv3":
        pan_channels, lms_channels = 1, 8
    else:
        pan_channels, lms_channels = 1, 4

    model = EvoARFSNet(pan_channels, lms_channels).to(device)
    checkpoint = torch.load(ckpath, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def unpack_model_outputs(outputs):
    if not isinstance(outputs, (tuple, list)):
        raise RuntimeError("Model outputs must be tuple/list.")

    if len(outputs) == 8 and isinstance(outputs[-1], dict):
        preds = outputs[:-1]
        feat_dict = outputs[-1]
        return preds, feat_dict

    raise RuntimeError(
        "Model does not return feature dict. Please modify EvoARFSNet.forward to return "
        "an extra dict when return_features=True."
    )


def analyze_one_sample(
    model,
    pan: torch.Tensor,
    lms: torch.Tensor,
    sample_idx: int,
    args,
    save_dir: str,
):
    sample_dir = os.path.join(save_dir, f"sample_{sample_idx:03d}")
    ensure_dir(sample_dir)

    with torch.no_grad():
        outputs = model(
            pan.unsqueeze(0),
            lms.unsqueeze(0),
            args.epoch,
            hw_range=args.hw_range,
            return_features=True,
        )
    _, feat_dict = unpack_model_outputs(outputs)

    ar_feat = feat_dict["x5_ar"]
    fd_feat = feat_dict["x5_fd"]
    fused_feat = feat_dict["x_refined"]

    ar_mean = feature_mean_map(ar_feat)
    fd_mean = feature_mean_map(fd_feat)
    fused_mean = feature_mean_map(fused_feat)
    ar_fft = feature_fft_map(ar_feat, args.fft_quantile)
    fd_fft = feature_fft_map(fd_feat, args.fft_quantile)
    fused_fft = feature_fft_map(fused_feat, args.fft_quantile)

    save_heatmap(ar_mean, "Spatial Branch Mean Activation", os.path.join(sample_dir, "ar_mean_map.png"))
    save_heatmap(fd_mean, "Frequency Branch Mean Activation", os.path.join(sample_dir, "fd_mean_map.png"))
    save_heatmap(fused_mean, "Fused Feature Mean Activation", os.path.join(sample_dir, "fused_mean_map.png"))
    save_heatmap(ar_fft, "Spatial Branch FFT Energy", os.path.join(sample_dir, "ar_fft_map.png"), cmap="magma")
    save_heatmap(fd_fft, "Frequency Branch FFT Energy", os.path.join(sample_dir, "fd_fft_map.png"), cmap="magma")
    save_heatmap(fused_fft, "Fused Feature FFT Energy", os.path.join(sample_dir, "fused_fft_map.png"), cmap="magma")
    save_channel_grid(ar_feat, "Spatial", os.path.join(sample_dir, "ar_top_channels.png"), args.topk_channels)
    save_channel_grid(fd_feat, "Frequency", os.path.join(sample_dir, "fd_top_channels.png"), args.topk_channels)
    save_pca_scatter(ar_feat, fd_feat, os.path.join(sample_dir, "branch_pca.png"))

    stats = {
        "index": sample_idx,
        "cosine": cosine_similarity(ar_feat, fd_feat),
        "pearson": pearson_correlation(ar_feat, fd_feat),
        "cka": linear_cka(ar_feat, fd_feat),
        "ar_high_freq_ratio": high_frequency_ratio(ar_feat),
        "fd_high_freq_ratio": high_frequency_ratio(fd_feat),
        "ar_sparsity": spatial_sparsity(ar_feat),
        "fd_sparsity": spatial_sparsity(fd_feat),
    }
    save_similarity_bar(stats, os.path.join(sample_dir, "similarity_bar.png"))
    return stats


def main():
    args = parse_args()
    ensure_dir(args.save_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lms, pan, _ = load_set(args.test_data_path)
    num_samples = lms.shape[0]
    if args.indices is None or len(args.indices) == 0:
        indices = list(range(min(args.max_samples, num_samples)))
    else:
        indices = [idx for idx in args.indices if 0 <= idx < num_samples]

    if not indices:
        raise ValueError("No valid sample indices were selected.")

    model = get_model(args.task, args.ckpath, device)

    stats_rows = []
    for idx in indices:
        print(f"Analyzing sample {idx}...")
        row = analyze_one_sample(
            model=model,
            pan=pan[idx].to(device),
            lms=lms[idx].to(device),
            sample_idx=idx,
            args=args,
            save_dir=args.save_dir,
        )
        stats_rows.append(row)

    csv_path = os.path.join(args.save_dir, "feature_stats.csv")
    save_csv(stats_rows, csv_path)

    summary = summarize(stats_rows)
    summary_path = os.path.join(args.save_dir, "feature_summary.txt")
    write_summary_text(summary, summary_path)

    print(f"Saved per-sample statistics to: {csv_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
