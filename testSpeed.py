import argparse
import csv
import json
import os
import time

import h5py
import numpy as np
import torch
from thop import clever_format, profile

from models import EvoARFSNet


def load_set(file_path):
    with h5py.File(file_path, "r") as data:
        lms = (
            torch.from_numpy(np.array(data["lms"][...], dtype=np.float32) / 2047.0)
            .unsqueeze(dim=0)
            .permute([1, 0, 2, 3, 4])
            .float()
        )
        ms = (
            torch.from_numpy(np.array(data["ms"][...], dtype=np.float32) / 2047.0)
            .unsqueeze(dim=0)
            .permute([1, 0, 2, 3, 4])
            .float()
        )
        pan = (
            torch.from_numpy(np.array(data["pan"][...], dtype=np.float32) / 2047.0)
            .unsqueeze(dim=0)
            .permute([1, 0, 2, 3, 4])
            .float()
        )
    return lms, ms, pan


def sync_if_needed(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def save_summary(summary_path, result_dict):
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    file_exists = os.path.exists(summary_path)
    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result_dict.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(result_dict)


def build_model(task, fusion_type, num_refine, device):
    if task == "wv3":
        pan_channels, lms_channels = 1, 8
    elif task in ["qb", "gf2"]:
        pan_channels, lms_channels = 1, 4
    else:
        raise ValueError(f"Unsupported task: {task}")

    return EvoARFSNet(
        pan_channels,
        lms_channels,
        fusion_type=fusion_type,
        num_refine=num_refine,
    ).to(device)


def main():
    parser = argparse.ArgumentParser(description="Test the inference speed of EvoARFSNet.")
    parser.add_argument("--ckpath", type=str, required=True, help="Path to the checkpoint file.")
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="pansharpening/test_data/WV3/test_wv3_OrigScale_multiExm1.h5",
        help="Path to the test data file.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="2_DL_Result/PanCollection/EvoARFSNet",
        help="Retained for compatibility; speed logs are saved under speed_analysis/.",
    )
    parser.add_argument(
        "--task",
        default="wv3",
        type=str,
        choices=["wv3", "qb", "gf2"],
        help="Dataset type.",
    )
    parser.add_argument(
        "--fusion_type",
        default="implicit",
        type=str,
        choices=["add", "concat", "explicit", "implicit"],
        help="Fusion type used by the checkpoint.",
    )
    parser.add_argument(
        "--num_refine",
        default=2,
        type=int,
        choices=[1, 2, 3],
        help="Number of autoregressive refinement steps used by the checkpoint.",
    )
    parser.add_argument(
        "--hw_range",
        nargs=2,
        type=int,
        default=[1, 18],
        help="Range used by ARConv during inference.",
    )
    parser.add_argument("--warmup_runs", type=int, default=50, help="Number of warm-up iterations.")
    parser.add_argument("--test_runs", type=int, default=200, help="Number of timed iterations.")
    parser.add_argument("--sample_idx", type=int, default=0, help="Index of the test sample used for repeated timing.")
    parser.add_argument("--epoch", type=int, default=1000, help="Epoch argument passed to the model.")
    parser.add_argument("--save_name", type=str, default="EvoARFSNet", help="Name used when saving speed records.")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    speed_dir = os.path.join("speed_analysis", args.save_name)
    os.makedirs(speed_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.task, args.fusion_type, args.num_refine, device)
    checkpoint = torch.load(args.ckpath, map_location=device)
    model.load_state_dict(checkpoint["model"])

    param_size = sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
    print(f"Model Size (MB): {param_size:.2f}")
    model.eval()

    lms, ms, pan = load_set(args.test_data_path)
    del ms
    pan = pan.to(device)
    lms = lms.to(device)
    sample_idx = min(args.sample_idx, pan.shape[0] - 1)
    print(f"pan[i].shape: {pan[sample_idx].shape}, lms[i].shape: {lms[sample_idx].shape}")

    flops, params = profile(
        model,
        inputs=(pan[sample_idx], lms[sample_idx], args.epoch, args.hw_range),
        verbose=False,
    )
    flops, params = clever_format([flops, params], "%.3f")
    print(f"FLOPs: {flops}, Params: {params}")

    times = []

    with torch.no_grad():
        input_pan = pan[sample_idx]
        input_lms = lms[sample_idx]

        for _ in range(args.warmup_runs):
            _ = model(input_pan, input_lms, args.epoch, args.hw_range)
        sync_if_needed(device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        for _ in range(args.test_runs):
            sync_if_needed(device)
            start = time.perf_counter()
            _ = model(input_pan, input_lms, args.epoch, args.hw_range)
            sync_if_needed(device)
            end = time.perf_counter()
            times.append(end - start)

    times_ms = np.array(times) * 1000.0
    avg_time_ms = float(times_ms.mean())
    std_time_ms = float(times_ms.std())
    min_time_ms = float(times_ms.min())
    max_time_ms = float(times_ms.max())
    fps = 1000.0 / avg_time_ms
    peak_mem_mb = 0.0
    if device.type == "cuda":
        peak_mem_mb = float(torch.cuda.max_memory_allocated(device) / 1024 / 1024)

    print(f"Warm-up Runs: {args.warmup_runs}")
    print(f"Timed Runs: {args.test_runs}")
    print(f"Average Inference Time: {avg_time_ms:.2f} +- {std_time_ms:.2f} ms")
    print(f"Min/Max Inference Time: {min_time_ms:.2f} / {max_time_ms:.2f} ms")
    print(f"FPS: {fps:.2f}")
    if device.type == "cuda":
        print(f"Peak GPU Memory: {peak_mem_mb:.2f} MB")

    result_dict = {
        "model_name": args.save_name,
        "task": args.task,
        "fusion_type": args.fusion_type,
        "num_refine": args.num_refine,
        "hw_range": str(tuple(args.hw_range)),
        "device": str(device),
        "sample_idx": sample_idx,
        "warmup_runs": args.warmup_runs,
        "test_runs": args.test_runs,
        "test_data_path": args.test_data_path,
        "ckpath": args.ckpath,
        "input_pan_shape": str(tuple(pan[sample_idx].shape)),
        "input_lms_shape": str(tuple(lms[sample_idx].shape)),
        "model_size_mb": f"{param_size:.6f}",
        "flops": flops,
        "params": params,
        "avg_time_ms": f"{avg_time_ms:.6f}",
        "std_time_ms": f"{std_time_ms:.6f}",
        "min_time_ms": f"{min_time_ms:.6f}",
        "max_time_ms": f"{max_time_ms:.6f}",
        "fps": f"{fps:.6f}",
        "peak_gpu_mem_mb": f"{peak_mem_mb:.6f}",
    }

    raw_record = {
        "summary": result_dict,
        "times_ms": times_ms.tolist(),
    }
    raw_path = os.path.join(speed_dir, f"{args.task}_sample{sample_idx:03d}.json")
    with open(raw_path, "w") as f:
        json.dump(raw_record, f, indent=2)

    summary_path = os.path.join("speed_analysis", "speed_summary.csv")
    save_summary(summary_path, result_dict)
    print(f"Saved raw timing record to: {raw_path}")
    print(f"Appended summary record to: {summary_path}")


if __name__ == "__main__":
    main()
