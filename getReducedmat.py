import argparse
import os

import h5py
import numpy as np
import scipy.io as sio
import torch
from einops import rearrange

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
    parser = argparse.ArgumentParser(
        description="Export reduced-resolution pansharpening outputs to .mat files."
    )
    parser.add_argument("--ckpath", type=str, required=True, help="Path to the checkpoint file.")
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="pansharpening/test_data/WV3/test_wv3_multiExm1.h5",
        help="Path to the reduced-resolution test data file.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="2_DL_Result/PanCollection/WV3_Reduced/EvoARFS/results/",
        help="Directory to save exported .mat files.",
    )
    parser.add_argument(
        "--task",
        default="wv3",
        type=str,
        choices=["wv3", "qb", "gf2"],
        help="Dataset type.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=1000,
        help="Epoch argument passed to the model during inference.",
    )
    parser.add_argument(
        "--hw_range",
        nargs=2,
        type=int,
        default=[1, 18],
        help="Range used by ARConv during inference.",
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
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(args.task, args.fusion_type, args.num_refine, device)
    checkpoint = torch.load(args.ckpath, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    lms, ms, pan = load_set(args.test_data_path)
    del ms
    pan = pan.to(device)
    lms = lms.to(device)

    print(f"Exporting reduced-resolution results from: {args.test_data_path}")
    print(f"Model config: fusion_type={args.fusion_type}, num_refine={args.num_refine}, hw_range={args.hw_range}")

    with torch.no_grad():
        for i in range(pan.shape[0]):
            *_, output = model(pan[i], lms[i], args.epoch, args.hw_range)
            output = rearrange(output, "b c h w -> b h w c") * 2047.0
            output_np = output[0].cpu().numpy()
            save_mat_path = os.path.join(args.save_dir, f"output_mulExm_{i}.mat")
            sio.savemat(save_mat_path, {"sr": output_np})
            print(f"Saved .mat to {save_mat_path}")


if __name__ == "__main__":
    main()
