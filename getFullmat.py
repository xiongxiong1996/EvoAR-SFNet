import torch
import torch.nn as nn
import os
import scipy.io as sio
from einops import rearrange
from models import EvoARFSNet
import h5py
import numpy as np
import argparse
def load_set(file_path):
    data = h5py.File(file_path)
    lms = torch.from_numpy(np.array(data['lms'][...], dtype=np.float32) / 2047.).unsqueeze(dim=0).permute(
        [1, 0, 2, 3, 4]).float()
    ms = torch.from_numpy(np.array(data['ms'][...], dtype=np.float32) / 2047.).unsqueeze(dim=0).permute(
        [1, 0, 2, 3, 4]).float()
    pan = torch.from_numpy(np.array(data['pan'][...], dtype=np.float32) / 2047.).unsqueeze(dim=0).permute(
        [1, 0, 2, 3, 4]).float()
    return lms, ms, pan


def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='Process checkpoint file to get reduced matrix.')
    # 添加--ckpath参数
    parser.add_argument('--ckpath', type=str, required=True, help='Path to the checkpoint file')
    # 添加其他路径参数，设置默认值
    parser.add_argument('--test_data_path', type=str, default=r'pansharpening/test_data/WV3/test_wv3_OrigScale_multiExm1.h5', 
                        help='Path to the test data file')
    parser.add_argument('--save_dir', type=str, default=r'2_DL_Result/PanCollection/WV3_Full/EvoARFDSFNet727/results/',
                        help='Directory to save the results')
    parser.add_argument(
        "--task",
        default="wv3",
        type=str,
        choices=["wv3", "qb", "gf2"],
        help="Model to train (choices: wv3, qb, gf2).",
    )


    # 解析命令行参数
    args = parser.parse_args()
    
    # 获取所有路径
    checkpoint_path = args.ckpath
    test_data_path = args.test_data_path
    save_dir = args.save_dir
    task = args.task
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    if task == "wv3":
        pan_channels, lms_channels = 1, 8
    elif task in ["qb", "gf2"]:
        pan_channels, lms_channels = 1, 4

    # 加载模型
    model = EvoARFSNet(pan_channels, lms_channels).cuda()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # 加载测试数据
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lms, ms, pan = load_set(test_data_path)
    pan = pan.to(device)
    lms = lms.to(device)
    ms = ms.to(device)

    # 推理所有图像
    with torch.no_grad():
        print('Running model inference...')
        for i in range(pan.shape[0]):
            _,_,_,_,_,_,output = model(pan[i], lms[i], 1000, [1, 18])
            output = rearrange(output, 'b c h w -> b h w c') * 2047
            output_np = output[0].cpu().numpy()
            
            save_mat_path = os.path.join(save_dir, f'output_mulExm_{i}.mat')
            sio.savemat(save_mat_path, {'sr': output_np})
            print(f"Saved .mat to {save_mat_path}")

if __name__ == '__main__':
    main()