<div align="center">
</div>


<div align="center">

# EvoAR-SFNet: Evolutionary Autoregressive Spatial-Frequency Network for Remote Sensing Pansharpening

</div>

<!--ts-->

Pytorch implementation of the paper "[EvoAR-SFNet: Evolutionary Autoregressive Spatial-Frequency Network for Remote Sensing Pansharpening]()" (The paper has been submitted IEEE Transactions on Geoscience and Remote Sensing（TGRS） ).

## Introduction
![fig1](.\image\fig1.png)
- We design a Spatial-Frequency Dual-Stream framework that integrates ARConv and FDConv, with the BDFB module further enhancing the fusion of spatial and frequency features.
- We develop an EvoAR module, where multi-stage predictions are progressively refined through autoregressive feedback, while an evolutionary strategy adaptively adjusts the loss weights of each stage.
- Comprehensive experiments on multiple benchmark datasets (WV3, QB, and GF2) show that our method achieves competitive performance compared with existing approaches.

## Installation

### Prerequisites
Only test on Ubuntu 20.04 with:
- Python >= 3.10 (tested with Python3.10.16)
- PyTorch >= 2.1 (tested with torch 2.1.1)
- CUDA (tested with cuda_11.3)
- Other dependencies described in `requirements.txt`

### Clone this repository
Clone this code to your workspace. 
We call this directory as `$EvoARFS_ROOT`

```Shell
git clone https://github.com/xiongxiong1996/EvoARFD-Net.git
```

### Create a conda virtual environment and activate it (conda is optional)

```Shell
conda create -n EvoARFS python=3.10 -y
conda activate EvoARFS
```

### Install dependencies

```Shell
# Install pytorch via pip firstly.
pip install torch==2.1.1+cu118 torchvision==0.16.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install python packages
pip install -r requirements.txt
```

### Data preparation

In this study, we employ three pansharpening datasets (WV3, QB, and GF2) provided by **PanCollection**. This platform is specifically designed for research on remote sensing image fusion and mainly includes:

- Standardized training/testing data from WorldView-3, QuickBird, and GaoFen-2 (GF2)
- A unified PyTorch-based deep learning framework that supports rapid experimental reproduction
- A standardized evaluation toolkit for both traditional and deep learning methods

Dataset download link: [PanCollection Official Website](https://liangjiandeng.github.io/PanCollection.html)

The dataset file structure is as follows:

```
EvoARFS_ROOT/pansharpening
├── validation_data
│   ├── valid_gf2.h5
│   ├── valid_qb.h5
│   └── valid_wv3.h5
├── training_data
│   ├── train_gf2.h5
│   ├── train_qb.h5
│   └── train_wv3.h5
└── test_data
    ├── QB
    │   ├── test_qb_multiExm1.h5
    │   └── test_qb_OrigScale_multiExm1.h5
    ├── WV3
    │   ├── test_wv3_multiExm1.h5
    │   └── test_wv3_OrigScale_multiExm1.h5
    └── GF2
        ├── test_gf2_multiExm1.h5
        └── test_gf2_OrigScale_multiExm1.h5
```

```Shell
cd $EvoARFS_ROOT
ln -s $pansharpening_ROOT pansharpening
```


## Getting Started

### Training
For training, run
```Shell
python trainerEvoARFSNet.py --batch_size [batch_size] --epochs [epochs] --lr [learning_rate] --ckpt [save_model_every_ckpt_epochs] --train_set_path [path_to_your_train_set_path] --val_set_path [path_to_your_val_set_path] --checkpoint_save_path [path_to_your_save_path] --task [model_to_train (choices: wv3, qb, gf2)] --hw_range [the_range_of_the_height_and_width]
```

For example, run

```Shell
CUDA_VISIBLE_DEVICES=0 python trainerEvoARFSNet.py --batch_size 16 --epochs 560 --lr 0.0006 --ckpt 10 --train_set_path ./pansharpening/training_data/train_wv3.h5 --val_set_path ./pansharpening/validation_data/valid_wv3.h5 --checkpoint_save_path ./workdir/EvoARFSNet_wv3 --task 'wv3' --hw_range 0 18
```

### Validation

1. **Select the optimal model parameters for inference.** The [pre-trained model weights and corresponding `.mat` files](https://pan.baidu.com/s/17g_SrZVx9taQmi6FFngXeA?pwd=pncw ) can be downloaded from the provided cloud drive.

For testing, run
```Shell
python getReducedmat.py/getFullmat.py --ckpath [the_checkpoint_file_path] --test_data_path [test_data_file_path] --save_dir [results_data_file_path]
```

For example on low-resolution data, run

```Shell
python getReducedmat.py --ckpath ./pth/checkpoint_wv3.pth --test_data_path pansharpening/test_data/WV3/test_wv3_multiExm1.h5 --save_dir 2_DL_Result/PanCollection/WV3_Reduced/EvoARFS/results/ --task wv3
```

For example on full-resolution data, run

```Shell
python getFullmat.py --ckpath ./pth/checkpoint_wv3.pth --test_data_path pansharpening/test_data/WV3/test_wv3_OrigScale_multiExm1.h5 --save_dir 2_DL_Result/PanCollection/WV3_Full/EvoARFS/results/ --task wv3
```

2. **Evaluate the results using MATLAB.** (All experiments in this paper were tested with **MATLAB R2024a**.)

For reference, please check the GitHub repository: [AFAR-Net](https://github.com/xiongxiong1996/AFAR-Net)

## Results

We present both qualitative and quantitative analyses of pansharpening results obtained by different methods on the WV3, QB, and GF2 datasets, covering both low-resolution and full-resolution cases.

**Visual comparison of pansharpening results obtained by different methods on low-resolution**

![fige1](.\image\fige1.png)

**Visual comparison of pansharpening results obtained by different methods on low-resolution**

![fige2](.\image\fige2.png)

**Performance on the WV3 Dataset (Mean±Std). Best in bold, second-best in italics**

|       Method        |       SAM       |      ERGAS      |          Q          |          D_λ          |        D_s        |         QNR         |
| :-----------------: | :-------------: | :-------------: | :-----------------: | :-------------------: | :---------------: | :-----------------: |
|  BDSD-PC(2019)[32]  |   5.429±1.823   |   4.698±1.617   |     0.829±0.097     |     0.0625±0.0235     |   0.0730±0.0356   |     0.870±0.053     |
|  CVPR19(2019)[35]   |   5.207±1.574   |   5.484±1.505   |     0.764±0.088     |     0.0297±0.0059     |   0.0410±0.0136   |     0.931±0.018     |
| LRTCFPan(2023)[36]  |   4.737±1.412   |   4.315±1.442   |     0.846±0.091     |     0.0176±0.0066     |   0.0528±0.0258   |     0.931±0.031     |
|   DiCNN(2018)[24]   |   3.593±0.762   |   2.673±0.663   |     0.900±0.087     |     0.0362±0.0111     |   0.0462±0.0175   |     0.920±0.026     |
| FusionNet(2021)[40] |   3.325±0.698   |   2.467±0.645   |     0.904±0.090     |     0.0239±0.0090     |   0.0364±0.0137   |     0.941±0.020     |
|  DCFNet(2021)[25]   |   3.038±0.585   |   2.165±0.499   |     0.913±0.087     |     0.0187±0.0072     |   0.0337±0.0054   |     0.948±0.012     |
|  LAGConv(2022)[27]  |   3.104±0.559   |   2.300±0.613   |     0.910±0.091     |     0.0368±0.0148     |   0.0418±0.0152   |     0.923±0.025     |
|  HMPNet(2023)[41]   |   3.063±0.577   |   2.229±0.545   |     0.916±0.087     |     0.0184±0.0073     |   0.0530±0.0555   |     0.930±0.011     |
|    CMT(2024)[26]    |   2.994±0.607   |   2.214±0.516   |     0.917±0.085     |     0.0207±0.0082     |   0.0370±0.0078   |     0.943±0.014     |
|  CANNet(2024)[44]   |   2.930±0.593   |   2.158±0.515   |     0.920±0.084     |     0.0196±0.0083     |   0.0301±0.0074   |     0.951±0.013     |
|  ARConv(2025) [28]  |  _2.885±0.590_  |  _2.139±0.528_  | **0.921**±**0.083** | **0.0146**±**0.0059** |  _0.0279±0.0068_  |    *0.958±0.010*    |
|  EvoAR-SFNet(Ours)  | **2.877±0.591** | **2.119±0.527** |   **0.922±0.084**   |   *0.0151± 0.0059*    | **0.0274±0.0040** | **0.958**±**0.008** |

**Performance on the QB Dataset (Mean±Std). Best in bold, second-best in italics**

|       Method        |       SAM       |        ERGAS        |          Q          |          D_λ          |        D_s        |         QNR         |
| :-----------------: | :-------------: | :-----------------: | :-----------------: | :-------------------: | :---------------: | :-----------------: |
|  BDSD-PC(2019)[32]  |   8.089±1.980   |     7.515±0.800     |     0.831±0.090     |     0.1975±0.0334     |   0.1636±0.0483   |     0.672±0.058     |
|  CVPR19(2019)[35]   |   7.998±.820    |     9.359±1.268     |     0.737±0.087     |     0.0498±0.0119     |   0.0783±0.0170   |     0.876±0.023     |
| LRTCFPan(2023)[36]  |   7.187±1.711   |     6.928±0.812     |     0.855±0.087     | **0.0226**±**0.0117** |   0.0705±0.0351   |     0.909±0.044     |
|   DiCNN(2018)[24]   |   5.380±1.027   |     5.135±0.488     |     0.904±0.094     |     0.0947±0.0145     |   0.1067±0.0210   |     0.809±0.031     |
| FusionNet(2021)[40] |   4.923±0.908   |     4.159±0.321     |     0.925±0.090     |     0.0572±0.0182     |   0.0522±0.0088   |     0.894±0.021     |
|  DCFNet(2021)[25]   |   4.512±0.773   |     3.809±0.336     |     0.934±0.087     |     0.0469±0.0150     |   0.1239±0.0269   |     0.835±0.016     |
|  LAGConv(2022)[27]  |   4.547±0.830   |     3.826±0.420     |     0.934±0.088     |     0.0859±0.0237     |   0.0676±0.0136   |     0.852±0.018     |
|  HMPNet(2023)[41]   |   4.617±0.404   | **3.404**±**0.478** |     0.936±0.102     |     0.1832±0.0542     |   0.0793±0.0245   |     0.753±0.065     |
|    CMT(2024)[26]    |   4.535±0.822   |     3.744±0.321     |     0.935±0.086     |     0.0504±0.0122     |  _0.0368±0.0075_  |     0.915±0.016     |
|  CANNet(2024)[44]   |   4.507±0.835   |     3.652±0.327     |     0.937±0.083     |    _0.0370±0.0129_    |   0.0499±0.0092   |     0.915±0.012     |
|  ARConv(2025)[28]   | **4.430±0.811** |     3.633±0.327     | **0.939**±**0.081** |     0.0384±0.0148     |   0.0396±0.0090   |    _0.924±0.019_    |
|  EvoAR-SFNet(Ours)  |  *4.470±0.831*  |    *3.603±0.334*    |    *0.939±0.082*    |     0.0413±0.0122     | **0.0319±0.0186** | **0.928**±**0.028** |

**Performance on the GF2 Dataset (Mean±Std). Best in bold, second-best in italics**

|       Method        |       SAM       |        ERGAS        |        Q        |        D_λ        |        D_s        |         QNR         |
| :-----------------: | :-------------: | :-----------------: | :-------------: | :---------------: | :---------------: | :-----------------: |
|  BDSD-PC(2019)[32]  |   1.681±0.360   |     1.667±0.445     |   0.892±0.035   |   0.0759±0.0301   |   0.1548±0.0280   |     0.781±0.041     |
|  CVPR19(2019)[35]   |   1.598±0.353   |     1.877±0.448     |   0.886±0.028   |   0.0307±0.0127   |   0.0622±0.0101   |     0.909±0.017     |
| LRTCFPan(2023)[36]  |   1.315±0.283   |     1.301±0.313     |   0.932±0.033   |   0.0325±0.0269   |   0.0896±0.0141   |     0.881±0.023     |
|   DiCNN(2018)[24]   |   1.053±0.231   |     1.081±0.254     |   0.959±0.010   |   0.0369±0.0132   |   0.0992±0.0131   |     0.868±0.016     |
| FusionNet(2021)[40] |   0.974±0.212   |     0.988±0.222     |   0.964±0.009   |   0.0350±0.0124   |   0.1013±0.0134   |     0.867±0.018     |
|  DCFNet(2021)[25]   |   0.872±0.169   |     0.784±0.146     |   0.974±0.009   |   0.0240±0.0115   |   0.0659±0.0096   |     0.912±0.012     |
|  LAGConv(2022)[27]  |   0.786±0.148   |     0.687±0.113     |   0.981±0.008   |   0.0284±0.0130   |   0.0792±0.0136   |     0.895±0.020     |
|  HMPNet(2023)[41]   |   0.803±0.141   | **0.564**±**0.099** |   0.981±0.020   |   0.0819±0.0499   |   0.1146±0.0126   |     0.813±0.049     |
|    CMT(2024)[26]    |   0.753±0.138   |     0.648±0.109     |   0.982±0.007   |   0.0225±0.0116   |  _0.0433±0.0096_  |    _0.935±0.014_    |
|  CANNet(2024)[44]   |  *0.707±0.148*  |     0.630±0.128     |  _0.983±0.006_  |   0.0194±0.0101   |   0.0630±0.0094   |     0.919±0.011     |
|  ARConv(2025)[28]   | **0.698±0.149** |    *0.626±0.127*    | **0.983±0.007** |  *0.0189±0.0097*  |   0.0515±0.0099   |     0.931±0.012     |
|  EvoAR-SFNet(Ours)  |   0.845±0.168   |     0.772±0.141     |   0.977±0.008   | **0.0188±0.0102** | **0.0366±0.0110** | **0.945**±**0.011** |

## Citation

If our paper and code are beneficial to your work, please consider citing:
```
@InProceedings{**,
    author    = {**,
    title     = {**,
    booktitle = {**,
    month     = {**,
    year      = {**,
    pages     = {**
**
```

<!--te-->
