# TransUNet
This repo holds code for [TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation](https://arxiv.org/pdf/2102.04306.pdf)

## 📰 News
- [7/26/2024] TransUNet, which supports both 2D and 3D data and incorporates a Transformer encoder and decoder, has been featured in the journal Medical Image Analysis ([link](https://www.sciencedirect.com/science/article/pii/S1361841524002056)).
```bibtex
@article{chen2024transunet,
  title={TransUNet: Rethinking the U-Net architecture design for medical image segmentation through the lens of transformers},
  author={Chen, Jieneng and Mei, Jieru and Li, Xianhang and Lu, Yongyi and Yu, Qihang and Wei, Qingyue and Luo, Xiangde and Xie, Yutong and Adeli, Ehsan and Wang, Yan and others},
  journal={Medical Image Analysis},
  pages={103280},
  year={2024},
  publisher={Elsevier}
}
```

- [10/15/2023] 🔥 3D version of TransUNet is out! Our 3D TransUNet surpasses nn-UNet with 88.11% Dice score on the BTCV dataset and outperforms the top-1 solution in the BraTs 2021 challenge and secure the second place in BraTs 2023 challenge. Please take a look at the [code](https://github.com/Beckschen/3D-TransUNet/tree/main) and [paper](https://arxiv.org/abs/2310.07781).


## Usage

### MyoPS++ ROI128 (local dataset)

The training/evaluation scripts support the existing MyoPS380 NPZ/H5 data and
the aligned `MyoPSpp_UNet2D_ROI128_1mm` NIfTI export. The TransUNet architecture,
four output channels, loss (`0.5 * CE + 0.5 * Dice`), and optimizer are unchanged.

```text
MyoPSpp_UNet2D_ROI128_1mm/
  splits.json
  cases/CenterB/Case2001/Case2001_{C0,LGE,T2,gd}.nii.gz
  cases/CenterC/Case3001/Case3001_{C0,LGE,T2,gd}.nii.gz
```

`splits.json` determines the patient split; no random re-splitting or validation
set is introduced. This export contains 64 training patients (296 slices) and
16 test patients (63 slices). Every slice is retained. Images must already be
aligned, 128x128 in-plane, and normalized to [0,1]. Missing modalities, mismatched
geometry, unknown label IDs, and overlapping patient splits raise errors.

Input channels are always **C0 (bSSFP), LGE, T2**. Raw masks are converted once:

| Raw NIfTI ID | Training target / output channel |
| --- | --- |
| 0, 500 (LV), 600 (RV) | 0: background |
| 200 | 1: normal myocardium |
| 1220 | 2: edema (exclusive of scar) |
| 2221 | 3: scar |

Both CE and Dice receive the same `torch.long` target. MyoPS++ never passes
through the legacy MyoPS380 scar/edema swap. Spatial augmentations are shared
across all three images and the mask; masks use nearest-neighbor interpolation.
Both datasets use the same original MyoPS380 augmentation (`im_mseg_legacy255_v1`):
brightness clips to [0,255], gamma computes `(image / 255) ** gamma * 255`, and
there is no additional [0,1] clipping after resize. The source images remain in
[0,1]; they are not multiplied by 255 before augmentation. Rotation, flipping,
noise, translation and cropping also follow the same implementation and RNG draws.

Train from PowerShell (small batch for the local 4 GB GPU):

```powershell
python train_myops.py --dataset myopspp --data-root "D:\NCKH\Dataset\MyoPSpp_UNet2D_ROI128_1mm" --batch-size 1 --num-workers 0 --amp fp16
```

The default budget remains 300 epochs; batch size and learning rate can be set
with `--batch-size` and `--lr`. `--pretrained` is optional; omitting it trains from
scratch. Dataset format can also be detected automatically from `--data-root`.
`--list-dir` is only used for MyoPS380.

The default MyoPS++ run directory is `outputs/runs/transunet_myopspp_seed1234`.
Checkpoints include the dataset identity, split hash, input order and raw label
mapping and augmentation protocol; resuming with a different dataset/split/mapping
or augmentation is rejected. Use the same arguments to auto-resume the last
checkpoint, or pass `--resume <checkpoint>`.
If a MyoPS++ checkpoint used the earlier intensity-max=1 implementation, start a
fresh run, e.g. `--run-name transunet_myopspp_aug380_seed1234`. Original
MyoPS380 checkpoints retain their existing augmentation and remain resumable.

```powershell
python evaluate_myops.py --dataset myopspp --data-root "D:\NCKH\Dataset\MyoPSpp_UNet2D_ROI128_1mm" --checkpoint "outputs/runs/transunet_myopspp_seed1234/checkpoints/transunet_seed1234.pt"
python -m unittest discover -s tests -v
```

Evaluation uses the same canonical class order and XYZ/HWD axes as training.
Saved predictions are canonical IDs 0..3. The report describes **cropped ROI**
metrics, with HD95/ASD in voxels; it does not claim native-image millimeter distances
or official MyoPS++ challenge results.

### 1. Download Google pre-trained ViT models
* [Get models in this link](https://console.cloud.google.com/storage/vit_models/): R50-ViT-B_16, ViT-B_16, ViT-L_16...
```bash
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz &&
mkdir ../model/vit_checkpoint/imagenet21k &&
mv {MODEL_NAME}.npz ../model/vit_checkpoint/imagenet21k/{MODEL_NAME}.npz
```

[Update 2026/02] The official ViT weights appear to have expired. 
You can still download a copy from the [project folder](https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd?usp=sharing) (same to BTCV preprocessed data). After extraction, find the file at:
`../model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz`

### 2. Prepare data (All data are available!)

All data are available so no need to send emails for data. Please use the [BTCV preprocessed data](https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd?usp=sharing) and [ACDC data](https://drive.google.com/drive/folders/1KQcrci7aKsYZi1hQoZ3T3QUtcy7b--n4?usp=drive_link).

### 3. Environment

Use Python >=3.10 and `pip install -r requirements.txt`. The current MyoPS scripts
are tested with Python 3.12, PyTorch 2.6.0 and torchvision 0.21.0 (CUDA 12.4 on the
local machine); the old PyTorch 1.4 environment does not support their AMP APIs.

### 4. Train/Test

- Run the train script on synapse dataset. The batch size can be reduced to 12 or 6 to save memory (please also decrease the base_lr linearly), and both can reach similar performance.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset Synapse --vit_name R50-ViT-B_16
```

- Run the test script on synapse dataset. It supports testing for both 2D images and 3D volumes.

```bash
python test.py --dataset Synapse --vit_name R50-ViT-B_16
```

## Reference
* [Google ViT](https://github.com/google-research/vision_transformer)
* [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
* [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)

## Citations


```bibtex
@article{chen2021transunet,
  title={TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation},
  author={Chen, Jieneng and Lu, Yongyi and Yu, Qihang and Luo, Xiangde and Adeli, Ehsan and Wang, Yan and Lu, Le and Yuille, Alan L., and Zhou, Yuyin},
  journal={arXiv preprint arXiv:2102.04306},
  year={2021}
}
```
