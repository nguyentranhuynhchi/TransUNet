#!/bin/bash
# ==============================================================================
# Script cài đặt môi trường và chuẩn bị dữ liệu cho TransUNet trên Colab CLI
# Repository: https://github.com/nguyentranhuynhchi/TransUNet.git
# Protocol: myops380_train304_test76_last_v1 (No-Val, 300 epochs)
# ==============================================================================

set -e

echo "=== [1/6] Kiểm tra và Kích hoạt GPU NVIDIA ==="
cd /content

# Tự động liên kết thư viện driver NVIDIA nếu có trên Colab (/usr/lib64-nvidia)
if [ -d "/usr/lib64-nvidia" ]; then
    if ! grep -q "/usr/lib64-nvidia" /etc/ld.so.conf.d/nvidia.conf 2>/dev/null; then
        echo "/usr/lib64-nvidia" >> /etc/ld.so.conf.d/nvidia.conf
        ldconfig 2>/dev/null || true
    fi
    export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH}
fi
if ! grep -q "/usr/lib64-nvidia" /root/.bashrc 2>/dev/null; then
    echo 'export LD_LIBRARY_PATH=/usr/lib64-nvidia:$LD_LIBRARY_PATH' >> /root/.bashrc
fi

if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️ CẢNH BÁO: Chưa tìm thấy NVIDIA GPU. Hãy kiểm tra lại Runtime trên Colab!"
else
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
fi

echo "=== [2/6] Kiểm tra Mount Google Drive ==="
python3 -c "
import os
if not os.path.exists('/content/drive/MyDrive'):
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print('✅ Mount Google Drive thành công!')
    except Exception as e:
        print('⚠️ Drive chưa mount tự động (bỏ qua nếu đã mount qua colab drivemount):', e)
else:
    print('✅ Google Drive đã được mount sẵn tại /content/drive/MyDrive')
" || true

echo "=== [3/6] Cài đặt Dependencies cho TransUNet ==="
python3 -m pip install --upgrade pip -q
python3 -m pip install ml_collections medpy SimpleITK scipy h5py gdown -q

echo "=== [4/6] Chuẩn bị Pretrained Checkpoint (R50-ViT-B_16.npz) ==="
mkdir -p /content/TransUNet/model/vit_checkpoint/imagenet21k
CKPT_FILE="/content/TransUNet/model/vit_checkpoint/imagenet21k/R50-ViT-B_16.npz"
DRIVE_CKPT="/content/drive/MyDrive/NCKH_V2/i_mmseg/ViT_checkpoint/R50-ViT-B_16.npz"

if [ -s "$CKPT_FILE" ]; then
    echo "✅ Checkpoint đã có sẵn tại: $CKPT_FILE"
elif [ -s "$DRIVE_CKPT" ]; then
    echo "Đang sao chép checkpoint từ Google Drive đã mount..."
    cp "$DRIVE_CKPT" "$CKPT_FILE"
    echo "✅ Sao chép checkpoint từ Google Drive thành công!"
else
    echo "Đang tải R50-ViT-B_16.npz từ Google Drive qua gdown..."
    gdown --id "1qJI7m6sM6deBZsSmcZjltHNWygRYagdD" -O "$CKPT_FILE" || {
        echo "Tải từ Drive thất bại, fallback sang server Google ViT Storage..."
        wget -c -O "$CKPT_FILE" https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz
    }
    echo "✅ Tải checkpoint hoàn tất!"
fi

echo "=== [5/6] Tải Dataset từ Thư mục Google Drive về SSD Cục bộ ==="
DATASET_FOLDER_ID="13T6892xy-pUwC1BSjGl4sRxoypMu-SIS"
DOWNLOAD_DIR="/content/dataset_download"
TARGET_DATASET_DIR="/content/MyoPS380_dataset"

mkdir -p "$DOWNLOAD_DIR"
mkdir -p "$TARGET_DATASET_DIR"

# Tải toàn bộ thư mục Google Drive nếu chưa tải
if [ -z "$(ls -A "$DOWNLOAD_DIR" 2>/dev/null)" ] && [ -z "$(ls -A "$TARGET_DATASET_DIR/Processed_data" 2>/dev/null)" ]; then
    echo "Đang tải toàn bộ thư mục dataset từ Google Drive (Folder ID: $DATASET_FOLDER_ID)..."
    gdown --folder --id "$DATASET_FOLDER_ID" -O "$DOWNLOAD_DIR" --remaining-ok || {
        echo "⚠️ Tải folder qua gdown gặp cảnh báo, tiếp tục kiểm tra các tệp đã tải..."
    }
fi

# Tự động giải nén bất kỳ tệp zip nào có trong thư mục tải về
for zipfile in $(find "$DOWNLOAD_DIR" -maxdepth 2 -name "*.zip" 2>/dev/null); do
    echo "Đang giải nén $zipfile vào $TARGET_DATASET_DIR ..."
    command -v unzip &>/dev/null || (apt-get update -qq && apt-get install -y unzip)
    unzip -q -o "$zipfile" -d "$TARGET_DATASET_DIR/"
done

# Nếu trong thư mục tải về đã là thư mục dữ liệu giải nén sẵn
if [ -d "$DOWNLOAD_DIR/Processed_data" ]; then
    cp -r "$DOWNLOAD_DIR/Processed_data" "$TARGET_DATASET_DIR/"
fi

# Cơ chế tự động định vị thư mục dữ liệu chuẩn (thư mục chứa bSSFP, LGE, T2w)
FOUND_BSSFP=$(find "$TARGET_DATASET_DIR" "$DOWNLOAD_DIR" -type d -name "bSSFP" 2>/dev/null | head -n 1)
if [ -n "$FOUND_BSSFP" ]; then
    LOCAL_DATASET=$(dirname "$FOUND_BSSFP")
    echo "✅ Đã xác định thư mục dataset chuẩn tại SSD: $LOCAL_DATASET"
else
    LOCAL_DATASET="/content/MyoPS380_dataset/Processed_data"
    echo "⚠️ Chưa phát hiện thư mục bSSFP, đặt mặc định: $LOCAL_DATASET"
fi

echo "=== [6/6] Kiểm tra PyTorch & Môi trường Hoàn tất ==="
command -v tmux &> /dev/null || (apt-get update -qq && apt-get install -y tmux)
python3 -c "import torch; print(f'PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo ""
echo "================================================================================"
echo "✅ THIẾT LẬP MÔI TRƯỜNG TRANSUNET HOÀN TẤT 100%!"
echo "Dataset chuẩn sẵn sàng tại: $LOCAL_DATASET"
echo "Pretrained weights sẵn sàng tại: $CKPT_FILE"
echo "================================================================================"
