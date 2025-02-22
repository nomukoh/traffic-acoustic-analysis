"""
オートエンコーダモデル(auto-encoder.py, CNN_x.py)を対象として、ラベルごとの再構成誤差を計算するプログラム。
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from encoder.auto_encoder.CNN_s import AutoEncoder, prepare_dataloader, output_settings

# タグのマッピングを明示
vehicle_map = {'car': 0, 'cv': 1}
direction_map = {'right': 0, 'left': 1}

# パラメータ設定
MODEL_PATH = "./encoder/best_model_s.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# logger の初期化
logger = output_settings()

# データローダの準備
_, val_loader = prepare_dataloader()

# モデルのロード
model = AutoEncoder().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()

# ラベル別再構成誤差の計算
def compute_reconstruction_error(model, dataloader, device):
    model.eval()
    errors = []
    locs = []
    speeds = []
    errors_by_vd = {
        (0, 0): [],
        (0, 1): [],
        (1, 0): [],
        (1, 1): []
    }

    with torch.no_grad():
        for data, speed, vtype, direc, loc in dataloader:
            data = data.to(device)
            reconstructed = model(data)
            error = torch.mean((reconstructed - data) ** 2, dim=(1, 2, 3))
            errors.extend(error.cpu().numpy())
            # ラベルと速度を保存
            locs.extend(loc.cpu().numpy())
            speeds.extend(speed.cpu().numpy())
            # (vtype, direc) ごとにエラーを追記
            for i in range(len(error)):
                vt = vtype[i].item()
                dr = direc[i].item()
                errors_by_vd[(vt, dr)].append(error[i].cpu().numpy())

    # vtype × direction ごとのエラー平均を計算
    errors_by_vd_mean = {}
    for key, val in errors_by_vd.items():
        if len(val) > 0:
            errors_by_vd_mean[key] = np.mean(val)
        else:
            errors_by_vd_mean[key] = None  # データがなければ None

    return np.array(errors), np.array(locs), np.array(speeds), errors_by_vd_mean

# 再構成誤差の計算
errors, locs, speeds, errors_by_vd_mean = compute_reconstruction_error(model, val_loader, DEVICE)

# ================================
#   ラベル(loc)ごとの統計量算出
# ================================
unique_labels = np.unique(locs)
locs_stats = {label: errors[locs == label].mean() for label in unique_labels}

# 速度範囲の設定
speed_bins = [0, 40, 60, 80, 100, float('inf')]
speed_labels = ['~40', '40~60', '60~80', '80~100', '100~']
speed_categories = np.digitize(speeds, speed_bins, right=True)

# 速度範囲ごとの統計量を計算
speed_stats = {speed_labels[i]: errors[speed_categories == i + 1].mean() 
               for i in range(len(speed_bins) - 1)}

# =============================
#   ログへの出力
# =============================
logger.info("Label-wise reconstruction errors:")
for label, mean_error in locs_stats.items():
    logger.info(f"Label {label}: Mean error = {mean_error:.4f}")

logger.info("Speed-wise reconstruction errors:")
for speed_range, mean_error in speed_stats.items():
    logger.info(f"Speed {speed_range}: Mean error = {mean_error:.4f}")

logger.info("Vtype-Direction-wise reconstruction errors (vtype, direction):")
# マッピング用の逆引き関数を作成
inv_vehicle_map = {v: k for k, v in vehicle_map.items()}
inv_direction_map = {v: k for k, v in direction_map.items()}

for key, mean_error in errors_by_vd_mean.items():
    # key は (vtype_id, direction_id) のタプル
    vtype_str = inv_vehicle_map[key[0]]
    direction_str = inv_direction_map[key[1]]
    logger.info(f"(vtype={vtype_str}, direction={direction_str}): Mean error = {mean_error}")


# =============================
#   グラフプロット
# =============================

# ラベルごとのヒストグラムのプロット
plt.figure(figsize=(10, 6))
for label in unique_labels:
    plt.hist(errors[locs == label], bins=50, alpha=0.6, label=f"Label {label}")
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.title("Reconstruction Error by Label")
plt.legend()
plt.savefig(os.path.join(logger.output_dir, 'label_wise_output.png'))
plt.close()

# 速度範囲ごとのヒストグラムのプロット
plt.figure(figsize=(10, 6))
error_min = errors.min()
error_max = errors.max()
bins = np.linspace(error_min, error_max, 50)  # ヒストグラムのビンを50個に設定
for i, speed_range in enumerate(speed_labels):
    plt.hist(
        errors[speed_categories == i + 1],
        bins=bins,
        alpha=0.6,
        label=f"Speed {speed_range}"
    )
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.title("Reconstruction Error by Speed Range")
plt.legend()
plt.savefig(os.path.join(logger.output_dir, 'speed_wise_output_fixed.png'))
plt.close()