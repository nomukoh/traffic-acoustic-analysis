"""
poetry run python -m encoder.auto-encoder
"""

import os
import sys
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.cm as cm
from sklearn.decomposition import PCA
import logging
import random
from sklearn.manifold import TSNE

# ハイパーパラメータの設定
DATA_CSV_PATH = './encoder/datasets/data_1-6.csv'
# DATA_CSV_PATH = './encoder/datasets/loc1_cut/data.csv' # data1のみを対象とする場合（新たにcsvファイルを作成する必要あり）
MEL = 'OFF' # ON or OFF 注: Decoder を手動で変更する必要あり
SAMPLING_RATE = 16000
N_FFT = 2048  # 1024 or 2048 がよく使われる（小さいほうが軽い）
HOP_LENGTH = 512
TEST_DATASET_PERCENTAGE = 0.2
BATCH_SIZE = 32
SEED = 42
LEARNING_RATE = 0.001
EPOCHS = 60
SAVE_LATENT_SPACE = 'n' # y or n : 潜在空間を保存する 

# 実行時間を使って出力ディレクトリを作成
current_time = datetime.now().strftime('%Y-%m-%d/%H-%M-%S')
output_dir = os.path.join('./outputs', current_time)
os.makedirs(output_dir, exist_ok=True)

# ログの設定
log_file_path = os.path.join(output_dir, 'training.log')

# ロガーの作成
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ハンドラの作成
file_handler = logging.FileHandler(log_file_path)
console_handler = logging.StreamHandler(sys.stdout)

# フォーマットの設定
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# ハンドラをロガーに追加
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# ハイパーパラメータをログに記録
logger.info('Hyperparameters:')
hyperparameters = {
    'DATA_CSV_PATH': DATA_CSV_PATH,
    'MEL': MEL,
    'SAMPLING_RATE': SAMPLING_RATE,
    'N_FFT': N_FFT,
    'HOP_LENGTH': HOP_LENGTH,
    'TEST_DATASET_PERCENTAGE': TEST_DATASET_PERCENTAGE,
    'BATCH_SIZE': BATCH_SIZE,
    'SEED': SEED,
    'LEARNING_RATE': LEARNING_RATE,
    'EPOCHS': EPOCHS
}
for key, value in hyperparameters.items():
    logger.info(f'{key}: {value}')

# デバイスの確認
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if device == 'cuda':
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# データの読み込み
data = pd.read_csv(DATA_CSV_PATH)

# 音声データを読み込む関数（スペクトログラムではなく生データ）
def load_audio(file_path, sr=SAMPLING_RATE):
    y, sr = librosa.load(file_path, sr=sr)
    return y


# すべてのリストを作成
waveforms = []
speeds = []
vehicle_types_list = []
directions_list = []

# すべての音声データを読み込み
logger.info("音声データをロードしています...")
max_length = 0
raw_data = []
for i, row in data.iterrows():
    file_path = os.path.join('./encoder/datasets/', row['path'])
    y = load_audio(file_path, sr=SAMPLING_RATE)
    raw_data.append((y, row['speed'], row['vehicle_type'], row['direction']))
    if len(y) > max_length:
        max_length = len(y)

logger.info("音声データの読み込みが完了しました。")
logger.info(f"最大波形長: {max_length}")


# データの準備
# 全てのデータを同一長さに揃える(ゼロパディング)
for (y, sp, vt, dr) in raw_data:
    if len(y) < max_length:
        # 後ろにゼロでパディング
        padded = np.pad(y, (0, max_length - len(y)), 'constant')
    else:
        padded = y[:max_length]  # もし超過していたら切り詰める(稀なら対処)
    waveforms.append(padded)
    speeds.append(sp)
    vehicle_types_list.append(vt)
    directions_list.append(dr)

waveforms = np.array(waveforms, dtype=np.float32)
speeds = np.array(speeds, dtype=np.float32)
vehicle_types_list = np.array(vehicle_types_list)
directions_list = np.array(directions_list)

# 正規化(0~1)
wave_min = waveforms.min()
wave_max = waveforms.max()
waveforms = (waveforms - wave_min) / (wave_max - wave_min)

# vehicle_type と direction をエンコード（例：car=0, cv=1 / right=0, left=1）
vehicle_map = {'car':0, 'cv':1}
direction_map = {'right':0, 'left':1}

# データの分割
train_data, val_data, speeds_train, speeds_val, vehicle_train, vehicle_val, direction_train, direction_val = train_test_split(
    waveforms, speeds, vehicle_types_list, directions_list,
    test_size=TEST_DATASET_PERCENTAGE, random_state=SEED
)

vehicle_train_encoded = torch.tensor([vehicle_map.get(v,0) for v in vehicle_train], dtype=torch.long)
vehicle_val_encoded = torch.tensor([vehicle_map.get(v,0) for v in vehicle_val], dtype=torch.long)
direction_train_encoded = torch.tensor([direction_map.get(d,0) for d in direction_train], dtype=torch.long)
direction_val_encoded = torch.tensor([direction_map.get(d,0) for d in direction_val], dtype=torch.long)

# テンソルへの変換 (1次元なのでunsqueeze(0)して [B, 1, L] にする)
train_tensors = [torch.tensor(x, dtype=torch.float32).unsqueeze(0) for x in train_data]
val_tensors = [torch.tensor(x, dtype=torch.float32).unsqueeze(0) for x in val_data]
train_speeds = torch.tensor(speeds_train, dtype=torch.float32)
val_speeds = torch.tensor(speeds_val, dtype=torch.float32)

# Datasetの作成
train_dataset = TensorDataset(torch.stack(train_tensors), train_speeds, vehicle_train_encoded, direction_train_encoded)
val_dataset = TensorDataset(torch.stack(val_tensors), val_speeds, vehicle_val_encoded, direction_val_encoded)

# DataLoaderの作成
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

logger.info("データの準備が完了しました。")


# モデルの定義（1次元Conv/Deconv）
class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv1d(1, 8, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(8, 16, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(16, 32, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
        )
    def forward(self, x):
        # x: [B, 1, L]
        x = self.encoder(x) # [B, 32, L/8程度]
        return x

class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose1d(16, 8, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose1d(8, 1, kernel_size=4, stride=2, padding=1),
            torch.nn.Sigmoid()
        )
    def forward(self, x):
        # x: [B, 32, L/8]
        x = self.decoder(x) # [B, 1, L]
        return x

class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder()
        self.dec = Decoder()
    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x

logger.info("モデルの定義が完了しました。")


# モデルの初期化と学習
model = AutoEncoder().to(device)
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# 保存済みのモデルとオプティマイザの状態を読み込む
# model.load_state_dict(torch.load('./encoder/trained.pth', map_location=device))  # モデルの重みを読み込む
# optimizer.load_state_dict(torch.load('./encoder/optimizer.pth', map_location=device))  # オプティマイザの状態を読み込む

logger.info("学習を開始します。")
best_valid_loss = float('inf')

for epoch in range(EPOCHS):
    # 学習ループ
    model.train()
    train_loss = 0.0
    for data, speed, vtype, direc in tqdm(train_loader, desc=f'Train Epoch {epoch+1}/{EPOCHS}'):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    logger.info(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_train_loss:.4f}")

    # 検証ループ
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data, speed, vtype, direc in val_loader:
            data = data.to(device)
            output = model(data)
            loss = criterion(output, data)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    logger.info(f"Validation Loss: {avg_val_loss:.4f}")

    # ベストモデルの保存
    if avg_val_loss < best_valid_loss:
        best_valid_loss = avg_val_loss
        torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
        torch.save(optimizer.state_dict(), os.path.join(output_dir, 'optimizer.pth'))
        logger.info(f"Best model saved with validation loss: {best_valid_loss:.4f}")

# 潜在空間の取得と可視化
logger.info("潜在空間を取得し、t-SNEで可視化します。")

model.eval()
latent_spaces = []
all_speeds = []
all_vehicle_types = []
all_directions = []
with torch.no_grad():
    for loader in [train_loader, val_loader]:
        for data, speed, vtype, direc in tqdm(loader, desc='Extracting Latent Spaces'):
            inputs = data.to(device)
            latents = model.enc(inputs)
            latents = latents.view(inputs.size(0), -1)
            latent_spaces.append(latents.cpu())
            all_speeds.extend(speed.numpy())
            all_vehicle_types.extend(vtype.numpy())
            all_directions.extend(direc.numpy())

latent_spaces = torch.cat(latent_spaces, dim=0).numpy()
all_speeds = np.array(all_speeds)
all_vehicle_types = np.array(all_vehicle_types)  # 0 or 1
all_directions = np.array(all_directions)        # 0 or 1

# t-SNEによる次元削減
logger.info("t-SNEによる次元削減を開始します。")
tsne = TSNE(n_components=2, random_state=SEED)
latent_2d = tsne.fit_transform(latent_spaces)

if SAVE_LATENT_SPACE == 'y':
    # latent_spaces の保存
    latent_spaces_df = pd.DataFrame(latent_spaces)
    latent_spaces_file = os.path.join(output_dir, 'latent_spaces.csv')
    latent_spaces_df.to_csv(latent_spaces_file, index=False)
    logger.info(f"latent_spaces を {latent_spaces_file} に保存しました。")

    # latent_2d の保存
    latent_2d_df = pd.DataFrame(latent_2d, columns=['Dimension_1', 'Dimension_2'])
    latent_2d_file = os.path.join(output_dir, 'latent_2d.csv')
    latent_2d_df.to_csv(latent_2d_file, index=False)
    logger.info(f"latent_2d を {latent_2d_file} に保存しました。")


# all_speeds, all_vehicle_types, all_directions を1つの DataFrame として保存
metadata_df = pd.DataFrame({
    'speed': all_speeds,
    'vehicle_type': all_vehicle_types,
    'direction': all_directions
})
metadata_file = os.path.join(output_dir, 'metadata.csv')
metadata_df.to_csv(metadata_file, index=False)
logger.info(f"speed, vehicle_type, direction を {metadata_file} に保存しました。")

norm = plt.Normalize(all_speeds.min(), all_speeds.max())

# ==========================================================
# 可視化1: 速度による色分けのみを表示した可視化
# ==========================================================
fig, ax = plt.subplots(figsize=(12, 8))

# 全データポイントをプロット、速度による色分けのみ
ax.scatter(latent_2d[:, 0], latent_2d[:, 1],
           c=all_speeds,
           cmap='viridis',
           norm=norm,
           alpha=0.7,
           edgecolors='none',
           linewidths=1)

# カラーマップ情報を持つ ScalarMappable を作成
sm = cm.ScalarMappable(norm=norm, cmap='viridis')
sm.set_array(all_speeds)

fig.colorbar(sm, ax=ax, label='Speed')

ax.set_title("Latent Space Visualization using t-SNE (Speed only)")
ax.set_xlabel("Dimension 1")
ax.set_ylabel("Dimension 2")

# 図の保存
plt.savefig(os.path.join(output_dir, 'tsne_plot_speed_only.png'))
plt.close()

logger.info("速度による可視化が完了しました。")

# ==========================================================
# 可視化2: 車両タイプと方向によるマーカーの違いを表示した可視化
# ==========================================================
fig, ax = plt.subplots(figsize=(12, 8))

# マーカーとエッジカラーの設定
# vehicle_type: 0->car, 1->cv
# direction: 0->right, 1->left
markers = {
    (0, 0): 'o',  # car/right -> 丸
    (0, 1): '^',  # car/left -> 三角
    (1, 0): 's',  # cv/right -> 四角
    (1, 1): 'D'   # cv/left -> ひし形
}

# カテゴリ別にプロット（4パターン: (car/right), (car/left), (cv/right), (cv/left)）
for vt in [0, 1]:
    for dr in [0, 1]:
        mask = (all_vehicle_types == vt) & (all_directions == dr)
        if np.any(mask):
            ax.scatter(latent_2d[mask, 0], latent_2d[mask, 1],
                       c=all_speeds[mask],
                       cmap='viridis',
                       norm=norm,
                       alpha=0.7,
                       marker=markers[(vt, dr)],
                       edgecolors='none',
                       linewidths=1,
                       label=f"{'car' if vt == 0 else 'cv'}/{'right' if dr == 0 else 'left'}")

# カラーマップ情報を持つ ScalarMappable を作成
sm = cm.ScalarMappable(norm=norm, cmap='viridis')
sm.set_array(all_speeds)

fig.colorbar(sm, ax=ax, label='Speed')

ax.set_title("Latent Space Visualization using t-SNE (Categorized by Vehicle Type and Direction)")
ax.set_xlabel("Dimension 1")
ax.set_ylabel("Dimension 2")
ax.legend()

# 図の保存
plt.savefig(os.path.join(output_dir, 'tsne_plot_categorized.png'))
plt.close()

logger.info("車両タイプと方向による可視化が完了しました。")
