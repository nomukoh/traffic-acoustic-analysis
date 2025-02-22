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
from mpl_toolkits.mplot3d import Axes3D  # 3D プロット用
import logging
import random
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import umap
from sklearn.manifold import MDS

from encoder.visualize import LatentSpaceVisualizer

# ハイパーパラメータの設定
ENCODER_MODEL = 'CNN_original'
DATA = 'loc1' # 使用するデータ: loc1 or loc1-6
DATA_CSV_PATH = './encoder/datasets/data_1-6.csv'
REAL_DATA_CSV_PATH = './encoder/datasets/real_data.csv'
MEL = 'OFF' # ON or OFF 注: Decoder を手動で変更する必要あり
SAMPLING_RATE = 16000
N_FFT = 1024  # 1024 or 2048 がよく使われる（小さいほうが軽い）
HOP_LENGTH = 160 # CNN_s = 512, baseline = 160
TEST_DATASET_PERCENTAGE = 0.2
BATCH_SIZE = 32
SEED = 42
LEARNING_RATE = 0.001
EPOCHS = 100
SAVE_LATENT_SPACE = 'n' # y or n : 潜在空間を保存する
DIMENSION_LATENT_SPACE = 2 # 潜在空間の次元
VISUALIZATION = 'PCA' # 可視化手法の選択 : t-SNE, PCA, LDA, UMAP, MDS
LATENT_3D_CSV_PATH = './encoder/latent_3d.csv'
PRINCIPAL_AXES_CSV_PATH = './encoder/principal_axes.csv'

def output_settings():
    # 実行時間を使って出力ディレクトリを作成
    current_time = datetime.now().strftime('%Y-%m-%d/%H-%M-%S')
    output_directory = os.path.join('./outputs', current_time)
    os.makedirs(output_directory, exist_ok=True)

    # ログの設定
    log_file_path = os.path.join(output_directory, 'training.log')

    # ロガーの作成
    logger = logging.getLogger()

    # ハンドラがすでに追加されていればスキップ
    if logger.hasHandlers():
        logger.handlers.clear()

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

    # output_dirをカスタム属性として追加
    logger.output_dir = output_directory

    return logger

def hyperparameter():
    # ハイパーパラメータをログに記録
    logger.info('Hyperparameters:')
    hyperparameters = {
        'ENCODER_MODEL': ENCODER_MODEL,
        'DATA_CSV_PATH': DATA_CSV_PATH,
        'DATA': DATA,
        'MEL': MEL,
        'SAMPLING_RATE': SAMPLING_RATE,
        'N_FFT': N_FFT,
        'HOP_LENGTH': HOP_LENGTH,
        'TEST_DATASET_PERCENTAGE': TEST_DATASET_PERCENTAGE,
        'BATCH_SIZE': BATCH_SIZE,
        'SEED': SEED,
        'LEARNING_RATE': LEARNING_RATE,
        'EPOCHS': EPOCHS,
        'DIMENSION_of_the_LATENT_SPACE': DIMENSION_LATENT_SPACE,
        'VISUALIZATION': VISUALIZATION,
        'LATENT_3D_CSV_PATH': LATENT_3D_CSV_PATH,
        'PRINCIPAL_AXES_CSV_PATH': PRINCIPAL_AXES_CSV_PATH
    }
    for key, value in hyperparameters.items():
        logger.info(f'{key}: {value}')

logger = output_settings()
output_dir = logger.output_dir

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


# スペクトログラムの作成
def create_spectrogram(file_path, n_fft=N_FFT, hop_length=HOP_LENGTH):
    y, sr = librosa.load(file_path, sr=SAMPLING_RATE)
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, center=True)
    S_db = librosa.amplitude_to_db(np.abs(S))
    return S_db

# メルスペクトログラムの作成
def create_mel_spectrogram(file_path, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=128):
    y, sr = librosa.load(file_path, sr=SAMPLING_RATE)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, center=True)
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    return S_db

def prepare_dataloader(
        data_num=DATA, sampling_rate=SAMPLING_RATE,
        batch_size=BATCH_SIZE, test_split=TEST_DATASET_PERCENTAGE,
        seed=SEED, data_csv_path=DATA_CSV_PATH
        ):
    
    # データの読み込み
    data = pd.read_csv(data_csv_path)

    # locのリストを作成
    if  data_num ==  'loc1-6':
        locations = [f'loc{i}' for i in range(1, 7)]
    else:
        locations = [f'{data_num}']


    # すべてのリストを作成
    spectrograms = []
    speeds = []
    vehicle_types_list = []
    directions_list = []
    locations_list = []

    # 各locについて音声データのスペクトログラムを生成
    for loc in locations:
        # locに応じてdataをフィルタリング (regex=False)
        loc_data = data[data['path'].str.contains(loc, regex=False)]

        for i, row in loc_data.iterrows():
            # ファイルパスを動的に作成
            file_path = os.path.join(f'./encoder/datasets/', row['path'])
            
            # スペクトログラムの生成
            if MEL == 'ON':
                S_db = create_mel_spectrogram(file_path)
            else:
                S_db = create_spectrogram(file_path)
            
            # リストに追加
            spectrograms.append(S_db)
            speeds.append(row['speed'])
            vehicle_types_list.append(row['vehicle_type'])
            directions_list.append(row['direction'])
            locations_list.append(loc)
            
            # 最初のスペクトログラムを保存（loc1の1つ目を例として保存）
            if loc == 'loc1' and i == 0:
                logger.info(f"sample_spectrogram: speed {row['speed']}, {loc}, {row['vehicle_type']}, {row['direction']}")
                plt.figure(figsize=(10, 4))
                if MEL == 'ON':
                    librosa.display.specshow(S_db, sr=sampling_rate, hop_length=512, x_axis='time', y_axis='mel', cmap='magma')
                else:
                    librosa.display.specshow(S_db, sr=sampling_rate, hop_length=512, x_axis='time', y_axis='hz', cmap='magma')
                plt.colorbar(format='%+2.0f dB')
                plt.title('Spectrogram')
                plt.xlabel("Time")
                plt.ylabel("Frequency")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'sample_spectrogram.png'))
                plt.close()


    # データの準備
    spectrograms = np.array(spectrograms, dtype=np.float32)
    speeds = np.array(speeds, dtype=np.float32)
    vehicle_types_list = np.array(vehicle_types_list)
    directions_list = np.array(directions_list)
    locations_list = np.array(locations_list)

    # スペクトログラムの最小値と最大値を取得
    original_spectrogram_min = spectrograms.min()
    original_spectrogram_max = spectrograms.max()

    # スペクトログラムを0-1に正規化
    spectrograms = (spectrograms - original_spectrogram_min) / (original_spectrogram_max - original_spectrogram_min)

    # データの分割
    train_data, val_data, speeds_train, speeds_val, vehicle_train, vehicle_val, direction_train, direction_val, location_train, location_val = train_test_split(
        spectrograms, speeds, vehicle_types_list, directions_list, locations_list,
        test_size=test_split, random_state=seed
    )

    # vehicle_type と direction をエンコード（例：car=0, cv=1 / right=0, left=1）
    vehicle_map = {'car':0, 'cv':1}
    direction_map = {'right':0, 'left':1}
    vehicle_train_encoded = torch.tensor([vehicle_map.get(v,0) for v in vehicle_train], dtype=torch.long)
    vehicle_val_encoded = torch.tensor([vehicle_map.get(v,0) for v in vehicle_val], dtype=torch.long)
    direction_train_encoded = torch.tensor([direction_map.get(d,0) for d in direction_train], dtype=torch.long)
    direction_val_encoded = torch.tensor([direction_map.get(d,0) for d in direction_val], dtype=torch.long)
    # locationを数値にエンコードするためのマップ作成
    location_map = {
        'loc1': 0,
        'loc2': 1,
        'loc3': 2,
        'loc4': 3,
        'loc5': 4,
        'loc6': 5
    }
    location_train_encoded = torch.tensor([location_map.get(l,0) for l in location_train], dtype=torch.long)
    location_val_encoded = torch.tensor([location_map.get(l,0) for l in location_val], dtype=torch.long)
    # テンソルへの変換
    train_tensors = [torch.tensor(s, dtype=torch.float32).unsqueeze(0) for s in train_data]
    val_tensors = [torch.tensor(s, dtype=torch.float32).unsqueeze(0) for s in val_data]
    train_speeds = torch.tensor(speeds_train, dtype=torch.float32)
    val_speeds = torch.tensor(speeds_val, dtype=torch.float32)
    # Datasetの作成
    train_dataset = TensorDataset(torch.stack(train_tensors), train_speeds, vehicle_train_encoded, direction_train_encoded, location_train_encoded)
    val_dataset = TensorDataset(torch.stack(val_tensors), val_speeds, vehicle_val_encoded, direction_val_encoded, location_val_encoded)
    # DataLoaderの作成
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, original_spectrogram_min, original_spectrogram_max


# モデルの定義
class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU()
        )
        
    def forward(self, x):
        # print(f"Encoder Input image size: {x.size()}\n")
        x = self.encoder(x)
        # print(f"Encoder Output image size: {x.size()}\n")
        return x

class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=(0, 0)),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=(0, 0)),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=(0, 0)),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        # print(f"Decoder Input image size: {x.size()}\n")
        x = self.decoder(x)
        # print(f"Decoder Output image size: {x.size()}\n")
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



def main():
    hyperparameter()
    train_loader, val_loader, original_spectrogram_min, original_spectrogram_max = prepare_dataloader()
    logger.info("データの準備が完了しました。")

    # モデルの初期化と学習
    model = AutoEncoder().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    logger.info("モデルの定義が完了しました。")

    # 保存済みのモデルとオプティマイザの状態を読み込む
    # model.load_state_dict(torch.load(f'./encoder/best_model_{ENCODER_MODEL}.pth', map_location=device, weights_only=True))  # モデルの重みを読み込む
    # optimizer.load_state_dict(torch.load(f'./encoder/optimizer_{ENCODER_MODEL}.pth', map_location=device, weights_only=True))  # オプティマイザの状態を読み込む
    
    
    logger.info("学習を開始します。")
    best_valid_loss = float('inf')

    for epoch in range(EPOCHS):
        # 学習ループ
        model.train()
        train_loss = 0.0
        for data, speed, vtype, direc, locs in tqdm(train_loader, desc=f'Train Epoch {epoch+1}/{EPOCHS}'):
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
            for data, speed, vtype, direc, locs in val_loader:
                data = data.to(device)
                output = model(data)
                loss = criterion(output, data)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        logger.info(f"Validation Loss: {avg_val_loss:.4f}")

        # ベストモデルの保存
        if avg_val_loss < best_valid_loss:
            best_valid_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, f'best_model_{ENCODER_MODEL}.pth'))
            torch.save(optimizer.state_dict(), os.path.join(output_dir, f'optimizer_{ENCODER_MODEL}.pth'))
            logger.info(f"Best model saved with validation loss: {best_valid_loss:.4f}")

    
    logger.info("潜在空間を取得します。")

    # 潜在空間の取得
    model.eval()
    latent_spaces = []
    all_speeds = []
    all_vehicle_types = []
    all_directions = []
    all_locations = []
    all_classes = [] # 車種と方向による4分類(0-3)

    with torch.no_grad():
        for data, speed, vtype, direc, locs in tqdm(train_loader, desc='Extracting Latent Spaces (Train)'):
            inputs = data.to(device)
            latents = model.enc(inputs)
            latents = latents.view(inputs.size(0), -1)
            latent_spaces.append(latents.cpu())
            all_speeds.extend(speed.numpy())
            all_vehicle_types.extend(vtype.numpy())
            all_directions.extend(direc.numpy())
            all_locations.extend(locs.numpy())
        for data, speed, vtype, direc, locs in tqdm(val_loader, desc='Extracting Latent Spaces (Validation)'):
            inputs = data.to(device)
            latents = model.enc(inputs)
            latents = latents.view(inputs.size(0), -1)
            latent_spaces.append(latents.cpu())
            all_speeds.extend(speed.numpy())
            all_vehicle_types.extend(vtype.numpy())
            all_directions.extend(direc.numpy())
            all_locations.extend(locs.numpy())
    



    
    logger.info("潜在空間を可視化します。")

    latent_spaces = torch.cat(latent_spaces, dim=0).numpy() # フラット化
    all_speeds = np.array(all_speeds)
    all_vehicle_types = np.array(all_vehicle_types)  # 0 or 1
    all_directions = np.array(all_directions)        # 0 or 1
    all_locations = np.array(all_locations)
    all_classes = (all_vehicle_types * 2) + all_directions

    # all_speeds, all_vehicle_types, all_directions を1つの DataFrame として保存
    metadata_df = pd.DataFrame({
        'speed': all_speeds,
        'vehicle_type': all_vehicle_types,
        'direction': all_directions,
        'location': all_locations
    })
    metadata_file = os.path.join(output_dir, 'metadata.csv')
    metadata_df.to_csv(metadata_file, index=False)
    logger.info(f"speed, vehicle_type, direction, location を {metadata_file} に保存しました。")

    norm = plt.Normalize(all_speeds.min(), all_speeds.max())


    # 可視化
    if VISUALIZATION == 't-SNE':
        # t-SNEによる次元削減
        logger.info("t-SNEによる次元削減を開始します。")
        tsne = TSNE(n_components=DIMENSION_LATENT_SPACE, random_state=SEED)
        latent_2d = tsne.fit_transform(latent_spaces)
    elif VISUALIZATION == 'PCA':
        # PCAによる次元削減
        logger.info("PCAによる次元削減を開始します。")
        pca = PCA(n_components=DIMENSION_LATENT_SPACE, random_state=SEED)
        latent_2d = pca.fit_transform(latent_spaces)
    elif VISUALIZATION == 'LDA':
        # 線形判別分析法(LDA)による次元削減
        logger.info("LDAによる次元削減を開始します。")
        lda = LinearDiscriminantAnalysis(n_components=DIMENSION_LATENT_SPACE)
        latent_2d = lda.fit_transform(latent_spaces, all_classes)
    elif VISUALIZATION == 'UMAP':
        # UMAPによる次元削減
        logger.info("UMAPによる次元削減を開始します。")
        fit = umap.UMAP(n_components=DIMENSION_LATENT_SPACE, random_state=SEED)
        latent_2d = fit.fit_transform(latent_spaces)
    elif VISUALIZATION == 'MDS':
        # UMAPによる次元削減
        logger.info("MDSによる次元削減を開始します。")
        mds = MDS(n_components=DIMENSION_LATENT_SPACE, random_state=SEED)
        latent_2d = mds.fit_transform(latent_spaces)


    if SAVE_LATENT_SPACE == 'y':
        # latent_spaces の保存
        latent_spaces_df = pd.DataFrame(latent_spaces)
        latent_spaces_file = os.path.join(output_dir, 'latent_spaces.csv')
        latent_spaces_df.to_csv(latent_spaces_file, index=False)
        logger.info(f"latent_spaces を {latent_spaces_file} に保存しました。")

        if DIMENSION_LATENT_SPACE==2:
            # latent_2d の保存
            latent_2d_df = pd.DataFrame(latent_2d, columns=['Dimension_1', 'Dimension_2'])
            latent_2d_file = os.path.join(output_dir, 'latent_2d.csv')
            latent_2d_df.to_csv(latent_2d_file, index=False)
            logger.info(f"latent_2d を {latent_2d_file} に保存しました。")
        elif DIMENSION_LATENT_SPACE==3:
            # latent_3d の保存
            latent_3d_df = pd.DataFrame(latent_2d, columns=['Dimension_1', 'Dimension_2', 'Dimension_3'])
            latent_3d_file = os.path.join(output_dir, 'latent_3d.csv')
            latent_3d_df.to_csv(latent_3d_file, index=False)
            logger.info(f"latent_3d を {latent_3d_file} に保存しました。")
        
    # 潜在空間の可視化
    visualizer = LatentSpaceVisualizer(
        dimension_latent_space=DIMENSION_LATENT_SPACE,
        latent_data=latent_2d,
        speeds=all_speeds,
        vehicle_types=all_vehicle_types,
        directions=all_directions,
        locations=all_locations,
        output_dir=output_dir,
        visualization=VISUALIZATION
    )

    # 可視化をまとめて実行 (3Dでインタラクティブに操作したければ show_plot=True)
    visualizer.visualize_all(show_plot=True)
    
if __name__ == "__main__":
    main()