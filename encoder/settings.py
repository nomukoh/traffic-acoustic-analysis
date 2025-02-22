import os
import sys
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset

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

# スペクトログラムの作成
def create_spectrogram(file_path, n_fft, hop_length, sampling_rate):
    y, sr = librosa.load(file_path, sr=sampling_rate)
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, center=True)
    S_db = librosa.amplitude_to_db(np.abs(S))
    return S_db

# メルスペクトログラムの作成
def create_mel_spectrogram(file_path, n_fft, hop_length, sampling_rate, n_mels=128):
    y, sr = librosa.load(file_path, sr=sampling_rate)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, center=True)
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    return S_db

def prepare_dataloader(
        logger, hop_length, batch_size, data_csv_path, main_data_dir,
        n_fft=1024, data_num='loc1-6', mel='OFF',
        sampling_rate=16000, test_split=0.2, seed=42
        ):
    
    output_dir = logger.output_dir

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
            file_path = os.path.join(main_data_dir, row['path'])
            
            # スペクトログラムの生成
            if mel == 'ON':
                S_db = create_mel_spectrogram(file_path, n_fft=n_fft, hop_length=hop_length, sampling_rate=sampling_rate)
            else:
                S_db = create_spectrogram(file_path, n_fft=n_fft, hop_length=hop_length, sampling_rate=sampling_rate)
            
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
                if mel == 'ON':
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
