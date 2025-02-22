"""
Circle Loss を使用した深層距離学習の実装。
"""
import os
import pandas as pd
import librosa
import librosa.display
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import random
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from encoder.settings import output_settings, prepare_dataloader
from encoder.visualize import LatentSpaceVisualizer
from encoder.auto_encoder.base_model import Encoder_Original, Encoder_Small
from encoder.loss.circle_loss import CircleLoss, convert_label_to_similarity

# ハイパーパラメータの設定
ENCODER_MODEL = 'CNN_s' # CNN_s or Original
DATA = 'loc1-6' # 使用するデータ: loc1 or loc1-6
DATA_CSV_PATH = './encoder/datasets/data_1-6.csv'
MEL = 'ON' # ON or OFF 注: Decoder を手動で変更する必要あり
SAMPLING_RATE = 16000
N_FFT = 1024  # 1024 or 2048 がよく使われる（小さいほうが軽い）
HOP_LENGTH = 512 # CNN_s = 512, baseline = 160
if ENCODER_MODEL == 'Original':
    HOP_LENGTH = 160
TEST_DATASET_PERCENTAGE = 0.2
BATCH_SIZE = 32
SEED = 42
LEARNING_RATE = 0.001
EPOCHS = 80
SAVE_LATENT_SPACE = 'n' # y or n : 潜在空間を保存する
DIMENSION_LATENT_SPACE = 2 # 潜在空間の次元
VISUALIZATION = 't-SNE' # 可視化手法の選択 : t-SNE, PCA, Projection-based PCA, UMAP, MDS

def hyperparameter(logger):
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
        'VISUALIZATION': VISUALIZATION
    }
    for key, value in hyperparameters.items():
        logger.info(f'{key}: {value}')

def main():
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

    hyperparameter(logger)
    train_loader, val_loader, *others = prepare_dataloader(
        logger, hop_length=HOP_LENGTH, batch_size=BATCH_SIZE, data_csv_path=DATA_CSV_PATH, main_data_dir = f'./encoder/datasets/',
        n_fft=N_FFT, data_num=DATA, mel=MEL,
        sampling_rate=SAMPLING_RATE, test_split=TEST_DATASET_PERCENTAGE, seed=SEED
        )
    logger.info("データの準備が完了しました。")

    # モデルの初期化と学習
    if ENCODER_MODEL == "CNN_s":
        model = Encoder_Small().to(device)
    elif ENCODER_MODEL == "Original":
        model = Encoder_Original().to(device) 
    criterion = CircleLoss(m=0.25, gamma=80).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    logger.info("モデルの定義が完了しました。")

    # model.load_state_dict(torch.load(f'./encoder/last_model_{ENCODER_MODEL}.pth', map_location=device, weights_only=True))  # モデルの重みを読み込む

    
    logger.info("学習を開始します。")
    best_valid_loss = float('inf')
    for epoch in range(EPOCHS):
        # 学習ループ
        model.train()
        train_loss = 0.0
        for data, speed, vtype, direc, locs in tqdm(train_loader, desc=f'Train Epoch {epoch+1}/{EPOCHS}'):
            data = data.to(device)
            optimizer.zero_grad()

            # 4クラス用に (vtype, direc) -> label を作成
            labels = vtype * 2 + direc
            labels = labels.to(device)

            # モデルの出力を埋め込みベクトルとして取得し，平均プーリング & 正規化
            embedding = model(data)
            embedding = embedding.mean(dim=[2, 3])          # Global Average Pool
            embedding = F.normalize(embedding, p=2, dim=1)  # L2ノルム正規化

            # CircleLoss 用のスコア算出
            sp, sn = convert_label_to_similarity(embedding, labels)
            loss = criterion(sp, sn)

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
                labels = vtype * 2 + direc
                labels = labels.to(device)
                embedding = model(data)
                embedding = embedding.mean(dim=[2, 3])
                embedding = F.normalize(embedding, p=2, dim=1)
                sp, sn = convert_label_to_similarity(embedding, labels)
                loss = criterion(sp, sn)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        logger.info(f"Validation Loss: {avg_val_loss:.4f}")

        # ベストモデルの保存
        if avg_val_loss < best_valid_loss:
            best_valid_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, f'best_model_{ENCODER_MODEL}.pth'))
            logger.info(f"Best model saved with validation loss: {best_valid_loss:.4f}")

    # ラストモデルの保存
    torch.save(model.state_dict(), os.path.join(output_dir, f'last_model_{ENCODER_MODEL}.pth'))
    logger.info(f"Last model saved with validation loss: {best_valid_loss:.4f}")
    

    # 学習終了後に可視化用の推論を行う
    model.eval()
    all_features = []
    all_speeds = []
    all_vehicle_types = []
    all_directions = []
    all_locations = []

    with torch.no_grad():
        for data, speed, vtype, direc, locs in tqdm(train_loader, desc="Generating embeddings for visualization"):
            data = data.to(device)
            embedding = model(data)
            embedding = embedding.mean(dim=[2, 3])
            embedding = F.normalize(embedding, p=2, dim=1)
            all_features.append(embedding.cpu().numpy())
            all_speeds.extend(speed.cpu().numpy())
            all_vehicle_types.extend(vtype.cpu().numpy())
            all_directions.extend(direc.cpu().numpy())
            all_locations.extend(locs.cpu().numpy())
        for data, speed, vtype, direc, locs in tqdm(val_loader, desc="Generating embeddings for visualization"):
            data = data.to(device)
            embedding = model(data)
            embedding = embedding.mean(dim=[2, 3])
            embedding = F.normalize(embedding, p=2, dim=1)
            all_features.append(embedding.cpu().numpy())
            all_speeds.extend(speed.cpu().numpy())
            all_vehicle_types.extend(vtype.cpu().numpy())
            all_directions.extend(direc.cpu().numpy())
            all_locations.extend(locs.cpu().numpy())

    all_features = np.concatenate(all_features, axis=0)
    all_speeds = np.array(all_speeds)
    all_vehicle_types = np.array(all_vehicle_types)  # 0 or 1
    all_directions = np.array(all_directions)        # 0 or 1
    all_locations = np.array(all_locations)

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

    # 次元に圧縮
    if VISUALIZATION == 't-SNE':
        # t-SNEによる次元削減
        logger.info("t-SNEによる次元削減を開始します。")
        tsne = TSNE(n_components=DIMENSION_LATENT_SPACE, init='pca', random_state=SEED)
        features_2d = tsne.fit_transform(all_features)
    elif VISUALIZATION == 'PCA':
        # PCAによる次元削減
        logger.info("PCAによる次元削減を開始します。")
        pca = PCA(n_components=DIMENSION_LATENT_SPACE, random_state=SEED)
        features_2d = pca.fit_transform(all_features)
        # PCA 軸を保存
        pca_components = pca.components_
        pca_components_path = os.path.join(output_dir, 'pca_components.csv')
        df_components = pd.DataFrame(pca_components)
        df_components.to_csv(pca_components_path, index=False)

    if SAVE_LATENT_SPACE == 'y':
        # latent_spaces の保存
        latent_spaces_df = pd.DataFrame(all_features)
        latent_spaces_file = os.path.join(output_dir, 'latent_spaces.csv')
        latent_spaces_df.to_csv(latent_spaces_file, index=False)
        logger.info(f"latent_spaces を {latent_spaces_file} に保存しました。")

        if DIMENSION_LATENT_SPACE==2:
            # latent_2d の保存
            latent_2d_df = pd.DataFrame(features_2d, columns=['Dimension_1', 'Dimension_2'])
            latent_2d_file = os.path.join(output_dir, 'latent_2d.csv')
            latent_2d_df.to_csv(latent_2d_file, index=False)
            logger.info(f"latent_2d を {latent_2d_file} に保存しました。")
        elif DIMENSION_LATENT_SPACE==3:
            # latent_3d の保存
            latent_3d_df = pd.DataFrame(features_2d, columns=['Dimension_1', 'Dimension_2', 'Dimension_3'])
            latent_3d_file = os.path.join(output_dir, 'latent_3d.csv')
            latent_3d_df.to_csv(latent_3d_file, index=False)
            logger.info(f"latent_3d を {latent_3d_file} に保存しました。")


    # 潜在空間の可視化
    visualizer = LatentSpaceVisualizer(
        dimension_latent_space=DIMENSION_LATENT_SPACE,
        latent_data=features_2d,
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