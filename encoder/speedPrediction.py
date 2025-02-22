"""
深層距離学習をさせたエンコーダモデルを基盤として、
速度による分類モデルを作成、学習させるプログラム。
"""
import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

from encoder.auto_encoder.base_model import Encoder_Original, Encoder_Small
from encoder.auto_encoder.CNN_s import output_settings, prepare_dataloader
from encoder.auto_encoder.CNN_s import MEL, SAMPLING_RATE, N_FFT, HOP_LENGTH, TEST_DATASET_PERCENTAGE

# ハイパーパラメータの設定
DATA_CSV_PATH = './encoder/datasets/data_1-6.csv'
DATA = 'loc1-6'
BATCH_SIZE = 32
SEED = 42
LEARNING_RATE = 1e-3
EPOCHS = 300
PRETRAINED_CNN_PATH = './encoder/last_model_Metric.pth'  # 学習済みCNNのパラメータ保存ファイル
CNN_MODEL = 'CNN_s' # original, CNN_s
LOSS_TYPE = "MSE" # MSE, MAE, HUBER

MODEL_PATH = f'./encoder/last_speed_model_{LOSS_TYPE}.pth' # 学習済み speed Prediction モデルのパラメータ保存ファイル

def hyperparameter(logger, DATA_CSV_PATH, DATA, BATCH_SIZE, SEED, LEARNING_RATE, EPOCHS, LOSS_TYPE, CNN_MODEL):
    # ハイパーパラメータをログに記録
    logger.info('Hyperparameters:')
    hyperparameters = {
        'MODEL': 'Speed prediction',
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
        'LOSS_TYPE': LOSS_TYPE,
        'CNN_MODEL' : CNN_MODEL
    }
    for key, value in hyperparameters.items():
        logger.info(f'{key}: {value}')

# 新たな回帰モデルの定義
class SpeedRegressor(torch.nn.Module):
    """
    事前学習済みのCNNを特徴抽出器として使い、その出力を受けて速度を回帰するモデル
    """
    def __init__(self, cnn):
        super().__init__()
        # 事前学習済み CNN
        self.cnn = cnn
        # CNN部分のパラメータをフリーズ（勾配更新しない）
        # for param in self.cnn.parameters():
        #     param.requires_grad = False

        # Global Average Pooling で次元を (B, 64) に圧縮
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = torch.nn.Flatten()

        # DNN部分（回帰器）：最後の出力次元は 1（速度のスカラー値）
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            # torch.nn.Dropout(0.5),
            torch.nn.Linear(128, 1)
        )

    def forward(self, x):
        features = self.cnn(x)            # (B, 64, H', W')
        pooled   = self.pool(features)    # (B, 64, 1, 1)
        flat     = self.flatten(pooled)   # (B, 64)
        # flat     = F.normalize(flat, p=2, dim=1)  # L2正規化
        out      = self.regressor(flat)   # (B, 1)
        return out

# 損失関数を選択するための関数
def LossFunction(loss_type="MSE"):
    """
    loss_type で選択肢を切り替える:
      - "MSE"       : Mean Squared Error (torch.nn.MSELoss)
      - "MAE"       : Mean Absolute Error (torch.nn.L1Loss)
      - "HUBER"     : Smooth L1 (Huber) Loss (torch.nn.SmoothL1Loss)
    """
    loss_type = loss_type.upper()
    if loss_type == "MSE":
        return torch.nn.MSELoss()
    elif loss_type == "MAE":
        return torch.nn.L1Loss()
    elif loss_type in ["HUBER", "SMOOTHL1"]:
        return torch.nn.SmoothL1Loss()
    else:
        raise ValueError("Invalid loss_type. Choose from 'MSE','MAE','Huber'.")
    
def auditor(train_loader, val_loader, logger):
    # 1. train_loader から全サンプルの速度を収集
    train_speeds = []
    for data, speed, _, _, _ in train_loader:
        train_speeds.extend(speed.cpu().numpy())  # speed は (batch_size,) なので extend で追加

    # 2. val_loader から全サンプルの速度を収集
    val_speeds = []
    for data, speed, _, _, _ in val_loader:
        val_speeds.extend(speed.cpu().numpy())

    # 3. numpy 配列に変換
    train_speeds = np.array(train_speeds)
    val_speeds = np.array(val_speeds)

    # 4. 統計量を計算
    train_mean = train_speeds.mean()
    train_std  = train_speeds.std()
    train_min  = train_speeds.min()
    train_max  = train_speeds.max()

    val_mean = val_speeds.mean()
    val_std  = val_speeds.std()
    val_min  = val_speeds.min()
    val_max  = val_speeds.max()

    # 5. ログに出力
    logger.info("=== Training Data Speed Stats ===")
    logger.info(f"Count    : {len(train_speeds)}")
    logger.info(f"Mean     : {train_mean:.4f}")
    logger.info(f"Std      : {train_std:.4f}")
    logger.info(f"Min      : {train_min:.4f}")
    logger.info(f"Max      : {train_max:.4f}")

    logger.info("=== Validation Data Speed Stats ===")
    logger.info(f"Count    : {len(val_speeds)}")
    logger.info(f"Mean     : {val_mean:.4f}")
    logger.info(f"Std      : {val_std:.4f}")
    logger.info(f"Min      : {val_min:.4f}")
    logger.info(f"Max      : {val_max:.4f}")


# ===============================
# 評価＆CSV保存用の関数
# ===============================
def evaluate_and_save(model, loader, device, criterion, output_csv):
    """
    model: 学習済みの回帰モデル
    loader: 推論を行う DataLoader (全データ or テストデータ)
    device: CPU or CUDA
    criterion: 損失関数 (MSELoss, L1Loss, SmoothL1Lossなど)
    output_csv: CSV ファイルパス (予測結果を保存する)
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    # 予測値と元の速度を保存するリスト
    all_preds = []
    all_speeds = []

    with torch.no_grad():
        for data, speed, _, _, _ in tqdm(loader, desc="Evaluating"):
            data = data.to(device)
            speed = speed.to(device)

            outputs = model(data).squeeze(1)  # (B,1)->(B,)
            loss = criterion(outputs, speed)

            batch_size = data.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # リストに追加 (CPUに戻してnumpy化)
            all_preds.extend(outputs.cpu().numpy().tolist())
            all_speeds.extend(speed.cpu().numpy().tolist())

    avg_loss = total_loss / total_samples

    # 予測値と真値をCSVに保存
    df = pd.DataFrame({
        'original_speed': all_speeds,
        'predicted_speed': all_preds
    })
    df.to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")
    print(f"Average Loss: {avg_loss:.4f}")

    return avg_loss

def main():
    # ログおよびデバイスの設定
    logger = output_settings()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    if device == 'cuda':
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    hyperparameter(logger, DATA_CSV_PATH, DATA, BATCH_SIZE,
                   SEED, LEARNING_RATE, EPOCHS, LOSS_TYPE, CNN_MODEL)

    # データローダの準備
    # ※ ここでは既存の prepare_dataloader を再利用（スペクトログラム画像の正規化済みテンソルが返る）
    train_loader, val_loader, _, _ = prepare_dataloader(
        data_num=DATA,
        data_csv_path=DATA_CSV_PATH,
        batch_size=BATCH_SIZE,
        seed=SEED
    )
    # データの偏りを可視化
    # auditor(train_loader, val_loader, logger)
    logger.info("データローダの準備が完了しました。")

    # モデルの定義
    if CNN_MODEL == 'original':
        cnn = Encoder_Original().to(device)
    elif CNN_MODEL == 'CNN_s':
        cnn = Encoder_Small().to(device)
    # cnn.load_state_dict(torch.load(PRETRAINED_CNN_PATH, map_location=device, weights_only=True))
    # logger.info("学習済みCNNのパラメータを読み込みました。")
    # CNN+回帰モデルの初期化（CNNは固定、DNN部分のみ学習）
    model = SpeedRegressor(cnn).to(device)


    # DNN部分（model.regressor）のみパラメータ更新
    optimizer = optim.Adam([
        {'params': model.cnn.parameters(), 'lr': LEARNING_RATE},
        # {'params': model.cnn.parameters(), 'lr': LEARNING_RATE * 0.1},  # CNNは低い学習率
        {'params': model.regressor.parameters(), 'lr': LEARNING_RATE}   # 回帰器は標準の学習率
    ])

    # 損失関数の設定
    criterion = LossFunction(LOSS_TYPE)
    logger.info(f"Selected Loss Function: {LOSS_TYPE}")
    
    # ===============================
    # 1. モデルの学習
    # ===============================
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for data, speed, _, _, _ in tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}'):
            # 入力をデバイスへ
            data = data.to(device)
            speed = speed.to(device)           # (B,) の連続値
            # 順伝搬
            outputs = model(data).squeeze(1)   # (B,1) -> (B,)
            loss = criterion(outputs, speed)
            # バックプロパゲーション
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 損失
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        logger.info(f"Epoch [{epoch+1}/{EPOCHS}] Training Loss: {avg_train_loss:.4f}")
        
        # 検証ループ
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, speed, _, _, _ in val_loader:
                data = data.to(device)
                speed = speed.to(device)
                outputs = model(data).squeeze(1)   # (B,1) -> (B,)
                loss = criterion(outputs, speed)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        logger.info(f"Epoch [{epoch+1}/{EPOCHS}] Validation Loss: {avg_val_loss:.4f}")

        # 今エポックのLossをリストに追加
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)    
    
        # 最良モデルの保存
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(logger.output_dir, f'best_speed_model_{LOSS_TYPE}.pth'))
            logger.info(f"Best model saved at epoch {epoch+1} with loss {best_val_loss:.4f}")
    
    # 最終モデルの保存
    torch.save(model.state_dict(), os.path.join(logger.output_dir, f'last_speed_model_{LOSS_TYPE}.pth'))
    logger.info("最終モデルを保存しました。")
    
    # 学習・検証損失のグラフ作成・保存
    # plt.figure()
    # plt.plot(range(1, EPOCHS+1), train_losses, label='Train Loss')
    # plt.plot(range(1, EPOCHS+1), val_losses, label='Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Train and Validation Loss')
    # plt.legend()
    # plt.ylim([0, 400])

    # # モデルのパラメータを保存しているディレクトリに同名ファイルで保存
    # graph_path = os.path.join(logger.output_dir, 'loss_graph.png')
    # plt.savefig(graph_path)
    # plt.close()

    # logger.info(f"損失の推移グラフを {graph_path} に保存しました。")

    """
    # ===============================
    # 2. 評価とcsvファイル保存
    # ===============================
    # 学習済みパラメータをロード
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    logger.info(f"Loaded model weights from {MODEL_PATH}")

    # 3-4. 評価と結果のCSV保存（train_loaderとval_loaderそれぞれに対して実施例）
    logger.info("Evaluating on train_loader...")
    train_csv_path = os.path.join(logger.output_dir, f"train_predictions_{LOSS_TYPE}.csv")
    train_loss = evaluate_and_save(model, train_loader, device, criterion, train_csv_path)
    logger.info(f"Train Loss: {train_loss:.4f}")

    logger.info("Evaluating on val_loader...")
    val_csv_path = os.path.join(logger.output_dir, f"val_predictions_{LOSS_TYPE}.csv")
    val_loss = evaluate_and_save(model, val_loader, device, criterion, val_csv_path)
    logger.info(f"Val Loss: {val_loss:.4f}")

    logger.info("Evaluation complete.")
    """

if __name__ == "__main__":
    main()
