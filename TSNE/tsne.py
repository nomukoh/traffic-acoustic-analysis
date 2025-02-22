import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# ハイパーパラメータの設定
DIMENSION_LATENT_SPACE = 2
SEED = 42

perplexity = 40
learning_rate = 200
n_iter = 1000
early_exaggeration = 12.0
angle = 0.5
method = 'barnes_hut'

# perplexity = 40       （5～50）
# learning_rate = 200   （200 前後）
# n_iter = 1000         （1000～3000）
# early_exaggeration = 12.0
# angle = 0.5
# method = 'barnes_hut'

def main():
    # 入力ファイルと出力ディレクトリのパス設定
    tsne_dir = "./TSNE"
    metadata_path = os.path.join(tsne_dir, "metadata.csv")
    latent_spaces_path = os.path.join(tsne_dir, "latent_spaces.csv")
    figure_dir = os.path.join(tsne_dir, "Figure")
    os.makedirs(figure_dir, exist_ok=True)

    # CSVファイルの読み込み
    metadata = pd.read_csv(metadata_path)
    latent_spaces = pd.read_csv(latent_spaces_path)
    
    # latent_spacesをnumpy配列に変換
    latent_features = latent_spaces.values

    # t-SNEによる次元削減
    tsne = TSNE(
        n_components=DIMENSION_LATENT_SPACE,
        init='pca',
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        early_exaggeration=early_exaggeration,
        angle=angle,
        method=method,
        random_state=SEED
    )
    features_2d = tsne.fit_transform(latent_features)

    # metadataの車両タイプと方向のカラムを取得
    # ※文字列の場合は数値に変換（例: 'car'->0, 'cv'->1, 'right'->0, 'left'->1）
    if metadata['vehicle_type'].dtype == 'object':
        vehicle_map = {'car': 0, 'cv': 1}
        metadata['vehicle_type'] = metadata['vehicle_type'].map(vehicle_map)
    if metadata['direction'].dtype == 'object':
        direction_map = {'right': 0, 'left': 1}
        metadata['direction'] = metadata['direction'].map(direction_map)
    
    vehicle_types = metadata['vehicle_type'].values
    directions = metadata['direction'].values

    # 可視化（車両タイプと方向による2Dプロット：スピード情報は含まない）
    visualize_tsne(features_2d, vehicle_types, directions, figure_dir)

def visualize_tsne(latent_data, vehicle_types, directions, output_dir):
    # フォントサイズの設定
    FONT_SIZE = 20
    plt.rcParams.update({'font.size': FONT_SIZE})

    fig, ax = plt.subplots(figsize=(12, 8))

    # マーカーの種類を定義（面積の大きい形状）
    markers = {
        (0, 0): 'o',  # 車両タイプ: car, 方向: right → 丸
        (0, 1): 's',  # car, left → 四角
        (1, 0): 'D',  # cv, right → ひし形
        (1, 1): 'p'   # cv, left → 五角形
    }
    marker_size = 50

    # (vehicle_type, direction) の組み合わせごとにプロット
    for vt in [0, 1]:
        for dr in [0, 1]:
            mask = (vehicle_types == vt) & (directions == dr)
            if np.any(mask):
                ax.scatter(
                    latent_data[mask, 0],
                    latent_data[mask, 1],
                    alpha=0.7,
                    edgecolors='none',
                    linewidths=1,
                    marker=markers[(vt, dr)],
                    s=marker_size,
                    label=f"{'car' if vt == 0 else 'cv'}/{'right' if dr == 0 else 'left'}"
                )
    
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.legend()

    # ファイル名にt-SNEのハイパーパラメータの数値を含める
    filename = f"tsne_p{perplexity}_lr{learning_rate}_iter{n_iter}_exp{early_exaggeration}_angle{angle}_{method}.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Figure saved to {save_path}")

if __name__ == "__main__":
    main()
