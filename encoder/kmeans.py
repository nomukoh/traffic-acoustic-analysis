import os
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D

def kmeans_clustering_and_visualize(train_loader, val_loader, logger, output_dir, n_clusters=4):
    """
    train_loader, val_loader からスペクトログラムを取り出し、
    K-meansクラスタリングを行った上でPCAで2次元に圧縮し可視化する。
    
    Parameters
    ----------
    train_loader : DataLoader
        スペクトログラムや速度などが格納された DataLoader（学習用）
    val_loader : DataLoader
        スペクトログラムや速度などが格納された DataLoader（検証用）
    logger : logging.Logger
        ログ出力用のロガー
    output_dir : str
        画像の出力先ディレクトリ
    n_clusters : int
        K-means で求めるクラスタ数
    """

    logger.info("=== K-meansクラスタリング & 可視化を開始します ===")

    # --------------------------------------------------
    # 1) DataLoader からスペクトログラムとメタ情報をまとめて取得
    # --------------------------------------------------
    all_spectrograms = []
    all_speeds = []
    all_vehicle_types = []
    all_directions = []
    all_locations = []

    # train_loader から取得
    for batch_data, batch_speeds, batch_vtypes, batch_direcs, batch_locs in train_loader:
        # batch_data.shape = (batch_size, 1, freq, time)
        # ここでは生のスペクトログラムをそのまま使用するため flatten して使う
        # CNN等に通すわけではないので2次元ベクトルに落とします: (batch_size, freq*time)
        # .view(batch_size, -1)でフラット化
        flattened = batch_data.view(batch_data.size(0), -1).cpu().numpy()

        all_spectrograms.append(flattened)
        all_speeds.extend(batch_speeds.cpu().numpy())
        all_vehicle_types.extend(batch_vtypes.cpu().numpy())
        all_directions.extend(batch_direcs.cpu().numpy())
        all_locations.extend(batch_locs.cpu().numpy())

    # val_loader から取得 (検証データも含めて一緒にクラスタリングしたい場合)
    for batch_data, batch_speeds, batch_vtypes, batch_direcs, batch_locs in val_loader:
        flattened = batch_data.view(batch_data.size(0), -1).cpu().numpy()

        all_spectrograms.append(flattened)
        all_speeds.extend(batch_speeds.cpu().numpy())
        all_vehicle_types.extend(batch_vtypes.cpu().numpy())
        all_directions.extend(batch_direcs.cpu().numpy())
        all_locations.extend(batch_locs.cpu().numpy())

    # リスト同士を縦方向（行方向）に連結 (shape: [N, freq*time])
    all_spectrograms = np.concatenate(all_spectrograms, axis=0)
    all_speeds = np.array(all_speeds)
    all_vehicle_types = np.array(all_vehicle_types)
    all_directions = np.array(all_directions)
    all_locations = np.array(all_locations)

    logger.info(f"スペクトログラム shape: {all_spectrograms.shape}")
    logger.info(f"速度データ数: {len(all_speeds)}")

    # --------------------------------------------------
    # 2) K-meansクラスタリングの実施
    # --------------------------------------------------
    logger.info(f"K-meansクラスタリングを実行します (n_clusters={n_clusters})")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(all_spectrograms)  # shape: [N]

    # --------------------------------------------------
    # 3) PCAによる2次元可視化
    # --------------------------------------------------
    logger.info("PCAにより次元削減し、2次元散布図を作成します。")
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(all_spectrograms)  # shape: [N, 2]

    # --------------------------------------------------
    # 4) プロット
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        pca_result[:, 0],
        pca_result[:, 1],
        c=cluster_labels,        # K-means で割り当てられたクラスタラベルを色分け
        cmap='tab10',            # 適宜好きなカラーマップに変更
        alpha=0.7
    )
    # クラスタラベルごとの凡例を作成
    unique_clusters = np.unique(cluster_labels)
    for cluster in unique_clusters:
        ax.scatter([], [], c=plt.cm.tab10(cluster / len(unique_clusters)), label=f"Cluster {cluster}")

    ax.legend(title="Clusters")
    #ax.set_title("K-means Clustering on Spectrogram (PCA 2D)")
    #ax.set_xlabel("PC1")
    #ax.set_ylabel("PC2")

    # カラーバー（クラスタラベル用）
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Cluster Label")

    # 出力保存
    output_path = os.path.join(output_dir, "kmeans_pca_scatter.png")
    plt.savefig(output_path)
    plt.close()
    logger.info(f"K-meansの結果を2次元PCAで可視化したプロットを保存しました: {output_path}")

    # --------------------------------------------------
    # 5) 速度を色に割り当てる例（任意）
    # --------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    speed_norm = (all_speeds - all_speeds.min()) / (all_speeds.max() - all_speeds.min() + 1e-9)
    scatter2 = ax2.scatter(
        pca_result[:, 0],
        pca_result[:, 1],
        c=speed_norm,         # 速度を正規化して色分け
        cmap='viridis',
        alpha=0.7
    )
    ax2.set_title("PCA 2D (colored by Speed)")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")

    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label("Normalized Speed")

    output_path2 = os.path.join(output_dir, "pca_scatter_speed.png")
    plt.savefig(output_path2)
    plt.close()
    logger.info(f"速度を色にした散布図を保存しました: {output_path2}")


    # --------------------------------------------------
    # 6) 車種・方向の例（マーカー・色分け）
    # --------------------------------------------------
    vehicle_color_map = {0: "blue", 1: "red"}  # 車種: car=0 (青), cv=1 (赤)
    direction_marker_map = {0: "o", 1: "s"}  # 方向: right=0 (○), left=1 (□)

    fig3, ax3 = plt.subplots(figsize=(8, 6))
    for i in range(len(pca_result)):
        vt = all_vehicle_types[i]  # 車種 (0=car, 1=cv)
        dr = all_directions[i]     # 方向 (0=right, 1=left)
        x = pca_result[i, 0]
        y = pca_result[i, 1]
        ax3.scatter(
            x, y,
            marker=direction_marker_map.get(dr, "x"),  # 方向に対応するマーカー
            color=vehicle_color_map.get(vt, "green"),  # 車種に対応する色
            alpha=0.6
        )

    # 車種・方向の凡例を追加
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='car (Right)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=10, label='car (Left)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='cv (Right)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=10, label='cv (Left)')
    ]
    ax3.legend(handles=legend_elements, loc='upper right', title="Vehicle & Direction")
    # ax3.set_title("PCA 2D by Vehicle Type & Direction")
    # ax3.set_xlabel("PC1")
    # ax3.set_ylabel("PC2")

    # 保存
    output_path3 = os.path.join(output_dir, "pca_scatter_vtype_direc.png")
    plt.savefig(output_path3)
    plt.close()
    logger.info(f"車種と走行方向で形状・色分けした散布図を保存しました: {output_path3}")


    # --------------------------------------------------
    # 7) 場所ごとに色分けして可視化する例
    # --------------------------------------------------
    # loc1～loc6が 0,1,2,3,4,5 の整数ラベルになっている前提です
    # （prepare_dataloader内の location_map = {'loc1':0, 'loc2':1, ...}）
    logger.info("場所ごとに色分けして可視化します。")
    unique_locs = np.unique(all_locations)  # [0,1,2,3,4,5] 等
    # カラーマップを用意 (例: tab10 の中から unique_locs の個数分を取得)
    colors = plt.cm.get_cmap('tab10', len(unique_locs))

    fig4, ax4 = plt.subplots(figsize=(8, 6))
    for loc_index, loc_value in enumerate(unique_locs):
        # この場所に該当するデータだけを取り出す
        idx = (all_locations == loc_value)
        ax4.scatter(
            pca_result[idx, 0],
            pca_result[idx, 1],
            color=colors(loc_index),
            alpha=0.6,
            label=f"loc{loc_value+1}"  # loc_valueが0ならloc1, 1ならloc2...という表記に
        )

    # ax4.set_title("PCA 2D (colored by Location)")
    # ax4.set_xlabel("PC1")
    # ax4.set_ylabel("PC2")
    ax4.legend()  # 場所の凡例表示
    output_path4 = os.path.join(output_dir, "pca_scatter_locations.png")
    plt.savefig(output_path4)
    plt.close()
    logger.info(f"場所ごとの色分け散布図を保存しました: {output_path4}")

    logger.info("=== K-meansクラスタリング & 可視化が完了しました ===")

if __name__ == "__main__":
    import encoder.CNN_s as CNN_s
    logger = CNN_s.logger
    output_dir = CNN_s.output_dir
    DATA = CNN_s.DATA # 使用するデータ: loc1 or loc1-6
    SEED = CNN_s.SEED

    CLUSTERS = 4 # K-meansクラスタの数

    CNN_s.hyperparameter()
    logger.info(f'CLUSTERS: {CLUSTERS}')
    logger.info(f'K-means による次元削減')
    train_loader, val_loader = CNN_s.prepare_dataloader()
    logger.info("データの準備が完了しました。")

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

    # K-means + 可視化の関数を呼び出す
    kmeans_clustering_and_visualize(
        train_loader=train_loader,
        val_loader=val_loader,
        logger=logger,
        output_dir=output_dir,
        n_clusters=CLUSTERS
    )