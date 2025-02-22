"""
汎用的な可視化クラス。
2次元または3次元に圧縮された
潜在空間表現とラベルを与えると、図による可視化を行う。
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class LatentSpaceVisualizer:
    """
    2次元または3次元の潜在空間可視化を行うクラス。

    以下の情報を与えてインスタンス化します:
        dimension_latent_space: 潜在空間の次元 (2 or 3)
        latent_data           : 潜在空間の座標データ (形状 [n, 2] or [n, 3])
        speeds                : 速度リスト (形状 [n, ])
        vehicle_types         : 車両タイプリスト (0->car, 1->cv) (形状 [n, ])
        directions            : 進行方向リスト (0->right, 1->left) (形状 [n, ])
        locations            : ロケーションIDリスト (形状 [n, ])
        output_dir            : 画像ファイルを出力するディレクトリ
        visualization         : 可視化の名前など (画像ファイル名に利用)
    """
    def __init__(
        self,
        dimension_latent_space,
        latent_data,
        speeds,
        vehicle_types,
        directions,
        locations,
        output_dir,
        visualization
    ):
        self.dimension_latent_space = dimension_latent_space
        self.latent_data = latent_data
        self.speeds = speeds
        self.vehicle_types = vehicle_types
        self.directions = directions
        self.locations = locations
        self.output_dir = output_dir
        self.visualization = visualization

        # カラーマップの正規化設定
        self.norm = cm.colors.Normalize(
            vmin=np.min(self.speeds),
            vmax=np.max(self.speeds)
        )

    def _visualize_2d_speed(self):
        """
        可視化1 (2D): 速度による色分けのみ
        """
        # FONT_SIZE= 20
        # plt.rcParams.update({'font.size': FONT_SIZE})

        fig, ax = plt.subplots(figsize=(12, 8))

        sc = ax.scatter(
            self.latent_data[:, 0],
            self.latent_data[:, 1],
            c=self.speeds,
            cmap='viridis',
            norm=self.norm,
            alpha=0.7,
            edgecolors='none',
            linewidths=1
        )

        # カラーバー
        sm = cm.ScalarMappable(norm=self.norm, cmap='viridis')
        sm.set_array(self.speeds)
        fig.colorbar(sm, ax=ax, label='Speed')

        ax.set_title(f"Latent Space Visualization using {self.visualization} (Speed only)")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")

        plt.savefig(os.path.join(self.output_dir, f'{self.visualization}_plot_speed_only.png'))
        plt.close()

        logger.info("速度による可視化が完了しました。(2D)")

    def _visualize_2d_vehicle_direction(self):
        """
        可視化2 (2D): 車両タイプと方向によるマーカーの違い
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        # マーカーの設定
        markers = {
            (0, 0): 'o',  # (car, right)
            (0, 1): '^',  # (car, left)
            (1, 0): 's',  # (cv, right)
            (1, 1): 'D'   # (cv, left)
        }

        # (vehicle_type, direction) の組み合わせでプロット
        for vt in [0, 1]:
            for dr in [0, 1]:
                mask = (self.vehicle_types == vt) & (self.directions == dr)
                if np.any(mask):
                    ax.scatter(
                        self.latent_data[mask, 0],
                        self.latent_data[mask, 1],
                        c=self.speeds[mask],
                        cmap='viridis',
                        norm=self.norm,
                        alpha=0.7,
                        marker=markers[(vt, dr)],
                        edgecolors='none',
                        linewidths=1,
                        label=f"{'car' if vt == 0 else 'cv'}/{'right' if dr == 0 else 'left'}"
                    )

        
        # カラーバー
        sm = cm.ScalarMappable(norm=self.norm, cmap='viridis')
        sm.set_array(self.speeds)
        fig.colorbar(sm, ax=ax, label='Speed')
        
        ax.set_title(f"Latent Space Visualization using {self.visualization} (Categorized by Vehicle Type & Direction)")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.legend()

        plt.savefig(os.path.join(self.output_dir, f'{self.visualization}_plot_categorized.png'))
        plt.close()

        logger.info("車両タイプと方向による可視化が完了しました。(2D)")

    def _visualize_2d_vehicle_direction_NOspeed(self):
        """
        可視化2-2 (2D): 車両タイプと方向によるマーカーの違い(スピード情報なし)
        """
        # FONT_SIZE= 20
        # plt.rcParams.update({'font.size': FONT_SIZE})

        fig, ax = plt.subplots(figsize=(12, 8))

        # (vehicle_type, direction) の組み合わせでプロット
        for vt in [0, 1]:
            for dr in [0, 1]:
                mask = (self.vehicle_types == vt) & (self.directions == dr)
                if np.any(mask):
                    ax.scatter(
                        self.latent_data[mask, 0],
                        self.latent_data[mask, 1],
                        alpha=0.7,
                        edgecolors='none',
                        linewidths=1,
                        label=f"{'car' if vt == 0 else 'cv'}/{'right' if dr == 0 else 'left'}"
                    )
        
        ax.set_title(f"Latent Space Visualization using {self.visualization} (Categorized by Vehicle Type & Direction)")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.legend()

        plt.savefig(os.path.join(self.output_dir, f'{self.visualization}_plot_categorized(NO_speed).png'))
        plt.close()

        logger.info("車両タイプと方向による可視化が完了しました。※NOspeed(2D)")

    def _visualize_2d_locations(self):
        """
        可視化3 (2D): location によるマーカーの違い
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        # location ID に応じてマーカーを切り替える
        markers = {
            0: 'o',
            1: '^',
            2: 's',
            3: 'D',
            4: 'v',
            5: 'P'
        }

        unique_locations = np.unique(self.locations)
        for loc in unique_locations:
            mask = (self.locations == loc)
            if np.any(mask):
                ax.scatter(
                    self.latent_data[mask, 0],
                    self.latent_data[mask, 1],
                    c=self.speeds[mask],
                    cmap='viridis',
                    norm=self.norm,
                    alpha=0.7,
                    marker=markers.get(loc, 'o'),
                    edgecolors='none',
                    linewidths=1,
                    label=f"Location {loc}"
                )

        sm = cm.ScalarMappable(norm=self.norm, cmap='viridis')
        sm.set_array(self.speeds)
        fig.colorbar(sm, ax=ax, label='Speed')

        ax.set_title(f"Latent Space Visualization using {self.visualization} (Categorized by Location)")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.legend()

        plt.savefig(os.path.join(self.output_dir, f'{self.visualization}_plot_categorized_by_location.png'))
        plt.close()

        logger.info("locations による可視化が完了しました。(2D)")

    # def _visualize_2d_locations(self):
    #     """
    #     可視化 (2D): location によるマーカー・色分け
    #     """
    #     fig, ax = plt.subplots(figsize=(12, 8))

    #     # location ID に応じてマーカーを切り替える
    #     markers = {
    #         0: 'o',
    #         1: '^',
    #         2: 's',
    #         3: 'D',
    #         4: 'v',
    #         5: 'P'
    #     }

    #     # location ID に応じて色を切り替える（必要に応じて増やしてください）
    #     colors = {
    #         0: 'blue',
    #         1: 'orange',
    #         2: 'green',
    #         3: 'red',
    #         4: 'purple',
    #         5: 'brown'
    #     }

    #     unique_locations = np.unique(self.locations)
    #     for loc in unique_locations:
    #         mask = (self.locations == loc)
    #         if np.any(mask):
    #             ax.scatter(
    #                 self.latent_data[mask, 0],
    #                 self.latent_data[mask, 1],
    #                 color=colors.get(loc, 'black'),      # location による色分け
    #                 marker=markers.get(loc, 'o'),        # location によるマーカー分け
    #                 alpha=0.7,
    #                 edgecolors='none',
    #                 linewidths=1,
    #                 label=f"Location {loc}"
    #             )

    #     # Speed によるカラーバーは使用しないため削除
    #     # fig.colorbar などの設定も削除

    #     ax.set_title(f"Latent Space Visualization ({self.visualization}) - Categorized by Location")
    #     ax.set_xlabel("Dimension 1")
    #     ax.set_ylabel("Dimension 2")
    #     ax.legend()

    #     plt.savefig(os.path.join(self.output_dir, f'{self.visualization}_plot_categorized_by_location.png'))
    #     plt.close()

    #     logger.info("locations による可視化が完了しました。(2D)")


    def _visualize_3d_speed(self, show_plot=False):
        """
        可視化1 (3D): 速度による色分けのみ
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        sc = ax.scatter(
            self.latent_data[:, 0],
            self.latent_data[:, 1],
            self.latent_data[:, 2],
            c=self.speeds,
            cmap='viridis',
            norm=self.norm,
            alpha=0.7,
            edgecolors='none',
            linewidths=1
        )

        # カラーバー
        cbar = fig.colorbar(sc, ax=ax, label='Speed')

        ax.set_title(f"Latent Space Visualization using {self.visualization} (Speed only) - 3D")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_zlabel("Dimension 3")

        # 3D可視化をインタラクティブ表示したい場合
        if show_plot:
            plt.show()

        plt.savefig(os.path.join(self.output_dir, f'{self.visualization}_plot_speed_only_3d.png'))
        logger.info("速度による可視化(3D)が完了しました。")
        plt.close()

    def _visualize_3d_vehicle_direction(self, show_plot=False):
        """
        可視化2 (3D): 車両タイプと方向によるマーカーの違い
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        markers = {
            (0, 0): 'o',  # car/right
            (0, 1): '^',  # car/left
            (1, 0): 's',  # cv/right
            (1, 1): 'D'   # cv/left
        }

        for vt in [0, 1]:
            for dr in [0, 1]:
                mask = (self.vehicle_types == vt) & (self.directions == dr)
                if np.any(mask):
                    ax.scatter(
                        self.latent_data[mask, 0],
                        self.latent_data[mask, 1],
                        self.latent_data[mask, 2],
                        c=self.speeds[mask],
                        cmap='viridis',
                        norm=self.norm,
                        alpha=0.7,
                        marker=markers[(vt, dr)],
                        edgecolors='none',
                        linewidths=1,
                        label=f"{'car' if vt == 0 else 'cv'}/{'right' if dr == 0 else 'left'}"
                    )

        sm = cm.ScalarMappable(norm=self.norm, cmap='viridis')
        sm.set_array(self.speeds)
        fig.colorbar(sm, ax=ax, label='Speed')

        ax.set_title(f"Latent Space Visualization using {self.visualization} (Vehicle & Direction) - 3D")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_zlabel("Dimension 3")
        ax.legend()

        # 3D可視化をインタラクティブ表示したい場合
        if show_plot:
            plt.show()
        
        plt.savefig(os.path.join(self.output_dir, f'{self.visualization}_plot_categorized_3d.png'))
        logger.info("車両タイプと方向による可視化(3D)が完了しました。")
        plt.close()

    def _visualize_3d_vehicle_direction_NOspeed(self, show_plot=False):
        """
        可視化2-2 (3D): 車両タイプと方向によるマーカーの違い（マーカー指定なし・スピードバー・色分けなし）
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        for vt in [0, 1]:
            for dr in [0, 1]:
                mask = (self.vehicle_types == vt) & (self.directions == dr)
                if np.any(mask):
                    ax.scatter(
                        self.latent_data[mask, 0],
                        self.latent_data[mask, 1],
                        self.latent_data[mask, 2],
                        alpha=0.7,
                        label=f"{'car' if vt == 0 else 'cv'}/{'right' if dr == 0 else 'left'}"
                    )

        ax.set_title(f"Latent Space Visualization using {self.visualization} (Vehicle & Direction) - 3D")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_zlabel("Dimension 3")
        ax.legend()

        # 3D可視化をインタラクティブ表示したい場合
        if show_plot:
            plt.show()
            
        plt.savefig(os.path.join(self.output_dir, f'{self.visualization}_plot_categorized_3d_NOspeed.png'))
        logger.info("車両タイプと方向による可視化(3D/NOspeed)が完了しました。")
        plt.close()

    def _visualize_3d_locations(self, show_plot=False):
        """
        可視化3 (3D): location によるマーカーの違い
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        markers = {
            0: 'o',
            1: '^',
            2: 's',
            3: 'D',
            4: 'v',
            5: 'P'
        }

        unique_locations = np.unique(self.locations)
        for loc in unique_locations:
            mask = (self.locations == loc)
            if np.any(mask):
                ax.scatter(
                    self.latent_data[mask, 0],
                    self.latent_data[mask, 1],
                    self.latent_data[mask, 2],
                    c=self.speeds[mask],
                    cmap='viridis',
                    norm=self.norm,
                    alpha=0.7,
                    marker=markers.get(loc, 'o'),
                    edgecolors='none',
                    linewidths=1,
                    label=f"Location {loc}"
                )

        sm = cm.ScalarMappable(norm=self.norm, cmap='viridis')
        sm.set_array(self.speeds)
        fig.colorbar(sm, ax=ax, label='Speed')

        ax.set_title(f"Latent Space Visualization using {self.visualization} (Location) - 3D")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_zlabel("Dimension 3")
        ax.legend()

        plt.savefig(os.path.join(self.output_dir, f'{self.visualization}_plot_categorized_by_location_3d.png'))
        logger.info("locations による可視化(3D)が完了しました。")

        # 3D可視化をインタラクティブ表示したい場合
        if show_plot:
            plt.show()
        
        plt.close()

    def visualize_all(self, show_plot=False):
        """
        引数 dimension_latent_space (2 or 3) に応じて、
        speed, vehicle_direction, locations の各可視化を順番に行う。
        show_plot = True にすると、3Dプロット時に plt.show() で表示（インタラクティブ操作）を有効化する。
        """
        if self.dimension_latent_space == 2:
            # ====== 2D ======
            self._visualize_2d_speed()
            self._visualize_2d_vehicle_direction()
            self._visualize_2d_vehicle_direction_NOspeed()
            self._visualize_2d_locations()

        elif self.dimension_latent_space == 3:
            # ====== 3D ======
            self._visualize_3d_speed(show_plot=False)
            self._visualize_3d_vehicle_direction(show_plot=False)
            self._visualize_3d_vehicle_direction_NOspeed(show_plot=show_plot)
            self._visualize_3d_locations(show_plot=show_plot)

        else:
            print("潜在空間の次元 (2 or 3) が正しく設定されていません。")
