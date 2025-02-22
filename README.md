# 環境音の特徴を捉える潜在空間表現の設計

このリポジトリは、卒業論文「環境音の特徴を捉える潜在空間表現の設計」で使用したプログラムを公開するためのものです。本コードは、論文に記載した実験結果の再現性を担保する目的で公開されていますが、研究プロトタイプとしての提供であり、今後のメンテナンスやサポートは行いません。  

## プロジェクトの目的

本プロジェクトは、従来の環境音解析手法が抱える録音条件依存やノイズの影響を克服し、環境条件に左右されない安定した特徴抽出を実現するための深層学習手法の検証を目的としています。具体的には、深層距離学習（Circle Loss や Log-ratio Loss）を用いて、環境音から低次元の潜在空間表現を獲得し、その有用性を評価しています。

## 背景

従来の環境音解析では、ノイズや録音環境の違いが大きな障壁となっており、直接的な特徴抽出が難しいという課題がありました。本研究では、CNN を用いた特徴抽出と深層距離学習を組み合わせることで、環境条件に依存しない特徴空間を取得することを試みました。

## 研究概要

本プロジェクトは、環境音から交通監視に有用な特徴を抽出するため、深層距離学習を用いた潜在空間表現の獲得手法を検証するものです。実験は以下の3種類を行い、プログラム実装面での詳細な検証を行いました。

- **実験１： Circle Loss による潜在空間の取得**  
   車種（car, cv）および進行方向（right, left）のラベルを用い、CNNエンコーダ（チャネル構成：1→16→32→64→64、すべて3×3カーネル、stride=2、padding=1）で低次元表現を学習しました。Adam（学習率1e-3、バッチサイズ32）を用いて300エポックで学習し、t-SNEによる可視化で4クラスタ（car_left, car_right, cv_left, cv_right）が明確に分離されることを確認しました。

- **実験２： Log-ratio Loss による潜在空間の取得**  
   Circle Lossで事前学習したモデルに対し、連続値の速度ラベルを反映させるためLog-ratio Lossを適用。Adam（学習率1e-3、バッチサイズ32）で100エポック学習し、t-SNEによる可視化で速度情報が保持された潜在空間が形成されることを確認しました。

- **実験３： 速度予測モデルの構築**  
   Log-ratio Lossで得られた潜在空間の特徴を活用し、CNNエンコーダの出力を全結合層で64次元に圧縮後、DNN（構成：64→128→1）を用いて速度を回帰予測するモデルを構築しました。Adam（学習率1e-3、バッチサイズ32）で100エポック学習した結果、事前学習済みCNNの有用性が示され、平均で約±5.4 km/hの誤差内で速度予測が可能であることが確認されました。

## 主な機能とモジュール

本プログラムは、以下の主要なモジュールで構成されています。

- **encoder/**  
  学習に関連する主要なプログラム群を収めたフォルダです。  
  - **auto_encoder/**  
    - *auto-encoder.py*：1次元オートエンコーダを学習させるためのプログラム（論文には記載せず、前段階実験用として実施）  
    - *base_model.py*：モデルの構造を定義するクラスを提供  
    - *CNN_o*：DCASEコンペティションのベースラインに近いCNNエンコーダモデルを学習させるプログラム（前段階実験用）  
    - *CNN_s*：CNNエンコーダモデル（CNN_oより大規模）の学習プログラム（前段階実験用）  
  - **datasets/**  
    学習に使用するデータを格納するフォルダです。  
    - *locx_cut/*：loc1_cut～loc6_cutまで、各ロケーションごとの前処理済みデータ  
    - *combain.py*：ダウンロードデータに含まれるメタデータを結合するスクリプト  
  - **loss/**  
    実験2で使用する深層距離学習用の損失関数を配置するフォルダです（こちらのコードは外部からダウンロードしてください）。  
  - **metric_learning/**  
    深層距離学習の実行に関するプログラム群です。  
    - *LogRatioLoss.py*：実験2で用いる、連続ラベル対応のLogRatio Lossによる学習プログラム  
    - *MetricLrarning.py*：実験1で用いる、Circle Lossによる学習プログラム  
  - *kmeans.py*：データ全体に対してK-meansによるクラスタリングを実行し、ラベル付けを行うプログラム  
  - *settings.py*：データ前処理および出力フォルダの設定を行うクラスを提供  
  - *speedPrediction.py*：実験3の速度予測モデルを学習させるためのプログラム  
  - *validation.py*：取得した潜在空間の評価（各ラベルごとの再構成誤差の算出など）を行うクラス  
  - *visualize.py*：潜在空間や学習結果の可視化を行うためのクラス  

- **TSNE/**  
  - *tsne.py*：学習済みの潜在空間をt-SNEを用いて2次元に可視化するためのプログラム。パラメータ調整など、詳細な検証に利用可能です。

- **workfolder/**  
  - **simulation/**：ダウンロードした学習用データを格納するためのフォルダです。

## 環境構築と実行方法

以下の手順で環境をセットアップし、プログラムを実行してください。

1. **リポジトリのクローン**  
    ```bash
    git clone https://github.com/nomukoh/traffic-acoustic-analysis.git
    cd traffic-acoustic-analysis
    ```

2. **プログラムと学習データのダウンロード**  
   - 深層距離学習の損失関数（CircleLoss）は、[Githubページ](https://github.com/TinyZeaMays/CircleLoss)からダウンロードし、`encoder/loss/`ディレクトリに`circle_loss.py`として配置してください。  
   - 深層距離学習の損失関数（LogRatioLoss）は、[Githubページ](https://github.com/tjddus9597/Beyond-Binary-Supervision-CVPR19)からダウンロードし、`main.py`、`utils.py`、`LogRatioLoss.py`を`encoder/loss/`ディレクトリに配置してください。  
   - 学習データは、[Zenodo](https://zenodo.org/records/10700792)からダウンロードし、`loc1`～`loc6`のフォルダを`workfolder/simulation/`に配置してください。

3. **データの準備**  
   - 以下のコマンドを実行して、走行音が最も大きい6秒間のデータをトリミングします。  
     ```bash
     poetry run python -m workfolder.simulation.cut
     ```  
   - 次に、以下のコマンドを実行して、`loc1`～`loc6`のメタデータを統合したメタファイルを作成します。  
     ```bash
     poetry run python -m encoder.datasets.combain
     ```

4. **プログラムの実行**  
   - **実験０（多様体取得）**  
     各実験前に、対象データ、可視化手法、ハイパーパラメータ等のパラメータ定義を必要に応じて変更してください。  
     - 1次元オートエンコーダの学習:  
       ```bash
       poetry run python -m encoder.auto_encoder.auto-encoder
       ```  
     - コンペティションのベースラインモデルに近いCNNエンコーダモデルの学習:  
       ```bash
       poetry run python -m encoder.auto_encoder.CNN_o
       ```  
     - 研究で主に使用したCNNエンコーダモデルの学習:  
       ```bash
       poetry run python -m encoder.auto_encoder.CNN_s
       ```

   - **実験１（CircleLossによる深層距離学習）**  
     車種と進行方向のラベルを用いてデータの分離を行います。  
     ```bash
     poetry run python -m encoder.metric_learning.MetricLearning
     ```

   - **実験２（LogRatioLossによる連続情報を保持した潜在空間の取得）**  
     速度ラベルに基づく連続情報を反映させる深層距離学習を行います。実験１からの継続学習を行う場合は、`outputs/`フォルダに保存された`.pth`ファイルを`encoder/`直下に配置し、プログラム内の該当部分のコメントを外してください。  
     ```bash
     poetry run python -m encoder.metric_learning.LogRatioLoss
     ```

   - **実験３（速度予測モデルの学習）**  
     事前学習済みの潜在空間を利用して、速度予測モデルの学習を行います。実験２からの継続学習を行う場合は、`outputs/`フォルダに保存された`.pth`ファイルを`encoder/`直下に配置し、プログラム内の該当部分のコメントを外してください。  
     ```bash
     poetry run python -m encoder.speedPrediction
     ```

5. **その他のプログラム**
   - t-SNEによる可視化
      `outputs/`フォルダに保存された`latent_spaces`ファイルと`metadata.csv`ファイルを`TSNE/`直下に配置し、以下のプログラムを実行することで、t-SNEによる可視化が行えます。`TSNE/tsne.py`の中でハイパパラメータを変更することができます。
     ```bash
     poetry run python -m TSNE.tsne
     ```

## 注意事項

本プログラムは研究目的で公開しているものであり、実運用を前提としたものではありません。動作保証やサポートは行っておりませんので、あらかじめご了承ください。

## 関連研究

本研究では、環境音の研究コミュニティーであるDCASEが2024年に主催した DCASE 2024 Task10 を参考にしています。以下に示すGitHubリポジトリとは互換性があり、１つのフォルダに統合してプログラムを動作させることができます。
- [ホームページ](https://dcase.community/challenge2024/task-acoustic-based-traffic-monitoring)
- [GitHub](https://github.com/boschresearch/acoustic-traffic-simulation-counting)

## ライセンス

本リポジトリ内の自作プログラムの著作権はすべて nomukoh に帰属します。ただし、外部からダウンロードしたファイルは、各公式ページに記載のライセンス条件に従ってご利用ください。

プログラムの実行に必要なコードのダウンロード先とライセンスを示します。
[circle_loss.py](https://github.com/TinyZeaMays/CircleLoss)（非公式実装・ライセンス未記載）
[LogRatioLoss.py](https://github.com/tjddus9597/Beyond-Binary-Supervision-CVPR19)（MITライセンス）
DCASE 2024 Challenge Task 10 Development Dataset [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10700792.svg)](https://doi.org/10.5281/zenodo.10700792)
（CC BY-NC-SA 4.0）
