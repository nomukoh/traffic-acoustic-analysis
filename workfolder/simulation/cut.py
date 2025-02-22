import os
import soundfile as sf

# locX, car/cv, left/rightのパラメータ
locations = [f'loc{i}' for i in range(2, 7)]
categories = ['car', 'cv']
directions = ['left', 'right']

# 処理を繰り返す
for loc in locations:
    for category in categories:
        for direction in directions:
            # 入力ディレクトリと出力ディレクトリのパスを生成
            input_dir = f'./workfolder/simulation/{loc}/{category}/{direction}'
            output_dir = f'../../encoder/datasets/{loc}_cut/{category}/{direction}'
            
            # 出力ディレクトリを作成（存在しない場合）
            os.makedirs(output_dir, exist_ok=True)
            
            # 入力ディレクトリ内のすべての .flac ファイルを処理
            for filename in os.listdir(input_dir):
                if filename.endswith('.flac'):
                    input_path = os.path.join(input_dir, filename)
                    output_path = os.path.join(output_dir, filename)
                    
                    # 音声ファイルを読み込む
                    data, samplerate = sf.read(input_path)
                    
                    # 12秒から18秒までの6秒間のサンプル数を計算
                    start_sample = int(12 * samplerate)
                    end_sample = int(18 * samplerate)
                    
                    # 音声データを切り取る
                    excerpt = data[start_sample:end_sample]
                    
                    # 結果を保存
                    sf.write(output_path, excerpt, samplerate)
