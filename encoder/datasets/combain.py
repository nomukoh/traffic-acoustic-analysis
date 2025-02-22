import os
import pandas as pd

# locX, car/cv, left/rightのパラメータ
locations = [f'loc{i}' for i in range(1, 7)]
categories = ['car', 'cv']
directions = ['left', 'right']

# 結合したデータを保持するリスト
combined_data = []

# 各CSVファイルを処理
for loc in locations:
    for category in categories:
        for direction in directions:
            # CSVファイルのパス
            csv_path = f'./workfolder/simulation/{loc}/{category}/{direction}/events-0000-0049.csv'
            
            # CSVファイルが存在するか確認
            if os.path.exists(csv_path):
                # CSVファイルを読み込む
                df = pd.read_csv(csv_path)
                
                # プレフィックス
                prefix = f'{loc}_cut/{category}/{direction}/'
                
                # path列が存在する場合はpath列の値にprefixを付与
                if 'path' in df.columns:
                    df['path'] = prefix + df['path'].astype(str)
                else:
                    # path列が存在しない場合、作成(ここでイベントファイル名がどの列にあるか要確認)
                    # 本来はイベントファイル名を保持する列名がわかっている場合、その列を利用
                    # 今回は例としてeventファイル名が"Path"または"FileName"のような列名だった場合を想定
                    # もし元CSVがpath列のみであれば下記は不要
                    # df['path'] = prefix + df['Path'].astype(str)
                    # あるいはファイル名自体が他列に無い場合はprefixのみを使うことも可能
                    pass
                
                # 列の順序を指定: path,index,speed,vehicle_type,direction
                desired_cols = ['path', 'index', 'speed', 'vehicle_type', 'direction']
                # 存在する列のみ抽出し、存在しない列はスキップ
                existing_cols = [col for col in desired_cols if col in df.columns]
                df = df[existing_cols]
                
                combined_data.append(df)

# 全てのデータフレームを結合
if combined_data:
    result = pd.concat(combined_data, ignore_index=True)
    
    # 結果を保存
    output_path = './encoder/datasets/data_1-6.csv'
    result.to_csv(output_path, index=False)
    print(f"結合したCSVを {output_path} に保存しました。")
else:
    print("対象のCSVファイルが見つかりませんでした。")
