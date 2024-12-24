import numpy as np
import pandas as pd

# パラメータ設定
data_size = 2000  # データ数
time_steps = 20   # 時系列長
dimensions = 512  # 次元数

# 合成データ生成
def generate_synthetic_series(data_size, time_steps, dimensions):
    """
    シンプルなサイン波とノイズを含む合成データを生成。
    """
    data = []
    for _ in range(data_size):
        series = []
        for _ in range(dimensions):
            # 時系列データの生成（サイン波 + ノイズ）
            time = np.linspace(0, 10, time_steps)
            sine_wave = np.sin(time * np.random.uniform(0.5, 2.0))  # ランダムな周波数のサイン波
            noise = np.random.normal(0, 0.1, time_steps)            # ノイズ
            series.append(sine_wave + noise)
        data.append(np.array(series).T)  # 転置して (time_steps, dimensions) の形に
    return np.array(data)  # 最終的な形は (data_size, time_steps, dimensions)

# データ生成
synthetic_data = generate_synthetic_series(data_size, time_steps, dimensions)

# データの確認
print("Synthetic Data Shape:", synthetic_data.shape)  # (2000, 20, 512)

# CSV形式で保存（必要に応じて）
flattened_data = synthetic_data.reshape(data_size, -1)  # (2000, 20 * 512)
columns = [f"t{t}_d{d}" for t in range(time_steps) for d in range(dimensions)]
df = pd.DataFrame(flattened_data, columns=columns)

# 保存
output_file = "synthetic_time_series.csv"
df.to_csv(output_file, index=False)
print(f"Synthetic time series data saved to {output_file}")
