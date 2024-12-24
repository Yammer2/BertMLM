import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split

# パラメータ設定
data_size = 8000  # データ数
time_steps = 40   # 時系列長
train_ratio = 0.7
val_ratio = 0.2

# 合成データ生成（サインとコサインの位相を変えたノイズ波形）
def generate_sine_cosine_series(data_size, time_steps):
    """
    サイン波とコサイン波の位相を変えたノイズ波形データを生成。
    """
    data = []
    for _ in range(data_size):
        time = np.linspace(0, 10, time_steps)
        phase_shift = np.random.uniform(0, 2 * np.pi)  # ランダムな位相シフト
        sine_wave = np.sin(time + phase_shift) + np.random.normal(0, 0.1, time_steps)  # サイン波 + ノイズ
        cosine_wave = np.cos(time + phase_shift) + np.random.normal(0, 0.1, time_steps)  # コサイン波 + ノイズ
        series = np.stack([sine_wave, cosine_wave], axis=-1)  # (time_steps, 2)
        data.append(series)
    return np.array(data)  # (data_size, time_steps, 2)

# データ生成
synthetic_data = generate_sine_cosine_series(data_size, time_steps)

# データの確認
print("Synthetic Data Shape:", synthetic_data.shape)  # (2000, 20, 2)

# Datasetクラスの定義
class TimeSeriesDataset(Dataset):
    def __init__(self, data):
        """
        :param data: NumPy配列 (data_size, time_steps, 2)
        """
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# データローダの定義
def get_data_loaders(data, batch_size=32, train_ratio=0.7, val_ratio=0.2):
    dataset = TimeSeriesDataset(data)
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# データをCSV形式で保存
def save_to_csv(data, file_path):
    flat_data = data.reshape(data.shape[0], -1)  # (data_size, time_steps * 2)
    columns = [f"t{t}_dim{d}" for t in range(data.shape[1]) for d in range(data.shape[2])]
    df = pd.DataFrame(flat_data, columns=columns)
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")

# CSV形式で保存
output_file = "synthetic_sine_cosine_series.csv"
save_to_csv(synthetic_data, output_file)

# DataLoaderの作成
batch_size = 32
train_loader, val_loader, test_loader = get_data_loaders(synthetic_data, batch_size=batch_size)

# DataLoaderのテスト
for batch in train_loader:
    print("Train Batch Shape:", batch.shape)  # (batch_size, time_steps, 2)
    break
for batch in val_loader:
    print("Validation Batch Shape:", batch.shape)  # (batch_size, time_steps, 2)
    break
for batch in test_loader:
    print("Test Batch Shape:", batch.shape)  # (batch_size, time_steps, 2)
    break
