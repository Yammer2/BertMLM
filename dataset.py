import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split

class TimeSeriesDataset(Dataset):
    def __init__(self, file_path, time_steps, dimensions):
        """
        :param file_path: データが保存されているCSVファイルのパス
        :param time_steps: 時系列長
        :param dimensions: 次元数
        """
        self.data = pd.read_csv(file_path).values
        self.time_steps = time_steps
        self.dimensions = dimensions
        self.data = self.data.reshape(-1, time_steps, dimensions)  # (data_size, time_steps, dimensions)
        self.data = torch.tensor(self.data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_data_loaders(file_path, time_steps, dimensions, batch_size=32, train_ratio=0.7, val_ratio=0.2):
    dataset = TimeSeriesDataset(file_path, time_steps, dimensions)
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    return train_loader, val_loader, test_loader