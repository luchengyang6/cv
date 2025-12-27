# data/dataset.py
import numpy as np
from typing import Tuple, Optional


class TimeSeriesDataset:
    """
    时间序列数据集类
    提供标准化的数据集接口
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        初始化数据集

        参数:
            X: 输入数据 (n_samples, seq_len, n_features)
            y: 目标数据 (n_samples, seq_len, n_targets)
        """
        self.X = X
        self.y = y
        self.n_samples = X.shape[0]
        self.seq_len = X.shape[1]
        self.n_features = X.shape[2]

    def __len__(self):
        """返回数据集大小"""
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """获取单个样本"""
        return self.X[idx], self.y[idx]

    def split(self, train_ratio: float = 0.8, shuffle: bool = True) -> Tuple:
        """
        分割数据集

        参数:
            train_ratio: 训练集比例
            shuffle: 是否打乱数据

        返回:
            (train_dataset, test_dataset)
        """
        indices = np.arange(self.n_samples)
        if shuffle:
            np.random.shuffle(indices)

        split_idx = int(self.n_samples * train_ratio)
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        train_X = self.X[train_indices]
        train_y = self.y[train_indices]
        test_X = self.X[test_indices]
        test_y = self.y[test_indices]

        return (TimeSeriesDataset(train_X, train_y),
                TimeSeriesDataset(test_X, test_y))

    def get_batch(self, batch_size: int, shuffle: bool = True):
        """
        生成批次数据

        参数:
            batch_size: 批次大小
            shuffle: 是否打乱数据

        返回:
            批次生成器
        """
        indices = np.arange(self.n_samples)
        if shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, self.n_samples, batch_size):
            end_idx = min(start_idx + batch_size, self.n_samples)
            batch_indices = indices[start_idx:end_idx]

            yield self.X[batch_indices], self.y[batch_indices]