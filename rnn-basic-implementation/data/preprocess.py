# data/preprocess.py
import numpy as np


def normalize_minmax(data: np.ndarray, feature_range: tuple = (0, 1)) -> np.ndarray:
    """
    最小-最大归一化

    参数:
        data: 输入数据
        feature_range: 归一化范围

    返回:
        归一化后的数据
    """
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)

    # 避免除以零
    data_range = data_max - data_min
    data_range[data_range == 0] = 1

    normalized = (data - data_min) / data_range
    normalized = normalized * (feature_range[1] - feature_range[0]) + feature_range[0]

    return normalized


def standardize(data: np.ndarray) -> np.ndarray:
    """
    标准化 (z-score 标准化)

    参数:
        data: 输入数据

    返回:
        标准化后的数据
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    # 避免除以零
    std[std == 0] = 1

    return (data - mean) / std


def add_noise(data: np.ndarray, noise_level: float = 0.05) -> np.ndarray:
    """
    添加高斯噪声

    参数:
        data: 输入数据
        noise_level: 噪声水平

    返回:
        添加噪声后的数据
    """
    noise = np.random.randn(*data.shape) * noise_level
    return data + noise


def create_sequences(data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    从时间序列数据创建输入-输出对

    参数:
        data: 原始时间序列 (n_timesteps, n_features)
        seq_length: 序列长度

    返回:
        X: 输入序列 (n_samples, seq_length, n_features)
        y: 输出序列 (n_samples, seq_length, n_features)
    """
    n_samples = len(data) - seq_length

    X = np.zeros((n_samples, seq_length, data.shape[1]))
    y = np.zeros((n_samples, seq_length, data.shape[1]))

    for i in range(n_samples):
        X[i] = data[i:i + seq_length]
        y[i] = data[i + 1:i + seq_length + 1]

    return X, y