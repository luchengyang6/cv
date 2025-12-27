# utils/initializers.py
import numpy as np


def xavier_init(fan_in: int, fan_out: int) -> float:
    """
    Xavier/Glorot 初始化

    参数:
        fan_in: 输入单元数
        fan_out: 输出单元数

    返回:
        初始化尺度
    """
    return np.sqrt(2.0 / (fan_in + fan_out))


def he_init(fan_in: int) -> float:
    """
    He 初始化 (适用于ReLU激活函数)

    参数:
        fan_in: 输入单元数

    返回:
        初始化尺度
    """
    return np.sqrt(2.0 / fan_in)


def random_normal_init(shape: tuple, mean: float = 0.0, std: float = 0.01) -> np.ndarray:
    """
    正态分布初始化

    参数:
        shape: 权重形状
        mean: 均值
        std: 标准差

    返回:
        初始化后的权重矩阵
    """
    return np.random.randn(*shape) * std + mean


def zeros_init(shape: tuple) -> np.ndarray:
    """
    零初始化

    参数:
        shape: 权重形状

    返回:
        全零矩阵
    """
    return np.zeros(shape)


def ones_init(shape: tuple) -> np.ndarray:
    """
    一初始化

    参数:
        shape: 权重形状

    返回:
        全一矩阵
    """
    return np.ones(shape)