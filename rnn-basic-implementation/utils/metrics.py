# utils/metrics.py
import numpy as np


def calculate_mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    计算均方误差 (Mean Squared Error)

    参数:
        y_pred: 预测值
        y_true: 真实值

    返回:
        MSE值
    """
    return np.mean((y_pred - y_true) ** 2)


def calculate_mae(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    计算平均绝对误差 (Mean Absolute Error)

    参数:
        y_pred: 预测值
        y_true: 真实值

    返回:
        MAE值
    """
    return np.mean(np.abs(y_pred - y_true))


def calculate_r2_score(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    计算R²分数 (Coefficient of Determination)

    参数:
        y_pred: 预测值
        y_true: 真实值

    返回:
        R²分数
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-10))


def calculate_explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    计算解释方差分数

    参数:
        y_pred: 预测值
        y_true: 真实值

    返回:
        解释方差分数
    """
    return 1 - np.var(y_true - y_pred) / np.var(y_true)