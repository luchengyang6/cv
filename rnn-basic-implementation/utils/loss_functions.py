import numpy as np


def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    均方误差损失函数

    参数:
        y_pred: 预测值
        y_true: 真实值

    返回:
        MSE损失值
    """
    return np.mean((y_pred - y_true) ** 2)


def mse_gradient(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    均方误差的梯度

    参数:
        y_pred: 预测值
        y_true: 真实值

    返回:
        梯度值
    """
    return 2 * (y_pred - y_true) / y_pred.size