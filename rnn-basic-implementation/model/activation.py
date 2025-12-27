# model/activation.py
import numpy as np

def tanh(x: np.ndarray) -> np.ndarray:
    """双曲正切激活函数"""
    return np.tanh(x)

def tanh_derivative(x: np.ndarray) -> np.ndarray:
    """双曲正切函数的导数"""
    return 1 - np.tanh(x) ** 2

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid激活函数"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """Sigmoid函数的导数"""
    sig = sigmoid(x)
    return sig * (1 - sig)

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU激活函数"""
    return np.maximum(0, x)

def relu_derivative(x: np.ndarray) -> np.ndarray:
    """ReLU函数的导数"""
    return (x > 0).astype(float)