import numpy as np
import os


def load_simple_dataset():
    """
    生成简单的二维分类数据集用于演示
    """
    np.random.seed(42)

    # 生成三个类别的数据
    n_samples = 300
    n_features = 2

    # 类别1
    X1 = np.random.normal([2, 2], 1, (n_samples // 3, n_features))
    y1 = np.zeros(n_samples // 3)

    # 类别2
    X2 = np.random.normal([-2, -2], 1, (n_samples // 3, n_features))
    y2 = np.ones(n_samples // 3)

    # 类别3
    X3 = np.random.normal([2, -2], 1, (n_samples // 3, n_features))
    y3 = np.full(n_samples // 3, 2)

    X = np.vstack([X1, X2, X3])
    y = np.hstack([y1, y2, y3])

    return X, y


def save_dataset(X, y, filename='simple_dataset.npz'):
    """保存数据集"""
    np.savez(filename, X=X, y=y)


def load_dataset(filename='simple_dataset.npz'):
    """加载数据集"""
    data = np.load(filename)
    return data['X'], data['y']