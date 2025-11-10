"""
数据加载和处理工具 - 使用相对路径
"""

import numpy as np
import os
import struct


def load_mnist_local(data_dir):
    """
    从本地文件加载MNIST数据集 - 修复版本

    Args:
        data_dir: 包含MNIST文件的目录（绝对路径或相对路径）

    Returns:
        train_data, train_labels, val_data, val_labels, test_data, test_labels
    """
    import struct

    def read_images(filename):
        """读取MNIST图像文件"""
        with open(filename, 'rb') as f:
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
            return images

    def read_labels(filename):
        """读取MNIST标签文件"""
        with open(filename, 'rb') as f:
            magic, num = struct.unpack('>II', f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)
            return labels

    # 使用您的实际文件名
    train_images_path = os.path.join(data_dir, 'train-images.idx3-ubyte')
    train_labels_path = os.path.join(data_dir, 'train-labels.idx1-ubyte')
    test_images_path = os.path.join(data_dir, 't10k-images.idx3-ubyte')
    test_labels_path = os.path.join(data_dir, 't10k-labels.idx1-ubyte')

    print(f"尝试加载MNIST文件:")
    print(f"  {train_images_path}")
    print(f"  {train_labels_path}")
    print(f"  {test_images_path}")
    print(f"  {test_labels_path}")

    # 检查所有文件是否存在
    for path in [train_images_path, train_labels_path, test_images_path, test_labels_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到MNIST文件: {path}")

    print("所有MNIST文件找到，开始加载...")

    # 加载数据
    train_data = read_images(train_images_path)
    train_labels = read_labels(train_labels_path)
    test_data = read_images(test_images_path)
    test_labels = read_labels(test_labels_path)

    # 数据预处理
    train_data = train_data.astype(np.float32) / 255.0
    test_data = test_data.astype(np.float32) / 255.0

    # 划分验证集 (从训练集中取5000个样本)
    val_data = train_data[55000:]
    val_labels = train_labels[55000:]
    train_data = train_data[:55000]
    train_labels = train_labels[:55000]

    print(f"数据加载完成:")
    print(f"  训练集: {train_data.shape[0]} 样本")
    print(f"  验证集: {val_data.shape[0]} 样本")
    print(f"  测试集: {test_data.shape[0]} 样本")

    return train_data, train_labels, val_data, val_labels, test_data, test_labels
def normalize_data(X, mean=None, std=None):
    """
    数据标准化

    Args:
        X: 输入数据
        mean: 均值 (如果为None则计算)
        std: 标准差 (如果为None则计算)

    Returns:
        X_normalized: 标准化后的数据
        mean: 使用的均值
        std: 使用的标准差
    """
    if mean is None:
        mean = np.mean(X, axis=0)
    if std is None:
        std = np.std(X, axis=0)
        std[std == 0] = 1  # 避免除零

    X_normalized = (X - mean) / std
    return X_normalized, mean, std