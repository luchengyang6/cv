import numpy as np
import os
from PIL import Image


def load_image_dataset(data_path, img_size=(28, 28)):
    """
    加载图像数据集

    参数:
        data_path (str): 数据集根目录
        img_size (tuple): 图像缩放尺寸

    返回:
        tuple: (特征矩阵, 标签向量)
    """
    X = []
    y = []
    class_names = []

    # 遍历类别文件夹
    for class_idx, class_name in enumerate(sorted(os.listdir(data_path))):
        class_path = os.path.join(data_path, class_name)
        if not os.path.isdir(class_path):
            continue

        class_names.append(class_name)

        # 遍历类别中的图像
        for img_file in os.listdir(class_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_path, img_file)

                # 加载并预处理图像
                img = Image.open(img_path).convert('L')  # 转为灰度
                img = img.resize(img_size)
                img_array = np.array(img).flatten()  # 展平为向量

                X.append(img_array)
                y.append(class_idx)

    return np.array(X), np.array(y), class_names


def load_feature_dataset(feature_file, label_file):
    """
    加载特征数据集

    参数:
        feature_file (str): 特征文件路径
        label_file (str): 标签文件路径

    返回:
        tuple: (特征矩阵, 标签向量)
    """
    X = np.load(feature_file) if feature_file.endswith('.npy') else np.genfromtxt(feature_file, delimiter=',')
    y = np.load(label_file) if label_file.endswith('.npy') else np.genfromtxt(label_file, delimiter=',')

    return X, y