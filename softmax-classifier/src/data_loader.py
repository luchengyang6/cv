import numpy as np
import os
import gzip
import pickle
from urllib.request import urlretrieve


class DataLoader:
    def __init__(self, data_dir='./data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def load_mnist(self):
        """
        从多个镜像源下载MNIST数据集，确保可靠性
        """
        # 多个镜像源
        mirrors = [
            'http://yann.lecun.com/exdb/mnist/',
            'https://ossci-datasets.s3.amazonaws.com/mnist/',
            'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
        ]

        files = {
            'train_images': 'train-images-idx3-ubyte.gz',
            'train_labels': 'train-labels-idx1-ubyte.gz',
            'test_images': 't10k-images-idx3-ubyte.gz',
            'test_labels': 't10k-labels-idx1-ubyte.gz'
        }

        # 尝试从不同镜像下载
        for mirror in mirrors:
            try:
                print(f"尝试从 {mirror} 下载MNIST数据集...")
                success = True

                for filename in files.values():
                    filepath = os.path.join(self.data_dir, filename)
                    if not os.path.exists(filepath):
                        try:
                            urlretrieve(mirror + filename, filepath)
                            print(f"成功下载: {filename}")
                        except Exception as e:
                            print(f"下载 {filename} 失败: {e}")
                            success = False
                            break

                if success:
                    print("所有文件下载成功!")
                    break

            except Exception as e:
                print(f"镜像 {mirror} 失败: {e}")
                continue
        else:
            # 所有镜像都失败，使用虚拟数据
            print("所有镜像都失败，使用虚拟数据进行演示...")
            return self._create_dummy_data()

        # 解析MNIST文件
        try:
            return self._parse_mnist_files(files)
        except Exception as e:
            print(f"解析MNIST数据失败: {e}")
            return self._create_dummy_data()

    def _parse_mnist_files(self, files):
        """解析MNIST文件"""
        # 加载训练图像
        with gzip.open(os.path.join(self.data_dir, files['train_images']), 'rb') as f:
            X_train = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28 * 28)

        # 加载训练标签
        with gzip.open(os.path.join(self.data_dir, files['train_labels']), 'rb') as f:
            y_train = np.frombuffer(f.read(), np.uint8, offset=8)

        # 加载测试图像
        with gzip.open(os.path.join(self.data_dir, files['test_images']), 'rb') as f:
            X_test = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28 * 28)

        # 加载测试标签
        with gzip.open(os.path.join(self.data_dir, files['test_labels']), 'rb') as f:
            y_test = np.frombuffer(f.read(), np.uint8, offset=8)

        print(f"MNIST数据加载成功:")
        print(f"训练集: {X_train.shape} (样本数: {X_train.shape[0]}, 特征数: {X_train.shape[1]})")
        print(f"测试集: {X_test.shape} (样本数: {X_test.shape[0]}, 特征数: {X_test.shape[1]})")
        print(f"类别数: {len(np.unique(y_train))}")

        return X_train, y_train, X_test, y_test

    def _create_dummy_data(self):
        """创建更真实的虚拟数据"""
        print("创建更真实的虚拟MNIST数据...")
        np.random.seed(42)

        # 创建类似MNIST的数据分布
        n_train, n_test = 6000, 1000
        n_features = 784

        # 为每个类别创建不同的数据分布
        X_train = []
        y_train = []

        for class_id in range(10):
            # 每个类别有不同的均值和方差
            mean = np.random.randn(n_features) * 0.5 + class_id * 0.1
            cov = np.eye(n_features) * 0.3

            # 生成该类别的样本
            n_samples = n_train // 10
            class_samples = np.random.multivariate_normal(mean, cov, n_samples)
            class_samples = np.clip(class_samples, 0, 1)  # 限制在[0,1]范围

            X_train.append(class_samples)
            y_train.extend([class_id] * n_samples)

        X_train = np.vstack(X_train)
        y_train = np.array(y_train)

        # 同样方式创建测试集
        X_test = []
        y_test = []

        for class_id in range(10):
            mean = np.random.randn(n_features) * 0.5 + class_id * 0.1
            cov = np.eye(n_features) * 0.3

            n_samples = n_test // 10
            class_samples = np.random.multivariate_normal(mean, cov, n_samples)
            class_samples = np.clip(class_samples, 0, 1)

            X_test.append(class_samples)
            y_test.extend([class_id] * n_samples)

        X_test = np.vstack(X_test)
        y_test = np.array(y_test)

        # 打乱数据
        train_shuffle = np.random.permutation(len(X_train))
        test_shuffle = np.random.permutation(len(X_test))

        X_train, y_train = X_train[train_shuffle], y_train[train_shuffle]
        X_test, y_test = X_test[test_shuffle], y_test[test_shuffle]

        print(f"虚拟数据创建完成:")
        print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

        return X_train, y_train, X_test, y_test


def normalize_data(X):
    """数据归一化到[0,1]范围"""
    X = X.astype(np.float32)
    X_min = X.min()
    X_max = X.max()
    if X_max > X_min:
        X = (X - X_min) / (X_max - X_min)
    return X


def preprocess_data(X_train, X_test):
    """数据预处理"""
    # 归一化
    X_train_norm = normalize_data(X_train)
    X_test_norm = normalize_data(X_test)

    print(f"Data preprocessing completed:")
    print(f"Training set shape: {X_train_norm.shape}, range: [{X_train_norm.min():.3f}, {X_train_norm.max():.3f}]")
    print(f"Test set shape: {X_test_norm.shape}, range: [{X_test_norm.min():.3f}, {X_test_norm.max():.3f}]")

    return X_train_norm, X_test_norm