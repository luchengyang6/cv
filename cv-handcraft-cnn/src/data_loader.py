import numpy as np
import urllib.request
import gzip
import os
from config.config import DATA_PATH


class MNISTLoader:
    def __init__(self):
        # 使用可靠的MNIST数据源
        self.mirrors = [
            'https://ossci-datasets.s3.amazonaws.com/mnist/',
            'https://storage.googleapis.com/cvdf-datasets/mnist/'
        ]

        self.files = {
            'train_images': 'train-images-idx3-ubyte.gz',
            'train_labels': 'train-labels-idx1-ubyte.gz',
            'test_images': 't10k-images-idx3-ubyte.gz',
            'test_labels': 't10k-labels-idx1-ubyte.gz'
        }

        # 创建数据目录
        if not os.path.exists(DATA_PATH):
            os.makedirs(DATA_PATH)

    def download_with_retry(self, filename, max_retries=3):
        """带重试的下载函数"""
        for mirror in self.mirrors:
            for attempt in range(max_retries):
                try:
                    filepath = os.path.join(DATA_PATH, filename)
                    url = mirror + filename

                    print(f'尝试从 {mirror} 下载 {filename} (尝试 {attempt + 1}/{max_retries})...')

                    # 下载文件
                    urllib.request.urlretrieve(url, filepath)
                    print(f'✅ 下载成功: {filename}')
                    return True

                except Exception as e:
                    print(f'❌ 下载失败: {e}')
                    if attempt < max_retries - 1:
                        print('等待1秒后重试...')
                        import time
                        time.sleep(1)
                    continue

        return False

    def download_data(self):
        """下载MNIST数据集"""
        print("开始下载MNIST数据集...")

        for key, filename in self.files.items():
            filepath = os.path.join(DATA_PATH, filename)

            # 如果文件已存在，跳过下载
            if os.path.exists(filepath):
                print(f'文件已存在: {filename}')
                continue

            if not self.download_with_retry(filename):
                raise Exception(f"无法下载 {filename}，请检查网络连接或手动下载数据")

    def load_images(self, filename):
        """加载图像数据"""
        filepath = os.path.join(DATA_PATH, filename)

        with gzip.open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)

        # 重塑为 (样本数, 高度, 宽度, 通道数) 并归一化
        images = data.reshape(-1, 28, 28, 1).astype(np.float32) / 255.0
        return images

    def load_labels(self, filename):
        """加载标签数据"""
        filepath = os.path.join(DATA_PATH, filename)

        with gzip.open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)

        return data

    def load_data(self):
        """加载完整数据集"""
        self.download_data()

        print("加载训练数据...")
        train_images = self.load_images(self.files['train_images'])
        train_labels = self.load_labels(self.files['train_labels'])

        print("加载测试数据...")
        test_images = self.load_images(self.files['test_images'])
        test_labels = self.load_labels(self.files['test_labels'])

        print(f"训练集: {train_images.shape[0]} 个样本")
        print(f"测试集: {test_images.shape[0]} 个样本")

        return (train_images, train_labels), (test_images, test_labels)

    def get_batches(self, data, batch_size=32):
        """生成批次数据 - 修复版本"""
        images, labels = data
        n_samples = images.shape[0]

        # 确保有数据
        if n_samples == 0:
            yield np.array([]), np.array([])
            return

        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]

            # 确保批次不为空
            if len(batch_indices) > 0:
                yield images[batch_indices], labels[batch_indices]