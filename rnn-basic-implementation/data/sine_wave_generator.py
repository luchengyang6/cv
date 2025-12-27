import numpy as np
from typing import Tuple, Optional


class SineWaveGenerator:
    """
    正弦波时间序列生成器
    用于生成训练RNN的时间序列数据
    """

    def __init__(self,
                 frequency: float = 1.0,
                 amplitude: float = 1.0,
                 noise_level: float = 0.05,
                 random_seed: Optional[int] = None):
        """
        初始化正弦波生成器

        参数:
            frequency: 正弦波频率
            amplitude: 正弦波振幅
            noise_level: 噪声水平
            random_seed: 随机种子
        """
        self.frequency = frequency
        self.amplitude = amplitude
        self.noise_level = noise_level

        if random_seed is not None:
            np.random.seed(random_seed)

    def generate_sequence(self,
                          n_samples: int = 1000,
                          sequence_length: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成正弦波时间序列

        参数:
            n_samples: 样本数量
            sequence_length: 序列长度

        返回:
            X: 输入序列 (n_samples, sequence_length, 1)
            y: 目标值 (n_samples, sequence_length, 1)
        """
        # 生成时间点
        t = np.linspace(0, 4 * np.pi, n_samples + sequence_length)

        # 生成正弦波
        sine_wave = self.amplitude * np.sin(self.frequency * t)

        # 添加噪声
        noise = self.noise_level * np.random.randn(*sine_wave.shape)
        noisy_sine_wave = sine_wave + noise

        # 创建序列数据
        X, y = [], []

        for i in range(n_samples):
            # 输入序列
            input_seq = noisy_sine_wave[i:i + sequence_length].reshape(-1, 1)
            # 目标序列（下一个时间步）
            target_seq = noisy_sine_wave[i + 1:i + sequence_length + 1].reshape(-1, 1)

            X.append(input_seq)
            y.append(target_seq)

        return np.array(X), np.array(y)

    def generate_train_test_split(self,
                                  total_samples: int = 2000,
                                  sequence_length: int = 50,
                                  test_size: float = 0.2) -> Tuple:
        """
        生成训练集和测试集

        参数:
            total_samples: 总样本数
            sequence_length: 序列长度
            test_size: 测试集比例

        返回:
            (X_train, y_train, X_test, y_test)
        """
        X, y = self.generate_sequence(total_samples, sequence_length)

        # 分割数据集
        split_idx = int(total_samples * (1 - test_size))

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    # 示例用法
    generator = SineWaveGenerator(frequency=1.0, amplitude=1.0, noise_level=0.05)
    X, y = generator.generate_sequence(n_samples=100, sequence_length=20)

    print(f"输入数据形状: {X.shape}")
    print(f"目标数据形状: {y.shape}")
    print(f"第一个样本的前5个时间步: {X[0, :5, 0]}")