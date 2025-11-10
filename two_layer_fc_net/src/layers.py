"""
两层全连接神经网络的核心层实现
使用纯Numpy实现，不依赖深度学习框架
"""

import numpy as np


class LinearLayer:
    """
    全连接层实现
    """

    def __init__(self, input_size, output_size, weight_scale=1e-3):
        """
        初始化全连接层

        Args:
            input_size: 输入维度
            output_size: 输出维度
            weight_scale: 权重初始化缩放因子
        """
        self.W = weight_scale * np.random.randn(input_size, output_size)
        self.b = np.zeros(output_size)
        self.x = None  # 前向传播的输入缓存
        self.dW = None  # 权重梯度
        self.db = None  # 偏置梯度

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入数据，形状 (N, input_size)

        Returns:
            out: 输出数据，形状 (N, output_size)
        """
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        """
        反向传播

        Args:
            dout: 上游梯度，形状 (N, output_size)

        Returns:
            dx: 对输入的梯度，形状 (N, input_size)
        """
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx

    def get_params(self):
        """返回参数"""
        return {'W': self.W, 'b': self.b}

    def set_params(self, params):
        """设置参数"""
        self.W = params['W']
        self.b = params['b']


class ReLU:
    """
    ReLU激活函数层
    """

    def __init__(self):
        self.mask = None

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入数据

        Returns:
            out: ReLU激活后的输出
        """
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        """
        反向传播

        Args:
            dout: 上游梯度

        Returns:
            dx: 对输入的梯度
        """
        dx = dout.copy()
        dx[self.mask] = 0
        return dx


class Softmax:
    """
    Softmax层
    """

    def __init__(self):
        self.out = None

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入数据，形状 (N, C)

        Returns:
            out: softmax输出，形状 (N, C)
        """
        # 数值稳定性处理：减去最大值
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_shifted)
        self.out = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.out

    def backward(self, dout):
        """
        反向传播

        Args:
            dout: 上游梯度，形状 (N, C)

        Returns:
            dx: 对输入的梯度，形状 (N, C)
        """
        dx = np.zeros_like(dout)

        for i in range(dout.shape[0]):
            # 对每个样本单独处理
            s = self.out[i].reshape(-1, 1)
            jacobian = np.diagflat(s) - np.dot(s, s.T)
            dx[i] = np.dot(jacobian, dout[i])

        return dx