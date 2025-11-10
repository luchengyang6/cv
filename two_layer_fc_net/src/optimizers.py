"""
优化器实现
"""

import numpy as np


class SGD:
    """
    随机梯度下降优化器
    """

    def __init__(self, lr=0.01):
        """
        初始化SGD优化器

        Args:
            lr: 学习率
        """
        self.lr = lr

    def update(self, params, grads):
        """
        更新参数

        Args:
            params: 参数字典
            grads: 梯度字典
        """
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class SGDMomentum:
    """
    带动量的随机梯度下降
    """

    def __init__(self, lr=0.01, momentum=0.9):
        """
        初始化带动量的SGD

        Args:
            lr: 学习率
            momentum: 动量系数
        """
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        """
        更新参数

        Args:
            params: 参数字典
            grads: 梯度字典
        """
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]