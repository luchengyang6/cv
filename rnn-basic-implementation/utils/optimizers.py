import numpy as np
from typing import Dict, Any


class SGD:
    """
    随机梯度下降优化器
    """

    def __init__(self, learning_rate: float = 0.01):
        """
        初始化SGD优化器

        参数:
            learning_rate: 学习率
        """
        self.learning_rate = learning_rate

    def update(self, parameters: Dict, gradients: Dict) -> Dict:
        """
        更新参数

        参数:
            parameters: 模型参数
            gradients: 参数梯度

        返回:
            更新后的参数
        """
        updated_params = {}

        for key in parameters:
            updated_params[key] = parameters[key] - self.learning_rate * gradients[key]

        return updated_params


class Adam:
    """
    Adam优化器（可选实现）
    """

    def __init__(self, learning_rate: float = 0.001,
                 beta1: float = 0.9, beta2: float = 0.999,
                 epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, parameters: Dict, gradients: Dict) -> Dict:
        """Adam参数更新"""
        if self.m is None:
            self.m = {k: np.zeros_like(v) for k, v in parameters.items()}
            self.v = {k: np.zeros_like(v) for k, v in parameters.items()}

        self.t += 1
        updated_params = {}

        for key in parameters:
            # 更新一阶矩估计
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * gradients[key]

            # 更新二阶矩估计
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (gradients[key] ** 2)

            # 偏差校正
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            # 更新参数
            updated_params[key] = parameters[key] - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return updated_params