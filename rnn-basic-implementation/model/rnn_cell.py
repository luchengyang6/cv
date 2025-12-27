# model/rnn_cell.py (修复版本)
import numpy as np
from typing import Tuple, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RNNCell:
    """
    基础的RNN单元实现
    增加了数值稳定性处理
    """

    def __init__(self, input_size: int, hidden_size: int,
                 weight_scale: float = 0.01):
        """
        初始化RNN单元

        参数:
            input_size: 输入特征维度
            hidden_size: 隐藏状态维度
            weight_scale: 权重缩放因子，用于防止梯度爆炸
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_scale = weight_scale

        # 使用更小的随机初始化防止梯度爆炸
        self.Wxh = np.random.randn(hidden_size, input_size) * weight_scale
        self.Whh = np.random.randn(hidden_size, hidden_size) * weight_scale
        self.bh = np.zeros((hidden_size, 1))

        self.Why = np.random.randn(1, hidden_size) * weight_scale
        self.by = np.zeros((1, 1))

        # 缓存用于反向传播
        self.cache = {}

    def forward(self, x: np.ndarray, h_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        单个时间步的前向传播
        添加了数值稳定性处理
        """
        # 数值稳定性：确保输入数据不是太大
        x = np.clip(x, -10, 10)
        h_prev = np.clip(h_prev, -10, 10)

        # 计算新的隐藏状态
        h_raw = np.dot(self.Wxh, x) + np.dot(self.Whh, h_prev) + self.bh

        # 数值稳定性：防止tanh输入太大
        h_raw = np.clip(h_raw, -50, 50)

        # 使用更稳定的tanh实现
        h_next = self._stable_tanh(h_raw)

        # 计算输出
        y = np.dot(self.Why, h_next) + self.by

        # 数值稳定性：限制输出范围
        y = np.clip(y, -10, 10)

        # 缓存用于反向传播
        self.cache['x'] = x.copy()
        self.cache['h_prev'] = h_prev.copy()
        self.cache['h_raw'] = h_raw.copy()
        self.cache['h_next'] = h_next.copy()

        return h_next, y

    def _stable_tanh(self, x: np.ndarray) -> np.ndarray:
        """更稳定的tanh实现，防止数值溢出"""
        # 对于大数值，tanh趋近于±1
        x_clipped = np.clip(x, -20, 20)
        return np.tanh(x_clipped)

    def backward(self, dh_next: np.ndarray, dy: np.ndarray) -> Tuple:
        """
        单个时间步的反向传播
        添加梯度裁剪防止梯度爆炸
        """
        # 从缓存中获取前向传播的值
        x = self.cache['x']
        h_prev = self.cache['h_prev']
        h_raw = self.cache['h_raw']
        h_next = self.cache['h_next']

        # 梯度裁剪：限制梯度大小
        dh_next = np.clip(dh_next, -1, 1)
        dy = np.clip(dy, -1, 1)

        # 计算tanh的导数
        dtanh = 1 - h_next ** 2

        # 计算隐藏状态的梯度
        dh = np.dot(self.Why.T, dy) + dh_next
        dh_raw = dh * dtanh

        # 防止梯度爆炸：裁剪中间梯度
        dh_raw = np.clip(dh_raw, -5, 5)

        # 计算参数梯度
        dWxh = np.dot(dh_raw, x.T)
        dWhh = np.dot(dh_raw, h_prev.T)
        dbh = dh_raw

        dWhy = np.dot(dy, h_next.T)
        dby = dy

        # 梯度裁剪：限制参数梯度大小
        dWxh = np.clip(dWxh, -1, 1)
        dWhh = np.clip(dWhh, -1, 1)
        dWhy = np.clip(dWhy, -1, 1)
        dbh = np.clip(dbh, -1, 1)
        dby = np.clip(dby, -1, 1)

        # 计算传递给前一个时间步和前一个层的梯度
        dx = np.dot(self.Wxh.T, dh_raw)
        dh_prev = np.dot(self.Whh.T, dh_raw)

        # 梯度裁剪：限制传播梯度大小
        dx = np.clip(dx, -1, 1)
        dh_prev = np.clip(dh_prev, -1, 1)

        gradients = {
            'Wxh': dWxh,
            'Whh': dWhh,
            'bh': dbh,
            'Why': dWhy,
            'by': dby
        }

        return dx, dh_prev, gradients

    def get_parameters(self) -> dict:
        """获取所有参数"""
        return {
            'Wxh': self.Wxh,
            'Whh': self.Whh,
            'bh': self.bh,
            'Why': self.Why,
            'by': self.by
        }

    def set_parameters(self, params: dict):
        """设置所有参数"""
        for key in params:
            if hasattr(self, key):
                setattr(self, key, params[key])