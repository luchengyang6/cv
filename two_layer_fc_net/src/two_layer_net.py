"""
两层全连接神经网络实现
"""

import numpy as np
from .layers import LinearLayer, ReLU, Softmax
from .utils import cross_entropy_loss, l2_regularization, accuracy


class TwoLayerNet:
    """
    两层全连接神经网络
    结构: 输入层 -> 隐藏层(ReLU) -> 输出层(Softmax)
    """

    def __init__(self, input_size, hidden_size, output_size, reg_strength=0.0, weight_scale=1e-3):
        """
        初始化网络

        Args:
            input_size: 输入维度
            hidden_size: 隐藏层维度
            output_size: 输出维度
            reg_strength: 正则化强度
            weight_scale: 权重初始化缩放因子
        """
        self.reg_strength = reg_strength

        # 初始化网络层
        self.layers = {}
        self.layers['linear1'] = LinearLayer(input_size, hidden_size, weight_scale)
        self.layers['relu1'] = ReLU()
        self.layers['linear2'] = LinearLayer(hidden_size, output_size, weight_scale)
        self.layers['softmax'] = Softmax()

        # 层顺序
        self.layer_order = ['linear1', 'relu1', 'linear2', 'softmax']

    def forward(self, X):
        """
        前向传播

        Args:
            X: 输入数据，形状 (N, input_size)

        Returns:
            scores: 输出分数，形状 (N, output_size)
        """
        out = X
        for layer_name in self.layer_order:
            out = self.layers[layer_name].forward(out)
        return out

    def backward(self, dscores):
        """
        反向传播

        Args:
            dscores: 输出层梯度

        Returns:
            grads: 参数字典的梯度
        """
        # 反向传播经过所有层
        dout = dscores
        for layer_name in reversed(self.layer_order):
            dout = self.layers[layer_name].backward(dout)

        # 收集所有参数的梯度
        grads = {}
        grads['W1'] = self.layers['linear1'].dW
        grads['b1'] = self.layers['linear1'].db
        grads['W2'] = self.layers['linear2'].dW
        grads['b2'] = self.layers['linear2'].db

        return grads

    def loss(self, X, y):
        """
        计算损失

        Args:
            X: 输入数据
            y: 真实标签

        Returns:
            total_loss: 总损失
            grads: 梯度字典
        """
        # 前向传播
        scores = self.forward(X)

        # 计算数据损失
        data_loss, dscores = cross_entropy_loss(scores, y)

        # 计算正则化损失
        params = self.get_params()
        reg_loss, reg_grads = l2_regularization(params, self.reg_strength)

        # 总损失
        total_loss = data_loss + reg_loss

        # 反向传播
        grads = self.backward(dscores)

        # 添加正则化梯度
        for key in reg_grads:
            grads[key] += reg_grads[key]

        return total_loss, grads

    def predict(self, X):
        """
        预测

        Args:
            X: 输入数据

        Returns:
            y_pred: 预测类别
        """
        scores = self.forward(X)
        y_pred = np.argmax(scores, axis=1)
        return y_pred

    def get_params(self):
        """获取所有参数"""
        params = {}
        params['W1'] = self.layers['linear1'].W
        params['b1'] = self.layers['linear1'].b
        params['W2'] = self.layers['linear2'].W
        params['b2'] = self.layers['linear2'].b
        return params

    def set_params(self, params):
        """设置所有参数"""
        self.layers['linear1'].W = params['W1']
        self.layers['linear1'].b = params['b1']
        self.layers['linear2'].W = params['W2']
        self.layers['linear2'].b = params['b2']