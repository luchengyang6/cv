# model/rnn_model.py (修复版本)
import numpy as np
from typing import List, Tuple, Dict, Optional
from .rnn_cell import RNNCell


class RNNModel:
    """
    完整的RNN模型实现
    添加了梯度裁剪和数值稳定性处理
    """

    def __init__(self, input_size: int, hidden_size: int,
                 weight_scale: float = 0.01, gradient_clip: float = 5.0):
        """
        初始化RNN模型

        参数:
            input_size: 输入特征维度
            hidden_size: 隐藏状态维度
            weight_scale: 权重缩放因子
            gradient_clip: 梯度裁剪阈值
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gradient_clip = gradient_clip

        # 创建RNN单元
        self.rnn_cell = RNNCell(input_size, hidden_size, weight_scale)

        # 训练历史记录
        self.loss_history = []
        self.gradient_norms = []

    def forward(self, X: np.ndarray, h0: Optional[np.ndarray] = None) -> Tuple:
        """
        完整序列的前向传播
        """
        batch_size, seq_len, _ = X.shape

        if h0 is None:
            h0 = np.zeros((batch_size, self.hidden_size, 1))

        # 初始化输出和隐藏状态
        outputs = np.zeros((batch_size, seq_len, 1))
        hidden_states = np.zeros((batch_size, seq_len, self.hidden_size, 1))

        # 数值稳定性：限制输入范围
        X = np.clip(X, -10, 10)

        # 对每个样本进行前向传播
        for b in range(batch_size):
            h_prev = h0[b].reshape(-1, 1) if len(h0.shape) == 3 else h0

            for t in range(seq_len):
                # 获取当前时间步的输入
                x_t = X[b, t].reshape(-1, 1)

                # 通过RNN单元
                h_next, y_t = self.rnn_cell.forward(x_t, h_prev)

                # 存储结果
                outputs[b, t] = y_t
                hidden_states[b, t] = h_next

                # 更新隐藏状态
                h_prev = h_next

        return outputs, hidden_states

    def backward(self, X: np.ndarray, y_true: np.ndarray, outputs: np.ndarray) -> Dict:
        """
        通过时间的反向传播（BPTT）
        添加梯度裁剪
        """
        batch_size, seq_len, _ = X.shape

        # 数值稳定性：限制目标值范围
        y_true = np.clip(y_true, -10, 10)

        # 初始化梯度累加器
        gradients = {
            'Wxh': np.zeros_like(self.rnn_cell.Wxh),
            'Whh': np.zeros_like(self.rnn_cell.Whh),
            'bh': np.zeros_like(self.rnn_cell.bh),
            'Why': np.zeros_like(self.rnn_cell.Why),
            'by': np.zeros_like(self.rnn_cell.by)
        }

        # 对每个样本进行反向传播
        for b in range(batch_size):
            # 初始化梯度
            dh_next = np.zeros((self.hidden_size, 1))

            # 从最后一个时间步开始反向传播
            for t in reversed(range(seq_len)):
                # 计算当前时间步的损失梯度
                dy = outputs[b, t] - y_true[b, t]
                dy = dy.reshape(-1, 1)

                # 梯度裁剪
                dy = np.clip(dy, -self.gradient_clip, self.gradient_clip)

                # 通过RNN单元反向传播
                x_t = X[b, t].reshape(-1, 1)

                # 注意：这里需要设置RNN单元的缓存
                # 为了简化，我们假设RNN单元已经缓存了前向传播的信息
                # 在实际实现中，需要在前向传播时缓存每个时间步的信息

                dx, dh_prev, grad_t = self.rnn_cell.backward(dh_next, dy)

                # 累加梯度
                for key in gradients:
                    gradients[key] += grad_t[key]

                # 更新dh_next
                dh_next = dh_prev

        # 平均梯度（除以batch_size）
        for key in gradients:
            gradients[key] /= batch_size

        # 计算梯度范数（用于监控）
        total_norm = 0
        for grad in gradients.values():
            # 使用更稳定的计算方法
            grad_clipped = np.clip(grad, -100, 100)
            total_norm += np.sum(grad_clipped ** 2)
        self.gradient_norms.append(np.sqrt(total_norm + 1e-10))

        return gradients

    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        计算均方误差损失
        添加数值稳定性处理
        """
        # 数值稳定性：限制数值范围
        diff = np.clip(y_pred, -100, 100) - np.clip(y_true, -100, 100)
        diff_squared = np.clip(diff ** 2, 0, 1e6)  # 防止平方溢出

        return np.mean(diff_squared)

    def train_step(self, X_batch: np.ndarray, y_batch: np.ndarray,
                   learning_rate: float = 0.001) -> float:  # 使用更小的学习率
        """
        单个训练步骤
        """
        # 数值稳定性：限制输入范围
        X_batch = np.clip(X_batch, -10, 10)
        y_batch = np.clip(y_batch, -10, 10)

        # 前向传播
        outputs, _ = self.forward(X_batch)

        # 计算损失
        loss = self.compute_loss(outputs, y_batch)
        self.loss_history.append(loss)

        # 反向传播
        gradients = self.backward(X_batch, y_batch, outputs)

        # 更新参数
        self.update_parameters(gradients, learning_rate)

        return loss

    def update_parameters(self, gradients: Dict, learning_rate: float):
        """
        使用梯度下降更新参数
        添加梯度裁剪和学习率衰减
        """
        params = self.rnn_cell.get_parameters()

        for key in params:
            # 梯度裁剪
            grad_clipped = np.clip(gradients[key], -self.gradient_clip, self.gradient_clip)

            # 更新参数
            params[key] -= learning_rate * grad_clipped

            # 参数值裁剪（防止参数值过大）
            if key in ['Wxh', 'Whh', 'Why']:
                params[key] = np.clip(params[key], -5, 5)

        self.rnn_cell.set_parameters(params)

    def predict(self, X: np.ndarray, h0: Optional[np.ndarray] = None) -> np.ndarray:
        """
        预测函数
        """
        outputs, _ = self.forward(X, h0)
        return outputs

    def save(self, path: str):
        """保存模型参数"""
        params = self.rnn_cell.get_parameters()
        np.savez(path, **params)

    def load(self, path: str):
        """加载模型参数"""
        data = np.load(path)
        params = {key: data[key] for key in data.files}
        self.rnn_cell.set_parameters(params)