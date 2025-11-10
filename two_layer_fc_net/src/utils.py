"""
工具函数
"""

import numpy as np
import pickle
import os


def cross_entropy_loss(scores, y):
    """
    计算交叉熵损失

    Args:
        scores: 模型输出分数，形状 (N, C)
        y: 真实标签，形状 (N,)

    Returns:
        loss: 交叉熵损失
        dscores: 对scores的梯度
    """
    N = scores.shape[0]

    # 数值稳定性处理
    scores_shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores_shifted)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # 计算损失
    correct_logprobs = -np.log(probs[np.arange(N), y])
    loss = np.sum(correct_logprobs) / N

    # 计算梯度
    dscores = probs.copy()
    dscores[np.arange(N), y] -= 1
    dscores /= N

    return loss, dscores


def l2_regularization(params, reg_strength):
    """
    计算L2正则化损失和梯度

    Args:
        params: 参数字典
        reg_strength: 正则化强度

    Returns:
        reg_loss: 正则化损失
        reg_grads: 正则化梯度
    """
    reg_loss = 0
    reg_grads = {}

    for key, param in params.items():
        if 'W' in key:
            reg_loss += 0.5 * reg_strength * np.sum(param * param)
            reg_grads[key] = reg_strength * param

    return reg_loss, reg_grads


def accuracy(scores, y):
    """
    计算准确率

    Args:
        scores: 模型输出分数
        y: 真实标签

    Returns:
        acc: 准确率
    """
    predicted_class = np.argmax(scores, axis=1)
    acc = np.mean(predicted_class == y)
    return acc


def save_model(model, filepath):
    """
    保存模型

    Args:
        model: 模型实例
        filepath: 保存路径
    """
    with open(filepath, 'wb') as f:
        pickle.dump(model.get_params(), f)


def load_model(model, filepath):
    """
    加载模型

    Args:
        model: 模型实例
        filepath: 加载路径
    """
    with open(filepath, 'rb') as f:
        params = pickle.load(f)
    model.set_params(params)


def create_mini_batches(X, y, batch_size, shuffle=True):
    """
    创建mini-batch

    Args:
        X: 输入数据
        y: 标签
        batch_size: batch大小
        shuffle: 是否打乱数据

    Returns:
        batches: 批数据生成器
    """
    N = X.shape[0]

    if shuffle:
        indices = np.random.permutation(N)
        X = X[indices]
        y = y[indices]

    for start_idx in range(0, N, batch_size):
        end_idx = min(start_idx + batch_size, N)
        yield X[start_idx:end_idx], y[start_idx:end_idx]