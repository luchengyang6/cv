"""
网络测试
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.two_layer_net import TwoLayerNet
from src.utils import cross_entropy_loss


def test_network_initialization():
    """测试网络初始化"""
    net = TwoLayerNet(10, 5, 3)
    params = net.get_params()

    assert 'W1' in params
    assert 'b1' in params
    assert 'W2' in params
    assert 'b2' in params

    assert params['W1'].shape == (10, 5)
    assert params['b1'].shape == (5,)
    assert params['W2'].shape == (5, 3)
    assert params['b2'].shape == (3,)

    print("Network initialization test passed!")


def test_forward_pass():
    """测试前向传播"""
    net = TwoLayerNet(2, 3, 2)
    X = np.array([[1.0, 2.0], [3.0, 4.0]])

    scores = net.forward(X)

    assert scores.shape == (2, 2)
    assert np.all(np.isfinite(scores))

    print("Forward pass test passed!")


def test_backward_pass():
    """测试反向传播"""
    net = TwoLayerNet(2, 3, 2)
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0, 1])

    loss, grads = net.loss(X, y)

    assert isinstance(loss, float)
    assert np.isfinite(loss)

    assert 'W1' in grads
    assert 'b1' in grads
    assert 'W2' in grads
    assert 'b2' in grads

    # 检查梯度形状
    assert grads['W1'].shape == net.get_params()['W1'].shape
    assert grads['b1'].shape == net.get_params()['b1'].shape
    assert grads['W2'].shape == net.get_params()['W2'].shape
    assert grads['b2'].shape == net.get_params()['b2'].shape

    print("Backward pass test passed!")


def test_gradient_check():
    """梯度检查"""
    input_size, hidden_size, output_size = 3, 2, 2
    net = TwoLayerNet(input_size, hidden_size, output_size)

    X = np.random.randn(5, input_size)
    y = np.random.randint(0, output_size, 5)

    # 数值梯度计算
    def f(W):
        old_W = net.get_params()['W1'].copy()
        net.get_params()['W1'] = W
        loss, _ = net.loss(X, y)
        net.get_params()['W1'] = old_W
        return loss

    # 计算数值梯度
    params = net.get_params()
    analytic_grad = net.loss(X, y)[1]['W1']

    # 简单的梯度检查
    h = 1e-5
    numerical_grad = np.zeros_like(params['W1'])

    it = np.nditer(params['W1'], flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        old_val = params['W1'][idx]

        params['W1'][idx] = old_val + h
        fxh1 = f(params['W1'])

        params['W1'][idx] = old_val - h
        fxh2 = f(params['W1'])

        numerical_grad[idx] = (fxh1 - fxh2) / (2 * h)
        params['W1'][idx] = old_val

        it.iternext()

    # 比较数值梯度和解析梯度
    diff = np.linalg.norm(analytic_grad - numerical_grad) / np.linalg.norm(analytic_grad + numerical_grad)
    print(f"Gradient difference: {diff}")

    if diff < 1e-7:
        print("Gradient check passed!")
    else:
        print("Gradient check failed!")


if __name__ == '__main__':
    test_network_initialization()
    test_forward_pass()
    test_backward_pass()
    test_gradient_check()