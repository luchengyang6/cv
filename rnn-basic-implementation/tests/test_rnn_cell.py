import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.rnn_cell import RNNCell


def test_rnn_cell_initialization():
    """测试RNN单元初始化"""
    input_size = 10
    hidden_size = 20

    cell = RNNCell(input_size, hidden_size)

    # 检查参数形状
    assert cell.Wxh.shape == (
    hidden_size, input_size), f"Expected Wxh shape {(hidden_size, input_size)}, got {cell.Wxh.shape}"
    assert cell.Whh.shape == (
    hidden_size, hidden_size), f"Expected Whh shape {(hidden_size, hidden_size)}, got cell.Whh.shape"
    assert cell.bh.shape == (hidden_size, 1), f"Expected bh shape {(hidden_size, 1)}, got {cell.bh.shape}"
    assert cell.Why.shape == (1, hidden_size), f"Expected Why shape {(1, hidden_size)}, got {cell.Why.shape}"
    assert cell.by.shape == (1, 1), f"Expected by shape {(1, 1)}, got {cell.by.shape}"

    print("✓ RNN单元初始化测试通过")


def test_rnn_cell_forward():
    """测试RNN单元前向传播"""
    np.random.seed(42)

    input_size = 5
    hidden_size = 10

    cell = RNNCell(input_size, hidden_size)

    # 生成测试输入
    x = np.random.randn(input_size, 1)
    h_prev = np.random.randn(hidden_size, 1)

    # 前向传播
    h_next, y = cell.forward(x, h_prev)

    # 检查输出形状
    assert h_next.shape == (hidden_size, 1), f"Expected h_next shape {(hidden_size, 1)}, got {h_next.shape}"
    assert y.shape == (1, 1), f"Expected y shape {(1, 1)}, got {y.shape}"

    # 检查输出值范围（tanh输出应该在[-1, 1]之间）
    assert np.all(h_next >= -1) and np.all(h_next <= 1), "h_next should be in range [-1, 1]"

    print("✓ RNN单元前向传播测试通过")


def test_rnn_cell_backward():
    """测试RNN单元反向传播"""
    np.random.seed(42)

    input_size = 3
    hidden_size = 5

    cell = RNNCell(input_size, hidden_size)

    # 生成测试输入
    x = np.random.randn(input_size, 1)
    h_prev = np.random.randn(hidden_size, 1)

    # 前向传播以填充缓存
    h_next, y = cell.forward(x, h_prev)

    # 生成梯度
    dh_next = np.random.randn(hidden_size, 1)
    dy = np.random.randn(1, 1)

    # 反向传播
    dx, dh_prev, gradients = cell.backward(dh_next, dy)

    # 检查梯度形状
    assert dx.shape == x.shape, f"Expected dx shape {x.shape}, got {dx.shape}"
    assert dh_prev.shape == h_prev.shape, f"Expected dh_prev shape {h_prev.shape}, got {dh_prev.shape}"

    # 检查梯度字典
    expected_keys = {'Wxh', 'Whh', 'bh', 'Why', 'by'}
    assert set(
        gradients.keys()) == expected_keys, f"Expected gradients keys {expected_keys}, got {set(gradients.keys())}"

    print("✓ RNN单元反向传播测试通过")


def test_gradient_check():
    """梯度检查（数值梯度 vs 解析梯度）"""
    np.random.seed(42)

    input_size = 2
    hidden_size = 3

    cell = RNNCell(input_size, hidden_size)

    # 生成测试输入
    x = np.random.randn(input_size, 1)
    h_prev = np.random.randn(hidden_size, 1)

    # 前向传播
    h_next, y = cell.forward(x, h_prev)

    # 设置目标输出（用于损失计算）
    y_target = np.random.randn(1, 1)

    # 计算损失梯度
    dy = 2 * (y - y_target) / y.size

    # 解析梯度
    _, _, gradients_analytic = cell.backward(np.zeros_like(h_next), dy)

    # 数值梯度
    epsilon = 1e-7
    params = cell.get_parameters()

    for param_name in ['Wxh', 'Whh', 'Why']:
        param = params[param_name]
        grad_numeric = np.zeros_like(param)

        # 对每个参数计算数值梯度
        it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index

            # 保存原始值
            original = param[idx]

            # f(x + epsilon)
            param[idx] = original + epsilon
            cell.set_parameters(params)
            h_next_plus, y_plus = cell.forward(x, h_prev)
            loss_plus = np.sum((y_plus - y_target) ** 2)

            # f(x - epsilon)
            param[idx] = original - epsilon
            cell.set_parameters(params)
            h_next_minus, y_minus = cell.forward(x, h_prev)
            loss_minus = np.sum((y_minus - y_target) ** 2)

            # 数值梯度
            grad_numeric[idx] = (loss_plus - loss_minus) / (2 * epsilon)

            # 恢复原始值
            param[idx] = original
            it.iternext()

        # 恢复原始参数
        params[param_name] = param
        cell.set_parameters(params)

        # 计算相对误差
        grad_analytic = gradients_analytic[param_name]
        relative_error = np.abs(grad_numeric - grad_analytic) / (np.abs(grad_numeric) + np.abs(grad_analytic) + 1e-10)

        # 检查相对误差
        max_error = np.max(relative_error)
        print(f"  {param_name}最大相对误差: {max_error:.2e}")
        assert max_error < 1e-5, f"梯度检查失败: {param_name}的相对误差太大"

    print("✓ 梯度检查测试通过")


if __name__ == "__main__":
    print("开始运行RNN单元测试...")
    print("-" * 50)

    test_rnn_cell_initialization()
    test_rnn_cell_forward()
    test_rnn_cell_backward()
    test_gradient_check()

    print("-" * 50)
    print("所有测试通过! 🎉")