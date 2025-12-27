# scripts/train_stable.py
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from data.sine_wave_generator import SineWaveGenerator
from model.rnn_model import RNNModel


def train_rnn_stable():
    """更稳定的RNN训练函数"""

    # 更保守的超参数配置
    config = {
        'sequence_length': 30,  # 减少序列长度
        'hidden_size': 64,  # 减少隐藏层大小
        'batch_size': 16,  # 减少批次大小
        'learning_rate': 0.001,  # 使用更小的学习率
        'epochs': 500,  # 减少训练轮数
        'test_size': 0.2,
        'total_samples': 1000,
        'weight_scale': 0.01,  # 权重缩放因子
        'gradient_clip': 1.0  # 梯度裁剪阈值
    }

    print("=" * 50)
    print("开始稳定训练RNN模型")
    print(f"超参数配置: {config}")
    print("=" * 50)

    # 生成更简单的数据
    print("\n1. 生成数据集...")
    generator = SineWaveGenerator(frequency=0.5, amplitude=0.5, noise_level=0.02)
    X_train, y_train, X_test, y_test = generator.generate_train_test_split(
        total_samples=config['total_samples'],
        sequence_length=config['sequence_length'],
        test_size=config['test_size']
    )

    print(f"训练集形状: X={X_train.shape}, y={y_train.shape}")
    print(f"测试集形状: X={X_test.shape}, y={y_test.shape}")

    # 创建更稳定的模型
    print("\n2. 初始化更稳定的RNN模型...")
    input_size = X_train.shape[2]
    model = RNNModel(
        input_size=input_size,
        hidden_size=config['hidden_size'],
        weight_scale=config['weight_scale'],
        gradient_clip=config['gradient_clip']
    )

    # 训练循环
    print("\n3. 开始训练...")
    losses = []
    test_losses = []

    for epoch in range(config['epochs']):
        epoch_loss = 0
        n_batches = 0

        # 随机选择批次
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        for i in range(0, len(X_train), config['batch_size']):
            X_batch = X_train_shuffled[i:i + config['batch_size']]
            y_batch = y_train_shuffled[i:i + config['batch_size']]

            # 数值稳定性：进一步限制输入范围
            X_batch = np.clip(X_batch, -5, 5)
            y_batch = np.clip(y_batch, -5, 5)

            loss = model.train_step(X_batch, y_batch, config['learning_rate'])
            epoch_loss += loss
            n_batches += 1

        avg_loss = epoch_loss / n_batches if n_batches > 0 else epoch_loss
        losses.append(avg_loss)

        # 在测试集上评估
        if epoch % 50 == 0:
            test_pred = model.predict(X_test)
            test_loss = model.compute_loss(test_pred, y_test)
            test_losses.append(test_loss)

            print(f"Epoch {epoch}/{config['epochs']}, Train Loss: {avg_loss:.6f}, Test Loss: {test_loss:.6f}")

            # 动态调整学习率
            if epoch > 0 and losses[-1] > losses[-2] * 1.5:
                config['learning_rate'] *= 0.5
                print(f"  损失上升，降低学习率到: {config['learning_rate']}")

    print(f"\n训练完成! 最终损失: {losses[-1]:.6f}")

    # 测试模型
    print("\n4. 在测试集上评估...")
    y_pred = model.predict(X_test)

    # 计算简单的误差
    diff = y_pred - y_test
    mse = np.mean(diff ** 2)
    mae = np.mean(np.abs(diff))

    print(f"测试集 MSE: {mse:.6f}")
    print(f"测试集 MAE: {mae:.6f}")

    # 可视化结果
    print("\n5. 生成可视化结果...")
    plot_results_stable(model, X_test, y_test, y_pred, losses, test_losses)

    # 保存模型
    print("\n6. 保存模型...")
    os.makedirs('experiments/checkpoints', exist_ok=True)
    model.save('experiments/checkpoints/rnn_model_stable.npz')
    print("模型已保存到 experiments/checkpoints/rnn_model_stable.npz")

    return model, losses, test_losses


def plot_results_stable(model, X_test, y_test, y_pred, losses, test_losses):
    """绘制稳定的训练结果"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 训练损失曲线
    axes[0, 0].plot(losses)
    axes[0, 0].set_title('Training Loss (log scale)')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True)

    # 2. 梯度范数
    if model.gradient_norms:
        axes[0, 1].plot(model.gradient_norms)
        axes[0, 1].set_title('Gradient Norms')
        axes[0, 1].set_xlabel('Training Step')
        axes[0, 1].set_ylabel('Gradient Norm')
        axes[0, 1].grid(True)

    # 3. 预测示例
    sample_idx = 0
    axes[1, 0].plot(y_test[sample_idx, :, 0], label='True', alpha=0.7, linewidth=2)
    axes[1, 0].plot(y_pred[sample_idx, :, 0], label='Predicted', alpha=0.7, linestyle='--')
    axes[1, 0].set_title('Sample Prediction')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # 4. 训练和测试损失对比
    if test_losses:
        test_epochs = range(0, len(test_losses) * 50, 50)
        axes[1, 1].plot(test_epochs[:len(test_losses)], test_losses, label='Test Loss', marker='o')
        train_epochs = range(0, len(losses), len(losses) // len(test_losses))
        if len(train_epochs) > len(test_losses):
            train_epochs = train_epochs[:len(test_losses)]
        axes[1, 1].plot(train_epochs[:len(test_losses)],
                        losses[::len(losses) // len(test_losses)][:len(test_losses)],
                        label='Train Loss', marker='s')
        axes[1, 1].set_title('Train vs Test Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

    plt.tight_layout()
    os.makedirs('experiments/results', exist_ok=True)
    plt.savefig('experiments/results/training_results_stable.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    np.random.seed(42)

    try:
        model, losses, test_losses = train_rnn_stable()
        print("\n✅ 训练成功完成!")
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {e}")
        print("尝试更简单的配置...")