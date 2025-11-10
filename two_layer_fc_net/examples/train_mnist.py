import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.two_layer_net import TwoLayerNet
from src.optimizers import SGDMomentum
from src.data_loader import load_mnist_local, normalize_data
from src.utils import create_mini_batches, accuracy


def train_mnist_relative():
    """使用相对路径训练两层全连接网络"""

    # 使用基于项目根目录的相对路径
    data_dir = os.path.join(project_root, 'data', 'mnist')

    print("=" * 50)
    print("两层全连接神经网络 - MNIST分类")
    print("=" * 50)

    print(f"项目根目录: {project_root}")
    print(f"数据目录: {data_dir}")

    # 从本地文件加载数据
    print(f"从路径加载MNIST数据集: {data_dir}")
    try:
        train_data, train_labels, val_data, val_labels, test_data, test_labels = load_mnist_local(data_dir)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("\n请确保:")
        print("1. MNIST文件位于项目根目录下的 data/mnist/ 文件夹中")
        print("2. 文件名如下:")
        print("   - train-images.idx3-ubyte")
        print("   - train-labels.idx1-ubyte")
        print("   - t10k-images.idx3-ubyte")
        print("   - t10k-labels.idx1-ubyte")
        print("\n当前工作目录:", os.getcwd())

        # 显示实际的文件检查
        print("\n文件检查:")
        expected_files = [
            'train-images.idx3-ubyte',
            'train-labels.idx1-ubyte',
            't10k-images.idx3-ubyte',
            't10k-labels.idx1-ubyte'
        ]
        for file in expected_files:
            file_path = os.path.join(data_dir, file)
            exists = os.path.exists(file_path)
            print(f"  {file}: {'存在' if exists else '不存在'}")

        return None, 0

    # 数据预处理
    print("\n数据预处理...")
    train_data, mean, std = normalize_data(train_data)
    val_data, _, _ = normalize_data(val_data, mean, std)
    test_data, _, _ = normalize_data(test_data, mean, std)

    # 超参数
    input_size = 28 * 28
    hidden_size = 128
    output_size = 10
    learning_rate = 1e-3
    reg_strength = 1e-4
    batch_size = 64
    num_epochs = 30

    print("\n网络参数:")
    print(f"  输入维度: {input_size}")
    print(f"  隐藏层大小: {hidden_size}")
    print(f"  输出维度: {output_size}")
    print(f"  学习率: {learning_rate}")
    print(f"  正则化强度: {reg_strength}")
    print(f"  批大小: {batch_size}")
    print(f"  训练轮数: {num_epochs}")

    # 初始化模型和优化器
    print("\n初始化模型...")
    model = TwoLayerNet(input_size, hidden_size, output_size, reg_strength)
    optimizer = SGDMomentum(lr=learning_rate, momentum=0.9)

    # 训练记录
    train_loss_history = []
    val_acc_history = []
    train_acc_history = []

    print("\n开始训练...")
    print("Epoch\tLoss\t\tTrain Acc\tVal Acc")
    print("-" * 50)

    for epoch in range(num_epochs):
        # 训练阶段
        epoch_loss = 0
        num_batches = 0

        for X_batch, y_batch in create_mini_batches(train_data, train_labels, batch_size):
            # 计算损失和梯度
            loss, grads = model.loss(X_batch, y_batch)
            epoch_loss += loss
            num_batches += 1

            # 更新参数
            params = model.get_params()
            optimizer.update(params, grads)
            model.set_params(params)

        # 计算准确率
        train_scores = model.forward(train_data[:1000])  # 使用部分数据计算训练准确率
        train_acc = accuracy(train_scores, train_labels[:1000])

        val_scores = model.forward(val_data)
        val_acc = accuracy(val_scores, val_labels)

        avg_loss = epoch_loss / num_batches
        train_loss_history.append(avg_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        if epoch % 5 == 0:
            print(f"{epoch}\t{avg_loss:.4f}\t\t{train_acc:.4f}\t\t{val_acc:.4f}")

    # 最终测试
    test_scores = model.forward(test_data)
    test_acc = accuracy(test_scores, test_labels)

    print("-" * 50)
    print(f"最终测试准确率: {test_acc:.4f}")

    # 绘制训练曲线
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(train_acc_history, label='Train Accuracy', linewidth=2)
    plt.plot(val_acc_history, label='Validation Accuracy', linewidth=2)
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # 确保输出目录存在
    output_dir = os.path.join(project_root, 'output')
    os.makedirs(output_dir, exist_ok=True)

    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    print(f"\n训练曲线已保存至: {os.path.join(output_dir, 'training_curves.png')}")
    plt.show()

    return model, test_acc


if __name__ == '__main__':
    # 直接使用相对路径运行，无需手动输入
    model, test_acc = train_mnist_relative()

    if model is not None:
        print("\n训练完成!")
        print(f"测试准确率: {test_acc:.4f}")

        # 保存模型
        from src.utils import save_model
        output_dir = os.path.join(project_root, 'output')
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, 'two_layer_net_model.pkl')
        save_model(model, model_path)
        print(f"模型已保存至: {model_path}")
    else:
        print("\n训练失败，请检查错误信息。")