import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 添加src路径到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.append(src_path)

try:
    from data_loader import MNISTLoader
    from cnn_model import SimpleCNN
    from train import Trainer
    from visualize import plot_training_history, visualize_filters, plot_sample_predictions
    from utils import compute_accuracy
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保所有必要的模块都存在")
    sys.exit(1)


def sample_data(data, labels, sample_size=1000, random_state=42):
    """从数据中随机采样指定数量的样本"""
    np.random.seed(random_state)
    indices = np.random.permutation(len(data))
    sampled_indices = indices[:sample_size]
    return data[sampled_indices], labels[sampled_indices]


def check_data_files():
    """检查数据文件是否存在"""
    data_dir = './data'
    required_files = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz'
    ]

    if not os.path.exists(data_dir):
        return False, "data/ 目录不存在"

    existing_files = os.listdir(data_dir)
    missing_files = [f for f in required_files if f not in existing_files]

    if missing_files:
        return False, f"缺少文件: {', '.join(missing_files)}"

    return True, "所有数据文件都存在"


def main():
    """主函数 - 运行完整的CNN训练流程"""
    print("=" * 60)
    print("       手搓CNN项目 - MNIST手写数字识别 (稳定版)")
    print("=" * 60)

    # 预先检查数据文件
    print("\n🔍 检查数据文件...")
    data_ok, data_message = check_data_files()
    if not data_ok:
        print(f"❌ {data_message}")
        return

    print("✅ 数据文件检查通过")

    # 步骤1: 加载数据
    print("\n📊 步骤1: 加载MNIST数据集...")
    loader = MNISTLoader()
    try:
        (train_images, train_labels), (test_images, test_labels) = loader.load_data()
        print(f"✅ 数据加载成功!")
        print(f"   原始训练集: {train_images.shape[0]} 个样本")
        print(f"   原始测试集: {test_images.shape[0]} 个样本")

        # 使用更少的数据以确保稳定性
        train_sample_size = 1000  # 使用1000个训练样本
        test_sample_size = 200  # 使用200个测试样本

        train_images, train_labels = sample_data(train_images, train_labels, train_sample_size)
        test_images, test_labels = sample_data(test_images, test_labels, test_sample_size)

        print(f"   采样后训练集: {train_images.shape[0]} 个样本")
        print(f"   采样后测试集: {test_images.shape[0]} 个样本")

    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    # 显示数据样本
    print("\n🖼️  显示数据样本...")
    plt.figure(figsize=(12, 4))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(train_images[i, :, :, 0], cmap='gray')
        plt.title(f'标签: {train_labels[i]}')
        plt.axis('off')
    plt.suptitle("MNIST训练样本示例 (采样后)")
    plt.tight_layout()
    plt.show()

    # 步骤2: 创建模型
    print("\n🔧 步骤2: 创建CNN模型...")
    try:
        model = SimpleCNN()
        print("✅ CNN模型创建成功!")
        print("   模型结构: Conv → ReLU → Pool → Conv → ReLU → Pool → Flatten → Dense → Dense")

    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return

    # 步骤3: 训练模型
    print("\n🚀 步骤3: 开始训练模型...")
    print("   注意: 使用稳定参数，训练时间较短...")

    try:
        trainer = Trainer(model, learning_rate=0.001)

        # 训练模型 - 使用更保守的参数
        history = trainer.train(
            (train_images, train_labels),
            (test_images, test_labels),
            epochs=3,  # 使用3个epoch
            batch_size=16  # 使用更小的批量大小
        )
        print("✅ 模型训练完成!")
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 步骤4: 最终评估
    print("\n📈 步骤4: 模型评估...")
    try:
        test_outputs = model.forward(test_images, training=False)
        final_accuracy = compute_accuracy(test_outputs, test_labels)
        print(f"✅ 最终测试准确率: {final_accuracy:.4f} ({final_accuracy * 100:.2f}%)")
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        return

    # 步骤5: 生成可视化结果
    print("\n🎨 步骤5: 生成可视化图表...")
    try:
        # 训练历史
        fig1 = plot_training_history(history)
        plt.savefig('stable_training_history.png', dpi=300, bbox_inches='tight')
        print("   ✅ 训练历史图表已保存: stable_training_history.png")

        # 卷积核可视化
        fig2 = visualize_filters(model, layer_idx=0)
        plt.savefig('stable_conv_filters.png', dpi=300, bbox_inches='tight')
        print("   ✅ 卷积核可视化已保存: stable_conv_filters.png")

        # 样本预测
        fig3 = plot_sample_predictions(model, test_images, test_labels, num_samples=10)
        plt.savefig('stable_predictions.png', dpi=300, bbox_inches='tight')
        print("   ✅ 样本预测结果已保存: stable_predictions.png")

    except Exception as e:
        print(f"❌ 可视化生成失败: {e}")
        return

    # 步骤6: 显示结果
    print("\n📋 步骤6: 显示训练结果...")
    try:
        plt.show()
        print("✅ 所有图表已显示!")
    except Exception as e:
        print(f"❌ 显示图表时出错: {e}")

    print("\n" + "=" * 60)
    print("🎉 稳定训练完成!")
    print("   生成的文件:")
    print("   - stable_training_history.png: 训练损失和准确率曲线")
    print("   - stable_conv_filters.png: 第一层卷积核可视化")
    print("   - stable_predictions.png: 测试样本预测结果")
    print("=" * 60)


if __name__ == "__main__":
    main()