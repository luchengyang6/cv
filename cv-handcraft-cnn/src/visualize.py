import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(history):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 损失曲线
    ax1.plot(history['train_losses'])
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)

    # 准确率曲线
    ax2.plot(history['train_accs'], label='Train Accuracy')
    ax2.plot(history['test_accs'], label='Test Accuracy')
    ax2.set_title('Training and Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    return fig


def visualize_filters(model, layer_idx=0):
    """可视化卷积核"""
    if hasattr(model.layers[layer_idx], 'weights'):
        weights = model.layers[layer_idx].weights
        n_filters = weights.shape[0]

        fig, axes = plt.subplots(1, n_filters, figsize=(15, 3))
        if n_filters == 1:
            axes = [axes]

        for i in range(n_filters):
            filter_img = weights[i, 0]  # 第一个输入通道
            axes[i].imshow(filter_img, cmap='viridis')
            axes[i].set_title(f'Filter {i + 1}')
            axes[i].axis('off')

        plt.tight_layout()
        return fig


def visualize_activations(model, sample_image, layer_names=None):
    """可视化各层激活"""
    if layer_names is None:
        layer_names = ['Conv1', 'ReLU1', 'Pool1', 'Conv2', 'ReLU2', 'Pool2']

    activations = []
    x = sample_image[np.newaxis, ...]  # 添加batch维度

    for i, layer in enumerate(model.layers[:6]):  # 只可视化前6层
        x = layer.forward(x)
        activations.append(x[0])  # 移除batch维度

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    for i, (act, name) in enumerate(zip(activations, layer_names)):
        if len(act.shape) == 3:  # 多通道特征图
            # 显示第一个通道
            axes[i].imshow(act[0], cmap='viridis')
        else:
            axes[i].imshow(act, cmap='viridis')
        axes[i].set_title(f'{name} Activation')
        axes[i].axis('off')

    plt.tight_layout()
    return fig


def plot_sample_predictions(model, test_images, test_labels, num_samples=10):
    """绘制样本预测结果"""
    predictions = model.forward(test_images[:num_samples], training=False)
    predicted_labels = np.argmax(predictions, axis=1)

    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()

    for i in range(num_samples):
        axes[i].imshow(test_images[i, :, :, 0], cmap='gray')
        true_label = test_labels[i]
        pred_label = predicted_labels[i]
        color = 'green' if true_label == pred_label else 'red'
        axes[i].set_title(f'True: {true_label}, Pred: {pred_label}', color=color)
        axes[i].axis('off')

    plt.tight_layout()
    return fig