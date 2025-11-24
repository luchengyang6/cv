import numpy as np


def compute_loss(predictions, labels):
    """计算交叉熵损失"""
    # 确保标签是整数索引
    if labels.dtype != np.int64:
        labels = labels.astype(np.int64)

    one_hot_labels = np.eye(10)[labels]
    epsilon = 1e-8
    return -np.mean(np.sum(one_hot_labels * np.log(predictions + epsilon), axis=1))


def compute_accuracy(predictions, labels):
    """计算准确率"""
    predicted_labels = np.argmax(predictions, axis=1)
    return np.mean(predicted_labels == labels)


def save_model(model, filepath):
    """保存模型"""
    params = model.get_parameters()
    np.savez(filepath, *[np.array([w, b]) for w, b in params])


def load_model(model, filepath):
    """加载模型"""
    data = np.load(filepath, allow_pickle=True)
    params = []
    for i in range(len(data.files)):
        weights, bias = data[f'arr_{i}']
        params.append((weights, bias))
    model.set_parameters(params)