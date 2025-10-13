import numpy as np


def accuracy_score(y_true, y_pred):
    """
    计算准确率

    参数:
        y_true (numpy.ndarray): 真实标签
        y_pred (numpy.ndarray): 预测标签

    返回:
        float: 准确率
    """
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred, labels=None):
    """
    计算混淆矩阵

    参数:
        y_true (numpy.ndarray): 真实标签
        y_pred (numpy.ndarray): 预测标签
        labels (list): 标签列表

    返回:
        numpy.ndarray: 混淆矩阵
    """
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))

    n_classes = len(labels)
    cm = np.zeros((n_classes, n_classes), dtype=int)

    label_to_index = {label: idx for idx, label in enumerate(labels)}

    for true, pred in zip(y_true, y_pred):
        cm[label_to_index[true], label_to_index[pred]] += 1

    return cm


def classification_report(y_true, y_pred, labels=None):
    """
    生成分类报告

    参数:
        y_true (numpy.ndarray): 真实标签
        y_pred (numpy.ndarray): 预测标签
        labels (list): 标签列表

    返回:
        dict: 分类报告
    """
    cm = confusion_matrix(y_true, y_pred, labels)

    report = {}
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        report[label] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': cm[i, :].sum()
        }

    return report