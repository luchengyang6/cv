import numpy as np
from scipy import stats


def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    划分训练集和测试集

    参数:
        X (numpy.ndarray): 特征数据
        y (numpy.ndarray): 标签数据
        test_size (float): 测试集比例
        random_state (int): 随机种子

    返回:
        tuple: (X_train, X_test, y_train, y_test)
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(X)
    n_test = int(n_samples * test_size)

    # 随机打乱索引
    indices = np.random.permutation(n_samples)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test


def normalize(X, method='minmax'):
    """
    数据归一化

    参数:
        X (numpy.ndarray): 输入数据
        method (str): 归一化方法，'minmax'或'standard'

    返回:
        numpy.ndarray: 归一化后的数据
    """
    if method == 'minmax':
        # 最小-最大归一化到[0,1]
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        return (X - X_min) / (X_max - X_min + 1e-8)
    elif method == 'standard':
        # 标准化：均值0，方差1
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        return (X - X_mean) / (X_std + 1e-8)
    else:
        raise ValueError(f"不支持的归一化方法: {method}")


def cross_validate(model, X, y, cv=5):
    """
    交叉验证

    参数:
        model: 模型实例
        X: 特征数据
        y: 标签数据
        cv: 交叉验证折数

    返回:
        list: 每折的准确率
    """
    n_samples = len(X)
    fold_size = n_samples // cv
    accuracies = []

    for i in range(cv):
        start = i * fold_size
        end = (i + 1) * fold_size if i < cv - 1 else n_samples

        # 划分训练测试
        X_test_fold = X[start:end]
        y_test_fold = y[start:end]
        X_train_fold = np.vstack([X[:start], X[end:]])
        y_train_fold = np.hstack([y[:start], y[end:]])

        # 训练和评估
        model.fit(X_train_fold, y_train_fold)
        acc = model.score(X_test_fold, y_test_fold)
        accuracies.append(acc)

    return accuracies