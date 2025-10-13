import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.knn import KNeighborsClassifier
from src.metrics import accuracy_score, confusion_matrix


def test_knn_basic():
    """测试K-NN基本功能"""
    # 创建简单数据集
    X_train = np.array([[1, 1], [1, 2], [2, 2], [5, 5], [5, 6], [6, 6]])
    y_train = np.array([0, 0, 0, 1, 1, 1])

    X_test = np.array([[1.5, 1.5], [5.5, 5.5]])
    y_test = np.array([0, 1])

    # 训练模型
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    # 预测
    predictions = knn.predict(X_test)

    # 验证
    assert accuracy_score(y_test, predictions) == 1.0
    print("基本功能测试通过!")


def test_different_metrics():
    """测试不同的距离度量"""
    X = np.random.rand(10, 2)
    y = np.random.randint(0, 2, 10)

    # 欧氏距离
    knn_euclidean = KNeighborsClassifier(metric='euclidean')
    knn_euclidean.fit(X, y)

    # 曼哈顿距离
    knn_manhattan = KNeighborsClassifier(metric='manhattan')
    knn_manhattan.fit(X, y)

    predictions1 = knn_euclidean.predict(X[:2])
    predictions2 = knn_manhattan.predict(X[:2])

    print("不同距离度量测试完成!")


if __name__ == "__main__":
    test_knn_basic()
    test_different_metrics()
    print("所有测试通过!")