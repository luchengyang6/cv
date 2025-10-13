import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.knn import KNeighborsClassifier
from src.utils import train_test_split, normalize
from data.download_data import load_simple_dataset


def main():
    print("K-NN分类器演示")
    print("=" * 50)

    # 1. 加载数据
    print("1. 加载数据集...")
    X, y = load_simple_dataset()
    print(f"数据集形状: X={X.shape}, y={y.shape}")
    print(f"类别分布: {np.unique(y, return_counts=True)}")

    # 2. 数据预处理
    print("\n2. 数据预处理...")
    X_normalized = normalize(X, method='minmax')

    # 3. 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y, test_size=0.3, random_state=42
    )
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

    # 4. 训练模型
    print("\n3. 训练K-NN模型...")
    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean', weights='uniform')
    knn.fit(X_train, y_train)

    # 5. 预测和评估
    print("\n4. 模型评估...")
    accuracy = knn.score(X_test, y_test)
    print(f"模型准确率: {accuracy:.4f}")

    # 6. 不同参数比较
    print("\n5. 不同参数比较...")
    k_values = [1, 3, 5, 7, 9]
    for k in k_values:
        knn_temp = KNeighborsClassifier(n_neighbors=k)
        knn_temp.fit(X_train, y_train)
        acc = knn_temp.score(X_test, y_test)
        print(f"k={k}: 准确率 = {acc:.4f}")


if __name__ == "__main__":
    main()