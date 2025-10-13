import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.knn import KNeighborsClassifier
from src.utils import train_test_split, normalize
from src.metrics import accuracy_score, classification_report
from data.load_custom_data import load_image_dataset, load_feature_dataset


def main():
    print("自定义数据集K-NN分类演示")
    print("=" * 50)

    # 根据您的数据类型选择加载方式
    # 方式1: 图像数据集
    print("1. 加载图像数据集...")
    data_path = "path/to/your/image/dataset"  # 替换为您的数据集路径
    X, y, class_names = load_image_dataset(data_path, img_size=(28, 28))

    # 方式2: 特征数据集（取消注释使用）
    # print("1. 加载特征数据集...")
    # X, y = load_feature_dataset("features.npy", "labels.npy")

    print(f"数据集信息: 样本数={X.shape[0]}, 特征数={X.shape[1]}, 类别数={len(np.unique(y))}")
    print(f"类别分布: {dict(zip(*np.unique(y, return_counts=True)))}")

    # 数据预处理
    print("\n2. 数据预处理...")
    X_normalized = normalize(X, method='minmax')

    # 划分训练测试集
    print("\n3. 划分数据集...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y, test_size=0.3, random_state=42
    )
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

    # 训练模型
    print("\n4. 训练K-NN模型...")
    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean', weights='uniform')
    knn.fit(X_train, y_train)

    # 预测和评估
    print("\n5. 模型评估...")
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"模型准确率: {accuracy:.4f}")

    # 详细分类报告
    print("\n6. 详细分类报告...")
    report = classification_report(y_test, y_pred)
    for class_id, metrics in report.items():
        print(f"类别 {class_id}: 精确率={metrics['precision']:.3f}, "
              f"召回率={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f}")

    # 参数调优实验
    print("\n7. 参数调优实验...")
    k_values = [1, 3, 5, 7, 9]
    metrics_list = ['euclidean', 'manhattan']

    best_accuracy = 0
    best_params = {}

    for k in k_values:
        for metric in metrics_list:
            knn_temp = KNeighborsClassifier(n_neighbors=k, metric=metric)
            knn_temp.fit(X_train, y_train)
            acc = knn_temp.score(X_test, y_test)

            print(f"k={k}, metric={metric}: 准确率 = {acc:.4f}")

            if acc > best_accuracy:
                best_accuracy = acc
                best_params = {'k': k, 'metric': metric}

    print(f"\n最佳参数: {best_params}, 最佳准确率: {best_accuracy:.4f}")


if __name__ == "__main__":
    main()