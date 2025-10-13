import os
from dotenv import load_dotenv
from data.load_custom_data import load_image_dataset
from src.knn import KNeighborsClassifier
from src.utils import train_test_split, normalize

# 加载配置
load_dotenv('.env.example')


def main():
    # 从环境变量读取配置
    data_path = os.getenv('DATA_PATH')
    img_size = tuple(map(int, os.getenv('IMAGE_SIZE').split(',')))
    test_size = float(os.getenv('TEST_SIZE'))
    random_state = int(os.getenv('RANDOM_STATE'))

    k_neighbors = int(os.getenv('K_NEIGHBORS'))
    distance_metric = os.getenv('DISTANCE_METRIC')
    weights = os.getenv('WEIGHTS')

    # 加载数据
    X, y, class_names = load_image_dataset(data_path, img_size=img_size)
    X_normalized = normalize(X)

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y, test_size=test_size, random_state=random_state
    )

    # 训练模型
    knn = KNeighborsClassifier(
        n_neighbors=k_neighbors,
        metric=distance_metric,
        weights=weights
    )
    knn.fit(X_train, y_train)

    # 评估
    accuracy = knn.score(X_test, y_test)
    print(f"配置化运行结果: 准确率 = {accuracy:.4f}")



if __name__ == "__main__":
    main()