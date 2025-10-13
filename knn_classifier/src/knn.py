import numpy as np
from collections import Counter
from scipy.spatial.distance import cdist
from .metrics import accuracy_score


class KNeighborsClassifier:
    """
    K-近邻分类器实现

    参数:
        n_neighbors (int): 近邻数量，默认为5
        metric (str): 距离度量方法，'euclidean'或'manhattan'
        weights (str): 权重类型，'uniform'或'distance'
    """

    def __init__(self, n_neighbors=5, metric='euclidean', weights='uniform'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        训练K-NN分类器

        参数:
            X (numpy.ndarray): 训练数据，形状为(n_samples, n_features)
            y (numpy.ndarray): 训练标签，形状为(n_samples,)
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self

    def predict(self, X):
        """
        预测样本类别

        参数:
            X (numpy.ndarray): 测试数据，形状为(n_samples, n_features)

        返回:
            numpy.ndarray: 预测标签
        """
        if self.X_train is None:
            raise ValueError("模型尚未训练，请先调用fit方法")

        X = np.array(X)
        # 计算距离矩阵
        distances = self._compute_distances(X)

        # 获取最近的k个邻居
        indices = self._get_neighbor_indices(distances)

        # 根据邻居预测类别
        predictions = self._predict_from_neighbors(indices, distances)

        return predictions

    def _compute_distances(self, X):
        """计算测试样本与训练样本之间的距离"""
        if self.metric == 'euclidean':
            # 使用cdist计算欧氏距离
            distances = cdist(X, self.X_train, metric='euclidean')
        elif self.metric == 'manhattan':
            distances = cdist(X, self.X_train, metric='cityblock')
        else:
            raise ValueError(f"不支持的度量方法: {self.metric}")

        return distances

    def _get_neighbor_indices(self, distances):
        """获取最近的k个邻居的索引"""
        # 按距离排序，获取前k个邻居的索引
        indices = np.argpartition(distances, self.n_neighbors, axis=1)
        indices = indices[:, :self.n_neighbors]
        return indices

    def _predict_from_neighbors(self, indices, distances):
        """根据邻居预测类别"""
        predictions = []

        for i, neighbor_indices in enumerate(indices):
            neighbor_labels = self.y_train[neighbor_indices]
            neighbor_distances = distances[i, neighbor_indices]

            if self.weights == 'uniform':
                # 均匀权重：简单多数投票
                most_common = Counter(neighbor_labels).most_common(1)[0][0]
                predictions.append(most_common)
            elif self.weights == 'distance':
                # 距离权重：距离越近权重越大
                weighted_votes = self._weighted_vote(neighbor_labels, neighbor_distances)
                predictions.append(weighted_votes)
            else:
                raise ValueError(f"不支持的权重类型: {self.weights}")

        return np.array(predictions)

    def _weighted_vote(self, labels, distances):
        """基于距离的加权投票"""
        # 避免除零错误，给距离加上一个很小的值
        weights = 1 / (distances + 1e-8)

        # 计算每个类别的权重和
        weight_sum = {}
        for label, weight in zip(labels, weights):
            weight_sum[label] = weight_sum.get(label, 0) + weight

        # 返回权重和最大的类别
        return max(weight_sum.items(), key=lambda x: x[1])[0]

    def score(self, X, y):
        """计算模型在测试集上的准确率"""
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)