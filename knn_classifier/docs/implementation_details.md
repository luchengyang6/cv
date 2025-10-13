# 📄 implementation_details.md

# K-近邻算法实现细节

## 🧮 数学原理详解

### 距离度量

#### 欧氏距离 (Euclidean Distance)
```math
d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
```
**实现优化**：
```python
# 使用scipy的cdist进行向量化计算，避免手动循环
distances = cdist(X, self.X_train, metric='euclidean')
```

#### 曼哈顿距离 (Manhattan Distance)
```math
d(x, y) = \sum_{i=1}^{n} |x_i - y_i|
```
**实现特点**：
```python
# 对高维稀疏数据更有效
distances = cdist(X, self.X_train, metric='cityblock')
```

### 投票机制

#### 均匀投票 (Uniform Voting)
```math
\hat{y} = \arg\max_{c} \sum_{i=1}^{k} \mathbb{I}(y_i = c)
```

#### 距离加权投票 (Distance-Weighted Voting)
```math
\hat{y} = \arg\max_{c} \sum_{i=1}^{k} \frac{1}{d(x, x_i) + \epsilon} \cdot \mathbb{I}(y_i = c)
```
其中 $\epsilon = 10^{-8}$ 用于避免除零错误。

## 🔧 核心代码解析

### KNeighborsClassifier 类设计

```python
class KNeighborsClassifier:
    def __init__(self, n_neighbors=5, metric='euclidean', weights='uniform'):
        # 参数验证和初始化
        self._validate_parameters(n_neighbors, metric, weights)
        
    def fit(self, X, y):
        """存储训练数据，K-NN是惰性学习算法"""
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self
        
    def predict(self, X):
        # 1. 计算距离矩阵
        distances = self._compute_distances(X)
        # 2. 获取最近邻居索引
        indices = self._get_neighbor_indices(distances)
        # 3. 基于邻居进行预测
        predictions = self._predict_from_neighbors(indices, distances)
        return predictions
```

### 关键算法实现

#### 距离计算优化
```python
def _compute_distances(self, X):
    """使用scipy的cdist进行高效的距离计算"""
    if self.metric == 'euclidean':
        # 欧氏距离的向量化实现，比手动循环快100倍以上
        return cdist(X, self.X_train, metric='euclidean')
    elif self.metric == 'manhattan':
        return cdist(X, self.X_train, metric='cityblock')
```

#### 邻居选择策略
```python
def _get_neighbor_indices(self, distances):
    """使用argpartition进行部分排序，时间复杂度O(n)"""
    # 只对前k个最小距离进行排序，而不是全部排序
    indices = np.argpartition(distances, self.n_neighbors, axis=1)
    return indices[:, :self.n_neighbors]
```

#### 加权投票实现
```python
def _weighted_vote(self, labels, distances):
    """距离加权投票，处理边界情况"""
    # 避免除零错误
    weights = 1 / (distances + 1e-8)
    
    # 使用字典累加权重，时间复杂度O(k)
    weight_sum = {}
    for label, weight in zip(labels, weights):
        weight_sum[label] = weight_sum.get(label, 0) + weight
    
    # 返回权重最大的类别
    return max(weight_sum.items(), key=lambda x: x[1])[0]
```

## ⚡ 性能优化技巧

### 1. 向量化运算
- 使用 `cdist` 代替手动循环计算距离
- 利用 `argpartition` 进行高效的部分排序
- 使用布尔索引和数组操作代替Python循环

### 2. 内存优化
```python
# 分批处理大数据集
def predict_batch(self, X, batch_size=1000):
    predictions = []
    for i in range(0, len(X), batch_size):
        batch = X[i:i+batch_size]
        pred_batch = self.predict(batch)
        predictions.extend(pred_batch)
    return np.array(predictions)
```

### 3. 缓存优化
```python
# 缓存距离计算中的中间结果
def _compute_squared_distances(self, X):
    """计算平方距离，避免开方运算"""
    if not hasattr(self, '_X_train_squared'):
        self._X_train_squared = np.sum(self.X_train**2, axis=1)
    
    X_squared = np.sum(X**2, axis=1, keepdims=True)
    distances_squared = X_squared + self._X_train_squared - 2 * X.dot(self.X_train.T)
    return np.maximum(distances_squared, 0)  # 避免数值误差导致的负数
```

## 🎯 算法复杂度分析

### 时间复杂度
- **训练阶段**: O(1) - 只是存储数据
- **预测阶段**: O(nd + n log k) 
  - n: 测试样本数
  - d: 特征维度
  - k: 近邻数

### 空间复杂度
- **训练数据存储**: O(nd)
- **距离矩阵**: O(nm) - m为训练样本数

## 🔍 边界情况处理

### 1. 距离为零的情况
```python
# 在加权投票中处理距离为零的邻居
weights = 1 / (distances + 1e-8)  # 添加小常数避免除零
```

### 2. 平票处理
```python
def _break_tie(self, weight_sum):
    """在平票时选择距离更近的类别"""
    max_weight = max(weight_sum.values())
    candidates = [label for label, weight in weight_sum.items() 
                 if weight == max_weight]
    # 选择第一个出现的类别（或可以随机选择）
    return candidates[0]
```

### 3. 空数据集验证
```python
def fit(self, X, y):
    if len(X) == 0:
        raise ValueError("训练数据不能为空")
    if len(X) != len(y):
        raise ValueError("特征和标签数量不匹配")
    self.X_train = np.array(X)
    self.y_train = np.array(y)
    return self
```

## 📈 扩展功能

### 概率预测
```python
def predict_proba(self, X):
    """返回每个类别的预测概率"""
    distances = self._compute_distances(X)
    indices = self._get_neighbor_indices(distances)
    
    probas = []
    for i, neighbor_indices in enumerate(indices):
        neighbor_labels = self.y_train[neighbor_indices]
        neighbor_distances = distances[i, neighbor_indices]
        
        if self.weights == 'uniform':
            # 计算每个类别的比例
            class_counts = np.bincount(neighbor_labels)
            proba = class_counts / len(neighbor_labels)
        else:
            # 基于距离加权的概率
            weights = 1 / (neighbor_distances + 1e-8)
            class_weights = np.bincount(neighbor_labels, weights=weights)
            proba = class_weights / np.sum(weights)
        
        probas.append(proba)
    
    return np.array(probas)
```