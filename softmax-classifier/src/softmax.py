import numpy as np


def softmax(x):
    """
    手动实现softmax函数
    参数: x - 输入矩阵 (n_samples, n_classes)
    返回: softmax概率
    """
    # 数值稳定性处理：减去最大值
    x_exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return x_exp / np.sum(x_exp, axis=1, keepdims=True)


class SoftmaxClassifier:
    def __init__(self, input_dim, num_classes, learning_rate=0.01, reg_strength=0.001):
        """
        初始化Softmax分类器

        参数:
        - input_dim: 输入特征维度
        - num_classes: 类别数量
        - learning_rate: 学习率
        - reg_strength: 正则化强度
        """
        # 初始化权重和偏置
        self.W = np.random.randn(input_dim, num_classes) * 0.01
        self.b = np.zeros(num_classes)
        self.learning_rate = learning_rate
        self.reg_strength = reg_strength
        self.loss_history = []

    def forward(self, X):
        """
        前向传播
        """
        # 线性变换: X * W + b
        scores = np.dot(X, self.W) + self.b
        # Softmax概率
        probs = softmax(scores)
        return probs, scores

    def compute_loss(self, X, y):
        """
        计算交叉熵损失和正则化损失
        """
        num_samples = X.shape[0]
        probs, _ = self.forward(X)

        # 交叉熵损失
        correct_log_probs = -np.log(probs[np.arange(num_samples), y] + 1e-8)  # 加小值避免log(0)
        data_loss = np.sum(correct_log_probs) / num_samples

        # L2正则化损失
        reg_loss = 0.5 * self.reg_strength * np.sum(self.W * self.W)

        total_loss = data_loss + reg_loss
        return total_loss

    def compute_gradients(self, X, y):
        """
        计算梯度
        """
        num_samples = X.shape[0]
        probs, _ = self.forward(X)

        # 计算概率梯度
        dscores = probs.copy()
        dscores[np.arange(num_samples), y] -= 1
        dscores /= num_samples

        # 计算权重和偏置的梯度
        dW = np.dot(X.T, dscores) + self.reg_strength * self.W
        db = np.sum(dscores, axis=0)

        return dW, db

    def train(self, X, y, num_epochs=1000, batch_size=64, verbose=True):
        """
        训练模型
        """
        num_samples = X.shape[0]

        for epoch in range(num_epochs):
            # 随机选择mini-batch
            indices = np.random.choice(num_samples, batch_size, replace=False)
            X_batch = X[indices]
            y_batch = y[indices]

            # 计算梯度和损失
            dW, db = self.compute_gradients(X_batch, y_batch)
            loss = self.compute_loss(X_batch, y_batch)

            # 更新参数
            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db

            self.loss_history.append(loss)

            if verbose and epoch % 100 == 0:
                accuracy = self.predict_accuracy(X_batch, y_batch)
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    def predict(self, X):
        """
        预测类别
        """
        probs, _ = self.forward(X)
        return np.argmax(probs, axis=1)

    def predict_accuracy(self, X, y):
        """
        计算准确率
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def save_model(self, filepath):
        """
        保存模型参数
        """
        model_params = {
            'W': self.W,
            'b': self.b,
            'learning_rate': self.learning_rate,
            'reg_strength': self.reg_strength
        }
        np.save(filepath, model_params)
        print(f"模型已保存到: {filepath}")

    def load_model(self, filepath):
        """
        加载模型参数
        """
        model_params = np.load(filepath, allow_pickle=True).item()
        self.W = model_params['W']
        self.b = model_params['b']
        self.learning_rate = model_params['learning_rate']
        self.reg_strength = model_params['reg_strength']
        print(f"模型已从 {filepath} 加载")