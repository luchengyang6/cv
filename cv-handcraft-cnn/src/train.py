import numpy as np
from src.cnn_model import SimpleCNN
from src.data_loader import MNISTLoader
from src.utils import compute_loss, compute_accuracy
import time


class Trainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.learning_rate = learning_rate

    def train_epoch(self, train_loader, epoch):
        """训练一个epoch"""
        total_loss = 0
        total_acc = 0
        num_batches = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            # 确保图像形状正确 (batch_size, 28, 28, 1)
            if images.ndim == 3:
                images = images.reshape(-1, 28, 28, 1)

            # 跳过空的批次
            if len(images) == 0:
                continue

            # 前向传播
            outputs = self.model.forward(images, training=False)  # 关闭训练时的调试输出
            loss = compute_loss(outputs, labels)
            acc = compute_accuracy(outputs, labels)

            # 反向传播
            one_hot_labels = np.eye(10)[labels]
            dout = (outputs - one_hot_labels) / images.shape[0]
            grads = self.model.backward(dout)

            # 参数更新
            self.update_parameters(grads)

            total_loss += loss
            total_acc += acc
            num_batches += 1

            if batch_idx % 50 == 0:  # 减少打印频率
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss:.4f}, Acc: {acc:.4f}')

        # 防止除零错误
        if num_batches == 0:
            print("警告: 没有处理任何批次数据")
            return 0, 0

        return total_loss / num_batches, total_acc / num_batches

    def update_parameters(self, grads):
        """更新参数"""
        params = self.model.get_parameters()

        for i, ((dw, db), (weights, bias)) in enumerate(zip(grads, params)):
            # 添加梯度裁剪，防止梯度爆炸
            if dw is not None:
                dw = np.clip(dw, -1, 1)
                weights -= self.learning_rate * dw

            if db is not None:
                db = np.clip(db, -1, 1)
                bias -= self.learning_rate * db

    def train(self, train_data, test_data, epochs=10, batch_size=32):
        """完整训练流程"""
        train_images, train_labels = train_data
        test_images, test_labels = test_data

        # 确保测试图像形状正确
        if test_images.ndim == 3:
            test_images = test_images.reshape(-1, 28, 28, 1)

        train_losses = []
        train_accs = []
        test_accs = []

        for epoch in range(epochs):
            start_time = time.time()

            # 每个epoch重新创建数据加载器
            train_loader = MNISTLoader().get_batches((train_images, train_labels), batch_size)

            # 训练
            train_loss, train_acc = self.train_epoch(train_loader, epoch)

            # 测试
            test_outputs = self.model.forward(test_images, training=False)
            test_acc = compute_accuracy(test_outputs, test_labels)

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)

            epoch_time = time.time() - start_time

            print(f'Epoch {epoch + 1}/{epochs}, Time: {epoch_time:.2f}s')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
            print('-' * 50)

        return {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'test_accs': test_accs
        }