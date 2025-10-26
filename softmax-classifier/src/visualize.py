import numpy as np
import matplotlib.pyplot as plt
import os


class Visualizer:
    def __init__(self, output_dir='./outputs/plots'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 设置matplotlib参数，避免中文问题
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 使用支持的字体
        plt.rcParams['axes.unicode_minus'] = False

    def plot_loss_curve(self, loss_history, save_path=None):
        """Plot training loss curve"""
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history, 'b-', linewidth=2, label='Training Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Loss curve saved: {save_path}")
        plt.show()

    def plot_accuracy_comparison(self, train_acc, test_acc, save_path=None):
        """Plot training and test accuracy comparison"""
        plt.figure(figsize=(8, 6))
        categories = ['Training Set', 'Test Set']
        accuracies = [train_acc, test_acc]

        bars = plt.bar(categories, accuracies, color=['blue', 'orange'], alpha=0.7)
        plt.ylabel('Accuracy')
        plt.title('Training vs Test Accuracy')
        plt.ylim(0, 1)

        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{acc:.4f}', ha='center', va='bottom')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Accuracy comparison saved: {save_path}")
        plt.show()

    def plot_sample_predictions(self, X, y_true, y_pred, num_samples=12, save_path=None):
        """Plot sample predictions (for MNIST images)"""
        # Randomly select samples
        indices = np.random.choice(len(X), min(num_samples, len(X)), replace=False)

        n_cols = 4
        n_rows = (num_samples + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.ravel()

        for i, idx in enumerate(indices):
            if i < len(axes):
                # Reshape to 28x28 image
                img = X[idx].reshape(28, 28)
                axes[i].imshow(img, cmap='gray')

                # Set title color: green for correct, red for wrong
                color = 'green' if y_true[idx] == y_pred[idx] else 'red'
                axes[i].set_title(f'True: {y_true[idx]}, Pred: {y_pred[idx]}', color=color)
                axes[i].axis('off')

        # Hide extra subplots
        for i in range(len(indices), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Sample predictions saved: {save_path}")
        plt.show()

    def plot_confusion_matrix_simple(self, y_true, y_pred, save_path=None):
        """Simple confusion matrix visualization"""
        num_classes = len(np.unique(y_true))
        cm = np.zeros((num_classes, num_classes), dtype=int)

        # Calculate confusion matrix
        for true, pred in zip(y_true, y_pred):
            cm[true, pred] += 1

        plt.figure(figsize=(10, 8))
        plt.imshow(cm, cmap='Blues', interpolation='nearest')
        plt.colorbar()
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')

        # Add value annotations
        for i in range(num_classes):
            for j in range(num_classes):
                plt.text(j, i, str(cm[i, j]),
                         ha='center', va='center',
                         color='red' if cm[i, j] > cm.max() / 2 else 'black')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved: {save_path}")
        plt.show()

        return cm

    def plot_feature_histograms(self, X, y, save_path=None):
        """Plot feature histograms (first 4 features)"""
        n_features = min(4, X.shape[1])
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()

        unique_classes = np.unique(y)

        for i in range(n_features):
            for cls in unique_classes:
                axes[i].hist(X[y == cls, i], alpha=0.7, label=f'Class {cls}', bins=20)
            axes[i].set_xlabel(f'Feature {i + 1}')
            axes[i].set_ylabel('Frequency')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        # Hide extra subplots
        for i in range(n_features, 4):
            fig.delaxes(axes[i])

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature distribution saved: {save_path}")
        plt.show()

    def plot_training_progress(self, loss_history, train_acc_history, test_acc_history, save_path=None):
        """Plot comprehensive training progress"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot loss
        ax1.plot(loss_history, 'b-', linewidth=2, label='Training Loss')
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot accuracy
        ax2.plot(train_acc_history, 'g-', linewidth=2, label='Training Accuracy')
        ax2.plot(test_acc_history, 'r-', linewidth=2, label='Test Accuracy')
        ax2.set_xlabel('Iterations')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training Progress')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training progress saved: {save_path}")
        plt.show()