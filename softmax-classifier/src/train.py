import numpy as np
import os
import sys

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoader, preprocess_data
from src.softmax import SoftmaxClassifier
from src.visualize import Visualizer


def main():
    print("=== Softmax Classifier Training Started ===")

    # Initialize components
    data_loader = DataLoader('./data')
    visualizer = Visualizer('./outputs/plots')

    # Load data
    print("\n1. Loading data...")
    X_train, y_train, X_test, y_test = data_loader.load_mnist()

    # Data preprocessing
    print("\n2. Data preprocessing...")
    X_train_processed, X_test_processed = preprocess_data(X_train, X_test)

    # Visualize original data
    print("\n3. Generating data visualizations...")
    visualizer.plot_feature_histograms(
        X_train_processed[:1000], y_train[:1000],  # Use subset to avoid memory issues
        save_path=os.path.join(visualizer.output_dir, 'feature_distribution.png')
    )

    # Create model
    input_dim = X_train_processed.shape[1]
    num_classes = len(np.unique(y_train))

    print(f"\n4. Creating model...")
    print(f"   Input dimension: {input_dim}")
    print(f"   Number of classes: {num_classes}")

    model = SoftmaxClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        learning_rate=0.1,
        reg_strength=0.001
    )

    # Train model
    print("\n5. Starting training...")
    num_epochs = 1000
    batch_size = 128

    print(f"   Number of epochs: {num_epochs}")
    print(f"   Batch size: {batch_size}")

    # Track accuracy during training
    train_acc_history = []
    test_acc_history = []

    for epoch in range(num_epochs):
        # Random mini-batch
        indices = np.random.choice(len(X_train_processed), batch_size, replace=False)
        X_batch = X_train_processed[indices]
        y_batch = y_train[indices]

        # Forward and backward pass
        dW, db = model.compute_gradients(X_batch, y_batch)
        model.W -= model.learning_rate * dW
        model.b -= model.learning_rate * db

        # Record metrics
        if epoch % 100 == 0:
            train_acc = model.predict_accuracy(X_train_processed, y_train)
            test_acc = model.predict_accuracy(X_test_processed, y_test)
            train_acc_history.append(train_acc)
            test_acc_history.append(test_acc)

            loss = model.compute_loss(X_batch, y_batch)
            print(f"Epoch {epoch}: Loss = {loss:.4f}, Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}")

    # Final evaluation
    print("\n6. Model evaluation...")
    final_train_acc = model.predict_accuracy(X_train_processed, y_train)
    final_test_acc = model.predict_accuracy(X_test_processed, y_test)

    print(f"   Final training accuracy: {final_train_acc:.4f}")
    print(f"   Final test accuracy: {final_test_acc:.4f}")

    # Generate predictions
    y_pred = model.predict(X_test_processed)

    # Generate visualizations
    print("\n7. Generating result visualizations...")

    # Loss curve
    visualizer.plot_loss_curve(
        model.loss_history,
        save_path=os.path.join(visualizer.output_dir, 'loss_curve.png')
    )

    # Training progress
    visualizer.plot_training_progress(
        model.loss_history, train_acc_history, test_acc_history,
        save_path=os.path.join(visualizer.output_dir, 'training_progress.png')
    )

    # Accuracy comparison
    visualizer.plot_accuracy_comparison(
        final_train_acc, final_test_acc,
        save_path=os.path.join(visualizer.output_dir, 'accuracy_comparison.png')
    )

    # Sample predictions
    visualizer.plot_sample_predictions(
        X_test_processed, y_test, y_pred, num_samples=12,
        save_path=os.path.join(visualizer.output_dir, 'sample_predictions.png')
    )

    # Confusion matrix
    cm = visualizer.plot_confusion_matrix_simple(
        y_test, y_pred,
        save_path=os.path.join(visualizer.output_dir, 'confusion_matrix.png')
    )

    # Save model
    print("\n8. Saving model...")
    os.makedirs('./outputs/models', exist_ok=True)
    model.save_model('./outputs/models/softmax_model.npy')

    print("\n=== Training Completed ===")
    print(f"Final test accuracy: {final_test_acc:.4f}")


if __name__ == "__main__":
    main()