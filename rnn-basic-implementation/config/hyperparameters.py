# config/hyperparameters.py
"""
超参数配置
"""

# 模型超参数
MODEL_PARAMS = {
    'input_size': 1,
    'hidden_size': 128,
    'output_size': 1
}

# 训练超参数
TRAINING_PARAMS = {
    'learning_rate': 0.01,
    'batch_size': 32,
    'epochs': 1000,
    'seq_length': 50
}

# 数据超参数
DATA_PARAMS = {
    'frequency': 1.0,
    'amplitude': 1.0,
    'noise_level': 0.05,
    'total_samples': 2000,
    'train_ratio': 0.8
}

# 优化器配置
OPTIMIZER_PARAMS = {
    'type': 'sgd',  # 'sgd' 或 'adam'
    'sgd_lr': 0.01,
    'adam_lr': 0.001,
    'adam_beta1': 0.9,
    'adam_beta2': 0.999,
    'adam_epsilon': 1e-8
}