"""
超参数配置
"""

# 网络架构参数
INPUT_SIZE = 28 * 28  # MNIST图像大小
HIDDEN_SIZE = 128     # 隐藏层神经元数量
OUTPUT_SIZE = 10      # 类别数量 (0-9)

# 训练参数
LEARNING_RATE = 1e-3
REG_STRENGTH = 1e-4
BATCH_SIZE = 64
NUM_EPOCHS = 50

# 优化器参数
MOMENTUM = 0.9
WEIGHT_SCALE = 1e-3

# 数据参数
VAL_SIZE = 5000  # 验证集大小