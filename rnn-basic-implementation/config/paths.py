# config/paths.py
"""
路径配置
"""
import os

# 项目根目录
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据路径
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# 模型路径
MODEL_DIR = os.path.join(ROOT_DIR, 'experiments', 'checkpoints')

# 日志路径
LOG_DIR = os.path.join(ROOT_DIR, 'experiments', 'logs')

# 结果路径
RESULTS_DIR = os.path.join(ROOT_DIR, 'experiments', 'results')

# 创建必要的目录
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, LOG_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)