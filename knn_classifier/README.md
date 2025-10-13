# K-近邻(K-NN)图像分类器

## 📖 项目简介

本项目是《用纯Python手搓经典计算机视觉算法》开源教材的第一章，**纯手工实现K-近邻算法**，仅使用Python、Numpy和Scipy基础库，深入理解这一经典机器学习算法的数学原理和工程实现。

> 🎯 **学习目标**：从零理解K-NN算法的数学基础、实现细节和实际应用

## 🚀 项目特色

### 🔥 纯手搓实现
- ✅ 仅使用 **Python + Numpy + Scipy** 基础库
- ✅ 从零实现距离计算、邻居选择、投票机制
- ✅ 深入理解向量化运算和性能优化

### 📊 完整功能
- ✅ 支持欧氏距离和曼哈顿距离
- ✅ 支持均匀权重和距离加权投票
- ✅ 完整的交叉验证和参数调优
- ✅ 多种评估指标和可视化分析

### 🛠 工程规范
- ✅ 模块化设计，代码可读性强
- ✅ 完整的单元测试覆盖
- ✅ 配置驱动，易于复现实验
- ✅ 详细的文档和示例

## 📂 项目结构

```
knn_classifier/
├── 📄 README.md                  # 本文档
├── 📄 requirements.txt           # 项目依赖
├── 📄 .env                       # 环境配置模板
├── 📄 run_with_config.py         # 配置化运行入口
│
├── 📁 src/                       # 源代码
│   ├── 📄 __init__.py
│   ├── 📄 knn.py                 # K-NN核心算法实现
│   ├── 📄 utils.py               # 数据预处理和工具函数
│   ├── 📄 metrics.py             # 评估指标计算
│   └── 📄 hyperparameter_tuning.py # 超参数优化
│
├── 📁 data/                      # 数据管理
│   ├── 📄 __init__.py
│   ├── 📄 download_data.py       # 演示数据生成
│   └── 📄 load_custom_data.py    # 真实数据集加载
│
├── 📁 examples/                  # 使用示例
│   ├── 📄 basic_usage.py         # 基础使用演示
│   ├── 📄 mnist_example.py       # MNIST数据集示例
│   └── 📄 custom_dataset.py      # 自定义数据集示例
│
├── 📁 tests/                     # 单元测试
│   ├── 📄 __init__.py
│   └── 📄 test_knn.py            # 算法测试用例
├── 📁 mnist_jpg/                 # mnist数据集(需解压缩)
└── 📁 docs/                      # 文档记录
    ├── 📄 implementation_details.md  # 实现细节文档
    ├── 📄 llm_interaction_logs.md    # LLM交互记录
    └── 📄 experiment_results.md      # 实验结果分析
```

## 🧠 算法原理

### 数学基础
K-近邻算法的核心思想：**相似的样本在特征空间中距离相近**

**距离度量**：
- 欧氏距离：$d(x,y) = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}$
- 曼哈顿距离：$d(x,y) = \sum_{i=1}^n |x_i - y_i|$

**决策规则**：
- 多数投票：$y = \arg\max_{c} \sum_{i=1}^{k} I(y_i = c)$
- 距离加权：$y = \arg\max_{c} \sum_{i=1}^{k} \frac{1}{d(x,x_i)} \cdot I(y_i = c)$

## 🛠 安装依赖

### 环境要求
- Python 3.8+
- 虚拟Linux环境（WSL/Docker/虚拟机）

### 安装步骤
```bash
# 克隆项目
git clone https://github.com/your-username/computer-vision-from-scratch.git
cd computer-vision-from-scratch/knn_classifier

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 依赖列表
```txt
numpy>=1.21.0
scipy>=1.7.0
python-dotenv>=0.19.0
Pillow>=8.0.0    # 图像处理支持
```

## 🚀 快速开始

### 基础使用
```python
import numpy as np
from src.knn import KNeighborsClassifier
from src.utils import train_test_split, normalize

# 生成演示数据
from data.download_data import load_simple_dataset
X, y = load_simple_dataset()

# 数据预处理
X_normalized = normalize(X, method='minmax')

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y, test_size=0.3, random_state=42
)

# 创建和训练模型
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train, y_train)

# 预测和评估
accuracy = knn.score(X_test, y_test)
print(f"模型准确率: {accuracy:.4f}")
```

### 配置化运行
```bash
# 复制配置模板
cp .env.example .env

# 编辑配置文件（可选）
# 修改 DATA_PATH, K_NEIGHBORS 等参数

# 运行配置化脚本
python run_with_config.py
```

## 📊 高级功能

### 参数调优
```python
from src.hyperparameter_tuning import grid_search_knn

# 定义参数网格
param_grid = {
    'n_neighbors': [1, 3, 5, 7, 9],
    'metric': ['euclidean', 'manhattan'],
    'weights': ['uniform', 'distance']
}

# 执行网格搜索
best_params, best_score, all_results = grid_search_knn(X, y, param_grid, cv=5)
print(f"最佳参数: {best_params}, 最佳得分: {best_score:.4f}")
```

### 交叉验证
```python
from src.utils import cross_validate

knn = KNeighborsClassifier(n_neighbors=5)
cv_scores = cross_validate(knn, X, y, cv=5)
print(f"交叉验证得分: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
```

### 使用真实图像数据集
```python
from data.load_custom_data import load_image_dataset

# 加载图像数据集
X, y, class_names = load_image_dataset(
    data_path="path/to/your/images",
    img_size=(28, 28)  # 调整图像尺寸
)

# 后续流程与基础使用相同
```

## 📈 实验结果

### 性能基准
在标准数据集上的测试结果：

| 数据集 | 准确率 | 最佳k值 | 最佳距离度量 |
|--------|--------|---------|-------------|
| 演示数据集 | 95.6% | 3 | 欧氏距离 |
| MNIST  | 96.7% | 5 | 欧氏距离 |
| 自定义图像集 | 89.3% | 7 | 曼哈顿距离 |

### 参数影响分析
通过网格搜索发现：
- **小数据集**：较小的k值（3-5）表现更好
- **大数据集**：较大的k值（5-9）更稳定
- **噪声数据**：距离加权投票能提升鲁棒性

## 🤖 与LLM协作记录

### 协作流程
本项目与**百度文心大模型**深度协作开发，完整记录了AI辅助编程的过程：

1. **算法设计阶段**：讨论K-NN的数学原理和实现方案
2. **代码实现阶段**：分模块实现，实时调试和优化
3. **测试验证阶段**：设计测试用例，验证算法正确性
4. **文档编写阶段**：协同撰写技术文档和使用指南

### 关键交互示例
> **提示词**: "请用纯Python实现K近邻算法的predict方法，要求支持距离加权投票，并处理距离为0的边界情况"

> **文心回复**: 
> ```python
> def _weighted_vote(self, labels, distances):
>     # 避免除零错误
>     weights = 1 / (distances + 1e-8)
>     # 计算加权投票...
> ```

[查看完整交互记录](./docs/llm_interaction_logs.md)

## 🧪 运行测试

确保代码质量和正确性：
```bash
# 运行所有测试
python -m pytest tests/

# 运行特定测试文件
python tests/test_knn.py
```

## 📝 提交要求

### GitHub仓库包含
- [x] 完整的K-NN算法实现
- [x] 单元测试和示例代码
- [x] 详细的项目文档
- [x] 依赖管理配置
- [x] LLM交互记录

### 百度AI Studio
- [x] 可直接运行的项目副本
- [x] 模型训练和推理演示
- [x] 性能基准测试结果

## 🎓 学习收获

通过本项目，您将掌握：

### 理论知识
- K-NN算法的数学原理和推导
- 距离度量的选择和影响
- 模型评估和参数调优方法

### 实践技能
- 纯Python实现机器学习算法
- 模块化代码设计和工程规范
- 与LLM协作开发的能力
- 实验设计和结果分析

### 工程能力
- 版本控制和项目管理
- 测试驱动开发
- 文档编写和知识分享

## 📄 许可证

本项目采用 [MIT License](../LICENSE)，欢迎学习和使用。

---

**💡 下一步**：继续学习下一章节 [Softmax分类器](../chapter2-softmax/)

**🌟 学习建议**：建议先运行基础示例理解算法流程，再阅读源代码深入理解实现细节。

---
*项目完成时间: 2025年10月*  
*最后更新: 2025年10月*