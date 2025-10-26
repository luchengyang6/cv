# Softmax 分类器 - 线性分类器实现

## 📖 项目概述

本项目是《用纯Python手搓经典计算机视觉算法》开源教材的第二章节，实现了基于Softmax的线性分类器。我们使用最基础的Python、Numpy和Scipy库，从零开始构建完整的图像分类pipeline，深入理解线性分类器的数学原理和实现细节。

## 🎯 模型原理

### 数学基础

**Softmax函数**：
```
softmax(z_i) = exp(z_i) / Σ_j(exp(z_j))
```

**交叉熵损失函数**：
```
L = -1/N * Σ_i Σ_j y_ij * log(p_ij) + λ/2 * ||W||^2
```

其中：
- `z_i` 是第i个类别的得分
- `p_ij` 是样本i属于类别j的预测概率
- `y_ij` 是样本i属于类别j的真实标签（one-hot编码）
- `λ` 是L2正则化强度

### 算法流程

1. **前向传播**：
   - 线性变换：`scores = X * W + b`
   - Softmax概率：`probs = softmax(scores)`

2. **损失计算**：
   - 交叉熵损失 + L2正则化

3. **反向传播**：
   - 计算梯度：`dW, db`
   - 参数更新：`W = W - η * dW`, `b = b - η * db`

## 🏗️ 项目结构

```
softmax-classifier/
├── src/
│   ├── data_loader.py     # 数据加载和预处理
│   ├── softmax.py         # Softmax分类器核心实现
│   ├── train.py           # 训练脚本
│   └── visualize.py       # 可视化模块
│   ├── data/                  # 数据目录
│   ├── outputs/               # 输出目录
│   │   ├── models/            # 保存的模型
│   │   └── plots/             # 可视化图表
├── requirements.txt       # 项目依赖
└── README.md             # 项目说明
```

## 🚀 快速开始

### 环境配置

```bash
# 创建虚拟环境
python -m venv cv_env
source cv_env/bin/activate  # Linux/Mac
# 或 cv_env\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 运行训练

```bash
python src/train.py
```

## 📊 实验结果

### 在MNIST数据集上的表现

| 指标 | 数值 |
|------|------|
| 训练集准确率 | 85-92% |
| 测试集准确率 | 85-90% |
| 训练时间 | ~2分钟 (CPU) |

### 损失曲线
训练过程中损失函数平稳下降，表明模型收敛良好。

### 混淆矩阵
对角线元素明显高于非对角线，表明分类效果较好。

## 🔧 核心实现

### 数据加载 (`src/data_loader.py`)

- 自动从多个镜像源下载MNIST数据集
- 数据归一化到[0,1]范围
- 容错机制：下载失败时使用虚拟数据

### Softmax分类器 (`src/softmax.py`)

```python
class SoftmaxClassifier:
    def __init__(self, input_dim, num_classes, learning_rate=0.01, reg_strength=0.001):
        self.W = np.random.randn(input_dim, num_classes) * 0.01
        self.b = np.zeros(num_classes)
        # ... 其他初始化
    
    def forward(self, X):
        scores = np.dot(X, self.W) + self.b
        probs = self._softmax(scores)
        return probs, scores
    
    def compute_gradients(self, X, y):
        # 手动实现梯度计算
        # ...
```

### 可视化 (`src/visualize.py`)

- 训练损失曲线
- 准确率对比图
- 混淆矩阵
- 样本预测展示
- 特征分布直方图

## 📈 超参数调优

| 超参数 | 推荐值 | 影响 |
|--------|--------|------|
| 学习率 | 0.01-0.1 | 影响收敛速度和稳定性 |
| 正则化强度 | 0.001-0.01 | 控制过拟合 |
| 批大小 | 64-128 | 影响训练稳定性和速度 |
| 训练轮数 | 500-1000 | 确保充分收敛 |

## 🔍 实现亮点

1. **纯底层实现**：仅使用Numpy、Scipy和基础Python
2. **数值稳定性**：Softmax实现中减去最大值避免数值溢出
3. **模块化设计**：各组件职责单一，易于理解和扩展
4. **完整可视化**：多角度展示模型性能和训练过程
5. **错误容错**：网络问题不影响代码运行


# 🤖 与LLM协作详细记录

## 💬 协作对话全记录

### 第一阶段：项目架构设计

**我的提问**：
```
我需要为计算机视觉课程作业实现一个Softmax分类器，要求：
1. 仅使用Python、Numpy、Scipy基础库
2. 完整的项目结构，包括数据加载、模型训练、可视化
3. 遵循软件工程规范
4. 包含详细的文档和注释

请帮我设计项目架构，并解释每个模块的职责。
```

**LLM回复**：
```
基于你的需求，我建议采用以下模块化架构：

src/
├── data_loader.py    # 数据加载、预处理、归一化
├── softmax.py        # Softmax分类器核心算法
├── train.py          # 训练流程和超参数管理
└── visualize.py      # 结果可视化和分析

每个模块的职责：
1. data_loader: 负责MNIST数据下载、解析、预处理
2. softmax: 实现前向传播、损失计算、梯度计算、参数更新
3. train: 组织训练循环、模型评估、结果保存
4. visualize: 绘制损失曲线、准确率、混淆矩阵等

关键设计原则：
- 单一职责：每个模块只负责一个特定功能
- 松耦合：模块间通过清晰接口通信
- 可配置：超参数集中管理，便于调优
```

**我的验证与调整**：
- 验证了模块划分的合理性
- 增加了`utils.py`用于公共函数
- 调整了文件命名使其更符合Python惯例

### 第二阶段：数学原理推导

**我的提问**：
```
请详细推导Softmax分类器的数学原理，包括：
1. Softmax函数及其数值稳定性处理
2. 交叉熵损失的梯度计算
3. L2正则化的影响和梯度计算
4. 参数更新公式

请用LaTeX格式给出数学表达式，并解释每个步骤的物理意义。
```

**LLM回复**：
```
## 数学推导

### 1. Softmax函数
$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$

**数值稳定性**：实际实现中减去最大值
$$
\text{softmax}(z_i) = \frac{e^{z_i - \max(z)}}{\sum_{j=1}^{K} e^{z_j - \max(z)}}
$$

### 2. 交叉熵损失梯度
令 $p_i = \text{softmax}(z_i)$，对于真实类别 $k$：
$$
\frac{\partial L}{\partial z_i} = 
\begin{cases}
p_i - 1 & \text{if } i = k \\
p_i & \text{otherwise}
\end{cases}
$$

### 3. L2正则化
总损失：$L_{\text{total}} = L_{\text{CE}} + \frac{\lambda}{2} \|W\|^2$

梯度：$\frac{\partial L_{\text{total}}}{\partial W} = \frac{\partial L_{\text{CE}}}{\partial W} + \lambda W$
```

**我的验证与调整**：
- 手动验证了梯度计算公式的正确性
- 在代码实现中添加了数值稳定性处理
- 调整了正则化系数的默认值

### 第三阶段：数据加载实现

**我的提问**：
```
MNIST数据集下载经常失败，请帮我设计一个健壮的数据加载方案：
1. 多个备用下载镜像
2. 下载失败时的虚拟数据生成
3. 完整的数据预处理流程
4. 错误处理和日志记录

请提供具体的Python实现代码。
```

**LLM回复**：
```python
class DataLoader:
    def __init__(self, data_dir='./data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def load_mnist(self):
        mirrors = [
            'http://yann.lecun.com/exdb/mnist/',
            'https://ossci-datasets.s3.amazonaws.com/mnist/'
        ]
        
        for mirror in mirrors:
            try:
                # 尝试下载逻辑
                return self._download_and_parse(mirror)
            except Exception as e:
                print(f"镜像 {mirror} 失败: {e}")
                continue
        
        # 所有镜像都失败，使用虚拟数据
        return self._create_dummy_data()
```

**我的验证与调整**：
- 测试了多个镜像的可用性
- 改进了虚拟数据生成，使其更接近真实分布
- 增加了数据预处理的质量检查

### 第四阶段：可视化优化

**我的提问**：
```
我遇到了matplotlib中文显示问题，并且希望图表更加专业美观。
请帮我：
1. 解决中文字体显示问题
2. 设计一套完整的可视化方案
3. 优化图表样式和布局
4. 确保图表能够清晰传达模型性能

请提供修改后的可视化代码。
```

**LLM回复**：
```python
class Visualizer:
    def __init__(self, output_dir='./outputs/plots'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置matplotlib参数避免中文问题
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_loss_curve(self, loss_history, save_path=None):
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history, 'b-', linewidth=2, label='Training Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
```

**我的验证与调整**：
- 测试了不同系统的字体兼容性
- 调整了图表颜色方案和布局
- 增加了多种可视化图表类型

### 第五阶段：调试与优化

**我的提问**：
```
我的Softmax分类器在MNIST上的准确率只有60%，远低于预期。
可能的问题：
1. 学习率设置不当
2. 梯度计算错误
3. 数据预处理问题
4. 模型初始化问题

请帮我分析可能的原因并提供调试建议。
```

**LLM回复**：
```
## 问题诊断和解决方案

### 1. 学习率问题
- 症状：损失震荡或不收敛
- 调试：尝试学习率 0.001, 0.01, 0.1, 1.0
- 建议：使用0.1作为起点

### 2. 梯度检查
实现数值梯度验证：
```python
def gradient_check(X, y, model, param='W', epsilon=1e-7):
    # 数值梯度计算
    # 与解析梯度比较



## 🎥 视频讲解要点

- Softmax函数的数学原理和物理意义
- 梯度推导的手动计算过程
- 项目架构设计思路
- 调试过程中遇到的挑战和解决方案
- 与LLM协作的最佳实践

## 📝 扩展方向

1. **多分类任务扩展**：适配CIFAR-10等更复杂数据集
2. **优化算法改进**：实现动量、Adam等优化器
3. **学习率调度**：添加学习率衰减策略
4. **早停机制**：防止过拟合
5. **模型集成**：多个Softmax分类器集成

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。