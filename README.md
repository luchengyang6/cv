# 《用纯Python手搓经典计算机视觉算法》开源教材

## 📖 项目简介

本项目是23级实验班计算机视觉课程的大作业成果，旨在通过**纯Python实现经典计算机视觉算法**，深入理解算法原理，并与大模型协作完成开源教材创作。

> 🎯 **核心理念**：从底层理解算法，拒绝成为"调包侠"！

## 🚀 项目特色

### 🔥 纯手搓实现
- 仅使用 **Python + Numpy + Scipy** 基础库
- 从零实现每一个数学运算和算法组件
- 深入理解梯度计算、反向传播等底层原理

### 🤖 大模型协作开发
- 与 **百度文心大模型** 深度协作
- 体验 **LLM Guided Learning** 新型学习模式
- 培养与AI协作的核心竞争力

### 📚 开源教材创作
- 完整的算法原理讲解和代码解析
- 详细的实验分析和性能评估
- 社区贡献，知识共享

## 📋 作业进度

| 作业 | 模型 | 截止日期 | 状态 |
|------|------|----------|------|
| 小作业1 | K-近邻 (K-NN) | 2025.10.12 | ✅ **已完成** |
| 小作业2 | Softmax分类器 | 2025.10.26 | ✅ **已完成** |
| 小作业3 | 两层全连接神经网络 | 2025.11.09 | ✅ **已完成** |
| 小作业4 | 简化版CNN | 2025.11.23 | 📅 待开始 |
| 大作业 | 基础版RNN | 2025.12.28 | 📅 待开始 |

## 🛠 技术栈

### 核心要求
- **编程语言**: 纯 Python
- **核心库**: Numpy, Scipy
- **开发环境**: 虚拟 Linux 环境 (Docker/WSL/虚拟机)
- **版本控制**: Git, GitHub
- **环境管理**: venv/conda + requirements.txt
- **配置管理**: .env 文件
- **文档编写**: Markdown

### 项目结构
```
computer-vision-from-scratch/
├── 📁 knn_classifier/           # K-近邻分类器 ✅
├── 📁 softmax_classifier/       # Softmax分类器 ✅
├── 📁 two_layer_fc_net/         # 全连接神经网络 ✅
├── 📄 LICENSE                   # 开源许可证
└── 📄 README.md                 # 项目总览
```

## 🎯 最新完成：两层全连接神经网络

### 📂 项目结构
```
two_layer_fc_net/
├── 📄 README.md                 # 本章详细文档
├── 📄 requirements.txt          # 依赖列表
├── 📄 .gitignore               # Git忽略配置
│
├── 📁 src/                      # 源代码
│   ├── 📄 two_layer_net.py      # 主网络类
│   ├── 📄 layers.py             # 网络层实现
│   ├── 📄 optimizers.py         # 优化器
│   ├── 📄 utils.py              # 工具函数
│   └── 📄 data_loader.py        # 数据加载
│
├── 📁 examples/                 # 使用示例
│   ├── 📄 train_mnist.py        # 主要训练脚本
│   ├── 📄 train_mnist_relative.py # 相对路径训练
│── 📁 data/                     # 数据集
│ 
├── 📁 tests/                    # 单元测试
│   ├── 📄 test_network.py       # 网络测试
│   ├── 📄 test_layers.py        # 层测试
│   └── 📄 test_utils.py         # 工具测试
│
├── 📁 config/                   # 配置
│   └── 📄 hyperparameters.py    # 超参数配置
│
├── 📁 output/                   # 输出目录
│   ├── 📄 training_curves.png   # 训练曲线
│
```

### 🧠 算法特性
- ✅ **纯Numpy实现**: 手动实现前向传播、反向传播、梯度计算
- ✅ **模块化设计**: 分离的层实现、优化器、工具函数
- ✅ **完整训练流程**: 数据加载、预处理、训练、评估一体化
- ✅ **L2正则化**: 完整的正则化损失和梯度计算
- ✅ **多种优化器**: SGD、带动量的SGD
- ✅ **梯度检查**: 数值梯度验证确保反向传播正确性
- ✅ **完整可视化**: 训练曲线、混淆矩阵、样本预测等

### 🚀 快速开始

#### 1. 环境配置
```bash
# 进入全连接神经网络目录
cd two_layer_fc_net

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

#### 2. 检查项目结构
```bash
# 检查项目结构和数据文件
python check_project_structure.py
```

#### 3. 运行训练
```bash
# 训练模型并生成可视化结果
python examples/train_mnist.py
```

#### 4. 生成报告
```bash
# 生成详细项目报告
python examples/generate_report.py
```

### 📊 核心实现

#### 网络层实现 (`src/layers.py`)
```python
class LinearLayer:
    """全连接层实现"""
    def __init__(self, input_size, output_size, weight_scale=1e-3):
        # Xavier初始化
        self.W = weight_scale * np.random.randn(input_size, output_size)
        self.b = np.zeros(output_size)
    
    def forward(self, x):
        self.x = x  # 缓存输入用于反向传播
        return np.dot(x, self.W) + self.b
    
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx

class ReLU:
    """ReLU激活函数"""
    def forward(self, x):
        self.mask = (x <= 0)  # 缓存负值位置
        out = x.copy()
        out[self.mask] = 0
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0  # 负值位置梯度为0
        return dout
```

#### 主网络类 (`src/two_layer_net.py`)
```python
class TwoLayerNet:
    """两层全连接神经网络"""
    def __init__(self, input_size, hidden_size, output_size, reg_strength=0.0):
        self.reg_strength = reg_strength
        
        # 初始化网络层
        self.layers = {}
        self.layers['linear1'] = LinearLayer(input_size, hidden_size)
        self.layers['relu1'] = ReLU()
        self.layers['linear2'] = LinearLayer(hidden_size, output_size)
        
    def loss(self, X, y):
        # 前向传播
        scores = self.forward(X)
        
        # 计算数据损失
        data_loss, dscores = cross_entropy_loss(scores, y)
        
        # 计算正则化损失
        params = self.get_params()
        reg_loss, reg_grads = l2_regularization(params, self.reg_strength)
        
        # 总损失
        total_loss = data_loss + reg_loss
        
        # 反向传播
        grads = self.backward(dscores)
        
        # 添加正则化梯度
        for key in reg_grads:
            grads[key] += reg_grads[key]
        
        return total_loss, grads
```

### 📈 实验结果

#### 性能表现
- **MNIST数据集**: 准确率 **92.70%**
- **训练时间**: ~2分钟 (CPU)
- **验证准确率**: 93.52%
- **收敛性**: 损失函数从2.3平稳下降至1.55，模型收敛良好

#### 训练过程
```
Epoch	Loss		Train Acc	Val Acc
--------------------------------------------------
0	2.3025		0.1160		0.1062
5	2.0880		0.4470		0.4438
10	1.7264		0.7640		0.7698
15	1.6330		0.8770		0.8946
20	1.5625		0.9170		0.9316
25	1.5514		0.9270		0.9352
--------------------------------------------------
最终测试准确率: 0.9270
```

#### 可视化分析
- ✅ 训练损失曲线 - 显示损失平稳下降
- ✅ 训练/验证准确率对比 - 显示模型有效学习且无过拟合
- ✅ 混淆矩阵分析 - 数字"1"识别率最高(98%)，数字"5"和"8"容易混淆
- ✅ 样本预测展示 - 直观显示模型预测结果

### 🎯 实现亮点

1. **数学原理深度理解**
   - 手动推导反向传播链式法则
   - 实现数值稳定的Softmax计算
   - 完整的交叉熵损失和正则化梯度计算

2. **工程健壮性**
   - 模块化设计，各组件独立测试
   - 完整的梯度检查验证
   - 相对路径支持，便于部署

3. **完整软件工程流程**
   - 版本控制规范
   - 单元测试覆盖
   - 配置分离管理
   - 详细文档记录

## 📚 已完成作业概述

### 🎯 Softmax分类器
- **准确率**: 85-92%
- **核心特性**: 数值稳定的Softmax实现、L2正则化、多镜像数据源
- **亮点**: 完整的手动梯度计算、专业可视化分析
- [查看详细文档](./softmax_classifier/README.md)

### 🎯 K-近邻分类器
- **准确率**: ~96.7%
- **核心特性**: 支持多种距离度量、加权投票、交叉验证
- **亮点**: 高效的批量预测、参数调优自动化
- [查看详细文档](./knn_classifier/README.md)

## 🤝 与大模型协作记录

### 协作成果
通过多个阶段的深度协作，我们完成了：

1. **项目架构设计** - 模块化代码结构
2. **数学原理推导** - 梯度计算验证
3. **数据加载实现** - 多镜像容错机制
4. **可视化优化** - 专业图表和英文标签
5. **调试与优化** - 从基础实现到性能优化
6. **性能分析调优** - 超参数搜索和正则化优化

### 关键协作经验
- **明确约束条件**: 指定技术栈和限制条件
- **分步骤请求**: 将复杂问题分解为可管理任务
- **验证关键信息**: 手动验证数学公式和代码逻辑
- **迭代改进**: 基于运行结果持续优化

### 典型交互案例
**我的提问**：
```
我在实现反向传播时担心梯度计算是否正确。请提供一个数值梯度检查的实现方法，用于验证解析梯度的准确性。
```

**LLM回复**：
```
数值梯度检查实现要点：

def gradient_check(f, x, analytic_grad, h=1e-5):
    numerical_grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'])
    
    while not it.finished:
        idx = it.multi_index
        old_val = x[idx]
        
        # f(x + h)
        x[idx] = old_val + h
        fxh1 = f(x)
        
        # f(x - h)  
        x[idx] = old_val - h
        fxh2 = f(x)
        
        numerical_grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = old_val
        it.iternext()
    
    # 比较差异
    diff = np.linalg.norm(analytic_grad - numerical_grad)
    diff /= np.linalg.norm(analytic_grad + numerical_grad)
    
    return diff < 1e-7, diff

差异小于1e-7表示梯度计算正确。
```

## 🎓 学习价值

### 能力培养
- ✅ **深度理解**: 掌握神经网络数学原理和实现细节
- ✅ **工程能力**: 遵循软件工程规范，模块化开发
- ✅ **AI协作**: 学会与LLM高效沟通和协作
- ✅ **问题解决**: 独立调试和优化代码的能力

### 知识体系
1. **数学基础**: 线性代数、微积分、概率统计、优化理论
2. **编程技能**: Python高级特性、面向对象设计、算法优化
3. **工程实践**: 版本控制、测试驱动开发、文档编写、配置管理
4. **AI应用**: 提示工程、模型评估、结果分析、性能调优

### 技术收获
- 深入理解前向传播和反向传播机制
- 掌握梯度检查和方法验证
- 学会模块化设计和代码重构
- 掌握完整的机器学习项目流程

## 📝 提交要求

### GitHub仓库
- [x] 完整源代码
- [x] 详细文档和教程
- [x] 依赖管理文件
- [x] LLM交互记录
- [x] 单元测试覆盖
- [x] 配置管理示例

### 百度AI Studio
- [ ] 可运行的项目副本
- [ ] 训练和推理演示
- [ ] 性能基准测试

### B站视频
- [ ] 项目整体介绍
- [ ] 算法原理讲解
- [ ] 代码实现亮点
- [ ] LLM协作经验分享

## 📄 许可证

本项目采用 [MIT License](LICENSE)，鼓励知识共享和社区贡献。

## 🎯 下一步计划

1. **开始卷积神经网络**: 研究卷积层、池化层实现
2. **性能优化**: 添加学习率调度、早停等高级特性
3. **扩展数据集**: 在CIFAR-10等更复杂数据集上测试
4. **模型部署**: 实现模型保存加载和推理接口
5. **社区互动**: 收集反馈，持续改进项目质量


*最后更新: 2025年11月*  
*当前进度: 3/5 章节完成*  
*下一个目标: 卷积神经网络 (CNN) 实现*
