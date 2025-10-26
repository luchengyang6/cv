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
| 小作业3 | 两层全连接神经网络 | 2025.11.09 | 📅 待开始 |
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
├── 📁 chapter3-fc-net/        # 全连接神经网络
├── 📁 chapter4-cnn/           # 卷积神经网络
├── 📁 chapter5-rnn/           # 循环神经网络
├── 📁 docs/                   # 项目文档
├── 📁 datasets/               # 数据集管理
├── 📁 utils/                  # 共享工具函数
├── 📄 LICENSE                 # 开源许可证
└── 📄 README.md               # 项目总览
```

## 🎯 最新完成：Softmax分类器

### 📂 项目结构
```
chapter2-softmax/
├── 📄 README.md              # 本章详细文档
├── 📄 requirements.txt       # 依赖列表
├── 📄 .gitignore            # Git忽略配置
│
├── 📁 src/                   # 源代码
│   ├── 📄 softmax.py         # Softmax分类器核心实现
│   ├── 📄 data_loader.py     # 数据加载和预处理
│   ├── 📄 train.py           # 训练脚本
│   └── 📄 visualize.py       # 可视化模块
    ├── 📁 outputs/               # 输出目录
           ├── 📁 models/            # 保存的模型
           ├── 📁 plots/             # 可视化图表
           └── 📄 .gitkeep          # 保持目录结构
    │
└── 📁 docs/                  # 文档记录
    ├── 📄 llm_interaction.md # LLM交互记录
    └── 📄 implementation_details.md # 实现细节
```

### 🧠 算法特性
- ✅ **纯Numpy实现**: 手动实现Softmax函数和梯度计算
- ✅ **数值稳定性**: 处理指数运算的数值溢出问题
- ✅ **L2正则化**: 完整的正则化损失和梯度计算
- ✅ **多镜像数据源**: 健壮的MNIST数据加载机制
- ✅ **完整可视化**: 损失曲线、准确率对比、混淆矩阵等

### 🚀 快速开始

#### 1. 环境配置
```bash
# 进入Softmax分类器目录
cd chapter2-softmax

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

#### 2. 运行训练
```bash
# 自动训练并生成可视化结果
python src/train.py
```

### 📊 核心实现

#### Softmax分类器 (`src/softmax.py`)
```python
class SoftmaxClassifier:
    def __init__(self, input_dim, num_classes, learning_rate=0.01, reg_strength=0.001):
        self.W = np.random.randn(input_dim, num_classes) * 0.01
        self.b = np.zeros(num_classes)
        # 参数初始化
    
    def forward(self, X):
        scores = np.dot(X, self.W) + self.b
        probs = self._softmax(scores)  # 手动实现Softmax
        return probs, scores
    
    def compute_gradients(self, X, y):
        # 手动计算梯度
        num_samples = X.shape[0]
        probs, _ = self.forward(X)
        
        dscores = probs.copy()
        dscores[np.arange(num_samples), y] -= 1
        dscores /= num_samples
        
        dW = np.dot(X.T, dscores) + self.reg_strength * self.W
        db = np.sum(dscores, axis=0)
        
        return dW, db
```

### 📈 实验结果

#### 性能表现
- **MNIST数据集**: 准确率 **85-92%**
- **训练时间**: ~2分钟 (CPU)
- **收敛性**: 损失函数平稳下降，模型收敛良好

#### 可视化分析
- ✅ 训练损失曲线
- ✅ 训练/测试准确率对比
- ✅ 混淆矩阵分析
- ✅ 样本预测展示
- ✅ 特征分布直方图

### 🎯 实现亮点

1. **数学原理深度理解**
   - 手动推导Softmax梯度公式
   - 实现数值稳定的Softmax计算
   - 完整的交叉熵损失和正则化

2. **工程健壮性**
   - 多镜像数据源容错机制
   - 虚拟数据生成用于测试
   - 完整的错误处理和日志

3. **可视化完整性**
   - 英文标签避免字体问题
   - 专业美观的图表布局
   - 多角度模型性能分析

## 🎯 已完成：K-近邻分类器

### 核心特性
- ✅ 支持欧氏距离和曼哈顿距离
- ✅ 支持均匀权重和距离加权投票
- ✅ 完整的交叉验证和参数调优
- ✅ 模块化设计，易于扩展

### 性能表现
- **MNIST数据集**: 准确率 ~96.7%
- **计算效率**: 支持批量预测和并行优化

[查看K-NN详细文档](./knn_classifier/README.md)

## 🤝 与大模型协作记录

### 协作成果
通过6个阶段的深度协作，我们完成了：

1. **项目架构设计** - 模块化代码结构
2. **数学原理推导** - Softmax梯度计算验证
3. **数据加载实现** - 多镜像容错机制
4. **可视化优化** - 专业图表和英文标签
5. **调试与优化** - 从60%到90%准确率提升
6. **性能分析调优** - 超参数搜索和正则化优化

### 关键协作经验
- **明确约束条件**: 指定技术栈和限制条件
- **分步骤请求**: 将复杂问题分解为可管理任务
- **验证关键信息**: 手动验证数学公式和代码逻辑
- **迭代改进**: 基于运行结果持续优化


## 🎓 学习价值

### 能力培养
- ✅ **深度理解**: 掌握线性分类器数学原理和实现细节
- ✅ **工程能力**: 遵循软件工程规范，模块化开发
- ✅ **AI协作**: 学会与LLM高效沟通和协作
- ✅ **问题解决**: 独立调试和优化代码的能力

### 知识体系
1. **数学基础**: 线性代数、概率统计、优化理论
2. **编程技能**: Python高级特性、面向对象设计
3. **工程实践**: 版本控制、测试驱动开发、文档编写
4. **AI应用**: 提示工程、模型评估、结果分析

## 📝 提交要求

### GitHub仓库
- [x] 完整源代码
- [x] 详细文档和教程
- [x] 依赖管理文件
- [x] LLM交互记录
- [ ] 单元测试覆盖（进行中）

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

1. **完善Softmax章节**: 添加超参数调优和更多数据集测试
2. **开始全连接神经网络**: 研究激活函数和反向传播
3. **持续文档更新**: 记录学习心得和问题解决方案
4. **社区互动**: 收集反馈，持续改进项目质量

---

**💡 提示**: 本项目是学习过程的记录，重点在于理解原理和培养能力，而非追求极致的性能指标。

**🌟 让我们一起探索计算机视觉的奥秘，从底层开始，构建坚实的AI基础！**

---
*最后更新: 2025年10月*  
*当前进度: 2/5 章节完成*
