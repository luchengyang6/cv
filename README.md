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
| 小作业2 | Softmax分类器 | 2025.10.26 | 🚧 进行中 |
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
├── 📁 chapter1-knn/          # K-近邻分类器
├── 📁 chapter2-softmax/      # Softmax分类器
├── 📁 chapter3-fc-net/       # 全连接神经网络
├── 📁 chapter4-cnn/          # 卷积神经网络
├── 📁 chapter5-rnn/          # 循环神经网络
├── 📁 docs/                  # 项目文档
├── 📁 datasets/              # 数据集管理
├── 📁 utils/                 # 共享工具函数
├── 📄 LICENSE                # 开源许可证
└── 📄 README.md              # 项目总览
```

## 🎯 当前完成：K-近邻分类器

### 📂 项目结构
```
chapter1-knn/
├── 📄 README.md              # 本章详细文档
├── 📄 requirements.txt       # 依赖列表
├── 📄 .env.example           # 环境配置模板
├── 📄 run_with_config.py     # 配置化运行入口
│
├── 📁 src/                   # 源代码
│   ├── 📄 knn.py             # K-NN核心算法
│   ├── 📄 utils.py           # 工具函数
│   ├── 📄 metrics.py         # 评估指标
│   └── 📄 hyperparameter_tuning.py  # 超参数优化
│
├── 📁 data/                  # 数据管理
│   ├── 📄 download_data.py   # 演示数据生成
│   └── 📄 load_custom_data.py # 真实数据集加载
│
├── 📁 examples/              # 使用示例
│   ├── 📄 basic_usage.py     # 基础演示
│   ├── 📄 mnist_example.py   # MNIST示例
│   └── 📄 custom_dataset.py  # 自定义数据集
│
├── 📁 tests/                 # 单元测试
│   └── 📄 test_knn.py        # 算法测试
│
└── 📁 docs/                  # 文档记录
    ├── 📄 implementation_details.md  # 实现细节
    ├── 📄 llm_interaction_logs.md   # LLM交互记录
    └── 📄 experiment_results.md     # 实验结果
```

### 🧠 算法特性
- ✅ 支持欧氏距离和曼哈顿距离
- ✅ 支持均匀权重和距离加权投票
- ✅ 完整的交叉验证和参数调优
- ✅ 模块化设计，易于扩展

### 🚀 快速开始

#### 1. 环境配置
```bash
# 克隆项目
git clone https://github.com/your-username/computer-vision-from-scratch.git
cd computer-vision-from-scratch/chapter1-knn

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

#### 2. 基础使用
```python
from src.knn import KNeighborsClassifier
from src.utils import train_test_split, normalize

# 加载数据
X, y = load_your_data()

# 数据预处理
X_normalized = normalize(X)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y)

# 训练模型
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# 评估模型
accuracy = knn.score(X_test, y_test)
print(f"准确率: {accuracy:.4f}")
```

#### 3. 配置化运行
```bash
# 复制配置模板
cp .env.example .env
# 编辑 .env 文件配置参数

# 运行配置化脚本
python run_with_config.py
```

#### 4. 高级功能
```python
# 参数调优
from src.hyperparameter_tuning import grid_search_knn

param_grid = {
    'n_neighbors': [1, 3, 5, 7, 9],
    'metric': ['euclidean', 'manhattan'],
    'weights': ['uniform', 'distance']
}

best_params, best_score = grid_search_knn(X, y, param_grid)
```

## 📊 实验结果

### 性能表现
在标准数据集上的测试结果：
- **MNIST数据集**: 准确率 ~96.7%
- **自定义图像数据集**: 准确率 ~89.3%
- **计算效率**: 支持批量预测和并行优化

### 参数分析
通过网格搜索找到的最佳参数组合：
- 近邻数 k: 3-5（依赖具体数据集）
- 距离度量: 欧氏距离表现更稳定
- 权重策略: 距离加权在小数据集上效果更好

## 🤝 与大模型协作记录

### 协作流程
1. **需求分析**: 向文心大模型描述算法需求和约束条件
2. **架构设计**: 讨论项目结构和模块划分
3. **代码实现**: 分模块实现，实时验证和调试
4. **文档编写**: 协同撰写技术文档和使用指南

### 关键交互示例
> **提示词**: "请用纯Python和Numpy实现K近邻算法的predict方法，要求支持距离加权投票"
> 
> **文心回复**: [提供了完整的代码实现和数学原理解释]
> 
> **验证过程**: 手动推导距离加权公式，测试边界情况

[查看完整交互记录](./docs/llm_interaction_logs.md)

## 🎓 学习价值

### 能力培养
- ✅ **深度理解**: 掌握算法数学原理和实现细节
- ✅ **工程能力**: 遵循软件工程规范，模块化开发
- ✅ **AI协作**: 学会与LLM高效沟通和协作
- ✅ **问题解决**: 独立调试和优化代码的能力

### 知识体系
1. **数学基础**: 距离度量、概率统计、优化理论
2. **编程技能**: Python高级特性、面向对象设计
3. **工程实践**: 版本控制、测试驱动开发、文档编写
4. **AI应用**: 提示工程、模型评估、结果分析

## 📝 提交要求

### GitHub仓库
- [ ] 完整源代码
- [ ] 详细文档和教程
- [ ] 单元测试覆盖
- [ ] 依赖管理文件
- [ ] LLM交互记录

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

1. **完善K-NN章节**: 添加更多数据集测试和可视化分析
2. **开始Softmax分类器**: 研究线性分类器和损失函数
3. **持续文档更新**: 记录学习心得和问题解决方案
4. **社区互动**: 收集反馈，持续改进项目质量

---

**💡 提示**: 本项目是学习过程的记录，重点在于理解原理和培养能力，而非追求极致的性能指标。

**🌟 让我们一起探索计算机视觉的奥秘，从底层开始，构建坚实的AI基础！**

---
*最后更新: 2025年10月*  
