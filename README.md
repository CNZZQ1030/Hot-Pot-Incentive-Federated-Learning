# Federated Learning with Differentiated Model Rewards
# 基于差异化模型奖励的联邦学习激励机制

## 📋 目录 / Table of Contents

1. [项目概述](#项目概述--project-overview)
2. [核心创新](#核心创新--core-innovation)
3. [实验设计思路](#实验设计思路--experiment-design)
4. [安装说明](#安装说明--installation)
5. [快速开始](#快速开始--quick-start)
6. [实验执行流程](#实验执行流程--experiment-workflow)
7. [命令行参数](#命令行参数--command-line-arguments)
8. [实验结果分析](#实验结果分析--results-analysis)
9. [项目结构](#项目结构--project-structure)
10. [扩展与定制](#扩展与定制--customization)

---

## 项目概述 / Project Overview

本项目实现了一个**创新的联邦学习激励机制**，灵感来源于商业会员系统（特别是海底捞火锅餐厅的会员模式）。核心创新是**差异化模型奖励机制**：不同等级的客户端根据其贡献获得不同质量的全局模型，形成"贡献越大，模型越好"的正向激励循环。

This project implements an **innovative federated learning incentive mechanism** inspired by commercial membership systems (specifically Haidilao's hotpot restaurant membership model). The core innovation is a **differentiated model reward mechanism**: clients of different levels receive global models of varying quality based on their contributions, creating a positive incentive cycle of "greater contribution leads to better models."

### 主要特点 / Key Features

✅ **差异化模型奖励** - 高等级客户端获得更优质的聚合模型  
✅ **会员等级系统** - 铜、银、金、钻石四级体系  
✅ **时间片积分管理** - 5种时间片策略，积分有效期机制  
✅ **公平性保障** - 基于数据量、计算时间和模型质量的多维度评估  
✅ **完整可视化** - 训练曲线、等级分布、质量差距等多维度分析  

---

## 核心创新 / Core Innovation

### 🎯 差异化模型奖励机制

传统联邦学习中，所有客户端在每轮训练后都接收相同的全局模型。本项目创新性地引入**分层模型分发策略**：

```
服务器聚合阶段：
├── 收集所有客户端的模型更新
├── 创建标准全局模型（FedAvg）
└── 根据客户端等级创建差异化模型：
    ├── 钻石级模型：使用高质量客户端，权重×3.0
    ├── 金级模型：使用优质客户端，权重×2.0  
    ├── 银级模型：标准权重聚合
    └── 铜级模型：基础全局模型

客户端训练阶段：
├── 钻石级客户端 → 接收钻石级模型（最优）
├── 金级客户端 → 接收金级模型（优质）
├── 银级客户端 → 接收银级模型（标准）
└── 铜级客户端 → 接收铜级模型（基础）
```

### 💡 激励原理

1. **正向循环**：客户端贡献更多 → 积分增加 → 等级提升 → 获得更好模型 → 训练效果提升 → 更愿意贡献
2. **公平性**：贡献与回报匹配，避免"搭便车"问题
3. **可持续性**：积分有效期机制鼓励持续参与

---

## 实验设计思路 / Experiment Design

### 📊 实验目标

验证差异化模型奖励机制能够：
1. **提高系统整体性能** - 相比标准激励机制，全局模型准确率更高
2. **增强客户端参与度** - 高等级客户端参与率和活跃度提升
3. **加快模型收敛速度** - 达到目标准确率所需轮次减少
4. **形成明显的质量差异** - 不同等级模型之间存在可观测的性能差距

### 🔬 实验设置

#### 基础配置
- **客户端数量**：100个
- **每轮选择**：10个客户端
- **训练轮次**：100轮
- **本地训练**：5个epochs
- **数据集**：CIFAR-10, MNIST, Fashion-MNIST

#### 会员等级设置
| 等级 | 积分阈值 | 权益倍数 | 模型质量期望 |
|------|---------|---------|-------------|
| 钻石 (Diamond) | 15,000+ | 2.0× | 最优 (100%) |
| 金 (Gold) | 6,000+ | 1.5× | 优质 (75%) |
| 银 (Silver) | 2,000+ | 1.2× | 标准 (50%) |
| 铜 (Bronze) | 0+ | 1.0× | 基础 (30%) |

#### 差异化聚合策略

**加权策略 (Weighted)** - 推荐使用：
```python
钻石级模型权重分配：
- 钻石级客户端：3.0×
- 金级客户端：2.0×
- 银级客户端：1.0×
- 铜级客户端：0.5×
```

**严格策略 (Strict)**：
- 钻石级模型：仅使用钻石+金级客户端
- 金级模型：使用钻石+金+银级客户端
- 银/铜级模型：使用全局模型

### 📈 评估指标

1. **模型性能指标**
   - 全局模型准确率和损失
   - 各等级模型准确率
   - 收敛速度（达到95%最优性能的轮次）

2. **激励效果指标**
   - 客户端参与率
   - 系统活跃度
   - 会员等级分布演变

3. **差异化效果指标**
   - 质量差距（钻石级-铜级准确率差）
   - 等级提升速率
   - 高等级客户端留存率

---

## 安装说明 / Installation

### 环境要求

- Python 3.7+
- PyTorch 1.9+
- CUDA (可选，用于GPU加速)

### 安装步骤

1. **克隆仓库** / Clone repository:
```bash
git clone <repository_url>
cd federated_learning_incentive
```

2. **安装依赖** / Install dependencies:
```bash
pip install -r requirements.txt
```

3. **验证安装** / Verify installation:
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torchvision; print('Installation successful!')"
```

---

## 快速开始 / Quick Start

### 🚀 运行第一个实验

**实验1：标准激励机制（基线）**
```bash
python main.py \
    --dataset cifar10 \
    --num_clients 100 \
    --num_rounds 100 \
    --distribution iid \
    --experiment_name baseline_standard
```

**实验2：差异化模型奖励机制**
```bash
python main.py \
    --dataset cifar10 \
    --num_clients 100 \
    --num_rounds 100 \
    --distribution iid \
    --enable_tiered_rewards \
    --experiment_name baseline_tiered
```

**实验3：自动对比两种机制**
```bash
python main.py \
    --dataset cifar10 \
    --num_clients 100 \
    --num_rounds 100 \
    --compare_rewards
```

### 📊 查看结果

实验完成后，结果保存在以下位置：

```
outputs/
├── results/
│   ├── {experiment_name}_results.json      # 详细实验数据
│   └── plots/
│       ├── {experiment_name}_training_curves.png      # 训练曲线
│       ├── {experiment_name}_level_distribution.png   # 等级分布
│       ├── {experiment_name}_points_distribution.png  # 积分分布
│       └── {experiment_name}_summary_report.png       # 总结报告
├── checkpoints/
│   └── {experiment_name}_round_*.pt        # 模型检查点
└── logs/
    └── experiment.log                       # 运行日志
```

---

## 实验执行流程 / Experiment Workflow

### 完整实验流程图

```
┌─────────────────────────────────────────────────────────┐
│ 1. 初始化阶段 / Initialization Phase                     │
├─────────────────────────────────────────────────────────┤
│ • 加载数据集并分配给各客户端                               │
│ • 初始化全局模型                                          │
│ • 创建会员系统（所有客户端初始为铜级）                      │
│ • 设置时间片管理器和积分计算器                             │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│ 2. 训练循环 (重复100轮) / Training Loop (100 rounds)      │
└─────────────────────────────────────────────────────────┘
  ↓
  ├─→ 2.1 客户端选择 / Client Selection
  │   ├─ 获取所有客户端的会员等级和积分
  │   ├─ 计算优先级（高等级客户端有优先选择权）
  │   └─ 选择10个客户端参与本轮训练
  │
  ├─→ 2.2 模型分发 / Model Distribution
  │   ├─ 如果启用差异化奖励：
  │   │   ├─ 钻石级客户端 → 获取钻石级模型
  │   │   ├─ 金级客户端 → 获取金级模型
  │   │   ├─ 银级客户端 → 获取银级模型
  │   │   └─ 铜级客户端 → 获取铜级模型
  │   └─ 否则：所有客户端获取相同的全局模型
  │
  ├─→ 2.3 本地训练 / Local Training
  │   ├─ 每个客户端在本地数据上训练5个epochs
  │   ├─ 记录训练时间、损失、准确率
  │   └─ 计算模型质量 = 1/(1+loss)
  │
  ├─→ 2.4 积分计算 / Points Calculation
  │   ├─ 基础积分 = 0.3×数据量 + 0.3×计算时间 + 0.4×模型质量
  │   ├─ 应用等级倍数（钻石2.0×，金1.5×，银1.2×，铜1.0×）
  │   └─ 更新时间片积分
  │
  ├─→ 2.5 等级更新 / Level Update
  │   ├─ 计算有效积分（最近10个时间片内的积分）
  │   ├─ 根据积分阈值确定新等级
  │   └─ 记录等级变化历史
  │
  ├─→ 2.6 模型聚合 / Model Aggregation
  │   ├─ 标准聚合：创建全局模型（FedAvg或加权）
  │   └─ 如果启用差异化奖励：
  │       ├─ 创建钻石级模型（高质量客户端权重×3.0）
  │       ├─ 创建金级模型（优质客户端权重×2.0）
  │       ├─ 创建银级模型（标准权重）
  │       └─ 创建铜级模型（基础全局模型）
  │
  ├─→ 2.7 模型评估 / Model Evaluation
  │   ├─ 在测试集上评估全局模型
  │   ├─ 如果启用差异化奖励：评估各等级模型
  │   ├─ 计算质量差距（钻石级-铜级）
  │   └─ 记录所有性能指标
  │
  └─→ 2.8 指标更新 / Metrics Update
      ├─ 更新参与率、系统活跃度
      ├─ 记录会员等级分布
      └─ 清理过期积分（每10轮）
                            ↓
┌─────────────────────────────────────────────────────────┐
│ 3. 结果分析阶段 / Results Analysis Phase                  │
├─────────────────────────────────────────────────────────┤
│ • 生成训练曲线图                                          │
│ • 绘制等级分布演变图                                       │
│ • 分析质量差距趋势                                        │
│ • 创建总结报告                                            │
│ • 保存模型检查点和实验数据                                 │
└─────────────────────────────────────────────────────────┘
```

### 关键流程说明

#### 🔑 差异化模型创建流程

```python
# 伪代码示例
def create_tiered_models(client_weights, client_infos):
    """
    创建分层模型的核心逻辑
    """
    # 1. 按等级分组客户端
    diamond_clients = [id for id, info in client_infos.items() 
                      if info['level'] == 'diamond']
    gold_clients = [id for id, info in client_infos.items() 
                   if info['level'] == 'gold']
    # ... 其他等级
    
    # 2. 为钻石级创建最优模型
    diamond_model = aggregate_with_weights(
        client_weights,
        level_weights={'diamond': 3.0, 'gold': 2.0, 
                      'silver': 1.0, 'bronze': 0.5}
    )
    
    # 3. 为其他等级创建相应质量的模型
    # ...
    
    return tiered_models
```

---

## 命令行参数 / Command Line Arguments

### 基础参数 / Basic Parameters

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--dataset` | str | cifar10 | 数据集选择：mnist, fashion-mnist, cifar10, cifar100 |
| `--num_clients` | int | 100 | 客户端总数 |
| `--num_rounds` | int | 100 | 训练轮次 |
| `--clients_per_round` | int | 10 | 每轮选择的客户端数 |

### 激励机制参数 / Incentive Parameters

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--time_slice_type` | str | rounds | 时间片类型：rounds, days, phases, dynamic, completion |
| `--rounds_per_slice` | int | 10 | 每个时间片的轮次数 |
| `--enable_tiered_rewards` | flag | False | 启用差异化模型奖励 |
| `--tiered_strategy` | str | weighted | 分层策略：weighted（加权）或 strict（严格）|

### 数据分布参数 / Data Distribution Parameters

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--distribution` | str | iid | 数据分布：iid（独立同分布）或 non-iid |
| `--alpha` | float | 0.5 | Non-IID的Dirichlet分布参数（越小越不均衡）|

### 实验参数 / Experiment Parameters

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--experiment_name` | str | auto | 实验名称（自动生成包含时间戳）|
| `--num_runs` | int | 1 | 实验重复次数 |
| `--compare_methods` | flag | False | 对比不同时间片方法 |
| `--compare_rewards` | flag | False | 对比标准vs差异化奖励 |
| `--seed` | int | 42 | 随机种子 |
| `--device` | str | auto | 计算设备：auto, cpu, cuda |

---

## 完整实验命令示例 / Complete Command Examples

### 实验场景1：基础对比实验

**目标**：对比标准激励 vs 差异化奖励的效果

```bash
# 自动运行两种机制并生成对比报告
python main.py \
    --dataset cifar10 \
    --num_clients 100 \
    --num_rounds 100 \
    --clients_per_round 10 \
    --distribution iid \
    --compare_rewards
```

**预期输出**：
```
REWARDS COMPARISON SUMMARY
======================================================================
Metric                         Standard        Differentiated  Improvement    
----------------------------------------------------------------------
Final Accuracy                 0.8542          0.8891          +4.08%
Average Accuracy               0.7923          0.8234          +3.93%
Final Loss                     0.4521          0.3892          -13.91%
Participation Rate             0.7800          0.8400          +7.69%
System Activity                0.7234          0.7891          +9.08%
Convergence Round              72              61              -15.28%

Model Quality Gap (Diamond-Bronze): 0.0847
======================================================================
```

### 实验场景2：Non-IID数据分布

**目标**：在更现实的非均衡数据分布下测试机制

```bash
python main.py \
    --dataset cifar10 \
    --num_clients 100 \
    --num_rounds 100 \
    --distribution non-iid \
    --alpha 0.3 \
    --enable_tiered_rewards \
    --tiered_strategy weighted \
    --experiment_name noniid_tiered
```

### 实验场景3：不同数据集测试

```bash
# MNIST数据集（简单）
python main.py --dataset mnist --enable_tiered_rewards --num_rounds 50

# Fashion-MNIST数据集（中等难度）
python main.py --dataset fashion-mnist --enable_tiered_rewards --num_rounds 80

# CIFAR-10数据集（困难）
python main.py --dataset cifar10 --enable_tiered_rewards --num_rounds 100
```

### 实验场景4：严格分层策略

**目标**：测试更严格的等级隔离效果

```bash
python main.py \
    --dataset cifar10 \
    --enable_tiered_rewards \
    --tiered_strategy strict \
    --experiment_name strict_strategy_test
```

### 实验场景5：时间片方法对比

```bash
python main.py \
    --dataset cifar10 \
    --enable_tiered_rewards \
    --compare_methods
```

---

## 实验结果分析 / Results Analysis

### 📈 生成的可视化图表

#### 1. 训练曲线图 (training_curves.png)
- **准确率曲线**：展示全局模型准确率随轮次的变化
- **损失曲线**：训练损失的收敛过程
- **参与率曲线**：客户端参与率的演变
- **系统活跃度曲线**：整体系统健康度指标

#### 2. 等级分布图 (level_distribution.png)
- **堆叠面积图**：展示各等级客户端数量随时间的演变
- **颜色编码**：
  - 🟤 铜级 (Bronze)
  - ⚪ 银级 (Silver)
  - 🟡 金级 (Gold)
  - 🔵 钻石级 (Diamond)

#### 3. 积分分布图 (points_distribution.png)
- **直方图**：客户端积分的分布情况
- **箱线图**：识别异常值和分布特征

#### 4. 总结报告 (summary_report.png)
- 最终性能指标对比
- 收敛速度信息
- 详细统计表格

### 📊 结果JSON文件结构

```json
{
  "experiment_name": "FL_cifar10_tiered_rounds_20241006_143022",
  "config": {
    "dataset": "cifar10",
    "num_clients": 100,
    "num_rounds": 100,
    "enable_tiered_rewards": true
  },
  "final_accuracy_mean": 0.8891,
  "final_accuracy_std": 0.0023,
  "final_loss": 0.3892,
  "avg_participation_rate": 0.8400,
  "avg_system_activity": 0.7891,
  "convergence_round":