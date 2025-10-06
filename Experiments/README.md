"""
README.md
项目说明文档 / Project Documentation
"""

# Federated Learning with Hotpot-Style Incentive Mechanism
# 基于火锅会员模式的联邦学习激励机制

## Project Overview / 项目概述

This project implements a federated learning framework with an innovative incentive mechanism inspired by commercial membership systems (specifically Haidilao's hotpot restaurant membership model). The system uses time-slice based point calculation and membership levels to encourage continuous client participation.

本项目实现了一个带有创新激励机制的联邦学习框架，灵感来源于商业会员系统（特别是海底捞火锅餐厅的会员模式）。系统使用基于时间片的积分计算和会员等级来鼓励客户端的持续参与。

## Key Features / 主要特性

1. **Multiple Time-Slice Strategies / 多种时间片策略**
   - Rounds-based / 基于轮次
   - Days-based / 基于天数
   - Phase-based / 基于阶段
   - Dynamic / 动态调整
   - Completion-based / 基于完成度

2. **Membership System / 会员系统**
   - Bronze, Silver, Gold, Diamond levels / 铜、银、金、钻石等级
   - Level-based benefits / 基于等级的权益
   - Points validity period / 积分有效期

3. **Comprehensive Metrics / 全面的评估指标**
   - Model accuracy and loss / 模型准确率和损失
   - Participation rate / 参与率
   - System activity / 系统活跃度
   - Membership distribution / 会员分布

## Installation / 安装

1. Clone the repository / 克隆仓库:
```bash
git clone <repository_url>
cd federated_learning_incentive
```

2. Install dependencies / 安装依赖:
```bash
pip install -r requirements.txt
```

## Usage / 使用方法

### Basic Training / 基础训练

Run a single experiment with default settings:
```bash
python main.py --dataset cifar10 --num_clients 100 --num_rounds 100
```

### Compare Time-Slice Methods / 比较时间片方法

Compare different time-slice strategies:
```bash
python main.py --dataset cifar10 --compare_methods
```

### Custom Configuration / 自定义配置

```bash
python main.py \
    --dataset fashion-mnist \
    --num_clients 50 \
    --num_rounds 80 \
    --time_slice_type dynamic \
    --distribution non-iid \
    --alpha 0.5 \
    --experiment_name my_experiment
```

## Command Line Arguments / 命令行参数

- `--dataset`: Dataset to use (mnist, fashion-mnist, cifar10, cifar100)
- `--num_clients`: Number of clients in the federation
- `--num_rounds`: Number of training rounds
- `--clients_per_round`: Number of clients selected per round
- `--time_slice_type`: Type of time slice strategy (rounds, days, phases, dynamic, completion)
- `--distribution`: Data distribution type (iid, non-iid)
- `--alpha`: Dirichlet distribution parameter for non-iid setting
- `--experiment_name`: Name for the experiment
- `--num_runs`: Number of experiment repetitions
- `--compare_methods`: Compare different time slice methods
- `--seed`: Random seed for reproducibility
- `--device`: Device to use (auto, cpu, cuda)

## Project Structure / 项目结构

```
federated_learning_incentive/
├── main.py                    # Main entry point
├── config.py                  # Configuration file
├── datasets/                  # Data loading and preprocessing
│   └── data_loader.py
├── models/                    # Neural network models
│   └── cnn_model.py
├── federated/                 # Federated learning components
│   ├── client.py
│   └── server.py
├── incentive/                 # Incentive mechanism
│   ├── membership.py
│   ├── points_calculator.py
│   └── time_slice.py
├── utils/                     # Utilities
│   ├── metrics.py
│   └── visualization.py
└── experiments/               # Experiment management
    └── run_experiments.py
```

## Output / 输出

The system generates several types of output:

1. **Checkpoints / 检查点**: Saved in `checkpoints/` directory
2. **Visualizations / 可视化图表**: Saved in `results/plots/` directory
3. **Results JSON / 结果JSON**: Saved in `results/` directory
4. **Logs / 日志**: Saved in `outputs/logs/` directory

## Visualization Examples / 可视化示例

The system generates various plots including:
- Training curves (accuracy, loss)
- Participation rate over time
- System activity metrics
- Membership level distribution
- Points distribution analysis
- Comparison charts for different methods

## Research Background / 研究背景

This implementation is based on the paper "The Secret Sauce: How Hotpot Membership Schemes Can Spice Up Federated Learning Incentive Mechanisms", which proposes using commercial membership systems as inspiration for federated learning incentive design.

## Citation / 引用

If you use this code in your research, please cite:
```bibtex
@article{hotpot_fl_2024,
  title={The Secret Sauce: How Hotpot Membership Schemes Can Spice Up Federated Learning Incentive Mechanisms},
  author={Your Name},
  year={2024}
}
```

## License / 许可证

This project is licensed under the MIT License.

## Contact / 联系方式

For questions or issues, please open an issue on GitHub or contact the maintainers.