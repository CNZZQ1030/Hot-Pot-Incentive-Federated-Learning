"""
main.py
主程序入口 / Main Program Entry
"""

import torch
import random
import numpy as np
import argparse
import json
import os
from datetime import datetime

# 导入配置 / Import configurations
from config import (
    FederatedConfig, IncentiveConfig, DatasetConfig, 
    ModelConfig, ExperimentConfig, DEVICE, SEED
)

# 导入实验模块 / Import experiment module
from experiments.run_experiments import ExperimentRunner


def set_seed(seed: int):
    """设置随机种子 / Set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_arguments():
    """解析命令行参数 / Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Federated Learning with Incentive Mechanism')
    
    # 基本参数 / Basic parameters
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['mnist', 'fashion-mnist', 'cifar10', 'cifar100'],
                       help='Dataset to use')
    parser.add_argument('--num_clients', type=int, default=100,
                       help='Number of clients')
    parser.add_argument('--num_rounds', type=int, default=100,
                       help='Number of training rounds')
    parser.add_argument('--clients_per_round', type=int, default=10,
                       help='Number of clients selected per round')
    
    # 激励机制参数 / Incentive mechanism parameters
    parser.add_argument('--time_slice_type', type=str, default='rounds',
                       choices=['rounds', 'days', 'phases', 'dynamic', 'completion'],
                       help='Type of time slice strategy')
    parser.add_argument('--rounds_per_slice', type=int, default=10,
                       help='Rounds per time slice')
    
    # 数据分布参数 / Data distribution parameters
    parser.add_argument('--distribution', type=str, default='iid',
                       choices=['iid', 'non-iid'],
                       help='Data distribution type')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Dirichlet distribution parameter for non-iid')
    
    # 实验参数 / Experiment parameters
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Name of the experiment')
    parser.add_argument('--num_runs', type=int, default=1,
                       help='Number of experiment runs')
    parser.add_argument('--compare_methods', action='store_true',
                       help='Compare different time slice methods')
    
    # 其他参数 / Other parameters
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use')
    
    return parser.parse_args()


def main():
    """主函数 / Main function"""
    # 解析参数 / Parse arguments
    args = parse_arguments()
    
    # 设置随机种子 / Set random seed
    set_seed(args.seed)
    
    # 设置设备 / Set device
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # 更新配置 / Update configurations
    FederatedConfig.NUM_CLIENTS = args.num_clients
    FederatedConfig.NUM_ROUNDS = args.num_rounds
    FederatedConfig.CLIENTS_PER_ROUND = args.clients_per_round
    FederatedConfig.DATA_DISTRIBUTION = args.distribution
    FederatedConfig.NON_IID_ALPHA = args.alpha
    
    DatasetConfig.DATASET_NAME = args.dataset
    
    IncentiveConfig.TIME_SLICE_TYPE = args.time_slice_type
    IncentiveConfig.ROUNDS_PER_SLICE = args.rounds_per_slice
    
    # 设置实验名称 / Set experiment name
    if args.experiment_name is None:
        args.experiment_name = f"FL_{args.dataset}_{args.time_slice_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"Starting experiment: {args.experiment_name}")
    print(f"Configuration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Clients: {args.num_clients}")
    print(f"  Rounds: {args.num_rounds}")
    print(f"  Distribution: {args.distribution}")
    print(f"  Time Slice Type: {args.time_slice_type}")
    
    # 创建实验运行器 / Create experiment runner
    runner = ExperimentRunner(
        dataset_name=args.dataset,
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        clients_per_round=args.clients_per_round,
        time_slice_type=args.time_slice_type,
        distribution=args.distribution,
        device=device
    )
    
    if args.compare_methods:
        # 比较不同的时间片方法 / Compare different time slice methods
        print("\nComparing different time slice methods...")
        results = runner.compare_time_slice_methods()
        
        # 保存比较结果 / Save comparison results
        results_path = f"results/comparison_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs("results", exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Comparison results saved to {results_path}")
        
    else:
        # 运行单个实验 / Run single experiment
        print(f"\nRunning experiment with {args.time_slice_type} time slice...")
        results = runner.run_single_experiment(
            experiment_name=args.experiment_name,
            num_runs=args.num_runs
        )
        
        # 保存结果 / Save results
        results_path = f"results/{args.experiment_name}_results.json"
        os.makedirs("results", exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_path}")
        
        # 打印结果摘要 / Print results summary
        print("\n" + "="*50)
        print("EXPERIMENT RESULTS SUMMARY")
        print("="*50)
        print(f"Final Accuracy: {results['accuracy_final']:.4f}")
        print(f"Final Loss: {results['loss_final']:.4f}")
        print(f"Average Participation Rate: {results['participation_rate_avg']:.4f}")
        print(f"Average System Activity: {results['system_activity_avg']:.4f}")
        print(f"Convergence Round: {results.get('convergence_round', 'N/A')}")
        print("="*50)


if __name__ == "__main__":
    main()