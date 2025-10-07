"""
experiments/run_experiments.py
实验运行模块 / Experiment Running Module
完整集成差异化模型奖励机制
Complete integration of differentiated model reward mechanism
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
import copy
from tqdm import tqdm

# 导入项目模块 / Import project modules
from datasets.data_loader import FederatedDataLoader
from models.cnn_model import ModelFactory
from federated.client import FederatedClient
from federated.server import FederatedServer
from incentive.membership import MembershipSystem
from incentive.points_calculator import PointsCalculator
from incentive.time_slice import TimeSliceManager
from utils.metrics import MetricsCalculator
from utils.visualization import Visualizer
from config import IncentiveConfig, DatasetConfig


class ExperimentRunner:
    """
    实验运行器 / Experiment Runner
    负责运行和管理联邦学习实验
    Responsible for running and managing federated learning experiments
    """
    
    def __init__(self, dataset_name: str, num_clients: int, num_rounds: int,
                 clients_per_round: int, time_slice_type: str,
                 distribution: str = "iid", device: torch.device = torch.device("cpu")):
        """
        初始化实验运行器 / Initialize experiment runner
        
        Args:
            dataset_name: 数据集名称 / Dataset name
            num_clients: 客户端数量 / Number of clients
            num_rounds: 训练轮次 / Number of training rounds
            clients_per_round: 每轮选择的客户端数 / Clients per round
            time_slice_type: 时间片类型 / Time slice type
            distribution: 数据分布 / Data distribution
            device: 计算设备 / Computing device
        """
        self.dataset_name = dataset_name
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.clients_per_round = clients_per_round
        self.time_slice_type = time_slice_type
        self.distribution = distribution
        self.device = device
        
        # 初始化组件 / Initialize components
        self._initialize_components()
        
    def _initialize_components(self):
        """初始化实验组件 / Initialize experiment components"""
        # 加载数据 / Load data
        self.data_loader = FederatedDataLoader(
            dataset_name=self.dataset_name,
            num_clients=self.num_clients,
            batch_size=32,
            distribution=self.distribution
        )
        
        # 创建模型 / Create model
        num_classes = DatasetConfig.NUM_CLASSES[self.dataset_name]
        self.model = ModelFactory.create_model(self.dataset_name, num_classes)
        
        # 创建服务器 / Create server
        self.server = FederatedServer(self.model, self.device)
        
        # 创建客户端 / Create clients
        self.clients = {}
        for client_id in range(self.num_clients):
            client_dataloader = self.data_loader.get_client_dataloader(client_id)
            self.clients[client_id] = FederatedClient(
                client_id=client_id,
                model=copy.deepcopy(self.model),
                dataloader=client_dataloader,
                device=self.device
            )
        
        # 初始化激励系统 / Initialize incentive system
        self.membership_system = MembershipSystem(
            level_thresholds=IncentiveConfig.LEVEL_THRESHOLDS,
            level_multipliers=IncentiveConfig.LEVEL_MULTIPLIERS
        )
        
        self.points_calculator = PointsCalculator(
            alpha=IncentiveConfig.ALPHA,
            beta=IncentiveConfig.BETA,
            gamma=IncentiveConfig.GAMMA
        )
        
        self.time_slice_manager = TimeSliceManager(
            slice_type=self.time_slice_type,
            rounds_per_slice=IncentiveConfig.ROUNDS_PER_SLICE,
            days_per_slice=IncentiveConfig.DAYS_PER_SLICE,
            validity_slices=IncentiveConfig.POINTS_VALIDITY_SLICES
        )
        
        # 初始化指标计算器和可视化器 / Initialize metrics calculator and visualizer
        self.metrics_calculator = MetricsCalculator()
        self.visualizer = Visualizer()
        
    def select_clients(self, round_num: int) -> List[int]:
        """
        选择客户端参与训练 / Select clients for training
        
        Args:
            round_num: 当前轮次 / Current round
            
        Returns:
            选中的客户端ID列表 / List of selected client IDs
        """
        # 获取客户端的会员等级和积分 / Get client membership levels and points
        client_priorities = []
        
        for client_id in range(self.num_clients):
            membership_info = self.membership_system.get_client_membership_info(client_id)
            level = membership_info['level']
            points = membership_info['total_points']
            
            # 计算优先级（考虑等级权益） / Calculate priority (considering level benefits)
            benefits = self.membership_system.get_level_benefits(level)
            priority = points * benefits['points_multiplier']
            
            # 金钻客户端有优先选择权 / Gold and diamond clients have priority
            if benefits['priority_selection']:
                priority *= 2
            
            client_priorities.append((client_id, priority))
        
        # 按优先级排序并添加随机性 / Sort by priority and add randomness
        client_priorities.sort(key=lambda x: x[1], reverse=True)
        
        # 优先选择高等级客户端，但保持一定随机性 / Prioritize high-level clients with some randomness
        selected_clients = []
        
        # 先选择高优先级客户端（前20%） / First select high priority clients (top 20%)
        high_priority_count = min(self.clients_per_round // 2, len(client_priorities) // 5)
        for i in range(high_priority_count):
            selected_clients.append(client_priorities[i][0])
        
        # 剩余名额随机选择 / Random selection for remaining slots
        remaining_clients = [c[0] for c in client_priorities[high_priority_count:]]
        remaining_slots = self.clients_per_round - len(selected_clients)
        
        if remaining_slots > 0 and remaining_clients:
            random_selected = np.random.choice(
                remaining_clients, 
                min(remaining_slots, len(remaining_clients)),
                replace=False
            )
            selected_clients.extend(random_selected.tolist())
        
        return selected_clients
    
    def run_single_round(self, round_num: int) -> Tuple[float, float]:
        """
        运行单轮训练 / Run single training round
        
        Args:
            round_num: 轮次号 / Round number
            
        Returns:
            准确率和损失 / Accuracy and loss
        """
        print(f"\n--- Round {round_num}/{self.num_rounds} ---")
        
        # 选择客户端 / Select clients
        selected_clients = self.select_clients(round_num)
        print(f"Selected clients: {selected_clients}")
        
        # 客户端训练 / Client training
        client_weights = {}
        client_infos = {}
        
        for client_id in tqdm(selected_clients, desc="Training clients"):
            client = self.clients[client_id]
            
            # 获取客户端当前等级 / Get client current level
            membership_info = self.membership_system.get_client_membership_info(client_id)
            current_level = membership_info['level']
            
            # 根据等级获取对应的模型权重 / Get model weights based on level
            if IncentiveConfig.ENABLE_TIERED_REWARDS:
                global_weights = self.server.get_tiered_model_weights(current_level)
                print(f"  Client {client_id} (Level: {current_level}) receives {current_level}-tier model")
            else:
                global_weights = self.server.get_global_weights()
            
            # 本地训练 / Local training
            updated_weights, train_info = client.train(
                global_weights=global_weights,
                epochs=5,
                lr=0.01
            )
            
            # 计算积分 / Calculate points
            level_multiplier = self.membership_system.level_multipliers[current_level]
            
            points = self.points_calculator.calculate_points(
                data_size=train_info['data_size'],
                computation_time=train_info['computation_time'],
                model_quality=train_info['model_quality'],
                level_multiplier=level_multiplier
            )
            
            # 更新时间片积分 / Update time slice points
            self.time_slice_manager.update_client_slice_points(client_id, round_num, points)
            
            # 获取有效积分 / Get active points
            active_points = self.time_slice_manager.get_active_points(client_id, round_num)
            
            # 更新会员等级 / Update membership level
            new_level = self.membership_system.update_membership_level(client_id, active_points)
            
            # 记录信息 / Record information
            train_info['membership_level'] = new_level
            client_weights[client_id] = updated_weights
            client_infos[client_id] = train_info
            
            # 更新客户端参与信息 / Update client participation info
            client.update_participation(round_num, active_points, new_level)
        
        # 服务器聚合 / Server aggregation
        self.server.aggregate_models(client_weights, client_infos, "weighted")
        
        # 评估全局模型 / Evaluate global model
        test_loader = self.data_loader.get_test_dataloader()
        accuracy, loss = self.server.evaluate_global_model(test_loader)
        
        print(f"Round {round_num} - Global Model - Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
        
        # 如果启用差异化奖励，评估各等级模型 / If tiered rewards enabled, evaluate tiered models
        if IncentiveConfig.ENABLE_TIERED_REWARDS:
            tiered_results = self.server.evaluate_tiered_models(test_loader)
            print(f"Tiered Model Performance:")
            for level, (acc, lss) in tiered_results.items():
                print(f"  {level.capitalize()}: Accuracy={acc:.4f}, Loss={lss:.4f}")
        
        # 计算指标 / Calculate metrics
        participation_rate = len(selected_clients) / self.num_clients
        
        # 计算客户端参与度 / Calculate client participation
        client_participations = {}
        for client_id in selected_clients:
            client = self.clients[client_id]
            participation = self.metrics_calculator.calculate_client_participation(
                client_id=client_id,
                rounds_participated=len(client.participation_rounds),
                total_rounds=round_num,
                data_contribution=client_infos[client_id]['data_size'] / 10000,
                computation_time=client_infos[client_id]['computation_time'] / 100,
                model_quality=client_infos[client_id]['model_quality']
            )
            client_participations[client_id] = participation
        
        # 计算系统活跃度 / Calculate system activity
        system_activity = self.metrics_calculator.calculate_system_activity(
            client_participations=client_participations,
            active_clients=len(selected_clients),
            total_clients=self.num_clients
        )
        
        # 获取会员统计 / Get membership statistics
        membership_stats = self.membership_system.get_membership_statistics()
        
        # 更新指标历史 / Update metrics history
        self.metrics_calculator.update_metrics(
            round_num=round_num,
            accuracy=accuracy,
            loss=loss,
            participation_rate=participation_rate,
            system_activity=system_activity,
            level_distribution=membership_stats['level_distribution'],
            points_stats={'avg_points': membership_stats['average_points']}
        )
        
        # 如果启用差异化奖励，记录模型质量差距 / If tiered rewards enabled, record quality gaps
        if IncentiveConfig.ENABLE_TIERED_REWARDS and tiered_results:
            diamond_acc = tiered_results.get('diamond', (0, 0))[0]
            bronze_acc = tiered_results.get('bronze', (0, 0))[0]
            quality_gap = diamond_acc - bronze_acc
            self.metrics_calculator.record_quality_gap(quality_gap)
        
        # 清理过期积分 / Clean expired points
        if round_num % 10 == 0:
            self.time_slice_manager.clean_expired_points(round_num)
        
        return accuracy, loss
    
    def run_single_experiment(self, experiment_name: str, 
                             num_runs: int = 1) -> Dict:
        """
        运行单个实验 / Run single experiment
        
        Args:
            experiment_name: 实验名称 / Experiment name
            num_runs: 运行次数 / Number of runs
            
        Returns:
            实验结果 / Experiment results
        """
        all_results = []
        
        for run in range(num_runs):
            print(f"\n{'='*50}")
            print(f"Run {run + 1}/{num_runs}")
            print('='*50)
            
            # 重新初始化组件 / Reinitialize components
            if run > 0:
                self._initialize_components()
            
            # 运行所有轮次 / Run all rounds
            for round_num in range(1, self.num_rounds + 1):
                accuracy, loss = self.run_single_round(round_num)
                
                # 定期保存检查点 / Periodically save checkpoints
                if round_num % 10 == 0:
                    checkpoint_path = f"checkpoints/{experiment_name}_round_{round_num}.pt"
                    self.server.save_checkpoint(round_num, checkpoint_path)
            
            # 收集结果 / Collect results
            metrics_summary = self.metrics_calculator.get_metrics_summary()
            metrics_summary['convergence_round'] = self.metrics_calculator.calculate_convergence_round()
            all_results.append(metrics_summary)
            
            # 生成可视化 / Generate visualizations
            self.visualizer.plot_training_curves(
                self.metrics_calculator.metrics_history,
                f"{experiment_name}_run_{run + 1}"
            )
            
            self.visualizer.plot_level_distribution(
                self.metrics_calculator.metrics_history['level_distribution'],
                f"{experiment_name}_run_{run + 1}"
            )
            
            # 绘制积分分布 / Plot points distribution
            final_points = {}
            for client_id in range(self.num_clients):
                membership_info = self.membership_system.get_client_membership_info(client_id)
                final_points[client_id] = membership_info['total_points']
            
            self.visualizer.plot_points_distribution(
                final_points,
                f"{experiment_name}_run_{run + 1}"
            )
            
            # 创建总结报告 / Create summary report
            self.visualizer.create_summary_report(
                metrics_summary,
                f"{experiment_name}_run_{run + 1}"
            )
        
        # 汇总多次运行的结果 / Aggregate results from multiple runs
        final_results = self._aggregate_run_results(all_results)
        final_results['experiment_name'] = experiment_name
        final_results['num_runs'] = num_runs
        
        return final_results
    
    def _aggregate_run_results(self, all_results: List[Dict]) -> Dict:
        """
        汇总多次运行的结果 / Aggregate results from multiple runs
        
        Args:
            all_results: 所有运行的结果 / Results from all runs
            
        Returns:
            汇总结果 / Aggregated results
        """
        aggregated = {}
        
        # 计算所有指标的平均值和标准差 / Calculate mean and std for all metrics
        metrics = ['accuracy_final', 'accuracy_avg', 'loss_final', 'loss_avg',
                  'participation_rate_final', 'participation_rate_avg',
                  'system_activity_final', 'system_activity_avg', 'convergence_round']
        
        # 添加质量差距指标 / Add quality gap metrics
        if IncentiveConfig.ENABLE_TIERED_REWARDS:
            metrics.extend(['quality_gap_final', 'quality_gap_avg', 'quality_gap_max'])
        
        for metric in metrics:
            values = [r.get(metric, 0) for r in all_results if metric in r]
            if values:
                aggregated[metric] = np.mean(values)
                aggregated[f'{metric}_std'] = np.std(values)
        
        return aggregated
    
    def compare_time_slice_methods(self) -> Dict:
        """
        比较不同的时间片方法 / Compare different time slice methods
        
        Returns:
            比较结果 / Comparison results
        """
        time_slice_types = ['rounds', 'days', 'phases', 'dynamic', 'completion']
        comparison_results = {}
        experiments_data = {}
        
        for ts_type in time_slice_types:
            print(f"\n{'='*60}")
            print(f"Testing time slice type: {ts_type}")
            print('='*60)
            
            # 更新时间片类型 / Update time slice type
            self.time_slice_type = ts_type
            self._initialize_components()
            
            # 运行实验 / Run experiment
            experiment_name = f"comparison_{ts_type}"
            results = self.run_single_experiment(experiment_name, num_runs=1)
            
            comparison_results[ts_type] = results
            experiments_data[ts_type] = self.metrics_calculator.metrics_history
        
        # 生成比较图表 / Generate comparison charts
        self.visualizer.plot_comparison(experiments_data, 'accuracy')
        self.visualizer.plot_comparison(experiments_data, 'participation_rate')
        self.visualizer.plot_comparison(experiments_data, 'system_activity')
        
        # 创建比较表格 / Create comparison table
        print("\n" + "="*80)
        print("TIME SLICE METHODS COMPARISON")
        print("="*80)
        print(f"{'Method':<15} {'Final Acc':<12} {'Conv. Round':<12} {'Avg Part.':<12} {'Avg Activity':<12}")
        print("-"*80)
        
        for ts_type, results in comparison_results.items():
            print(f"{ts_type:<15} "
                  f"{results.get('accuracy_final', 0):<12.4f} "
                  f"{results.get('convergence_round', -1):<12} "
                  f"{results.get('participation_rate_avg', 0):<12.4f} "
                  f"{results.get('system_activity_avg', 0):<12.4f}")
        
        print("="*80)
        
        return comparison_results