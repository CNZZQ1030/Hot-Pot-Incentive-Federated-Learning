"""
utils/visualization.py
可视化模块 / Visualization Module
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional
import os


class Visualizer:
    """
    可视化工具 / Visualization Tool
    生成联邦学习实验的各种图表
    Generate various charts for federated learning experiments
    """
    
    def __init__(self, save_dir: str = "results/plots"):
        """
        初始化可视化器 / Initialize visualizer
        
        Args:
            save_dir: 图表保存目录 / Chart save directory
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置绘图风格 / Set plotting style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 12
    
    def plot_training_curves(self, metrics_history: Dict, 
                            experiment_name: str) -> None:
        """
        绘制训练曲线 / Plot training curves
        
        Args:
            metrics_history: 指标历史 / Metrics history
            experiment_name: 实验名称 / Experiment name
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        rounds = metrics_history.get('round_times', list(range(len(metrics_history['accuracy']))))
        
        # 准确率曲线 / Accuracy curve
        axes[0, 0].plot(rounds, metrics_history['accuracy'], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Round')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Model Accuracy over Rounds')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 损失曲线 / Loss curve
        axes[0, 1].plot(rounds, metrics_history['loss'], 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Round')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Model Loss over Rounds')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 参与率曲线 / Participation rate curve
        axes[1, 0].plot(rounds, metrics_history['participation_rate'], 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Round')
        axes[1, 0].set_ylabel('Participation Rate')
        axes[1, 0].set_title('Client Participation Rate over Rounds')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 系统活跃度曲线 / System activity curve
        axes[1, 1].plot(rounds, metrics_history['system_activity'], 'm-', linewidth=2)
        axes[1, 1].set_xlabel('Round')
        axes[1, 1].set_ylabel('System Activity')
        axes[1, 1].set_title('System Activity over Rounds')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Training Progress - {experiment_name}', fontsize=16)
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, f'{experiment_name}_training_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training curves saved to {save_path}")
    
    def plot_level_distribution(self, level_distributions: List[Dict],
                               experiment_name: str) -> None:
        """
        绘制会员等级分布 / Plot membership level distribution
        
        Args:
            level_distributions: 等级分布历史 / Level distribution history
            experiment_name: 实验名称 / Experiment name
        """
        if not level_distributions:
            return
        
        # 准备数据 / Prepare data
        levels = ['bronze', 'silver', 'gold', 'diamond']
        rounds = list(range(len(level_distributions)))
        
        level_counts = {level: [] for level in levels}
        for dist in level_distributions:
            for level in levels:
                level_counts[level].append(dist.get(level, 0))
        
        # 创建堆叠面积图 / Create stacked area plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 堆叠面积图 / Stacked area chart
        ax.stackplot(rounds, 
                    level_counts['bronze'],
                    level_counts['silver'],
                    level_counts['gold'],
                    level_counts['diamond'],
                    labels=levels,
                    colors=['#CD7F32', '#C0C0C0', '#FFD700', '#B9F2FF'],
                    alpha=0.8)
        
        ax.set_xlabel('Round')
        ax.set_ylabel('Number of Clients')
        ax.set_title(f'Membership Level Distribution - {experiment_name}')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        save_path = os.path.join(self.save_dir, f'{experiment_name}_level_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Level distribution plot saved to {save_path}")
    
    def plot_comparison(self, experiments_data: Dict[str, Dict],
                       metric_name: str = 'accuracy') -> None:
        """
        绘制不同实验的比较图 / Plot comparison between different experiments
        
        Args:
            experiments_data: 实验数据字典 / Experiments data dictionary
            metric_name: 要比较的指标名称 / Metric name to compare
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(experiments_data)))
        
        for i, (exp_name, metrics) in enumerate(experiments_data.items()):
            if metric_name in metrics:
                rounds = list(range(len(metrics[metric_name])))
                ax.plot(rounds, metrics[metric_name], 
                       label=exp_name, linewidth=2, color=colors[i])
        
        ax.set_xlabel('Round')
        ax.set_ylabel(metric_name.replace('_', ' ').title())
        ax.set_title(f'Comparison of {metric_name.replace("_", " ").title()} Across Experiments')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        save_path = os.path.join(self.save_dir, f'comparison_{metric_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Comparison plot saved to {save_path}")
    
    def plot_points_distribution(self, points_data: Dict[int, float],
                                experiment_name: str) -> None:
        """
        绘制积分分布 / Plot points distribution
        
        Args:
            points_data: 客户端积分数据 / Client points data
            experiment_name: 实验名称 / Experiment name
        """
        if not points_data:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 积分直方图 / Points histogram
        points_values = list(points_data.values())
        ax1.hist(points_values, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Points')
        ax1.set_ylabel('Number of Clients')
        ax1.set_title('Points Distribution')
        ax1.grid(True, alpha=0.3)
        
        # 积分箱线图 / Points boxplot
        ax2.boxplot(points_values, vert=True)
        ax2.set_ylabel('Points')
        ax2.set_title('Points Distribution (Boxplot)')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'Client Points Analysis - {experiment_name}', fontsize=14)
        
        save_path = os.path.join(self.save_dir, f'{experiment_name}_points_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Points distribution plot saved to {save_path}")
    
    def create_summary_report(self, metrics_summary: Dict, 
                             experiment_name: str) -> None:
        """
        创建总结报告图表 / Create summary report chart
        
        Args:
            metrics_summary: 指标摘要 / Metrics summary
            experiment_name: 实验名称 / Experiment name
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 准备数据 / Prepare data
        metrics = ['Accuracy', 'Participation Rate', 'System Activity']
        final_values = [
            metrics_summary.get('accuracy_final', 0),
            metrics_summary.get('participation_rate_final', 0),
            metrics_summary.get('system_activity_final', 0)
        ]
        avg_values = [
            metrics_summary.get('accuracy_avg', 0),
            metrics_summary.get('participation_rate_avg', 0),
            metrics_summary.get('system_activity_avg', 0)
        ]
        
        # 柱状图比较最终值和平均值 / Bar chart comparing final and average values
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, final_values, width, label='Final', color='steelblue')
        axes[0, 0].bar(x + width/2, avg_values, width, label='Average', color='lightcoral')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].set_title('Metrics Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(metrics, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 收敛速度指标 / Convergence speed metrics
        convergence_round = metrics_summary.get('convergence_round', -1)
        axes[0, 1].text(0.5, 0.5, f'Convergence Round:\n{convergence_round}',
                       ha='center', va='center', fontsize=20,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[0, 1].set_xlim(0, 1)
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].axis('off')
        axes[0, 1].set_title('Convergence Information')
        
        # 最终准确率和损失 / Final accuracy and loss
        final_acc = metrics_summary.get('accuracy_final', 0)
        final_loss = metrics_summary.get('loss_final', 0)
        
        axes[1, 0].pie([final_acc, 1-final_acc], 
                      labels=['Accuracy', 'Error'],
                      colors=['green', 'red'],
                      autopct='%1.1f%%',
                      startangle=90)
        axes[1, 0].set_title(f'Final Model Performance\nLoss: {final_loss:.4f}')
        
        # 统计信息表格 / Statistics table
        table_data = []
        for metric in ['accuracy', 'loss', 'participation_rate', 'system_activity']:
            table_data.append([
                metric.replace('_', ' ').title(),
                f"{metrics_summary.get(f'{metric}_final', 0):.4f}",
                f"{metrics_summary.get(f'{metric}_avg', 0):.4f}",
                f"{metrics_summary.get(f'{metric}_max', 0):.4f}",
                f"{metrics_summary.get(f'{metric}_min', 0):.4f}"
            ])
        
        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')
        table = axes[1, 1].table(cellText=table_data,
                                colLabels=['Metric', 'Final', 'Average', 'Max', 'Min'],
                                cellLoc='center',
                                loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        axes[1, 1].set_title('Detailed Statistics')
        
        plt.suptitle(f'Experiment Summary Report - {experiment_name}', fontsize=16)
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, f'{experiment_name}_summary_report.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Summary report saved to {save_path}")