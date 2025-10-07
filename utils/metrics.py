"""
utils/metrics.py
评估指标模块 / Evaluation Metrics Module
"""

import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime

class MetricsCalculator:
    """
    评估指标计算器 / Metrics Calculator
    计算联邦学习和激励机制的各种指标
    Calculate various metrics for federated learning and incentive mechanisms
    """
    
    def __init__(self):
        """初始化指标计算器 / Initialize metrics calculator"""
        self.metrics_history = {
            'accuracy': [],
            'loss': [],
            'participation_rate': [],
            'system_activity': [],
            'client_participation': {},
            'level_distribution': [],
            'points_distribution': [],
            'round_times': [],
            'quality_gap': []  # 新增：模型质量差距 / New: model quality gap
        }
    
    def record_quality_gap(self, gap: float):
        """
        记录模型质量差距 / Record model quality gap
        
        Args:
            gap: 钻石级与铜级模型的准确率差距 / Accuracy gap between diamond and bronze models
        """
        self.metrics_history['quality_gap'].append(gap)
    
    def calculate_participation_rate(self, participating_clients: int, 
                                    total_clients: int) -> float:
        """
        计算参与率 / Calculate participation rate
        
        Args:
            participating_clients: 参与的客户端数 / Number of participating clients
            total_clients: 总客户端数 / Total number of clients
            
        Returns:
            参与率 / Participation rate
        """
        if total_clients == 0:
            return 0
        return participating_clients / total_clients
    
    def calculate_client_participation(self, client_id: int, 
                                      rounds_participated: int,
                                      total_rounds: int,
                                      data_contribution: float,
                                      computation_time: float,
                                      model_quality: float,
                                      alpha: float = 0.3,
                                      beta: float = 0.3,
                                      gamma: float = 0.4) -> float:
        """
        计算单个客户端的参与度 / Calculate individual client participation
        
        Args:
            client_id: 客户端ID / Client ID
            rounds_participated: 参与的轮次数 / Number of rounds participated
            total_rounds: 总轮次数 / Total number of rounds
            data_contribution: 数据贡献 / Data contribution
            computation_time: 计算时间贡献 / Computation time contribution
            model_quality: 模型质量贡献 / Model quality contribution
            alpha, beta, gamma: 权重参数 / Weight parameters
            
        Returns:
            客户端参与度 / Client participation
        """
        # 参与频率 / Participation frequency
        frequency = rounds_participated / total_rounds if total_rounds > 0 else 0
        
        # 贡献度 / Contribution
        contribution = (alpha * data_contribution + 
                       beta * computation_time + 
                       gamma * model_quality)
        
        # 综合参与度 / Combined participation
        participation = 0.5 * frequency + 0.5 * contribution
        
        return participation
    
    def calculate_system_activity(self, client_participations: Dict[int, float],
                                 active_clients: int,
                                 total_clients: int) -> float:
        """
        计算系统活跃度 / Calculate system activity
        
        Args:
            client_participations: 客户端参与度字典 / Client participation dictionary
            active_clients: 活跃客户端数 / Number of active clients
            total_clients: 总客户端数 / Total number of clients
            
        Returns:
            系统活跃度 / System activity
        """
        if total_clients == 0:
            return 0
        
        # 活跃客户端比例 / Active client ratio
        active_ratio = active_clients / total_clients
        
        # 平均参与度 / Average participation
        if len(client_participations) > 0:
            avg_participation = sum(client_participations.values()) / len(client_participations)
        else:
            avg_participation = 0
        
        # 综合系统活跃度 / Combined system activity
        system_activity = 0.6 * active_ratio + 0.4 * avg_participation
        
        return system_activity
    
    def update_metrics(self, round_num: int, accuracy: float, loss: float,
                      participation_rate: float, system_activity: float,
                      level_distribution: Dict, points_stats: Dict) -> None:
        """
        更新指标历史 / Update metrics history
        
        Args:
            round_num: 轮次 / Round number
            accuracy: 准确率 / Accuracy
            loss: 损失 / Loss
            participation_rate: 参与率 / Participation rate
            system_activity: 系统活跃度 / System activity
            level_distribution: 等级分布 / Level distribution
            points_stats: 积分统计 / Points statistics
        """
        self.metrics_history['accuracy'].append(accuracy)
        self.metrics_history['loss'].append(loss)
        self.metrics_history['participation_rate'].append(participation_rate)
        self.metrics_history['system_activity'].append(system_activity)
        self.metrics_history['level_distribution'].append(level_distribution)
        self.metrics_history['points_distribution'].append(points_stats)
        self.metrics_history['round_times'].append(round_num)
    
    def get_metrics_summary(self) -> Dict:
        """
        获取指标摘要 / Get metrics summary
        
        Returns:
            指标摘要字典 / Metrics summary dictionary
        """
        summary = {}
        
        # 计算平均值和最终值 / Calculate averages and final values
        for key in ['accuracy', 'loss', 'participation_rate', 'system_activity']:
            if self.metrics_history[key]:
                summary[f'{key}_final'] = self.metrics_history[key][-1]
                summary[f'{key}_avg'] = np.mean(self.metrics_history[key])
                summary[f'{key}_max'] = np.max(self.metrics_history[key])
                summary[f'{key}_min'] = np.min(self.metrics_history[key])
        
        # 添加质量差距统计 / Add quality gap statistics
        if self.metrics_history['quality_gap']:
            summary['quality_gap_final'] = self.metrics_history['quality_gap'][-1]
            summary['quality_gap_avg'] = np.mean(self.metrics_history['quality_gap'])
            summary['quality_gap_max'] = np.max(self.metrics_history['quality_gap'])
        
        return summary
    
    def calculate_convergence_round(self, threshold: float = 0.95) -> int:
        """
        计算收敛轮次 / Calculate convergence round
        
        Args:
            threshold: 收敛阈值 / Convergence threshold
            
        Returns:
            收敛轮次 / Convergence round
        """
        if not self.metrics_history['accuracy']:
            return -1
        
        max_acc = max(self.metrics_history['accuracy'])
        target_acc = max_acc * threshold
        
        for i, acc in enumerate(self.metrics_history['accuracy']):
            if acc >= target_acc:
                return i
        
        return -1