"""
incentive/points_calculator.py
积分计算器 / Points Calculator
"""

class PointsCalculator:
    """
    积分计算器 / Points Calculator
    根据客户端贡献计算积分
    Calculate points based on client contributions
    """
    
    def __init__(self, alpha: float = 0.3, beta: float = 0.3, gamma: float = 0.4):
        """
        初始化积分计算器 / Initialize points calculator
        
        Args:
            alpha: 数据量权重 / Data size weight
            beta: 计算时间权重 / Computation time weight
            gamma: 模型质量权重 / Model quality weight
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # 归一化参数 / Normalization parameters
        self.max_data_size = 10000
        self.max_computation_time = 100
        
        # 积分历史 / Points history
        self.points_history = {}
        
    def calculate_points(self, data_size: int, computation_time: float, 
                        model_quality: float, level_multiplier: float = 1.0) -> float:
        """
        计算积分 / Calculate points
        
        Args:
            data_size: 数据量 / Data size
            computation_time: 计算时间(秒) / Computation time (seconds)
            model_quality: 模型质量(0-1) / Model quality (0-1)
            level_multiplier: 等级倍数 / Level multiplier
            
        Returns:
            计算得到的积分 / Calculated points
        """
        # 归一化 / Normalize
        norm_data_size = min(data_size / self.max_data_size, 1.0)
        norm_computation_time = min(computation_time / self.max_computation_time, 1.0)
        
        # 基础积分计算 / Base points calculation
        base_points = (self.alpha * norm_data_size * 1000 + 
                      self.beta * norm_computation_time * 1000 + 
                      self.gamma * model_quality * 1000)
        
        # 应用等级倍数 / Apply level multiplier
        total_points = base_points * level_multiplier
        
        return total_points
    
    def calculate_bonus_points(self, consecutive_rounds: int, 
                              performance_rank: int) -> float:
        """
        计算奖励积分 / Calculate bonus points
        
        Args:
            consecutive_rounds: 连续参与轮次 / Consecutive participation rounds
            performance_rank: 性能排名 / Performance rank
            
        Returns:
            奖励积分 / Bonus points
        """
        # 连续参与奖励 / Consecutive participation bonus
        consecutive_bonus = min(consecutive_rounds * 50, 500)
        
        # 排名奖励 / Ranking bonus
        rank_bonus = 0
        if performance_rank == 1:
            rank_bonus = 300
        elif performance_rank == 2:
            rank_bonus = 200
        elif performance_rank == 3:
            rank_bonus = 100
        elif performance_rank <= 10:
            rank_bonus = 50
        
        return consecutive_bonus + rank_bonus
    
    def record_points(self, client_id: int, round_num: int, points: float) -> None:
        """
        记录积分历史 / Record points history
        
        Args:
            client_id: 客户端ID / Client ID
            round_num: 轮次 / Round number
            points: 获得的积分 / Points earned
        """
        if client_id not in self.points_history:
            self.points_history[client_id] = []
        
        self.points_history[client_id].append({
            'round': round_num,
            'points': points,
            'timestamp': datetime.now()
        })
