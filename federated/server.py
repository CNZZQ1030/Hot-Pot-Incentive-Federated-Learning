"""
federated/server.py
服务器类定义 / Server Class Definition
支持差异化模型奖励机制
Supports differentiated model reward mechanism
"""

import torch
import torch.nn as nn
import copy
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader

class FederatedServer:
    """
    联邦学习服务器 / Federated Learning Server
    负责模型聚合和客户端协调，支持差异化模型分发
    Responsible for model aggregation and client coordination, supports tiered model distribution
    """
    
    def __init__(self, model: nn.Module, device: torch.device = torch.device("cpu")):
        """
        初始化服务器 / Initialize server
        
        Args:
            model: 全局模型 / Global model
            device: 计算设备 / Computing device
        """
        self.global_model = copy.deepcopy(model).to(device, dtype=torch.float32)
        # 确保所有参数使用相同的dtype
        for param in self.global_model.parameters():
            param.data = param.data.to(dtype=torch.float32)
        
        self.device = device
        
        # 训练历史 / Training history
        self.accuracy_history = []
        self.loss_history = []
        self.round_times = []
        
        # 客户端信息 / Client information
        self.client_info = {}
        
        # 分层模型存储 / Tiered models storage
        self.tiered_models = {
            'diamond': None,
            'gold': None,
            'silver': None,
            'bronze': None
        }
        
        # 各等级模型质量记录 / Model quality records by level
        self.tiered_model_qualities = {
            'diamond': [],
            'gold': [],
            'silver': [],
            'bronze': []
        }
        
    def get_global_weights(self) -> Dict:
        """
        获取全局模型权重 / Get global model weights
        
        Returns:
            全局模型权重字典 / Global model weights dictionary
        """
        return self.global_model.state_dict()
    
    def get_tiered_model_weights(self, client_level: str) -> Dict:
        """
        根据客户端等级获取对应的模型权重 / Get model weights based on client level
        
        Args:
            client_level: 客户端等级 / Client level
            
        Returns:
            对应等级的模型权重 / Model weights for the level
        """
        if self.tiered_models.get(client_level) is not None:
            return self.tiered_models[client_level].state_dict()
        else:
            # 如果还没有分层模型，返回全局模型
            # If tiered models not yet created, return global model
            return self.global_model.state_dict()
    
    def aggregate_models(self, client_weights: Dict[int, Dict], 
                        client_infos: Dict[int, Dict],
                        aggregation_method: str = "fedavg") -> None:
        """
        聚合客户端模型 / Aggregate client models
        
        Args:
            client_weights: 客户端模型权重 / Client model weights
            client_infos: 客户端训练信息 / Client training information
            aggregation_method: 聚合方法 / Aggregation method
        """
        from config import IncentiveConfig
        
        # 先执行标准聚合，生成基础全局模型
        # First perform standard aggregation to generate base global model
        if aggregation_method == "fedavg":
            self._fedavg_aggregation(client_weights, client_infos)
        elif aggregation_method == "weighted":
            self._weighted_aggregation(client_weights, client_infos)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
        
        # 如果启用差异化奖励，生成分层模型
        # If tiered rewards enabled, generate tiered models
        if IncentiveConfig.ENABLE_TIERED_REWARDS:
            self._create_tiered_models(client_weights, client_infos)
    
    def _create_tiered_models(self, client_weights: Dict[int, Dict], 
                             client_infos: Dict[int, Dict]) -> None:
        """
        创建分层模型 / Create tiered models
        根据客户端等级创建不同质量的模型
        Create models of different quality based on client levels
        """
        from config import IncentiveConfig
        
        # 按等级分组客户端 / Group clients by level
        clients_by_level = {
            'diamond': [],
            'gold': [],
            'silver': [],
            'bronze': []
        }
        
        for client_id, info in client_infos.items():
            level = info.get('membership_level', 'bronze')
            clients_by_level[level].append(client_id)
        
        strategy = IncentiveConfig.TIERED_AGGREGATION_STRATEGY
        
        if strategy == 'weighted':
            # 加权策略：高等级模型使用高质量客户端，权重更大
            # Weighted strategy: high-level models use high-quality clients with larger weights
            self._create_weighted_tiered_models(client_weights, client_infos, clients_by_level)
        else:  # 'strict'
            # 严格策略：每个等级只使用该等级及以上的客户端
            # Strict strategy: each level only uses clients of that level and above
            self._create_strict_tiered_models(client_weights, client_infos, clients_by_level)
    
    def _create_weighted_tiered_models(self, client_weights: Dict[int, Dict],
                                      client_infos: Dict[int, Dict],
                                      clients_by_level: Dict[str, List[int]]) -> None:
        """
        使用加权策略创建分层模型 / Create tiered models using weighted strategy
        """
        from config import IncentiveConfig
        
        aggregation_weights = IncentiveConfig.TIERED_AGGREGATION_WEIGHTS
        
        # 为每个等级创建模型 / Create model for each level
        for target_level in ['diamond', 'gold', 'silver', 'bronze']:
            level_weights = aggregation_weights[target_level]
            
            # 使用特定权重聚合所有客户端 / Aggregate all clients with specific weights
            tiered_model_weights = self._aggregate_with_level_weights(
                client_weights, client_infos, level_weights
            )
            
            # 创建并存储分层模型 / Create and store tiered model
            self.tiered_models[target_level] = copy.deepcopy(self.global_model)
            self.tiered_models[target_level].load_state_dict(tiered_model_weights)
    
    def _create_strict_tiered_models(self, client_weights: Dict[int, Dict],
                                    client_infos: Dict[int, Dict],
                                    clients_by_level: Dict[str, List[int]]) -> None:
        """
        使用严格策略创建分层模型 / Create tiered models using strict strategy
        每个等级只使用该等级及以上的客户端
        Each level only uses clients of that level and above
        """
        # 钻石级：只使用钻石和金级客户端 / Diamond: only diamond and gold clients
        diamond_clients = clients_by_level['diamond'] + clients_by_level['gold']
        if diamond_clients:
            diamond_weights = self._aggregate_selected_clients(
                client_weights, client_infos, diamond_clients
            )
            self.tiered_models['diamond'] = copy.deepcopy(self.global_model)
            self.tiered_models['diamond'].load_state_dict(diamond_weights)
        else:
            self.tiered_models['diamond'] = copy.deepcopy(self.global_model)
        
        # 金级：使用金、银、钻石级客户端 / Gold: diamond, gold, and silver clients
        gold_clients = clients_by_level['diamond'] + clients_by_level['gold'] + clients_by_level['silver']
        if gold_clients:
            gold_weights = self._aggregate_selected_clients(
                client_weights, client_infos, gold_clients
            )
            self.tiered_models['gold'] = copy.deepcopy(self.global_model)
            self.tiered_models['gold'].load_state_dict(gold_weights)
        else:
            self.tiered_models['gold'] = copy.deepcopy(self.global_model)
        
        # 银级和铜级：使用全局模型 / Silver and Bronze: use global model
        self.tiered_models['silver'] = copy.deepcopy(self.global_model)
        self.tiered_models['bronze'] = copy.deepcopy(self.global_model)
    
    def _aggregate_with_level_weights(self, client_weights: Dict[int, Dict],
                                client_infos: Dict[int, Dict],
                                level_multipliers: Dict[str, float]) -> Dict:
        """
        使用等级权重聚合 / Aggregate with level weights
        """
        aggregated_weights = {}
        total_weight = 0
        
        # 计算每个客户端的权重
        client_weight_factors = {}
        for client_id, weights in client_weights.items():
            info = client_infos[client_id]
            level = info.get('membership_level', 'bronze')
            level_mult = level_multipliers.get(level, 1.0)
            
            weight = info['num_samples'] * info['model_quality'] * level_mult
            client_weight_factors[client_id] = weight
            total_weight += weight
    
        # 归一化并聚合
        for client_id, weights in client_weights.items():
            factor = client_weight_factors[client_id] / total_weight if total_weight > 0 else 0
            
            for key in weights.keys():
                # 确保聚合权重使用与原始权重相同的dtype
                if key not in aggregated_weights:
                    aggregated_weights[key] = torch.zeros_like(weights[key], dtype=weights[key].dtype)
                
                # 根据权重类型选择不同的处理方式
                if weights[key].dtype in [torch.int32, torch.int64, torch.long]:
                    # 对于整数类型参数，先转换为float进行计算，再转回原类型
                    float_weights = weights[key].float()
                    weighted_update = (float_weights * factor).to(weights[key].dtype)
                else:
                    # 对于浮点类型参数，直接计算
                    weighted_update = weights[key] * factor
            
                # 确保类型匹配
                if weighted_update.dtype != aggregated_weights[key].dtype:
                    weighted_update = weighted_update.to(aggregated_weights[key].dtype)
            
                aggregated_weights[key] += weighted_update
    
        return aggregated_weights

    def _aggregate_selected_clients(self, client_weights: Dict[int, Dict],
                              client_infos: Dict[int, Dict],
                              selected_clients: List[int]) -> Dict:
        """
        聚合选定的客户端 / Aggregate selected clients
        """
        aggregated_weights = {}
        total_samples = sum(client_infos[cid]['num_samples'] for cid in selected_clients if cid in client_infos)
        
        for client_id in selected_clients:
            if client_id not in client_weights:
                continue
                
            weights = client_weights[client_id]
            info = client_infos[client_id]
            factor = info['num_samples'] / total_samples if total_samples > 0 else 0
            
            for key in weights.keys():
                if key not in aggregated_weights:
                    aggregated_weights[key] = torch.zeros_like(weights[key])
                
                # 类型转换：确保类型匹配
                if weights[key].dtype != aggregated_weights[key].dtype:
                    weighted_update = weights[key].to(aggregated_weights[key].dtype) * factor
                else:
                    weighted_update = weights[key] * factor
                
                aggregated_weights[key] += weighted_update
        
        return aggregated_weights

    def _fedavg_aggregation(self, client_weights: Dict[int, Dict], 
                       client_infos: Dict[int, Dict]) -> None:
        """
        FedAvg聚合算法 / FedAvg aggregation algorithm
        按照样本数量加权平均 / Weighted average by number of samples
        """
        total_samples = sum(info['num_samples'] for info in client_infos.values())
        aggregated_weights = {}
        
        for client_id, weights in client_weights.items():
            client_samples = client_infos[client_id]['num_samples']
            weight_factor = client_samples / total_samples if total_samples > 0 else 0
            
            for key in weights.keys():
                if key not in aggregated_weights:
                    aggregated_weights[key] = torch.zeros_like(weights[key])
                
                # 类型转换：确保类型匹配
                if weights[key].dtype != aggregated_weights[key].dtype:
                    weighted_update = weights[key].to(aggregated_weights[key].dtype) * weight_factor
                else:
                    weighted_update = weights[key] * weight_factor
            
                aggregated_weights[key] += weighted_update
    
        self.global_model.load_state_dict(aggregated_weights)

    def _weighted_aggregation(self, client_weights: Dict[int, Dict], 
                         client_infos: Dict[int, Dict]) -> None:
        """
        加权聚合算法 / Weighted aggregation algorithm
        考虑数据质量和会员等级 / Consider data quality and membership level
        """
        weight_factors = {}
        total_weight = 0
        
        level_multipliers = {
            'bronze': 1.0,
            'silver': 1.2,
            'gold': 1.5,
            'diamond': 2.0
        }
        
        # 计算权重因子
        for client_id, info in client_infos.items():
            level = info.get('membership_level', 'bronze')
            level_mult = level_multipliers.get(level, 1.0)
            weight = info['num_samples'] * info['model_quality'] * level_mult
            weight_factors[client_id] = weight
            total_weight += weight
    
        # 归一化权重
        for client_id in weight_factors:
            weight_factors[client_id] /= total_weight if total_weight > 0 else 1

        aggregated_weights = {}
        for client_id, weights in client_weights.items():
            factor = weight_factors[client_id]
            
            for key in weights.keys():
                # 确保聚合权重使用与原始权重相同的dtype
                if key not in aggregated_weights:
                    aggregated_weights[key] = torch.zeros_like(weights[key], dtype=weights[key].dtype)
                
                # 确保weighted_update使用正确的dtype
                if weights[key].dtype in [torch.int32, torch.int64, torch.long]:
                    # 对于整数类型的参数，先进行浮点运算再转回整数
                    weighted_update = (weights[key].float() * factor).to(weights[key].dtype)
                else:
                    # 对于浮点类型的参数，直接进行运算
                    weighted_update = weights[key] * factor
            
                # 确保类型匹配
                if weighted_update.dtype != aggregated_weights[key].dtype:
                    weighted_update = weighted_update.to(aggregated_weights[key].dtype)
            
                aggregated_weights[key] += weighted_update

        # 加载聚合后的权重
        self.global_model.load_state_dict(aggregated_weights)
    
    def evaluate_global_model(self, test_loader: DataLoader) -> Tuple[float, float]:
        """
        评估全局模型 / Evaluate global model
        
        Args:
            test_loader: 测试数据加载器 / Test data loader
            
        Returns:
            准确率和损失 / Accuracy and loss
        """
        self.global_model.eval()
        test_loss = 0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                test_loss += criterion(output, target).item()
                # 修改这里：确保预测结果为整数类型
                pred = output.argmax(dim=1, keepdim=True).to(torch.long)
                target = target.view_as(pred).to(torch.long)
                correct += pred.eq(target).sum().item()
                total += len(data)
        
        accuracy = correct / total if total > 0 else 0
        avg_loss = test_loss / len(test_loader) if len(test_loader) > 0 else 0
        
        self.accuracy_history.append(accuracy)
        self.loss_history.append(avg_loss)
        
        return accuracy, avg_loss
    
    def evaluate_tiered_models(self, test_loader: DataLoader) -> Dict[str, Tuple[float, float]]:
        """
        评估各等级模型的性能 / Evaluate performance of tiered models
        
        Args:
            test_loader: 测试数据加载器 / Test data loader
            
        Returns:
            各等级的(准确率, 损失) / (accuracy, loss) for each level
        """
        results = {}
        criterion = nn.CrossEntropyLoss()
        
        for level in ['diamond', 'gold', 'silver', 'bronze']:
            if self.tiered_models[level] is None:
                continue
            
            model = self.tiered_models[level]
            model.eval()
            
            test_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    test_loss += criterion(output, target).item()
                    # 修改这里：确保预测结果为整数类型
                    pred = output.argmax(dim=1, keepdim=True).to(torch.long)
                    target = target.view_as(pred).to(torch.long)
                    correct += pred.eq(target).sum().item()
                    total += len(data)
            
            accuracy = correct / total if total > 0 else 0
            avg_loss = test_loss / len(test_loader) if len(test_loader) > 0 else 0
            
            results[level] = (accuracy, avg_loss)
            self.tiered_model_qualities[level].append(accuracy)
        
        return results
    
    def save_checkpoint(self, round_num: int, save_path: str) -> None:
        """
        保存检查点 / Save checkpoint
        
        Args:
            round_num: 当前轮次 / Current round
            save_path: 保存路径 / Save path
        """
        # 确保save_path的完整目录路径存在
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        checkpoint = {
            'round': round_num,
            'model_state_dict': self.global_model.state_dict(),
            'accuracy_history': self.accuracy_history,
            'loss_history': self.loss_history,
            'tiered_models': {
                level: model.state_dict() if model is not None else None
                for level, model in self.tiered_models.items()
            },
            'tiered_model_qualities': self.tiered_model_qualities
        }

        try:
            torch.save(checkpoint, save_path)
            print(f"Checkpoint saved at round {round_num}: {save_path}")
        except Exception as e:
            print(f"Error saving checkpoint: {str(e)}")
            print(f"Attempting to save at: {save_path}")
            print(f"Directory exists: {os.path.exists(os.path.dirname(save_path))}")
