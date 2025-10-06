"""
federated/server.py
服务器类定义 / Server Class Definition
"""

class FederatedServer:
    """
    联邦学习服务器 / Federated Learning Server
    负责模型聚合和客户端协调 / Responsible for model aggregation and client coordination
    """
    
    def __init__(self, model: nn.Module, device: torch.device = torch.device("cpu")):
        """
        初始化服务器 / Initialize server
        
        Args:
            model: 全局模型 / Global model
            device: 计算设备 / Computing device
        """
        self.global_model = copy.deepcopy(model).to(device)
        self.device = device
        
        # 训练历史 / Training history
        self.accuracy_history = []
        self.loss_history = []
        self.round_times = []
        
        # 客户端信息 / Client information
        self.client_info = {}
        
    def get_global_weights(self) -> Dict:
        """
        获取全局模型权重 / Get global model weights
        
        Returns:
            全局模型权重字典 / Global model weights dictionary
        """
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
        if aggregation_method == "fedavg":
            self._fedavg_aggregation(client_weights, client_infos)
        elif aggregation_method == "weighted":
            self._weighted_aggregation(client_weights, client_infos)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
    
    def _fedavg_aggregation(self, client_weights: Dict[int, Dict], 
                           client_infos: Dict[int, Dict]) -> None:
        """
        FedAvg聚合算法 / FedAvg aggregation algorithm
        按照样本数量加权平均 / Weighted average by number of samples
        """
        # 计算总样本数 / Calculate total samples
        total_samples = sum(info['num_samples'] for info in client_infos.values())
        
        # 初始化聚合权重 / Initialize aggregated weights
        aggregated_weights = {}
        
        for client_id, weights in client_weights.items():
            client_samples = client_infos[client_id]['num_samples']
            weight_factor = client_samples / total_samples
            
            for key in weights.keys():
                if key not in aggregated_weights:
                    aggregated_weights[key] = torch.zeros_like(weights[key])
                aggregated_weights[key] += weights[key] * weight_factor
        
        # 更新全局模型 / Update global model
        self.global_model.load_state_dict(aggregated_weights)
    
    def _weighted_aggregation(self, client_weights: Dict[int, Dict], 
                             client_infos: Dict[int, Dict]) -> None:
        """
        加权聚合算法 / Weighted aggregation algorithm
        考虑数据质量和会员等级 / Consider data quality and membership level
        """
        # 计算权重因子 / Calculate weight factors
        weight_factors = {}
        total_weight = 0
        
        level_multipliers = {
            'bronze': 1.0,
            'silver': 1.2,
            'gold': 1.5,
            'diamond': 2.0
        }
        
        for client_id, info in client_infos.items():
            # 获取会员等级倍数 / Get membership level multiplier
            level = info.get('membership_level', 'bronze')
            level_mult = level_multipliers.get(level, 1.0)
            
            # 综合考虑样本数、质量和等级 / Consider samples, quality and level
            weight = info['num_samples'] * info['model_quality'] * level_mult
            weight_factors[client_id] = weight
            total_weight += weight
        
        # 归一化权重 / Normalize weights
        for client_id in weight_factors:
            weight_factors[client_id] /= total_weight
        
        # 聚合模型 / Aggregate models
        aggregated_weights = {}
        
        for client_id, weights in client_weights.items():
            factor = weight_factors[client_id]
            
            for key in weights.keys():
                if key not in aggregated_weights:
                    aggregated_weights[key] = torch.zeros_like(weights[key])
                aggregated_weights[key] += weights[key] * factor
        
        # 更新全局模型 / Update global model
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
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(data)
        
        accuracy = correct / total if total > 0 else 0
        avg_loss = test_loss / len(test_loader) if len(test_loader) > 0 else 0
        
        # 记录历史 / Record history
        self.accuracy_history.append(accuracy)
        self.loss_history.append(avg_loss)
        
        return accuracy, avg_loss
    
    def save_checkpoint(self, round_num: int, save_path: str) -> None:
        """
        保存检查点 / Save checkpoint
        
        Args:
            round_num: 当前轮次 / Current round
            save_path: 保存路径 / Save path
        """
        checkpoint = {
            'round': round_num,
            'model_state_dict': self.global_model.state_dict(),
            'accuracy_history': self.accuracy_history,
            'loss_history': self.loss_history
        }
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved at round {round_num}: {save_path}")