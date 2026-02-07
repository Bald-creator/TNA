"""
Adaptive Brain Cluster Module
自适应脑网络聚类模块 - 用于脑网络的深度聚类
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional
from .brain_region_assignment import BrainRegionAssignment


class AdaptiveBrainCluster(nn.Module):
    """
    自适应脑网络聚类模块
    
    将脑网络节点通过编码器映射到嵌入空间，然后进行软聚类分配
    使用自监督学习优化聚类质量
    """
    
    def __init__(
        self,
        num_clusters: int,
        hidden_dimension: int,
        encoder: torch.nn.Module,
        alpha: float = 1.0,
        temperature: float = 1.0,
        use_orthogonal: bool = True,
        freeze_centers: bool = True,
        assignment_method: str = 'projection',
        loss_reduction: str = 'mean'
    ):
        """
        初始化自适应脑网络聚类模块
        
        Args:
            num_clusters: 聚类数量（脑区数量）
            hidden_dimension: 隐藏层维度（编码器输出维度）
            encoder: 特征编码器
            alpha: t-分布自由度参数
            temperature: 温度参数，控制分配软硬程度
            use_orthogonal: 使用正交化初始化
            freeze_centers: 冻结聚类中心
            assignment_method: 分配方法 ('projection' 或 't_distribution')
            loss_reduction: 损失归约方式
        """
        super(AdaptiveBrainCluster, self).__init__()
        
        self.encoder = encoder
        self.hidden_dimension = hidden_dimension
        self.num_clusters = num_clusters
        self.alpha = alpha
        self.temperature = temperature
        self.loss_reduction = loss_reduction
        
        # 脑区分配模块
        self.region_assignment = BrainRegionAssignment(
            num_clusters=num_clusters,
            embedding_dimension=hidden_dimension,
            alpha=alpha,
            temperature=temperature,
            use_orthogonal=use_orthogonal,
            freeze_centers=freeze_centers,
            assignment_method=assignment_method
        )
        
        # KL散度损失
        self.loss_fn = nn.KLDivLoss(size_average=False)
    
    def forward(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            batch: 输入批次 [batch_size, num_nodes, hidden_dim]
            
        Returns:
            tuple: (cluster_repr, assignment)
                - cluster_repr: 聚类级别的表示 [batch_size, num_clusters, hidden_dim]
                - assignment: 节点的聚类分配 [batch_size, num_nodes, num_clusters]
        """
        batch_size = batch.size(0)
        num_nodes = batch.size(1)
        
        # Step 1: 展平并编码
        flattened_batch = batch.view(batch_size, -1)
        encoded = self.encoder(flattened_batch)
        encoded = encoded.view(batch_size * num_nodes, -1)
        
        # Step 2: 计算软分配
        assignment = self.region_assignment(encoded)
        assignment = assignment.view(batch_size, num_nodes, -1)
        encoded = encoded.view(batch_size, num_nodes, -1)
        
        # Step 3: 聚合得到聚类表示
        # 使用加权平均：根据分配概率聚合节点特征
        cluster_repr = torch.bmm(
            assignment.transpose(1, 2),  # [batch_size, num_clusters, num_nodes]
            encoded  # [batch_size, num_nodes, hidden_dim]
        )  # -> [batch_size, num_clusters, hidden_dim]
        
        return cluster_repr, assignment
    
    def compute_target_distribution(self, soft_assignment: torch.Tensor) -> torch.Tensor:
        """
        计算目标分布（用于自监督学习）
        
        通过平方和归一化提高高置信度分配的权重
        这是一种自训练策略，让模型学习更确定的聚类边界
        
        Args:
            soft_assignment: 当前软分配 [batch_size, num_nodes, num_clusters]
            
        Returns:
            目标分布 [batch_size, num_nodes, num_clusters]
        """
        # 展平为2D进行计算
        original_shape = soft_assignment.shape
        flat_assignment = soft_assignment.view(-1, soft_assignment.size(-1))  # [batch*nodes, clusters]
        
        # 平方增强高置信度
        squared = flat_assignment ** 2
        
        # 除以每个聚类的频率（跨样本归一化）
        cluster_freq = torch.sum(squared, dim=0, keepdim=True)  # [1, num_clusters]
        weight = squared / (cluster_freq + 1e-10)
        
        # 归一化使每个样本的概率和为1
        weight = (weight.t() / torch.sum(weight, dim=1)).t()
        
        # 恢复原始形状
        weight = weight.view(original_shape)
        
        return weight
    
    def loss(self, assignment: torch.Tensor) -> torch.Tensor:
        """
        计算聚类损失（KL散度）
        
        通过最小化当前分配与目标分配的KL散度来优化聚类
        
        Args:
            assignment: 聚类分配 [batch_size, num_nodes, num_clusters]
            
        Returns:
            标量损失值
        """
        # 展平
        flattened_assignment = assignment.view(-1, assignment.size(-1))
        
        # 计算目标分布（不需要梯度）
        target = self.compute_target_distribution(assignment).detach()
        flattened_target = target.view(-1, target.size(-1))
        
        # KL散度: KL(P||Q) = sum(P * log(P/Q))
        log_assignment = flattened_assignment.log()
        kl_loss = self.loss_fn(log_assignment, flattened_target)
        
        # 归约
        if self.loss_reduction == 'mean':
            kl_loss = kl_loss / flattened_assignment.size(0)
        
        return kl_loss
    
    def get_cluster_centers(self) -> torch.Tensor:
        """获取当前的聚类中心"""
        return self.region_assignment.get_cluster_centers()
    
    def predict_clusters(self, batch: torch.Tensor) -> torch.Tensor:
        """
        预测硬聚类标签
        
        Args:
            batch: 输入批次 [batch_size, num_nodes, hidden_dim]
            
        Returns:
            聚类标签 [batch_size, num_nodes]
        """
        _, assignment = self.forward(batch)
        cluster_labels = torch.argmax(assignment, dim=2)
        return cluster_labels

