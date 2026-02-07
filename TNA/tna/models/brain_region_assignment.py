"""
Brain Region Assignment Module
脑区分配模块 - 计算节点到聚类中心的软分配概率
"""
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from typing import Optional


class BrainRegionAssignment(nn.Module):
    """
    脑区软分配模块
    使用Student's t-分布或投影方法计算节点到各个聚类区域的分配概率
    支持正交化初始化和多种分配策略
    """
    
    def __init__(
        self,
        num_clusters: int,
        embedding_dimension: int,
        alpha: float = 1.0,
        temperature: float = 1.0,
        initial_centers: Optional[torch.Tensor] = None,
        use_orthogonal: bool = True,
        freeze_centers: bool = True,
        assignment_method: str = 'projection'
    ):
        """
        初始化脑区分配模块
        
        Args:
            num_clusters: 聚类区域数量
            embedding_dimension: 嵌入向量的维度
            alpha: t-分布的自由度参数
            temperature: 温度参数，控制分配的软硬程度（默认1.0）
            initial_centers: 初始聚类中心
            use_orthogonal: 是否对聚类中心进行正交化
            freeze_centers: 是否冻结聚类中心
            assignment_method: 分配方法 ('projection' 或 't_distribution')
        """
        super(BrainRegionAssignment, self).__init__()
        
        self.num_clusters = num_clusters
        self.embedding_dimension = embedding_dimension
        self.alpha = alpha
        self.temperature = temperature
        self.assignment_method = assignment_method
        
        # 初始化聚类中心
        if initial_centers is None:
            centers = torch.zeros(num_clusters, embedding_dimension, dtype=torch.float)
            nn.init.xavier_uniform_(centers)
        else:
            centers = initial_centers
        
        # 正交化处理
        if use_orthogonal:
            centers = self._orthogonalize_gram_schmidt(centers)
        
        self.cluster_centers = Parameter(centers, requires_grad=(not freeze_centers))
    
    def _orthogonalize_gram_schmidt(self, centers):
        """
        使用Gram-Schmidt方法正交化聚类中心
        减少聚类中心之间的冗余，提高区分度
        
        Args:
            centers: 原始中心 [num_clusters, embedding_dim]
        Returns:
            正交化后的中心
        """
        orthogonal_centers = torch.zeros_like(centers)
        orthogonal_centers[0] = centers[0]
        
        for i in range(1, self.num_clusters):
            projection_sum = 0
            for j in range(i):
                # 计算投影: proj_u(v) = (u·v / u·u) * u
                u = orthogonal_centers[j]
                v = centers[i]
                projection_sum = projection_sum + (torch.dot(u, v) / torch.dot(u, u)) * u
            
            # 减去投影得到正交分量
            orthogonal_component = centers[i] - projection_sum
            
            # L2归一化
            orthogonal_centers[i] = orthogonal_component / torch.norm(orthogonal_component, p=2)
        
        return orthogonal_centers
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        计算软分配概率
        
        Args:
            embeddings: 嵌入向量 [batch_size, embedding_dim]
        Returns:
            软分配概率 [batch_size, num_clusters]
        """
        if self.assignment_method == 'projection':
            return self._projection_based_assignment(embeddings)
        else:
            return self._t_distribution_assignment(embeddings)
    
    def _projection_based_assignment(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        基于投影的分配方法
        计算嵌入向量与聚类中心的相似度
        
        Args:
            embeddings: [batch_size, embedding_dim]
        Returns:
            软分配 [batch_size, num_clusters]
        """
        # 计算相似度: embeddings @ centers^T
        similarity = torch.matmul(embeddings, self.cluster_centers.T)
        
        # 平方增强
        similarity_squared = torch.pow(similarity, 2)
        
        # 归一化
        center_norms = torch.norm(self.cluster_centers, p=2, dim=1)
        normalized_similarity = similarity_squared / center_norms
        
        # 温度缩放 + softmax
        scaled_similarity = normalized_similarity * self.temperature
        soft_assignment = F.softmax(scaled_similarity, dim=1)
        
        return soft_assignment
    
    def _t_distribution_assignment(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        基于Student's t-分布的分配方法
        距离越近，分配概率越高
        
        Args:
            embeddings: [batch_size, embedding_dim]
        Returns:
            软分配 [batch_size, num_clusters]
        """
        # 计算欧氏距离平方
        embeddings_expanded = embeddings.unsqueeze(1)  # [batch_size, 1, embed_dim]
        squared_distance = torch.sum(
            (embeddings_expanded - self.cluster_centers) ** 2, dim=2
        )  # [batch_size, num_clusters]
        
        # Student's t-分布核
        numerator = 1.0 / (1.0 + squared_distance / self.alpha)
        power = (self.alpha + 1.0) / 2.0
        numerator = torch.pow(numerator, power)
        
        # 归一化
        denominator = torch.sum(numerator, dim=1, keepdim=True)
        soft_assignment = numerator / denominator
        
        return soft_assignment
    
    def get_cluster_centers(self) -> torch.Tensor:
        """获取聚类中心"""
        return self.cluster_centers

