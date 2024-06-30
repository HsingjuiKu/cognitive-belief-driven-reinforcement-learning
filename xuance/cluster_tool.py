import numpy as np
import torch
from kmeans_pytorch import kmeans

class ClusterTool:
    def __init__(self, state_space, action_space, n_clusters, device='cpu'):
        self.state_space = state_space
        self.action_space = action_space
        self.n_clusters = n_clusters
        self.device = torch.device(device)

        # 转换为torch张量并移动到指定设备
        state_space_tensor = torch.from_numpy(state_space.astype(np.float32)).to(self.device)

        # 使用kmeans_pytorch进行聚类
        self.cluster_ids_x, self.cluster_centers = kmeans(
            X=state_space_tensor, num_clusters=n_clusters, distance='euclidean', device=self.device
        )

        # 将聚类结果转换为numpy数组
        self.state_clusters = self.cluster_ids_x.cpu().numpy()
        self.action_counts = {k: {a: 0 for a in range(action_space)} for k in range(n_clusters)}

    def get_cluster(self, state):
        # 获取状态对应的簇
        if isinstance(state, torch.Tensor):
            state_tensor = state.float().to(self.device)
        else:
            state_tensor = torch.from_numpy(state.reshape(-1, state.shape[-1]).astype(np.float32)).to(self.device)

        # 确保state_tensor是2D的
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)

        # 计算距离
        dists = torch.cdist(state_tensor, self.cluster_centers)
        cluster = torch.argmin(dists, dim=1).item()
        return cluster

    def update_action_counts(self, state, action):
        # 更新动作选择频率
        cluster = self.get_cluster(state)
        self.action_counts[cluster][action] += 1

    def get_action_prob(self, state, action):
        # 计算动作选择概率分布 P_k(a)
        cluster = self.get_cluster(state)
        total_actions = sum(self.action_counts[cluster].values())
        if total_actions == 0:
            return 0
        return self.action_counts[cluster][action] / total_actions

    def compute_belief_distribution(self, state):
        # 返回当前状态所属簇的动作概率分布
        prior_probs = np.array([self.get_action_prob(state, a) for a in range(self.action_space)])
        return prior_probs
