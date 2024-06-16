import numpy as np
from sklearn.cluster import KMeans
import torch

class ClusterTool:
    def __init__(self, state_space, action_space, n_clusters):
        self.state_space = state_space
        self.action_space = action_space
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        self.state_clusters = self.kmeans.fit_predict(state_space.astype(np.float32))
        self.action_counts = {k: {a: 0 for a in range(action_space)} for k in range(n_clusters)}

    def get_cluster(self, state):
        # 获取状态对应的簇
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        tmp = state.reshape(-1, 4).astype(np.float32)
        cluster = self.kmeans.predict(tmp)[0]
        return cluster

    def update_action_counts(self, state, action):
        # 更新动作选择频率
        cluster = self.get_cluster(state)
        self.action_counts[cluster][action] += 1

    # def update_action_counts(self, states, actions):
    #     # 批量更新动作选择频率
    #     clusters = [self.get_cluster(state) for state in states]
    #     for cluster, action in zip(clusters, actions):
    #         self.action_counts[cluster][action] += 1

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
