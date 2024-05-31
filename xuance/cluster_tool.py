# cluster_tool.py
import numpy as np
from sklearn.cluster import KMeans

def clipped_softmax(logits, k):
    sorted_logits = np.sort(logits)[::-1]
    threshold = sorted_logits[k - 1] if k < len(logits) else sorted_logits[-1]
    clipped_logits = np.clip(logits - threshold, a_min=0, a_max=None)
    exp_logits = np.exp(clipped_logits)
    return exp_logits / exp_logits.sum()

class ClusterTool:
    def __init__(self, state_space, action_space, n_clusters):
        self.state_space = state_space
        self.action_space = action_space
        self.n_clusters = n_clusters

        # 使用KMeans对状态空间进行聚类
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        self.state_clusters = self.kmeans.fit_predict(state_space)

        # 初始化动作选择频率
        self.action_counts = {k: {a: 0 for a in range(action_space)} for k in range(n_clusters)}

    def get_cluster(self, state):
        # 获取状态对应的簇
        tmp = state.reshape(-1, 84)
        cluster = self.kmeans.predict(np.array(tmp, dtype=np.float64))[0]
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

    def compute_belief_distribution(self, state, beta_t, immediate_belief, k):
        # 计算综合概率分布 b_t(a | s_{t+1})，并应用Clipped Softmax
        prior_probs = np.array([self.get_action_prob(state, a) for a in range(self.action_space)])
        immediate_belief = clipped_softmax(immediate_belief, k)
        comprehensive_prob = beta_t * prior_probs + (1 - beta_t) * immediate_belief
        return comprehensive_prob