import numpy as np
from sklearn.cluster import MiniBatchKMeans
from collections import defaultdict

class StateCategorizer:
    def __init__(self, state_space, action_space, n_categories):
        self.state_space = np.array(state_space, dtype=np.float32)
        self.action_space = action_space
        self.n_categories = n_categories
        # 使用 MiniBatchKMeans 进行初始聚类
        flattened_states = self.state_space.reshape(len(state_space), -1)
        kmeans = MiniBatchKMeans(n_clusters=n_categories)
        kmeans.fit(flattened_states)
        # 预计算所有状态的类别并存储
        self.state_categories = {tuple(state): category for state, category in zip(flattened_states, kmeans.labels_)}
        # 计算每个类别的中心点
        self.category_centers = kmeans.cluster_centers_
        # 初始化动作偏好字典
        self.action_counts = defaultdict(lambda: defaultdict(int))

    def get_category(self, state):
        state_array = np.array(state, dtype=np.float32).flatten()
        state_tuple = tuple(state_array)
        if state_tuple in self.state_categories:
            return self.state_categories[state_tuple]
        else:
            distances = np.linalg.norm(self.category_centers - state_array, axis=1)
            nearest_category = np.argmin(distances)
            self.state_categories[state_tuple] = nearest_category
            return nearest_category

    def update_action_counts(self, state, action):
        category = self.get_category(state)
        self.action_counts[category][action] += 1

    def get_action_prob(self, state):
        category = self.get_category(state)
        total_actions = sum(self.action_counts[category].values())
        if total_actions == 0:
            return np.ones(self.action_space) / self.action_space

        probs = np.array([self.action_counts[category][action] / total_actions
                          for action in range(self.action_space)])
        return probs

    def compute_belief_distribution(self, state, immediate_belief=None, beta=0.5):
        prior_probs = self.get_action_prob(state)
        if immediate_belief is None:
            return prior_probs

        combined_probs = beta * prior_probs + (1 - beta) * immediate_belief
        return combined_probs / combined_probs.sum()
