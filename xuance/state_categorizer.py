# import numpy as np
# from sklearn.cluster import MiniBatchKMeans
# from collections import defaultdict
# from sklearn.cluster import KMeans

# class StateCategorizer:
#     def __init__(self, state_space, action_space, n_categories):
#         self.state_space = np.array(state_space, dtype=np.float32)
#         self.action_space = action_space
#         self.n_categories = n_categories

#         # 使用 MiniBatchKMeans 进行初始聚类
#         kmeans = KMeans(n_clusters=n_categories, random_state=0)
#         kmeans.fit(self.state_space)

#         # 预计算所有状态的类别并存储
#         self.state_categories = {tuple(state): category for state, category in zip(self.state_space, kmeans.labels_)}

#         # 计算每个类别的中心点
#         self.category_centers = kmeans.cluster_centers_

#         # 初始化动作偏好字典
#         self.action_counts = defaultdict(lambda: defaultdict(int))

#     def get_category(self, state):
#         state_tuple = tuple(np.array(state, dtype=np.float32).flatten())
#         if state_tuple in self.state_categories:
#             return self.state_categories[state_tuple]
#         else:
#             # 如果是新状态，找到最近的中心点
#             distances = np.linalg.norm(self.category_centers - state, axis=1)
#             nearest_category = np.argmin(distances)
#             self.state_categories[state_tuple] = nearest_category
#             return nearest_category

#     def update_action_counts(self, state, action):
#         category = self.get_category(state)
#         self.action_counts[category][action] += 1

#     def get_action_prob(self, state):
#         category = self.get_category(state)
#         total_actions = sum(self.action_counts[category].values())
#         if total_actions == 0:
#             return np.ones(self.action_space) / self.action_space  # 均匀分布

#         probs = np.array([self.action_counts[category][action] / total_actions
#                           for action in range(self.action_space)])
#         return probs

#     def compute_belief_distribution(self, state, immediate_belief=None, beta=0.5):
#         prior_probs = self.get_action_prob(state)
#         if immediate_belief is None:
#             return prior_probs

#         combined_probs = beta * prior_probs + (1 - beta) * immediate_belief
#         return combined_probs / combined_probs.sum()  # 归一化


# import numpy as np
# from sklearn.cluster import MiniBatchKMeans
# from collections import defaultdict

# class StateCategorizer:
#     def __init__(self, state_space, action_space, n_categories):
#         self.state_space = np.array(state_space, dtype=np.float32)
#         self.action_space = action_space
#         self.n_categories = n_categories
#         self.replay_buffer = []
#         self.initialized = False

#     def initialize_clusters(self):
#         flattened_states = np.array(self.replay_buffer).reshape(len(self.replay_buffer), -1)
#         kmeans = MiniBatchKMeans(n_clusters=self.n_categories)
#         kmeans.fit(flattened_states)
#         self.state_categories = {tuple(state): category for state, category in zip(flattened_states, kmeans.labels_)}
#         self.category_centers = kmeans.cluster_centers_
#         self.initialized = True
#         self.action_counts = defaultdict(lambda: defaultdict(int))

#     def add_to_replay_buffer(self, state, buffer_size):
#         self.replay_buffer.append(state)
#         if len(self.replay_buffer) >= buffer_size and not self.initialized:
#             self.initialize_clusters()

#     def get_category(self, state):
#         state_array = np.array(state, dtype=np.float32).flatten()
#         state_tuple = tuple(state_array)
#         if state_tuple in self.state_categories:
#             return self.state_categories[state_tuple]
#         else:
#             distances = np.linalg.norm(self.category_centers - state_array, axis=1)
#             nearest_category = np.argmin(distances)
#             self.state_categories[state_tuple] = nearest_category
#             return nearest_category

#     def update_action_counts(self, state, action):
#         category = self.get_category(state)
#         self.action_counts[category][action] += 1

#     def get_action_prob(self, state):
#         category = self.get_category(state)
#         total_actions = sum(self.action_counts[category].values())
#         if total_actions == 0:
#             return np.ones(self.action_space) / self.action_space

#         probs = np.array([self.action_counts[category][action] / total_actions
#                           for action in range(self.action_space)])
#         return probs

#     def compute_belief_distribution(self, state, immediate_belief=None, beta=0.5):
#         prior_probs = self.get_action_prob(state)
#         if immediate_belief is None:
#             return prior_probs

#         combined_probs = beta * prior_probs + (1 - beta) * immediate_belief
#         return combined_probs / combined_probs.sum()

import torch
from sklearn.cluster import MiniBatchKMeans
from collections import defaultdict

class StateCategorizer:
    def __init__(self, action_dim, n_categories, buffer_size, device):
        self.action_dim = action_dim
        self.n_categories = n_categories
        self.buffer_size = buffer_size
        self.device = device

        # 初始化状态缓冲区
        self.state_buffer = []
        self.initialized = False

        # 初始化动作偏好字典
        self.action_counts = defaultdict(lambda: defaultdict(int))
        self.belief_mu = defaultdict(lambda: torch.zeros(action_dim, device=device))  # Mean
        self.belief_sigma2 = defaultdict(lambda: torch.ones(action_dim, device=device))  # Variance
        self.counts = defaultdict(int)

    def initialize_clusters(self):
        flattened_states = torch.stack(self.state_buffer).view(len(self.state_buffer), -1).cpu().numpy()
        kmeans = MiniBatchKMeans(n_clusters=self.n_categories)
        kmeans.fit(flattened_states)
        self.category_centers = torch.tensor(kmeans.cluster_centers_).to(self.device)
        self.state_categories = {tuple(state): category for state, category in zip(flattened_states, kmeans.labels_)}
        self.initialized = True

    def add_to_state_buffer(self, state):
        state_tensor = torch.as_tensor(state).view(-1).to(self.device)
        if len(self.state_buffer) < self.buffer_size:
            self.state_buffer.append(state_tensor)
        if len(self.state_buffer) >= self.buffer_size and not self.initialized:
            self.initialize_clusters()

    def get_category(self, state):
        state_tensor = torch.as_tensor(state).view(-1).to(self.device)
        state_tuple = tuple(state_tensor.cpu().numpy())
        if state_tuple in self.state_categories:
            return self.state_categories[state_tuple]
        else:
            distances = torch.norm(self.category_centers - state_tensor, dim=1)
            nearest_category = torch.argmin(distances).item()
            self.state_categories[state_tuple] = nearest_category
            return nearest_category
            
    def get_categories_batch(self, states_batch):
        """Get categories for a batch of states."""
        categories = []
        for state in states_batch:
            category = self.get_category(state)
            categories.append(category)
        return torch.tensor(categories, device=self.device)

    def update_action_counts(self, state, action):
        category = self.get_category(state)
        self.action_counts[category][action] += 1

    def get_action_prob(self, state):
        category = self.get_category(state)
        total_actions = sum(self.action_counts[category].values())
        if total_actions == 0:
            return torch.ones(self.action_space, device=self.device) / self.action_space

        probs = torch.tensor([self.action_counts[category][action] / total_actions for action in range(self.action_space)], device=self.device)
        return probs

    def compute_belief_distribution(self, state, immediate_belief=None, beta=0.5):
        prior_probs = self.get_action_prob(state)
        if immediate_belief is None:
            return prior_probs

        combined_probs = beta * prior_probs + (1 - beta) * immediate_belief
        return combined_probs / combined_probs.sum()

    def compute_belief_distribution(self, state, immediate_belief=None, beta=0.5):
        prior_probs = self.get_action_prob(state)
        if immediate_belief is None:
            return prior_probs

        combined_probs = beta * prior_probs + (1 - beta) * immediate_belief
        return combined_probs / combined_probs.sum()

    # def update_belief(self, state, dist):
    #     """Apply Bayesian update to belief distribution parameters based on new action."""
    #     category = self.get_category(state)
    #     mu_b = self.belief_mu[category]
    #     sigma2_b = self.belief_sigma2[category]
    #
    #     # Bayesian update for Gaussian parameters
    #     sigma2_a = dist[1]
    #     mu_b_new = (sigma2_b * dist[0] + sigma2_a * mu_b) / (sigma2_b + sigma2_a)
    #     sigma2_b_new = (1 / (1 / sigma2_b + 1 / sigma2_a))
    #
    #     # Update the belief parameters for the category
    #     self.belief_mu[category] = mu_b_new
    #     self.belief_sigma2[category] = sigma2_b_new

    def update_belief(self, category, dist):
        """使用增量更新的贝叶斯更新方法，更新每个类别的均值和方差。"""
        # category = self.get_category(state)
        mu_b = self.belief_mu[category]
        sigma2_b = self.belief_sigma2[category]
        count = self.counts[category]

        # 新数据的均值和方差
        mu_a, sigma2_a= dist.get_param()
        # sigma2_a = dist.get_variance()

        # 更新计数器
        self.counts[category] += 1
        new_count = self.counts[category]

        # 增量更新均值和方差
        mu_b_new = (count * mu_b + mu_a) / new_count
        sigma2_b_new = (count * (sigma2_b + mu_b ** 2) + sigma2_a + mu_a ** 2) / new_count - mu_b_new ** 2

        # 更新均值和方差
        self.belief_mu[category] = mu_b_new
        self.belief_sigma2[category] = sigma2_b_new

    def get_belief_distribution(self, state):
        """Retrieve the current belief distribution (Gaussian) for the given state."""
        category = self.get_category(state)
        mu_b = self.belief_mu[category]
        sigma2_b = self.belief_sigma2[category]
        count_b = self.counts[category]
        return mu_b, sigma2_b, count_b
