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


import numpy as np
from sklearn.cluster import MiniBatchKMeans
from collections import defaultdict

class StateCategorizer:
    def __init__(self, state_space, action_space, n_categories):
        self.state_space = np.array(state_space, dtype=np.float32)
        self.action_space = action_space
        self.n_categories = n_categories
        self.replay_buffer = []
        self.initialized = False

    def initialize_clusters(self):
        flattened_states = np.array(self.replay_buffer).reshape(len(self.replay_buffer), -1)
        kmeans = MiniBatchKMeans(n_clusters=self.n_categories)
        kmeans.fit(flattened_states)
        self.state_categories = {tuple(state): category for state, category in zip(flattened_states, kmeans.labels_)}
        self.category_centers = kmeans.cluster_centers_
        self.initialized = True
        self.action_counts = defaultdict(lambda: defaultdict(int))

    def add_to_replay_buffer(self, state, buffer_size):
        self.replay_buffer.append(state)
        if len(self.replay_buffer) >= buffer_size and not self.initialized:
            self.initialize_clusters()

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

