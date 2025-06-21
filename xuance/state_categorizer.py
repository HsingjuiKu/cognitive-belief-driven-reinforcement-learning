
import torch
from sklearn.cluster import MiniBatchKMeans
from collections import defaultdict
import numpy as np
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

        # td3 initization
        # create a initial tensor list including mu and kappa value for TD3
        # self.mukappa = torch.zeros((self.n_categories,2), device=device)
        self.phi_batch = torch.zeros((self.n_categories,self.action_dim), device = device)
        self.kappa = torch.zeros(self.n_categories , device = device)

    def initialize_clusters(self):
        flattened_states = torch.stack(self.state_buffer).view(len(self.state_buffer), -1).cpu().numpy()
        kmeans = MiniBatchKMeans(n_clusters=self.n_categories)
        print(self.n_categories)
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

    # def get_category(self, state):
    #     state_tensor = torch.as_tensor(state).view(-1).to(self.device)
    #     state_tuple = tuple(state_tensor.cpu().numpy())
    #     if state_tuple in self.state_categories:
    #         return self.state_categories[state_tuple]
    #     else:
    #         distances = torch.norm(self.category_centers - state_tensor, dim=1)
    #         nearest_category = torch.argmin(distances).item()
    #         self.state_categories[state_tuple] = nearest_category
    #         return nearest_category


    def get_category(self, state):
        state_tensor = torch.as_tensor(state).view(-1).to(self.device)
        if self.initialized:
            # 直接计算与聚类中心的距离，不再记录新的键
            distances = torch.norm(self.category_centers - state_tensor, dim=1)
            return torch.argmin(distances).item()
        else:
            # 未初始化时，仍然用字典进行缓存
            state_tuple = tuple(state_tensor.cpu().numpy())
            if state_tuple in self.state_categories:
                return self.state_categories[state_tuple]
            else:
                # 当缓存未满时，添加新状态
                distances = torch.norm(self.category_centers - state_tensor, dim=1) if hasattr(self, 'category_centers') else None
                # 这里可以简单返回 0 或等待初始化
                new_category = 0 if distances is None else torch.argmin(distances).item()
                self.state_categories[state_tuple] = new_category
                return new_category

            
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
            return torch.ones(self.action_dim, device=self.device) / self.action_dim

        probs = torch.tensor([self.action_counts[category][action] / total_actions for action in range(self.action_dim)], device=self.device)
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

        # # 增量更新均值和方差
        # mu_b_new = (count * mu_b + mu_a) / new_count
        # sigma2_b_new = (count * (sigma2_b + mu_b ** 2) + sigma2_a + mu_a ** 2) / new_count - mu_b_new ** 2
        # 贝叶斯后验更新均值
        mu_b_new = (sigma2_b * mu_a + sigma2_a * mu_b) / (sigma2_b + sigma2_a)
        # 贝叶斯后验更新方差
        sigma2_b_new = 1 / (1 / sigma2_b + 1 / sigma2_a)

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
    
    def update_phi(self, category, grad):
        eta = 0.05
        for k in category: 
            new_phi = (1 - eta) * self.phi_batch[k] + eta * grad[k]
            new_phi = new_phi / (new_phi.norm() + 1e-6)
            self.phi_batch[k] = new_phi

    # def calcualte_beta(self, sigma0_sq, category, Qa, Qb):
    #     Qa = Qa.detach()  # 切断梯度
    #     Qb = Qb.detach()
        
    #     kappa = self.mukappa[category, 1]
    #     s_mem_sq = sigma0_sq / max(kappa, 1)
    #     s_det_sq = torch.clamp((Qa - Qb)**2, min=1e-6)  # 防止除以 0
        
    #     Beta = 1 / (1 + torch.sqrt(s_mem_sq / s_det_sq))
    #     return s_mem_sq, s_det_sq, kappa, Beta.detach()

    # def update_mukappa(self, Beta, Q, category):
    #     with torch.no_grad():  # 禁用梯度计算
    #         if category >= self.mukappa.shape[0]:  # 防止越界
    #             return
    #         self.mukappa[category, 1] = (1 - Beta) * self.mukappa[category, 1] + 1
    #         self.mukappa[category, 0] = self.mukappa[category, 0] * (1 - Beta) + Beta * Q.detach()