# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from collections import defaultdict

# # -------------------------------
# # 1) 定义简单的VQ模块 (vector quantization)
# # -------------------------------
# class VectorQuantizerEMA(nn.Module):
#     """
#     VQ-variations很多，这里演示的是EMA(Exponential Moving Average)版，如需纯勾股损失也可自行更改
#     """
#     def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, eps=1e-5):
#         super().__init__()
#         self.num_embeddings = num_embeddings   # n_categories
#         self.embedding_dim = embedding_dim     # latent维度
#         self.commitment_cost = commitment_cost
#         self.decay = decay
#         self.eps = eps

#         # codebook向量，shape=[n_categories, embed_dim]
#         self.embeddings = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
#         self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
#         self.ema_w = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

#     def forward(self, inputs):
#         """
#         inputs shape: [B, D, H, W] 或 [B, D] 或其他, 这里假定[Batch, C, H, W]格式
#         1. flatten -> 找到最近的 codebook 索引
#         2. 计算 VQ 损失
#         3. 返回 quantized 向量 + vq_loss + perplexity
#         """
#         # [B,C,H,W], flatten => [B*H*W, C]
#         flatten = inputs.permute(0,2,3,1).contiguous()  # => [B,H,W,C]
#         flatten = flatten.view(-1, self.embedding_dim)  # => [BHW, D]

#         # compute distances => [BHW, n_categories]
#         distances = (flatten**2).sum(dim=1, keepdim=True) \
#                     - 2*flatten @ self.embeddings.t() \
#                     + (self.embeddings**2).sum(dim=1, keepdim=True).t()
#         # 找最近索引
#         encoding_indices = torch.argmin(distances, dim=1)  # [BHW]

#         # 量化向量 => gather选embedding
#         quantized = self.embeddings[encoding_indices]      # [BHW, D]
#         quantized = quantized.view(*inputs.shape)  # reshape回 [B,C,H,W]

#         # 计算 perplexity (信息熵度量)
#         encodings_onehot = F.one_hot(encoding_indices, self.num_embeddings).float()  # [BHW, K]
#         avg_probs = torch.mean(encodings_onehot, dim=0)
#         perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

#         # EMA 更新, 仅训练时生效 (requires_grad=False)
#         if self.training:
#             self.ema_cluster_size.data.mul_(self.decay).add_(1 - self.decay, encodings_onehot.sum(dim=0))
#             n = self.ema_cluster_size.sum()
#             cluster_size = ((self.ema_cluster_size + self.eps) / (n + self.num_embeddings*self.eps) * n)
#             dw = encodings_onehot.t() @ flatten
#             self.ema_w.data.mul_(self.decay).add_(1 - self.decay, dw)
#             updated_embed = self.ema_w / cluster_size.unsqueeze(1)
#             # normalize
#             self.embeddings.data = updated_embed

#         # 计算 commitment 损失
#         vq_loss = self.commitment_cost * F.mse_loss(inputs, quantized.detach())
#         # straight-through: 让quantized的梯度回传时对embedding保持一致
#         quantized = inputs + (quantized - inputs).detach()

#         return quantized, vq_loss, perplexity, encoding_indices.view(inputs.shape[0], inputs.shape[2], inputs.shape[3])

# # -------------------------------
# # 2) 定义 Encoder 和 Decoder (卷积示例)
# # -------------------------------
# class AtariEncoder(nn.Module):
#     """简单卷积下采样，输出 latent空间 [B, latent_dim, H', W']。可以自行改深度、加BN、加skip conn等"""
#     def __init__(self, in_channels=4, out_channels=64):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels, 32, 4, stride=2, padding=1)  # => h/2
#         self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)           # => h/4
#         self.conv3 = nn.Conv2d(64, out_channels, 3, stride=1, padding=1) # => h/4
#         # 形状 rough: [B,64,H/4,W/4]

#     def forward(self, x):
#         # x shape: [B,4,84,84] or [B,3,84,84] etc.
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = self.conv3(x)  # 这里不加relu, 保留后面量化空间
#         return x

# class AtariDecoder(nn.Module):
#     """简单转置卷积上采样回到原图大小"""
#     def __init__(self, in_channels=64, out_channels=4):
#         super().__init__()
#         # 逆过程: x => [B,64,H/4,W/4] => [B,32,H/2,W/2] => [B,4,H,W]
#         self.deconv1 = nn.ConvTranspose2d(in_channels, 64, 4, stride=2, padding=1)
#         self.deconv2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
#         self.deconv3 = nn.ConvTranspose2d(32, out_channels, 3, stride=1, padding=1)

#     def forward(self, x):
#         x = F.relu(self.deconv1(x))
#         x = F.relu(self.deconv2(x))
#         x = self.deconv3(x)  # 最后可不加激活, 保留原图范围
#         return x

# # -------------------------------
# # 3) 整合成一个 VQVAE 模型
# # -------------------------------
# class VQVAE(nn.Module):
#     def __init__(self, in_channels=4, latent_dim=64, n_categories=512, commitment_cost=0.25):
#         super().__init__()
#         self.encoder = AtariEncoder(in_channels, latent_dim)
#         self.vq_layer = VectorQuantizerEMA(num_embeddings=n_categories,
#                                            embedding_dim=latent_dim,
#                                            commitment_cost=commitment_cost)
#         self.decoder = AtariDecoder(in_channels=latent_dim, out_channels=in_channels)

#     def forward(self, x):
#         # x: [B,C,H,W]
#         z = self.encoder(x)                          # => [B, latent_dim, H/4, W/4]
#         z_q, vq_loss, perp, idx_map = self.vq_layer(z)
#         x_recon = self.decoder(z_q)                  # => [B, C, H, W]
#         return x_recon, vq_loss, perp, idx_map

# # -------------------------------
# # 4) 在 StateCategorizer 中使用 VQVAE
# # -------------------------------
# class StateCategorizer:
#     def __init__(self, 
#                  action_space, 
#                  n_categories, 
#                  buffer_size, 
#                  device,
#                  in_channels=4,  # Atari 通常是 3 或 4 通道
#                  latent_dim=64,
#                  commitment_cost=0.25):
#         """
#         保持原有命名和大部分接口。只是在内部换成 VQ-VAE 的训练和离散索引作为类别。
#         """
#         self.action_dim = action_space
#         self.n_categories = n_categories  # 这里作为 codebook 大小
#         self.buffer_size = buffer_size
#         self.device = device

#         # 状态缓冲区 (保存图像)
#         self.state_buffer = []
#         self.initialized = False

#         # 新增: VQVAE 模型
#         self.vqvae = VQVAE(in_channels=in_channels,
#                            latent_dim=latent_dim,
#                            n_categories=n_categories,
#                            commitment_cost=commitment_cost).to(device)

#         # 用于后续离线训练
#         self.optimizer = torch.optim.Adam(self.vqvae.parameters(), lr=1e-4)
#         self.vqvae.train()

#         # 保存 (tuple(state), category) 映射
#         self.state_categories = dict()

#         # 其他 RL 相关的统计 (保持不变)
#         self.action_counts = defaultdict(lambda: defaultdict(int))
#         self.belief_mu = defaultdict(lambda: torch.zeros(self.action_dim, device=device)) 
#         self.belief_sigma2 = defaultdict(lambda: torch.ones(self.action_dim, device=device))
#         self.counts = defaultdict(int)

#     def add_to_state_buffer(self, state):
#         """
#         The expected state is of shape (C, H, W). If the state is in HWC format,
#         we convert it to CHW format.
#         """
#         if isinstance(state, np.ndarray):
#             state_tensor = torch.from_numpy(state).float()
#         else:
#             state_tensor = state.float()
    
#         # If the state shape is (H, W, C) instead of (C, H, W), transpose it:
#         if state_tensor.ndim == 3 and state_tensor.shape[-1] == 4 and state_tensor.shape[0] != 4:
#             state_tensor = state_tensor.permute(2, 0, 1)
    
#         if state_tensor.ndim != 3:
#             raise ValueError(f"Expected state shape (C, H, W), got {state_tensor.shape}")

#         self.state_buffer.append(state_tensor)
#         if len(self.state_buffer) >= self.buffer_size and not self.initialized:
#             self.initialize_clusters()


#     def initialize_clusters(self):
#         """
#         真正开始训练 VQ-VAE (离线)。和原先的 KMeans 不同，实际上是:
#          1) 收集到 self.state_buffer (大于 buffer_size)
#          2) 做若干 epoch 的重构训练
#          3) 训练完成后, 对所有状态做一次前向计算, 得到 codebook index => 作为类别
#          4) self.initialized = True
#         """
#         # 准备训练集
#         states = torch.stack(self.state_buffer, dim=0).to(self.device)
#         # shape: [N, C, H, W]

#         # 训练若干epoch (示例5)
#         n_epochs = 20
#         batch_size = 32
#         for ep in range(n_epochs):
#             perm = torch.randperm(states.size(0))
#             total_vq_loss = 0.0
#             total_recon_loss = 0.0
#             for i in range(0, states.size(0), batch_size):
#                 idx = perm[i: i+batch_size]
#                 batch = states[idx]  # [B,C,H,W]
#                 self.optimizer.zero_grad()

#                 x_recon, vq_loss, _, _ = self.vqvae(batch)
#                 recon_loss = F.mse_loss(x_recon, batch)

#                 loss = recon_loss + vq_loss
#                 loss.backward()
#                 self.optimizer.step()

#                 total_vq_loss += vq_loss.item()
#                 total_recon_loss += recon_loss.item()

#             print(f"[VQVAE init] Epoch {ep+1}/{n_epochs}, ReconLoss={total_recon_loss:.3f}, VQLoss={total_vq_loss:.3f}")

#         # 训练完后,我们给每个 state 分配一个 codebook index
#         self.vqvae.eval()
#         with torch.no_grad():
#             x_encoded = []
#             for i in range(0, states.size(0), batch_size):
#                 batch = states[i: i+batch_size]
#                 _, _, _, idx_map = self.vqvae(batch)
#                 # idx_map shape: [B,H/4,W/4], 这里我们可以简单的把它再 flatten
#                 # 也可以取 idx_map 的某种聚合, 但为了简化, 可以取 idx_map.mean dim=> int
#                 # 这里演示：对 idx_map 全部取平均 => round => 作为 "类别"
#                 # 也可保留 idx_map 全部像素 => 可能会非常多, 仅用于演示
#                 # 例如, 取 idx_map[:,0,0], 只取左上角的embedding, ...
#                 # 下面先简单 flatten => most_frequent index
#                 B, h, w = idx_map.shape
#                 idx_map_2d = idx_map.view(B, -1)  # shape [B, h*w]
#                 # 统计每张图出现最多的code index
#                 mode_idx = []
#                 for row in idx_map_2d:
#                     vals, counts = torch.unique(row, return_counts=True)
#                     max_id = vals[torch.argmax(counts)]
#                     mode_idx.append(int(max_id.item()))
#                 x_encoded.extend(mode_idx)

#             # 存到 self.state_categories
#             # 这里 state 的 key 用 tuple+hash(…) 也行
#             # 但图像非常大; 纯粹保存 tuple(…) 不现实. 仅演示
#             # 强烈建议实际使用时只把 index 存 replay buffer ID => category
#             for i in range(len(self.state_buffer)):
#                 # 这里演示把图像flatten成 tuple, 可能非常大,仅供测试
#                 key_tuple = tuple(self.state_buffer[i].cpu().numpy().ravel()[:50])  # 只取前50元素…
#                 cat_id = x_encoded[i]
#                 self.state_categories[key_tuple] = cat_id

#         self.initialized = True
#         self.vqvae.train()  # 恢复train模式

#     def get_category(self, state):
#         """
#         类似原先KMeans找最近中心。现在是: forward到 VQVAE encoder->quantize, 
#         取dominant index(或其他方式)当作类别ID
#         """
#         if not self.initialized:
#             # 未初始化就直接返回-1表示无类别
#             return -1

#         # 确保是tensor [C,H,W]
#         if isinstance(state, np.ndarray):
#             state_tensor = torch.from_numpy(state).float().to(self.device)
#         else:
#             state_tensor = state.float().to(self.device)

#         with torch.no_grad():
#             state_tensor = state_tensor.unsqueeze(0)  # => [1,C,H,W]
#             _, _, _, idx_map = self.vqvae(state_tensor)
#             idx_map_2d = idx_map.view(1, -1)  # => [1, h*w]
#             # 找出现最多的 index
#             vals, counts = torch.unique(idx_map_2d, return_counts=True)
#             cat_id = vals[torch.argmax(counts)].item()

#         return int(cat_id)

#     def get_categories_batch(self, states_batch):
#         """
#         批量获取类别。可以直接循环get_category，或者一次前向
#         """
#         cats = []
#         for s in states_batch:
#             cat = self.get_category(s)
#             cats.append(cat)
#         return torch.tensor(cats, device=self.device)

#     def update_action_counts(self, state, action):
#         cat = self.get_category(state)
#         if cat < 0:
#             return
#         self.action_counts[cat][action] += 1

#     def get_action_prob(self, state):
#         cat = self.get_category(state)
#         if cat < 0:
#             return torch.ones(self.action_dim, device=self.device) / self.action_dim

#         total_actions = sum(self.action_counts[cat].values())
#         if total_actions == 0:
#             return torch.ones(self.action_dim, device=self.device) / self.action_dim

#         probs = torch.tensor([self.action_counts[cat][a] / total_actions 
#                               for a in range(self.action_dim)], device=self.device)
#         return probs

#     def compute_belief_distribution(self, state, immediate_belief=None, beta=0.5):
#         prior_probs = self.get_action_prob(state)
#         if immediate_belief is None:
#             return prior_probs
#         combined_probs = beta * prior_probs + (1 - beta) * immediate_belief
#         return combined_probs / combined_probs.sum()

#     def update_belief(self, category, dist):
#         mu_b = self.belief_mu[category]
#         sigma2_b = self.belief_sigma2[category]
#         count = self.counts[category]

#         mu_a, sigma2_a = dist.get_param()
#         self.counts[category] += 1

#         # Bayesian Update
#         mu_b_new = (sigma2_b * mu_a + sigma2_a * mu_b) / (sigma2_b + sigma2_a)
#         sigma2_b_new = 1 / (1 / sigma2_b + 1 / sigma2_a)

#         self.belief_mu[category] = mu_b_new
#         self.belief_sigma2[category] = sigma2_b_new

#     def get_belief_distribution(self, state):
#         cat = self.get_category(state)
#         mu_b = self.belief_mu[cat]
#         sigma2_b = self.belief_sigma2[cat]
#         count_b = self.counts[cat]
#         return mu_b, sigma2_b, count_b





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