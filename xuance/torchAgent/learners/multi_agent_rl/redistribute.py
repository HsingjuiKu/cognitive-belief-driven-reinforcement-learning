# import torch.nn as nn
# import torch.nn.functional as F
# import torch



# class EnhancedCausalModel(nn.Module):
#     def __init__(self, num_agents, obs_dim, action_dim, device):
#         super().__init__()
#         self.num_agents = num_agents
#         self.obs_dim = obs_dim
#         self.action_dim = action_dim
#         self.device = device

#         self.network = nn.Sequential(
#             nn.Linear(obs_dim + action_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, action_dim)
#         ).to(device)

#     def predict_others_actions(self, obs, action):
#         return self.network(torch.cat([obs, action], dim=-1))

#     def calculate_social_influence(self, obs, actions):
#         batch_size, num_agents, obs_dim = obs.shape
#         influences = []

#         adaptive_factor = max(10, num_agents)

#         for k in range(num_agents):
#             agent_idx = k % num_agents
#             obs_k = obs[:, agent_idx]
#             action_k = actions[:, agent_idx]
#             p_with_k = self.predict_others_actions(obs_k, action_k)
#             p_without_k = self.predict_others_actions(obs_k, torch.zeros_like(action_k).to(self.device))
#             for _ in range(adaptive_factor):
#                 counterfactual_actions = torch.rand_like(action_k).to(self.device)  # Generate random actions
#                 p_without_k += self.predict_others_actions(obs_k, counterfactual_actions)
#             p_without_k /= (adaptive_factor + 1)

#             influence = F.kl_div(
#                 p_with_k.log_softmax(dim=-1),
#                 p_without_k.softmax(dim=-1),
#                 reduction='batchmean'
#             )
#             influences.append(influence.unsqueeze(-1))
#         influences = torch.stack(influences, dim=-1)
#         influences = F.softmax(influences, dim=-2)
#         influences = influences.unsqueeze(1)
#         return influences

#     def calculate_social_contribution_index(self, obs, actions):
#         influences = self.calculate_social_influence(obs, actions)
#         return influences

#     def calculate_tax_rates(self, social_contribution_index):
#         return torch.sigmoid(social_contribution_index)

#     def redistribute_rewards(self, original_rewards, social_contribution_index, tax_rates, beta=0.5, alpha=1.0):
#         central_pool = (tax_rates * original_rewards).sum(dim=1, keepdim=True)
#         normalized_contributions = social_contribution_index / (social_contribution_index.sum(dim=1, keepdim=True) + 1e-8)
#         redistributed_rewards = (1 - tax_rates) * original_rewards + beta * normalized_contributions * central_pool
#         # return alpha * redistributed_rewards + (1 - alpha) * original_rewards
#         redistributed_rewards = redistributed_rewards.sum(dim=-1, keepdim=True)
#         return redistributed_rewards


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len=1000):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, x, seq_len):
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb[None, :seq_len, :]

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, pos_emb):
    q = q + pos_emb
    k = k + pos_emb
    return q, k

class GLU(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.act = nn.GELU()
        self.fc = nn.Linear(d_model, d_model * 2)

    def forward(self, x):
        x, gate = self.fc(x).chunk(2, dim=-1)
        return x * self.act(gate)

class EnhancedTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.glu = GLU(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos_emb):
        q, k = apply_rotary_pos_emb(x, x, pos_emb)
        attn_output, _ = self.attention(q, k, x)
        x = self.norm1(x + self.dropout(attn_output))
        glu_output = self.glu(x)
        x = self.norm2(x + self.dropout(glu_output))
        return x

class EnhancedCausalModel(nn.Module):
    def __init__(self, num_agents, obs_dim, action_dim, device, d_model=64, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.d_model = d_model

        self.input_proj = nn.Linear(obs_dim + action_dim, d_model).to(device)
        self.pos_emb = RotaryPositionalEmbedding(d_model).to(device)
        self.transformer_layers = nn.ModuleList([
            EnhancedTransformerBlock(d_model, num_heads, dropout).to(device) for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(d_model, action_dim).to(device)

    def forward(self, obs, actions):
        x = torch.cat([obs, actions], dim=-1).to(self.device)
        x = self.input_proj(x)
        x = x.permute(1, 0, 2)  # (seq_len, batch, d_model)

        pos_emb = self.pos_emb(x, x.size(0))

        for layer in self.transformer_layers:
            x = layer(x, pos_emb)

        x = x.permute(1, 0, 2)  # (batch, seq_len, d_model)
        return self.output_proj(x)

    def predict_others_actions(self, obs, action):
        return self.forward(obs.unsqueeze(1), action.unsqueeze(1)).squeeze(1)

    def calculate_social_influence(self, obs, actions):
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        batch_size, num_agents, obs_dim = obs.shape
        influences = []

        adaptive_factor = max(10, num_agents)

        for k in range(num_agents):
            agent_idx = k % num_agents
            obs_k = obs[:, agent_idx]
            action_k = actions[:, agent_idx]
            p_with_k = self.predict_others_actions(obs_k, action_k)
            p_without_k = self.predict_others_actions(obs_k, torch.zeros_like(action_k).to(self.device))
            for _ in range(adaptive_factor):
                counterfactual_actions = torch.rand_like(action_k).to(self.device)
                p_without_k += self.predict_others_actions(obs_k, counterfactual_actions)
            p_without_k /= (adaptive_factor + 1)

            influence = F.kl_div(
                p_with_k.log_softmax(dim=-1),
                p_without_k.softmax(dim=-1),
                reduction='batchmean'
            )
            influences.append(influence.unsqueeze(-1))
        influences = torch.stack(influences, dim=-1).to(self.device)
        influences = F.softmax(influences, dim=-2)
        influences = influences.unsqueeze(1)
        return influences

    def calculate_social_contribution_index(self, obs, actions):
        influences = self.calculate_social_influence(obs, actions)
        return influences

    def calculate_tax_rates(self, social_contribution_index):
        return torch.sigmoid(social_contribution_index)

    def redistribute_rewards(self, original_rewards, social_contribution_index, tax_rates, beta=0.5, alpha=1.0):
        central_pool = (tax_rates * original_rewards).sum(dim=1, keepdim=True).to(self.device)
        normalized_contributions = social_contribution_index / (
                    social_contribution_index.sum(dim=1, keepdim=True) + 1e-8)
        redistributed_rewards = (1 - tax_rates) * original_rewards + beta * normalized_contributions * central_pool
        redistributed_rewards = redistributed_rewards.sum(dim=-1, keepdim=True).to(self.device)
        return redistributed_rewards
