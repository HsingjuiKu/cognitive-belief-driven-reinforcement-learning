from xuance.torchAgent.learners import *
import numpy as np
import torch
from xuance.state_categorizer import StateCategorizer


def clipped_softmax(x, beta, k):
    topk_indices = torch.topk(x, k=k, dim=1).indices
    clipped_x = torch.full_like(x, float('-inf'))
    for i, indices in enumerate(topk_indices):
        clipped_x[i, indices] = x[i, indices]
    e_x = torch.exp(beta * clipped_x - torch.max(beta * clipped_x, dim=1, keepdim=True).values)
    return e_x / e_x.sum(dim=1, keepdim=True)


# class CBDDQN_Learner(Learner):
#     def __init__(self,
#                  policy: nn.Module,
#                  optimizer: torch.optim.Optimizer,
#                  scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
#                  device: Optional[Union[int, str, torch.device]] = None,
#                  model_dir: str = "./",
#                  gamma: float = 0.99,
#                  sync_frequency: int = 100):
#         self.gamma = gamma
#         self.sync_frequency = sync_frequency
#         super(CBDDQN_Learner, self).__init__(policy, optimizer, scheduler, device, model_dir)

#     def update(self, obs_batch, act_batch, rew_batch, next_batch, terminal_batch, k, state_categorizer):
#         self.iterations += 1

#         # obs_batch = torch.tensor(obs_batch, device=self.device).float()
#         # act_batch = torch.tensor(act_batch, device=self.device).long()
#         # rew_batch = torch.tensor(rew_batch, device=self.device).float()
#         # terminal_batch = torch.tensor(terminal_batch, device=self.device).float()
#         # next_batch = torch.tensor(next_batch, device=self.device).float()

#         beta = min(0.1 + 0.0001 * self.iterations, 10.0)
#         beta_dynamic = min(0.0 + 0.00001 * self.iterations, 0.5)

#         obs_batch = torch.tensor(obs_batch, device=self.device)
#         act_batch = torch.tensor(act_batch, device=self.device)
#         rew_batch = torch.tensor(rew_batch, device=self.device)
#         ter_batch = torch.tensor(terminal_batch, device=self.device)
#         next_batch = torch.tensor(next_batch, device=self.device)

#         _, _, evalQ = self.policy(obs_batch)
#         _, _, targetQ = self.policy.target(next_batch)

#         # belief_distributions = clipped_softmax(targetQ, beta=beta, k = k)
#         # belief_distributions = torch.tensor(belief_distributions).clone().detach().float().to(self.device)
#         #
#         # # belief_distributions = torch.tensor(belief_distributions, device=self.device).float()
#         # targetQ = (targetQ * belief_distributions).sum(dim=-1)
#         # targetQ = rew_batch + self.gamma * (1 - ter_batch) * targetQ
#         # predictQ = (evalQ * F.one_hot(act_batch.long(), evalQ.shape[1]).float()).sum(dim=-1)

#         # prior_probs = np.array([cluster_tool.compute_belief_distribution(next_batch[i]) for i in range(len(next_batch))])
#         prior_probs = np.array(
#             [state_categorizer.get_action_prob(next_batch[i].cpu().numpy()) for i in range(len(next_batch))])
#         prior_probs = torch.tensor(prior_probs, device=self.device).float()
#         clipped_dist = clipped_softmax(targetQ, beta, k)
#         belief_distributions = beta_dynamic * prior_probs + (1 - beta_dynamic) * clipped_dist


#         targetQ = (targetQ * belief_distributions).sum(dim=-1)
#         targetQ = rew_batch + self.gamma * (1 - ter_batch) * targetQ
#         predictQ = (evalQ * F.one_hot(act_batch.long(), evalQ.shape[1]).float()).sum(dim=-1)

#         loss = F.mse_loss(predictQ, targetQ)
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
#         if self.scheduler is not None:
#             self.scheduler.step()
#         if self.iterations % self.sync_frequency == 0:
#             self.policy.copy_target()
#         lr = self.optimizer.state_dict()['param_groups'][0]['lr']

#         info = {
#             "Qloss": loss.item(),
#             "learning_rate": lr,
#             "predictQ": predictQ.mean().item()
#         }

#         return info


class CBDDQN_Learner(Learner):
    def __init__(self,
                 policy: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 device: Optional[Union[int, str, torch.device]] = None,
                 model_dir: str = "./",
                 gamma: float = 0.99,
                 sync_frequency: int = 100):
        self.gamma = gamma
        self.sync_frequency = sync_frequency
        super(CBDDQN_Learner, self).__init__(policy, optimizer, scheduler, device, model_dir)

    def update(self, obs_batch, act_batch, rew_batch, next_batch, terminal_batch, k, state_categorizer):
        self.iterations += 1

        beta = min(0.1 + 0.0001 * self.iterations, 10.0)

        act_batch = torch.tensor(act_batch, device=self.device)
        rew_batch = torch.tensor(rew_batch, device=self.device)
        ter_batch = torch.tensor(terminal_batch, device=self.device)

        _, _, evalQ = self.policy(obs_batch)
        _, _, targetQ = self.policy.target(next_batch)
        times = 0

        obs_batch = torch.tensor(obs_batch, device=self.device)
        next_batch = torch.tensor(next_batch, device=self.device)

        if state_categorizer.initialized:
            # beta_dynamic = min(0.5 + 0.00001 * times, 1)
            beta_dynamic = 1
            # prior_probs = np.array(
            #     [state_categorizer.get_action_prob(next_batch[i].cpu().numpy()) for i in range(len(next_batch))])
            # prior_probs = torch.tensor(prior_probs, device=self.device).float()

            prior_probs = torch.stack(
                [state_categorizer.get_action_prob(next_batch[i]) for i in range(len(next_batch))])
            prior_probs = prior_probs.to(self.device).float()
            clipped_dist = clipped_softmax(targetQ, beta, k)
            belief_distributions = beta_dynamic * prior_probs + (1 - beta_dynamic) * clipped_dist
            times += 1
        else:
            clipped_dist = clipped_softmax(targetQ, beta, k)
            belief_distributions = clipped_dist

        targetQ = (targetQ * belief_distributions).sum(dim=-1)
        targetQ = rew_batch + self.gamma * (1 - ter_batch) * targetQ
        predictQ = (evalQ * F.one_hot(act_batch.long(), evalQ.shape[1]).float()).sum(dim=-1)

        loss = F.mse_loss(predictQ, targetQ)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        info = {
            "Qloss": loss.item(),
            "learning_rate": lr,
            "predictQ": predictQ.mean().item()
        }

        return info
