from xuance.torchAgent.learners import *
from xuance.torchAgent.learners import Learner
import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, Union

class DoubleGum_Learner(Learner):
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
        super(DoubleGum_Learner, self).__init__(policy, optimizer, scheduler, device, model_dir)

    def beta_log_sum_exp(self, q_values, std):
        spread = std * torch.sqrt(torch.tensor(3.0)) / torch.pi
        exponents = q_values / spread
        log_sum_exp = torch.logsumexp(exponents, dim=-1)
        return log_sum_exp, spread

    def soft_v(self, target_network, obs):
        q_values, std = target_network(obs)
        log_sum_exp, spread = self.beta_log_sum_exp(q_values, std)
        return spread * log_sum_exp

    def update(self, obs_batch, act_batch, rew_batch, next_batch, terminal_batch):
        self.iterations += 1
        act_batch = torch.as_tensor(act_batch, device=self.device)
        rew_batch = torch.as_tensor(rew_batch, device=self.device)
        ter_batch = torch.as_tensor(terminal_batch, device=self.device)

        with torch.no_grad():
            target_v = self.soft_v(self.policy.target, next_batch)
            target_q = rew_batch + self.gamma * (1 - ter_batch) * target_v

        eval_q, std = self.policy(obs_batch)
        spread = std * torch.sqrt(torch.tensor(3.0)) / torch.pi
        eval_q = eval_q.gather(1, act_batch.long().unsqueeze(-1)).squeeze(-1)

        td_loss = (eval_q - target_q).detach()
        loss = torch.log(std) + 0.5 * (td_loss / std) ** 2
        loss = (spread.detach() * loss).mean()

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
            "predictQ": eval_q.mean().item()
        }

        return info
