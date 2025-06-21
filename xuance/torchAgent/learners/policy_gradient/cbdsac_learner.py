from xuance.torchAgent.learners import *
from xuance.state_categorizer import StateCategorizer

class CBDSAC_Learner(Learner):
    def __init__(self,
                 policy: nn.Module,
                 optimizers: Sequence[torch.optim.Optimizer],
                 schedulers: Sequence[torch.optim.lr_scheduler._LRScheduler],
                 device: Optional[Union[int, str, torch.device]] = None,
                 model_dir: str = "./",
                 **kwargs):
        self.tau = kwargs['tau']
        self.gamma = kwargs['gamma']
        self.alpha = kwargs['alpha']
        self.use_automatic_entropy_tuning = kwargs['use_automatic_entropy_tuning']
        super(CBDSAC_Learner, self).__init__(policy, optimizers, schedulers, device, model_dir)
        if self.use_automatic_entropy_tuning:
            self.target_entropy = kwargs['target_entropy']
            self.log_alpha = nn.Parameter(torch.zeros(1, requires_grad=True, device=device))
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=kwargs['lr_policy'])

    def update(self, obs_batch, act_batch, rew_batch, next_batch, terminal_batch, state_categorizer):
        self.iterations += 1
        Beta = min(0.5 + 0.5/1000000 * self.iterations, 1)
        act_batch = torch.as_tensor(act_batch, device=self.device)
        rew_batch = torch.as_tensor(rew_batch, device=self.device)
        ter_batch = torch.as_tensor(terminal_batch, device=self.device)

        # actor update
        p, policy_q_1, policy_q_2,dist = self.policy.Qpolicy(obs_batch)
        new_dist = dist
        

        # critic update
        action_q_1, action_q_2 = self.policy.Qaction(obs_batch, act_batch)
        log_pi_next, target_q,next_dist = self.policy.Qtarget(next_batch)
        new_next_dist = next_dist
        


        if state_categorizer.initialized:
            # 获取当前 obs_batch 中每个状态的均值和方差，并与类别均值和方差加权平均
            new_means_actor = []
            new_vars_actor = []

            new_means_critics = []
            new_vars_critics = []


            mean_current_actor, var_current_actor = dist.get_param()  # 当前状态的均值
            mean_current_critics, var_current_critics = next_dist.get_param()
    
            for i in range(len(obs_batch)):
                # 获取类别均值和方差
                belief_mu, belief_sigma2, _ = state_categorizer.get_belief_distribution(obs_batch[i])
                # 使用 Beta 平滑权重当前观测和信念均值、方差
                mean_new_actor = Beta * belief_mu + (1 - Beta) * mean_current_actor[i,:]
                var_new_actor = Beta * belief_sigma2 + (1 - Beta) * var_current_actor[i,:]

                mean_new_critics = Beta * belief_mu + (1 - Beta) * mean_current_critics[i,:]
                var_new_critics = Beta * belief_sigma2 + (1 - Beta) * var_current_critics[i,:]
                # 将结果添加到新的均值和方差列表中
                new_means_actor.append(mean_new_actor)
                new_vars_actor.append(var_new_actor)

                new_means_critics.append(mean_new_critics)
                new_vars_critics.append(var_new_critics)

            # 将更新后的均值和方差堆叠为张量
            updated_means_actor = torch.stack(new_means_actor)
            updated_vars_actor = torch.stack(new_vars_actor)

            updated_means_critics = torch.stack(new_means_critics)
            updated_vars_critics = torch.stack(new_vars_critics)
            # 更新 dist 为加权后的均值和方差
            new_dist.set_param(updated_means_actor, updated_vars_actor)

            new_next_dist.set_param(updated_means_critics, updated_vars_critics)
    
        _,log_pi = new_dist.activated_rsample_and_logprob()
        _,log_pi_next = new_next_dist.activated_rsample_and_logprob()
        
        policy_q = torch.min(policy_q_1, policy_q_2).reshape([-1])
        p_loss = (self.alpha * log_pi.reshape([-1]) - policy_q).mean()
        self.optimizer[0].zero_grad()
        p_loss.backward()
        self.optimizer[0].step()

        target_value = target_q - self.alpha * log_pi_next.reshape([-1])
        backup = rew_batch + (1 - ter_batch) * self.gamma * target_value
        q_loss = F.mse_loss(action_q_1, backup.detach()) + F.mse_loss(action_q_2, backup.detach())
        self.optimizer[1].zero_grad()
        q_loss.backward()
        self.optimizer[1].step()

        # automatic entropy tuning
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0

        if self.scheduler is not None:
            self.scheduler[0].step()
            self.scheduler[1].step()

        self.policy.soft_update(self.tau)

        actor_lr = self.optimizer[0].state_dict()['param_groups'][0]['lr']
        critic_lr = self.optimizer[1].state_dict()['param_groups'][0]['lr']

        info = {
            "Qloss": q_loss.item(),
            "Ploss": p_loss.item(),
            "Qvalue": policy_q.mean().item(),
            "actor_lr": actor_lr,
            "critic_lr": critic_lr,
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
        }

        return info
