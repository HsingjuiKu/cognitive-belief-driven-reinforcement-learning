# TD3 add three tricks to DDPG:
# 1. noisy action in target actor
# 2. double critic network
# 3. delayed actor update
from xuance.torchAgent.learners import *
from xuance.state_categorizer import StateCategorizer


class CBDTD3_Learner(Learner):
    def __init__(self,
                 policy: nn.Module,
                 optimizers: Sequence[torch.optim.Optimizer],
                 schedulers: Sequence[torch.optim.lr_scheduler._LRScheduler],
                 device: Optional[Union[int, str, torch.device]] = None,
                 model_dir: str = "./",
                 gamma: float = 0.99,
                 tau: float = 0.01,
                 delay: int = 3):
        self.tau = tau
        self.gamma = gamma
        self.delay = delay
        super(CBDTD3_Learner, self).__init__(policy, optimizers, schedulers, device, model_dir)

    def update(self, obs_batch, act_batch, rew_batch, next_batch, terminal_batch, state_categorizer):
        self.iterations += 1
        beta_dynamic = min(0.5 + 0.5/50000 * self.iterations, 1)

        act_batch = torch.as_tensor(act_batch, device=self.device)
        rew_batch = torch.as_tensor(rew_batch, device=self.device)
        ter_batch = torch.as_tensor(terminal_batch, device=self.device)
        obs_batch = torch.tensor(obs_batch, device=self.device)
        next_batch = torch.tensor(next_batch, device=self.device)

        # Critic upate step (same as TD3)
        # Generate double Q values
        action_q_A, action_q_B = self.policy.Qaction(obs_batch, act_batch)
        # next_q = self.policy.Qtarget(next_batch).reshape([-1])
        action_q_A = action_q_A.reshape([-1])
        action_q_B = action_q_B.reshape([-1])

        # category = state_categorizer.get_categories_batch(obs_batch)
        # print(category)

        # # update 
        # new_next_q = []
        # if state_categorizer.initialized:
        #     for i in range(len(obs_batch)):                 
        #         mem,det,kappa,Beta = state_categorizer.calcualte_beta(sigma0_sq, category[i], action_q_A[i].detach(),action_q_B[i].detach())
                
        #         new_next_q.append((1 - beta_dynamic) * next_q[i] + beta_dynamic * state_categorizer.mukappa[category[i],0])
        #         # print(category)
        #         state_categorizer.update_mukappa(Beta, next_q[i], category[i] )
           
        # new_q = torch.tensor(new_next_q,device = self.device)
        next_q = self.policy.Qtarget(next_batch).reshape(-1)
        target_q = rew_batch + self.gamma * (1 - ter_batch) * next_q
        q_loss = F.mse_loss(action_q_A, target_q.detach()) + F.mse_loss(action_q_B, target_q.detach())
        self.optimizer[1].zero_grad()
        q_loss.backward()
        self.optimizer[1].step()
        if self.scheduler is not None:
            self.scheduler[1].step()

        # Get the cluster category batch for each data
        category = state_categorizer.get_categories_batch(obs_batch)
        # actor update
        if self.iterations % self.delay == 0:
            # 1️⃣ Generate Action distribution (actor 参与训练)
            rep = self.policy.actor_representation(obs_batch)['state']
            a_det = self.policy.actor(rep)  # 这个保留计算图，用于 loss 和训练

            # 2️⃣ Estimate Gradient through current min Q and Action Distribution (Q 值和 loss 计算，a_blend 也是用它生成)
            q1 = self.policy.critic_A(rep, a_det)
            q2 = self.policy.critic_B(rep, a_det)
            q_min = torch.min(q1, q2).mean()
            grad = torch.autograd.grad(q_min, a_det, retain_graph=True)[0]  # ✅ 提取 grad 但保留图

            # 3️⃣ Experience Combination (用 grad 和 phi_batch 计算 delta，融合生成 a_blend)
            grad = grad.detach()
            phi_batch = torch.stack([
                state_categorizer.phi_batch[k].to(self.device) for k in category], dim=0)
            alpha = torch.sum(grad[:, -1:] * phi_batch[:, -1:], dim=1, keepdim=True) # 
            beta = torch.clamp(alpha, min=0.0, max=1.0)
            d_i = beta * phi_batch + (1 - beta) * grad
            d_i = d_i / (d_i.norm(dim=1, keepdim=True) + 1e-6)
            delta = 0.05 * d_i

            # 4️⃣ 限制 delta 范围
            rad_max = (2 * 0.02)**0.5 * 0.1
            delta_norm = delta.norm(dim=1, keepdim=True)
            delta = torch.where(delta_norm > rad_max, delta * (rad_max / (delta_norm + 1e-8)), delta)

            # 5️⃣ 构造 a_blend，并继续用于后续 loss
            a_blend = (a_det + delta).clamp(-1, 1)

            
            za = self.policy.critic_A_representation(obs_batch)['state']
            q1_blend = self.policy.critic_A(za , a_blend).reshape(-1)
            zb = self.policy.critic_B_representation(obs_batch)['state']
            q2_blend = self.policy.critic_B(zb , a_blend).reshape(-1)

            policy_q = torch.min(q1_blend,q2_blend)
            # policy_q = self.policy.Qpolicy(obs_batch)
            p_loss = -policy_q.mean()
            
            self.optimizer[0].zero_grad()
            p_loss.backward()
            self.optimizer[0].step()
            if self.scheduler is not None:
                self.scheduler[0].step()
            self.policy.soft_update(self.tau)

        # update gradient 
        rep = self.policy.critic_A_representation(obs_batch)['state']
        a_det = self.policy.actor(rep).detach().requires_grad_()
        # a_det.requires_grad_(True)
        q1 = self.policy.critic_A(rep,a_det)
        q2 = self.policy.critic_B(rep,a_det)
        q_min = torch.min(q1,q2).mean()
        grad = torch.autograd.grad(q_min , a_det)[0]

        grad = grad.detach()
        grad = grad/(grad.norm(p=2, dim=1, keepdim=True) + 1e-6)

        # 更新每个簇族
        state_categorizer.update_phi(category, grad)
        # a_det.requires_grad_(False)

        actor_lr = self.optimizer[0].state_dict()['param_groups'][0]['lr']
        critic_lr = self.optimizer[1].state_dict()['param_groups'][0]['lr']

        info = {
            "Qloss": q_loss.item(),
            "QvalueA": action_q_A.mean().item(),
            "QvalueB": action_q_B.mean().item(),
            "actor_lr": actor_lr,
            "critic_lr": critic_lr
        }
        if self.iterations % self.delay == 0:
            info["Ploss"] = p_loss.item()

        return info
