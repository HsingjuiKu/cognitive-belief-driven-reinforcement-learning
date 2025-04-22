from xuance.torchAgent.learners import *
from xuance.state_categorizer import StateCategorizer

class CBDPPO_Learner(Learner):
    def __init__(self,
                 policy: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 device: Optional[Union[int, str, torch.device]] = None,
                 model_dir: str = "./",
                 vf_coef: float = 0.25,
                 ent_coef: float = 0.005,
                 clip_range: float = 0.25,
                 clip_grad_norm: float = 0.25,
                 use_grad_clip: bool = True,
                 ):
        super(CBDPPO_Learner, self).__init__(policy, optimizer, scheduler, device, model_dir)
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.clip_range = clip_range
        self.clip_grad_norm = clip_grad_norm
        self.use_grad_clip = use_grad_clip

    def update(self, obs_batch, act_batch, ret_batch, value_batch, adv_batch, old_logp,state_categorizer):
        self.iterations += 1

        if self.iterations > 50000000/4:
            Beta = min(0.25 + 0.75/50000000 * self.iterations, 1)
        else:
            Beta = 0


        # Beta = min(0.5 + 0.5/50000000 * self.iterations, 1)
        
        act_batch = torch.as_tensor(act_batch, device=self.device)
        ret_batch = torch.as_tensor(ret_batch, device=self.device)
        value_batch = torch.as_tensor(value_batch, device=self.device)
        adv_batch = torch.as_tensor(adv_batch, device=self.device)
        old_logp_batch = torch.as_tensor(old_logp, device=self.device)

        outputs, a_dist, v_pred = self.policy(obs_batch)

        # a_dist.probs 表示网络输出的策略概率 (shape: [batch_size, num_actions])
        network_probs = a_dist.probs

        batch_size = obs_batch.shape[0]
        belief_list = []
        # 针对 batch 中的每个样本，利用 state_categorizer 获取累计信念（先验分布）
        for i in range(batch_size):
            # get_action_prob 返回一个 numpy 数组，形状 [num_actions]
            belief_np = state_categorizer.get_action_prob(obs_batch[i])
            # 转换为 tensor（确保数据类型和设备一致）
            belief_tensor = belief_np.clone().detach().to(device=self.device, dtype=network_probs.dtype)
            belief_list.append(belief_tensor.unsqueeze(0))
        belief_probs = torch.cat(belief_list, dim=0)  # shape: [batch_size, num_actions]

        # 融合概率：mix_probs = (1-beta)*network_probs + beta*belief_probs，再归一化
        mix_probs = (1 - Beta) * network_probs + Beta * belief_probs
        mix_probs = mix_probs / mix_probs.sum(dim=1, keepdim=True)
        # print(mix_probs.shape)
        # 构造融合后的离散分布
        a_dist.set_param(probs=mix_probs)
        
        log_prob = a_dist.log_prob(act_batch)
        
        # ppo-clip core implementations 
        ratio = (log_prob - old_logp_batch).exp().float()
        surrogate1 = ratio.clamp(1.0 - self.clip_range, 1.0 + self.clip_range) * adv_batch
        surrogate2 = adv_batch * ratio
        a_loss = -torch.minimum(surrogate1, surrogate2).mean()

        c_loss = F.mse_loss(v_pred, ret_batch)

        e_loss = a_dist.entropy().mean()
        loss = a_loss - self.ent_coef * e_loss + self.vf_coef * c_loss
        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        # Logger
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        cr = ((ratio < 1 - self.clip_range).sum() + (ratio > 1 + self.clip_range).sum()) / ratio.shape[0]
        
        info = {
            "actor-loss": a_loss.item(),
            "critic-loss": c_loss.item(),
            "entropy": e_loss.item(),
            "learning_rate": lr,
            "predict_value": v_pred.mean().item(),
            "clip_ratio": cr
        }

        return info


    # def update(self, obs_batch, act_batch, ret_batch, value_batch, adv_batch, old_logp):
    #     self.iterations += 1
    #     act_batch = torch.as_tensor(act_batch, device=self.device)
    #     ret_batch = torch.as_tensor(ret_batch, device=self.device)
    #     value_batch = torch.as_tensor(value_batch, device=self.device)
    #     adv_batch = torch.as_tensor(adv_batch, device=self.device)
    #     old_logp_batch = torch.as_tensor(old_logp, device=self.device)

    #     outputs, a_dist, v_pred = self.policy(obs_batch)
    #     log_prob = a_dist.log_prob(act_batch)

    #     # ppo-clip core implementations 
    #     ratio = (log_prob - old_logp_batch).exp().float()
    #     surrogate1 = ratio.clamp(1.0 - self.clip_range, 1.0 + self.clip_range) * adv_batch
    #     surrogate2 = adv_batch * ratio
    #     a_loss = -torch.minimum(surrogate1, surrogate2).mean()

    #     c_loss = F.mse_loss(v_pred, ret_batch)

    #     e_loss = a_dist.entropy().mean()
    #     loss = a_loss - self.ent_coef * e_loss + self.vf_coef * c_loss
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     if self.use_grad_clip:
    #         torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.clip_grad_norm)
    #     self.optimizer.step()
    #     if self.scheduler is not None:
    #         self.scheduler.step()
    #     # Logger
    #     lr = self.optimizer.state_dict()['param_groups'][0]['lr']
    #     cr = ((ratio < 1 - self.clip_range).sum() + (ratio > 1 + self.clip_range).sum()) / ratio.shape[0]
        
    #     info = {
    #         "actor-loss": a_loss.item(),
    #         "critic-loss": c_loss.item(),
    #         "entropy": e_loss.item(),
    #         "learning_rate": lr,
    #         "predict_value": v_pred.mean().item(),
    #         "clip_ratio": cr
    #     }

    #     return info
