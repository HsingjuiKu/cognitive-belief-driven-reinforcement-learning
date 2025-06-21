from xuance.torchAgent.agents import *
from xuance.torchAgent.learners import *
from xuance.torchAgent.learners.policy_gradient.cbdsac_learner import *
from xuance.state_categorizer import StateCategorizer
import torch


class CBDTD3_Agent(Agent):
    """The implementation of TD3 agent.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
        policy: the neural network modules of the agent.
        optimizer: the method of optimizing.
        scheduler: the learning rate decay scheduler.
        device: the calculating device of the model, such as CPU or GPU.
    """
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecEnv_Gym,
                 policy: nn.Module,
                 optimizer: Sequence[torch.optim.Optimizer],
                 scheduler: Optional[Sequence[torch.optim.lr_scheduler._LRScheduler]] = None,
                 device: Optional[Union[int, str, torch.device]] = None):
        self.render = config.render
        self.n_envs = envs.num_envs

        self.gamma = config.gamma
        self.train_frequency = config.training_frequency
        self.start_training = config.start_training
        self.start_noise = config.start_noise
        self.end_noise = config.end_noise
        self.noise_scale = config.start_noise

        self.observation_space = envs.observation_space
        self.action_space = envs.action_space
        self.auxiliary_info_shape = {}

        self.policy2 = policy

        memory = DummyOffPolicyBuffer(self.observation_space,
                                      self.action_space,
                                      self.auxiliary_info_shape,
                                      self.n_envs,
                                      config.buffer_size,
                                      config.batch_size)
        learner = CBDTD3_Learner(policy,
                              optimizer,
                              scheduler,
                              config.device,
                              config.model_dir,
                              config.gamma,
                              config.tau,
                              config.actor_update_delay)
        self.state_categorizer = StateCategorizer(
            action_dim=self.action_space.shape[0],
            n_categories=getattr(config, 'n_clusters', 1),
            buffer_size=1000,
            device=device
        )
        
        super(CBDTD3_Agent, self).__init__(config, envs, policy, memory, learner, device, config.log_dir, config.model_dir)
        self.generate_initial_states()

    def generate_initial_states(self):
        model_path = "models/td3/torchAgent/Humanoid-v4/seed_33_2025_0621_013116/final_train_model.pth"
        self.policy2.load_state_dict(torch.load(model_path, map_location=self.device))
        self.policy2.eval()
    
        # 重置环境，获取初始观测
        value_buffer = []
        obs = self.envs.reset()
        for _ in tqdm(range(5000)):
            with torch.no_grad():
                action = self._action(obs[0], self.noise_scale)
                obs_tensor = torch.as_tensor(obs[0], device=self.device).float()
                action_tensor = torch.as_tensor(action, device = self.device).float()
                
                action_q_A, action_q_B = self.policy.Qaction(obs_tensor, action_tensor)
                value_buffer.append(torch.min(action_q_A, action_q_B))

                next_obs, _, _, _, _ = self.envs.step(action)
                self.state_categorizer.add_to_state_buffer(next_obs[0]) # 只取环境返回的第一个元素
                obs = np.expand_dims(next_obs,axis = 0)
        # 使用 PyTorch 计算方差（更高效，且支持 GPU）
        values_tensor = torch.cat(value_buffer)  # 直接在 GPU 上拼接
        self.sigma0_sq = torch.var(values_tensor, unbiased=True).item()  # ddof=1 对应 unbiased=True

    def _action2(self, obs, noise_scale=0.0):
        _, action = self.policy2(obs)
        action = action.detach().cpu().numpy()
        action = action + np.random.normal(size=action.shape) * noise_scale
        return np.clip(action, -1, 1) 

    def _action(self, obs, noise_scale=0.0):
        _, action = self.policy(obs)
        action = action.detach().cpu().numpy()
        action = action + np.random.normal(size=action.shape) * noise_scale
        return np.clip(action, -1, 1)

    def train(self, train_steps):
        obs = self.envs.buf_obs
        for _ in tqdm(range(train_steps)):
            step_info = {}
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            acts = self._action(obs, self.noise_scale)
            if self.current_step < self.start_training:
                acts = [self.action_space.sample() for _ in range(self.n_envs)]
            next_obs, rewards, terminals, trunctions, infos = self.envs.step(acts)
            self.memory.store(obs, acts, self._process_reward(rewards), terminals, self._process_observation(next_obs))
            if self.current_step > self.start_training and self.current_step % self.train_frequency == 0:
                obs_batch, act_batch, rew_batch, terminal_batch, next_batch = self.memory.sample()
                step_info = self.learner.update(obs_batch, act_batch, rew_batch, next_batch, terminal_batch, self.state_categorizer, self.sigma0_sq)
                step_info["noise_scale"] = self.noise_scale
                self.log_infos(step_info, self.current_step)

            self.returns = self.gamma * self.returns + rewards
            obs = next_obs
            for i in range(self.n_envs):
                if terminals[i] or trunctions[i]:
                    obs[i] = infos[i]["reset_obs"]
                    self.ret_rms.update(self.returns[i:i + 1])
                    self.returns[i] = 0.0
                    self.current_episode[i] += 1
                    if self.use_wandb:
                        step_info["Episode-Steps/env-%d" % i] = infos[i]["episode_step"]
                        step_info["Train-Episode-Rewards/env-%d" % i] = infos[i]["episode_score"]
                    else:
                        step_info["Episode-Steps"] = {"env-%d" % i: infos[i]["episode_step"]}
                        step_info["Train-Episode-Rewards"] = {"env-%d" % i: infos[i]["episode_score"]}
                    self.log_infos(step_info, self.current_step)

            self.current_step += self.n_envs
            if self.noise_scale >= self.end_noise:
                self.noise_scale = self.noise_scale - (self.start_noise - self.end_noise) / self.config.running_steps

    def test(self, env_fn, test_episodes):
        test_envs = env_fn()
        num_envs = test_envs.num_envs
        videos, episode_videos = [[] for _ in range(num_envs)], []
        current_episode, scores, best_score = 0, [], -np.inf
        obs, infos = test_envs.reset()
        if self.config.render_mode == "rgb_array" and self.render:
            images = test_envs.render(self.config.render_mode)
            for idx, img in enumerate(images):
                videos[idx].append(img)

        while current_episode < test_episodes:
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            acts = self._action(obs, noise_scale=0.0)
            next_obs, rewards, terminals, trunctions, infos = test_envs.step(acts)
            if self.config.render_mode == "rgb_array" and self.render:
                images = test_envs.render(self.config.render_mode)
                for idx, img in enumerate(images):
                    videos[idx].append(img)

            obs = next_obs
            for i in range(num_envs):
                if terminals[i] or trunctions[i]:
                    obs[i] = infos[i]["reset_obs"]
                    scores.append(infos[i]["episode_score"])
                    current_episode += 1
                    if best_score < infos[i]["episode_score"]:
                        best_score = infos[i]["episode_score"]
                        episode_videos = videos[i].copy()
                    if self.config.test_mode:
                        print("Episode: %d, Score: %.2f" % (current_episode, infos[i]["episode_score"]))

        if self.config.render_mode == "rgb_array" and self.render:
            # time, height, width, channel -> time, channel, height, width
            videos_info = {"Videos_Test": np.array([episode_videos], dtype=np.uint8).transpose((0, 1, 4, 2, 3))}
            self.log_videos(info=videos_info, fps=self.fps, x_index=self.current_step)

        if self.config.test_mode:
            print("Best Score: %.2f" % (best_score))

        test_info = {
            "Test-Episode-Rewards/Mean-Score": np.mean(scores),
            "Test-Episode-Rewards/Std-Score": np.std(scores)
        }
        self.log_infos(test_info, self.current_step)

        test_envs.close()

        return scores
