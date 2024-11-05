from xuance.torchAgent.agents import *
from xuance.torchAgent.learners import *
from xuance.torchAgent.learners.policy_gradient.cbdsac_learner import *
from xuance.state_categorizer import StateCategorizer
import torch

class CBDSAC_Agent(Agent):
    """The implementation of SAC agent.

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
        learner = CBDSAC_Learner(policy, optimizer, scheduler, config.device, config.model_dir,
                              gamma=config.gamma,
                              tau=config.tau,
                              alpha=config.alpha,
                              use_automatic_entropy_tuning=config.use_automatic_entropy_tuning,
                              target_entropy=-np.prod(self.action_space.shape).item(),
                              lr_policy=config.actor_learning_rate)
        self.state_categorizer = StateCategorizer(
            action_dim=self.action_space.shape[0],
            n_categories=getattr(config, 'n_categories', 10),
            buffer_size=1000,
            device=device
        )
        
        super(CBDSAC_Agent, self).__init__(config, envs, policy, memory, learner, device, config.log_dir, config.model_dir)
        self.generate_initial_states()

    def generate_initial_states(self):
        model_path = "models/sac/torchAgent/Ant-v4/seed_1_2024_1101_154815/final_train_model.pth"
        self.policy2.load_state_dict(torch.load(model_path, map_location=self.device))
        self.policy2.eval()
        obs = self.envs.reset()
        for _ in tqdm(range(5000)):
            with torch.no_grad():
                _,action = self.policy2(obs[0])
                action = action.detach().cpu().numpy()
                next_obs, _, _, _, _ = self.envs.step(action)
                self.state_categorizer.add_to_state_buffer(next_obs[0]) # 只取环境返回的第一个元素
                obs = np.expand_dims(next_obs,axis = 0)

    def _action(self, obs):
        _, action = self.policy(obs)
        return action.detach().cpu().numpy()

    def train(self, train_steps):
        obs = self.envs.buf_obs
        for _ in tqdm(range(train_steps)):
            step_info = {}
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            acts = self._action(obs)
            if self.current_step < self.start_training:
                acts = [self.action_space.sample() for _ in range(self.n_envs)]

            next_obs, rewards, terminals, trunctions, infos = self.envs.step(acts)

            self.memory.store(obs, acts, self._process_reward(rewards), terminals, self._process_observation(next_obs))
            if (self.current_step > self.start_training) and (self.current_step % self.train_frequency == 0):
                obs_batch, act_batch, rew_batch, terminal_batch, next_batch = self.memory.sample()
                
                # Get the index based on categories
                idx_batch = self.state_categorizer.get_categories_batch(obs_batch)
                # 获取 index 中的唯一类别
                unique_indices = torch.unique(idx_batch)

                idx_batch = torch.as_tensor(idx_batch, device = self.device)
                obs_batch = torch.as_tensor(obs_batch, device = self.device)

                # 对每个唯一类别进行迭代
                for i in unique_indices:
                    # 获取对应类别的子集
                    sub_obs = obs_batch[idx_batch == i, :]
                    # sub_obs = sub_obs.to(self.device)
                    _, _, _ ,dist= self.policy.Qpolicy(sub_obs)
                    self.state_categorizer.update_belief(i, dist)
                step_info = self.learner.update(obs_batch, act_batch, rew_batch, next_batch, terminal_batch, self.state_categorizer)
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
            torch.cuda.empty_cache()

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
            acts = self._action(obs)
            next_obs, rewards, terminals, trunctions, infos = test_envs.step(acts)
            if self.config.render and self.config.render_mode == "rgb_array":
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
