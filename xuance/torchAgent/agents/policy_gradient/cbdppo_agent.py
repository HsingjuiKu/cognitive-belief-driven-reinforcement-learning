from xuance.torchAgent.agents import *
from xuance.torchAgent.learners import *
from xuance.torchAgent.learners.policy_gradient.cbdppo_learner import *
from xuance.state_categorizer import StateCategorizer
import torch

class CBDPPO_Agent(Agent):
    """The implementation of PPO agent.

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
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 device: Optional[Union[int, str, torch.device]] = None):
        print("\n===== Environment Vector Info =====")
        print(f"Type: {type(envs)}")
        print(f"Number of Environments: {envs.num_envs}")
        print(f"Observation Space: {envs.observation_space}")
        print(f"Action Space: {envs.action_space}")
        self.render = config.render
        self.n_envs = envs.num_envs
        self.horizon_size = config.horizon_size
        self.n_minibatch = config.n_minibatch
        self.n_epoch = config.n_epoch

        self.gamma = config.gamma
        self.gae_lam = config.gae_lambda
        self.observation_space = envs.observation_space
        self.action_space = envs.action_space
        # print(f"Action Space: {self.action_space}")
        self.auxiliary_info_shape = {"old_logp": ()}
        self.frequency = 0
        self.policy2 = policy

        self.atari = True if config.env_name == "Atari" else False
        Buffer = DummyOnPolicyBuffer_Atari if self.atari else DummyOnPolicyBuffer
        self.buffer_size = self.n_envs * self.horizon_size
        self.batch_size = self.buffer_size // self.n_minibatch
        memory = Buffer(self.observation_space,
                        self.action_space,
                        self.auxiliary_info_shape,
                        self.n_envs,
                        self.horizon_size,
                        config.use_gae,
                        config.use_advnorm,
                        self.gamma,
                        self.gae_lam)
        learner = CBDPPO_Learner(policy,
                                  optimizer,
                                  scheduler,
                                  config.device,
                                  config.model_dir,
                                  vf_coef=config.vf_coef,
                                  ent_coef=config.ent_coef,
                                  clip_range=config.clip_range,
                                  clip_grad_norm=config.clip_grad_norm,
                                  use_grad_clip=config.use_grad_clip)
 
        self.state_categorizer = StateCategorizer(
            action_dim=self.action_space.n,
            n_categories=getattr(config, 'n_categories', 10),
            buffer_size=5000,
            device=device
        )

        super(CBDPPO_Agent, self).__init__(config, envs, policy, memory, learner, device,
                                            config.log_dir, config.model_dir)
        self.generate_initial_states()

    def generate_initial_states(self):
        model_path = "/home/hui/cognitive-belief-driven-qlearning/models/ppo/torchAgent/LunarLander-v2/seed_33_2025_0518_125738/final_train_model.pth"
        self.policy2.load_state_dict(torch.load(model_path, map_location=self.device))
        self.policy2.eval()
    
        # 重置环境，获取初始观测
        obs = self.envs.reset()
        for _ in tqdm(range(10000)):
            with torch.no_grad():
                # obs = self._process_observation(obs)
                _, dists, vs = self.policy2(obs[0])
                acts = dists.stochastic_sample()
                acts = acts.detach().cpu().numpy()
                next_obs, _, _, _, _ = self.envs.step(acts)
                self.state_categorizer.add_to_state_buffer(next_obs[0]) # 只取环境返回的第一个元素
                obs = np.expand_dims(next_obs,axis = 0)

    def _action(self, obs):
        _, dists, vs = self.policy(obs)
        acts = dists.stochastic_sample()
        logps = dists.log_prob(acts)
        vs = vs.detach().cpu().numpy()
        acts = acts.detach().cpu().numpy()
        logps = logps.detach().cpu().numpy()
        return acts, vs, logps

    def train(self, train_steps):
        obs = self.envs.buf_obs
        for _ in tqdm(range(train_steps)):
            step_info = {}
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            
            acts, value, logps = self._action(obs)
            next_obs, rewards, terminals, trunctions, infos = self.envs.step(acts)

             # 更新动作选择频率
            for o, a in zip(obs, acts):
                if self.state_categorizer.initialized:
                    if self.frequency % 5000 == 0:
                        self.state_categorizer.update_action_counts(o, a)

            self.memory.store(obs, acts, self._process_reward(rewards), value, terminals, {"old_logp": logps})
            if self.memory.full:
                _, vals, _ = self._action(self._process_observation(next_obs))
                for i in range(self.n_envs):
                    if terminals[i]:
                        self.memory.finish_path(0.0, i)
                    else:
                        self.memory.finish_path(vals[i], i)
                indexes = np.arange(self.buffer_size)
                for _ in range(self.n_epoch):
                    np.random.shuffle(indexes)
                    for start in range(0, self.buffer_size, self.batch_size):
                        end = start + self.batch_size
                        sample_idx = indexes[start:end]
                        obs_batch, act_batch, ret_batch, value_batch, adv_batch, aux_batch = self.memory.sample(sample_idx)
                        step_info = self.learner.update(obs_batch, act_batch, ret_batch, value_batch, 
                        adv_batch, aux_batch['old_logp'], self.state_categorizer)
                
                self.log_infos(step_info, self.current_step)
                self.memory.clear()

            self.returns = (1 - terminals) * self.gamma * self.returns + rewards
            obs = next_obs
            for i in range(self.n_envs):
                if terminals[i] or trunctions[i]:
                    self.ret_rms.update(self.returns[i:i + 1])
                    self.returns[i] = 0.0
                    if self.atari and (~trunctions[i]):
                        pass
                    else:
                        if terminals[i]:
                            self.memory.finish_path(0.0, i)
                        else:
                            _, vals, _ = self._action(self._process_observation(next_obs))
                            self.memory.finish_path(vals[i], i)
                        obs[i] = infos[i]["reset_obs"]
                        self.current_episode[i] += 1
                        if self.use_wandb:
                            step_info["Episode-Steps/env-%d" % i] = infos[i]["episode_step"]
                            step_info["Train-Episode-Rewards/env-%d" % i] = infos[i]["episode_score"]
                        else:
                            step_info["Episode-Steps"] = {"env-%d" % i: infos[i]["episode_step"]}
                            step_info["Train-Episode-Rewards"] = {"env-%d" % i: infos[i]["episode_score"]}
                        self.log_infos(step_info, self.current_step)

            self.current_step += self.n_envs

    def test(self, env_fn, test_episode):
        test_envs = env_fn()
        num_envs = test_envs.num_envs
        videos, episode_videos = [[] for _ in range(num_envs)], []
        current_episode, scores, best_score = 0, [], -np.inf
        obs, infos = test_envs.reset()
        if self.config.render_mode == "rgb_array" and self.render:
            images = test_envs.render(self.config.render_mode)
            for idx, img in enumerate(images):
                videos[idx].append(img)

        while current_episode < test_episode:
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            acts, rets, logps = self._action(obs)
            next_obs, rewards, terminals, trunctions, infos = test_envs.step(acts)
            if self.config.render_mode == "rgb_array" and self.render:
                images = test_envs.render(self.config.render_mode)
                for idx, img in enumerate(images):
                    videos[idx].append(img)

            obs = next_obs
            for i in range(num_envs):
                if terminals[i] or trunctions[i]:
                    if self.atari and (~trunctions[i]):
                        pass
                    else:
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
