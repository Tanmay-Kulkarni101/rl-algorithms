import torch
from torch.optim import Adam
from time import time
from vanilla_policy_gradient_utils import cumulative_sum

class PlayBackBuffer:
    def __init__(self, gamma, lamb, size, obs_dim, act_dim):
        self.gamma = gamma
        self.lamb = lamb
        
        self.curr = 0
        self.trajectory_start = 0
        self.max_size = size

        self.observation_buffer = torch.zeros((size, *obs_dim))
        self.action_buffer = torch.zeros((size, *act_dim))
        self.reward_buffer = torch.zeros(size)
        self.value_buffer = torch.zeros(size)
        self.log_prob_buffer = torch.zeros(size)


    def store(self, observation, action, reward, value, log_prob):
        '''
        Add an interaction of the agent with the environment to the buffer
        '''
        assert self.curr < self.max_size, 'Buffer Overflow!'

        self.observation_buffer[self.curr] = observation
        self.action_buffer[self.curr] = action
        self.reward_buffer[self.curr] = reward
        self.value_buffer[self.curr] = value
        self.log_prob_buffer[self.curr] = log_prob

        self.ptr += 1

    def terminate_trajectory(self, terminal_value):
        '''
        Call this function at the end of a trajectory.
        It calculates the advantages and the returns corresponding to the episode.
        '''
        trajectory = slice(self.trajectory_start, self.curr)
        rewards = self.reward_buffer[trajectory]
        values = self.value_buffer[trajectory]
        
        # Bootstrap with the final value.
        # This is equivalent to obtaining the approximate discounted sum of rewards
        # after the end of the horizon
        terminal_value = torch.tensor([terminal_value])
        rewards = torch.concat((rewards, terminal_value))
        values = torch.concat((values, terminal_value))

        # Calculate the advantages
        delta = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.advantage_buffer = cumulative_sum(delta, self.gamma * self.lamb)

        # Calculate the returns
        self.returns_buffer = cumulative_sum(rewards, self.gamma)

        self.trajectory_start = self.curr
    
    def get(self):
        '''
        This function encapsulates the details of an epoch in a data dict
        It also resets the state of the buffer and standardizes the advantage values.
        '''
        assert self.curr == self.max_size

        # Reinitialize Buffer State
        self.curr = 0
        self.trajectory_start = 0

        variance, mean = torch.var_mean(self.advantage_buffer)
        
        # Standardize the Advantages
        self.advantage_buffer = (self.advantage_buffer - mean) / variance

        # Encapsulate the details of the trajectories in an epoch
        data = dict(obs=self.observation_buffer, act=self.action_buffer, ret=self.returns_buffer,
                    adv=self.advantage_buffer, logp=self.log_prob_buffer)
        
        return data

class VanillaPolicyGradient:
        def __init__(self, env_func, actor_critic, ac_kwargs, seed, steps_per_epoch, 
        epochs, gamma, pi_lr, vf_lr, train_v_iters, lamb, max_ep_len, save_freq):
            torch.manual_seed(seed)

            self.train_v_iters = train_v_iters

            # Instantiate the environment
            self.env = env_func()
            obs_dim = self.env.observation_space.shape
            act_dim = self.env.action_space.shape

            # Instantiate Actor Critic
            self.ac = actor_critic(self.env.observation_space, self.env.action_space, **ac_kwargs) 

            # Instantiate Buffer
            self.buffer = PlayBackBuffer(gamma, lamb, steps_per_epoch, obs_dim, act_dim)

            # Initialize Optimizer
            self.policy_optimizer = Adam(self.ac.policy.parameters(), lr=pi_lr)
            self.value_func_optimizer = Adam(self.ac.value_function.parameters(), lr=vf_lr)

            # Experiment Parameters
            self.epochs = epochs
            self.steps_per_epoch = steps_per_epoch
            self.max_ep_len = max_ep_len

        def compute_policy_loss(self, obs, act, adv, log_prob):
            action_distribution, new_log_prob = self.ac.policy(obs, act)
            policy_loss = - (new_log_prob * adv).mean()

            # Additional Information
            # p log p / q = p log p - p log q
            approx_kl = (log_prob - new_log_prob).mean()
            entropy = action_distribution.entropy().mean()
            action_distribution_info = dict(kl=approx_kl, entropy = entropy)

            return policy_loss, action_distribution_info
        
        def compute_value_func_loss(self, obs, returns):
            return ((self.ac.value_function(obs) - returns) ** 2).mean()
        
        def update(self):
            '''
            Update Model Paramters
            '''
            data = self.buffer.get()

            # Expand members of dict
            obs = data['obs']
            actions = data['act']
            returns = data['ret']
            advantages = data['adv']
            logp = data['logp']

            # Update Policy Paramters
            self.policy_optimizer.zero_grad()
            policy_loss, _ = self.compute_policy_loss(obs, actions, advantages, logp)
            print(f'Policy Loss: {policy_loss}')

            policy_loss.backward()
            self.policy_optimizer.step()

            value_func_loss_avg = 0
            # Update Value Function Paramters
            for i in range(self.train_v_iters):
                self.value_func_optimizer.zero_grad()
                value_func_loss = self.compute_value_func_loss(obs, returns)
                value_func_loss_avg += value_func_loss
                value_func_loss.backward()
                self.value_func_optimizer.step()
            
            print(f'Value Function Loss:{value_func_loss_avg}')


        def train(self):
            '''
            '''
            start_time = time()
            observation = self.env.reset()
            episode_return = 0
            episode_length = 0

            for epoch in self.epochs:
                print("########################################################")
                print(f"Epoch:{epoch}")
                for time_step in self.steps_per_epoch:
                    action, value, logp_a = self.ac.step(torch.as_tensor(observation))

                    next_observation, reward, done, _ = self.env.step(action)
                    episode_return += reward
                    episode_length += 1

                    self.buffer.store(observation, action, reward, value, logp_a)

                    # Update current observation
                    observation = next_observation

                    # Terminal Conditions
                    is_timeout = episode_length == self.max_ep_len - 1
                    is_term = is_timeout or done
                    is_epoch_end = time_step == self.steps_per_epoch - 1

                    if is_term or is_epoch_end:
                        if is_epoch_end and not is_term:
                            print(f"Trajectory cut short after {episode_length} steps")
                        if is_timeout or is_epoch_end:
                            value = self.ac.step(observation)[1]
                        else:
                            value = 0
                        
                        self.buffer.terminate_trajectory(value)

                        if is_term:
                            print(f'Episode Terminated')
                            print(f'Episode Return:{episode_return}')
                            print(f'Episode Length:{episode_length}')
                    
                # Update Model Params
                self.update()