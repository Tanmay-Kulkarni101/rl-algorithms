import pdb
import torch
import argparse
import gym
import logging
from torch.optim import Adam
from time import time
from vanilla_policy_gradient_utils import cumulative_sum, MLPActorCritic, str_to_activation, str_to_log_level

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
        self.advantage_buffer = torch.zeros(size)
        self.returns_buffer = torch.zeros(size)


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

        self.curr += 1

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
        self.advantage_buffer[trajectory] = cumulative_sum(delta, self.gamma * self.lamb)

        # Calculate the returns
        self.returns_buffer[trajectory] = cumulative_sum(rewards, self.gamma)[:-1]

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
    def __init__(self, env, actor_critic, ac_kwargs, seed, steps_per_epoch, 
        epochs, gamma, pi_lr, vf_lr, train_v_iters, lamb, max_ep_len, log_level):
        torch.manual_seed(seed)

        self.train_v_iters = train_v_iters

        # Instantiate the environment
        self.env = env
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

        logging.basicConfig(level=log_level)
        self.logger = logging.getLogger(__name__)

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
        self.logger.debug(f'Policy Loss: {policy_loss}')

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
        
        self.logger.debug(f'Value Function Loss:{value_func_loss_avg}')


    def train(self):
        '''
        '''
        start_time = time()
        observation = self.env.reset()
        episode_return = 0
        episode_length = 0

        for epoch in range(self.epochs):
            self.logger.info("########################################################")
            self.logger.info(f"Epoch:{epoch}")
            avg_return = 0
            episode_count = 0
            for time_step in range(self.steps_per_epoch):
                action, value, logp_a = self.ac.step(torch.as_tensor(observation))

                action = action.numpy()

                next_observation, reward, done, _ = self.env.step(action)
                episode_return += reward
                episode_length += 1

                # Convert to torch tensor form ndarray
                action = torch.tensor(action)
                observation = torch.tensor(observation)
                reward = torch.tensor(reward)

                self.buffer.store(observation, action, reward, value, logp_a)

                # Update current observation
                observation = next_observation

                # Terminal Conditions
                is_timeout = episode_length == self.max_ep_len - 1
                is_term = is_timeout or done
                is_epoch_end = time_step == self.steps_per_epoch - 1

                if is_term or is_epoch_end:
                    if is_epoch_end and not is_term:
                        self.logger.warning(f"Trajectory cut short after {episode_length} steps")                            
                    if is_timeout or is_epoch_end:
                        # Convert the last observation to a tensor
                        observation = torch.tensor(observation)
                        value = self.ac.step(observation)[1]
                    else:
                        value = 0
                    
                    self.buffer.terminate_trajectory(value)

                    if is_term:
                        self.logger.debug(f'Episode Terminated')
                        self.logger.debug(f'Episode Return:{episode_return}')
                        self.logger.debug(f'Episode Length:{episode_length}')
                        avg_return += episode_return
                        episode_count += 1
                    
                    observation = self.env.reset()
                    episode_return = 0
                    episode_length = 0
            
            # Update Model Params
            self.update()
            self.logger.info(f'Average Returns: {avg_return / episode_count}')
            self.logger.info(f'Episode Count: {episode_count}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str)
    parser.add_argument('--hidden_sizes', nargs='+', type=int)
    parser.add_argument('--activations', nargs='+', type=str)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--pi_lr', type=float)
    parser.add_argument('--v_lr', type=float)
    parser.add_argument('--value_func_iters', type=int)
    parser.add_argument('--lamb', type=float)
    parser.add_argument('--max_episode_length', type=int)
    parser.add_argument('--log_level', type=str)

    args = parser.parse_args()

    env = gym.make(args.env)
    actor_critic = MLPActorCritic
    hidden_sizes = args.hidden_sizes

    activations = args.activations
    activations = str_to_activation(activations)
    
    gamma = args.gamma
    seed = args.seed
    steps_per_epoch = args.steps
    epochs = args.epochs
    pi_lr = args.pi_lr
    v_lr = args.v_lr
    value_func_iters = args.value_func_iters
    lamb = args.lamb
    max_ep_len = args.max_episode_length
    log_level = str_to_log_level(args.log_level)

    algo = VanillaPolicyGradient(env, actor_critic, dict(hidden_sizes=hidden_sizes, activations=activations),
    gamma, steps_per_epoch, epochs, gamma, pi_lr, v_lr, value_func_iters, lamb, max_ep_len, log_level)

    algo.train()