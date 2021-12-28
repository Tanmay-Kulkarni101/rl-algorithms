import torch
import pdb
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from gym.spaces import Box, Discrete

def str_to_activation(activations):
    torch_activations = []

    for activation in activations:
        if activation.lower() == 'tanh':
            torch_activations.append(nn.Tanh)
        elif activation.lower() == 'relu':
            torch_activations.append(nn.ReLU)
        else:
            raise NotImplementedError
    
    return torch_activations

def init_mlp(sizes, activations):
    '''
    Initializes a neural network according to the sizes and activations.
    
    The neural network created in the following manner:
    (sizes[0],sizes[1]) actvation[0] (sizes[1], sizes[2]) activation[1] ... 
    (sizes[n-2], sizes[n-1]) activation[n-1]

    Note:
    The length of sizes and activations should be the same
    '''

    assert len(sizes) - 1 == len(activations), f'sizes - 1 {len(sizes) - 1} and activations {len(activations)} should have the same length'
    
    layers = []
    num_sizes = len(sizes)
    num_activations = len(activations)
    curr_size = 0
    curr_activation = 0

    while curr_size < num_sizes - 1 and curr_activation < num_activations:
        input_size = sizes[curr_size]
        output_size = sizes[curr_size + 1]
        activation = activations[curr_activation]

        layer = [nn.Linear(input_size, output_size), activation()]
        layers += layer

        curr_size += 1
        curr_activation += 1

    return nn.Sequential(*layers)

def cumulative_sum(array, discount_factor):
    discounted_sum = torch.zeros_like(array)
    discounted_sum[-1] = array[-1]

    for i in reversed(range(1, len(array))):
        discounted_sum[i - 1] = discount_factor * discounted_sum[i] + array[i - 1]
    
    return discounted_sum
    
class Actor(nn.Module):
    def _distribution(self, obs):
        '''
        Return distribution of actions corresponding to each observation
        '''
        raise NotImplementedError
    
    def _log_prob_from_distribution(self, pi, actions):
        '''
        Calculate the log likelihood of the actions for a set of action distributions
        '''
        raise NotImplementedError
    
    def forward(self, obs, actions=None):
        '''
        Produce action distribution for each observation and produce the corresponding log
        likelihood if the actions have been specified.
        '''
        pi = self._distribution(obs)
        logp_a = None

        if actions is not None:
            logp_a = self._log_prob_from_distribution(pi, actions)
        
        return pi, logp_a
    
class MLPCategoricalActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_dims, activations):
        super().__init__()

        # Input will the same dim as the observation space
        # Final output should match the action space
        # All the hidden layers will be inbetween
        sizes = [obs_dim] + list(hidden_dims) + [act_dim]
        self.logits_net = init_mlp(sizes, activations)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)
    
    def _log_prob_from_distribution(self, pi, actions):
        '''
        Here pi is an instance of torch.distributions
        actions is a tensor corresponding to the action at every step
        '''
        return pi.log_prob(actions)

class MLPGaussianFixedVarianceActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_dims, activations):
        super().__init__()
        sizes = [obs_dim] + list(hidden_dims) + [act_dim]

        # We start with one standard deviation
        log_std = torch.zeros(act_dim)

        # The standard deviation is the same across all parameters
        self.log_std = nn.parameter.Parameter(log_std)
        self.mu_net = init_mlp(sizes, activations)
    
    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)
    
    def _log_prob_from_distribution(self, pi, actions):
        '''
        We need to sum across the action_space. This is because we want an
        output of the form batch_size, and not batch_size, action_space
        '''
        return pi.log_prob(actions).sum(axis=-1)

class MLPGaussianDynamicVarianceActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_dims, activations):
        super().__init__()
        sizes = [obs_dim] + list(hidden_dims) + [act_dim]

        # We learn both the mean and the standard deviation
        self.log_std = init_mlp(sizes, activations)
        self.mu_net = init_mlp(sizes, activations)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)
    
    def _log_prob_from_distribution(self, pi, actions):
        '''
        We need to sum across the action_space. This is because we want an
        output of the form batch_size, and not batch_size, action_space
        '''
        return pi.log_prob(actions).sum(axis=-1)

class MLPCritic(nn.Module):
    def __init__(self, obs_dim, hidden_dims, activations):
        super().__init__()
        sizes = [obs_dim] + list(hidden_dims) + [1]
        self.value_net = init_mlp(sizes, activations)

    def forward(self, obs):
        return self.value_net(obs)

class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes, activations):
        super().__init__()

        assert len(observation_space.shape) == 1, 'the observation space should be a singleton tuple'
        obs_dim = observation_space.shape[0]

        if isinstance(action_space, Box):
            assert len(action_space.shape) == 1, 'the actions space should be a singleton tuple'
            act_dim = action_space.shape[0]
            self.policy = MLPGaussianFixedVarianceActor(obs_dim, act_dim, hidden_sizes, activations)
        elif isinstance(action_space, Discrete):
            act_dim = action_space.n 
            self.policy = MLPCategoricalActor(obs_dim, act_dim, hidden_sizes, activations)
        
        self.value_function = MLPCritic(obs_dim, hidden_sizes, activations)

    def step(self, obs):
        with torch.no_grad():
            action_distribution = self.policy._distribution(obs)
            action = action_distribution.sample()
            logp_a = self.policy._log_prob_from_distribution(action_distribution, action)
            value = self.value_function(obs)

        return action, value, logp_a
    
    def act(self, obs):
        return self.step(obs)[0]
