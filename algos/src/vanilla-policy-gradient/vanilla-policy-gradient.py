import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

def init_mlp(self, sizes, activations):
    layers = []
    num_sizes = len(sizes)
    num_activations = len(activations)
    curr_size = 0
    curr_activation = 0

    while curr_size < num_sizes - 1 and curr_activation < num_activations:
        input_size = sizes[curr_size]
        output_size = sizes[curr_size + 1]
        activation = activations[curr_activation]

        layer = nn.Linear(input_size, output_size, activation)
        layers.append(layer)

        curr_size += 1
        curr_activation += 1
    
    return nn.Sequential(*layers)
    
class Actor(nn.Module):
    pass

class Critic(nn.Module):
    pass


