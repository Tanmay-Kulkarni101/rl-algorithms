import torch

class PlayBackBuffer:
    def __init__(self):
        pass

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

    def terminate_trajectory(self):
        '''
        Call this function at the end of a trajectory.
        '''
        trajectory = slice(self.trajectory_start, self.curr)
        rewards = self.reward_buffer[trajectory]
        values = self.value_buffer[trajectory]

        delta = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.advantage_buffer = 0