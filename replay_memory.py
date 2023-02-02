import random
from collections import namedtuple
import numpy as np
import torch
torch.set_default_tensor_type('torch.DoubleTensor')

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state',
                                       'reward'))


class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def sample(self):
        return Transition(*zip(*self.memory))

    def __len__(self):
        return len(self.memory)


class ReplayBuffer(object):
    """Buffer to store environment transitions."""

    def __init__(self, obs, actions, rewards, next_obs, not_dones, device='cpu'):
        self.capacity = obs.shape[0]
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        self.obs = obs
        self.actions = actions
        self.rewards = rewards
        self.next_obs = next_obs
        self.not_dones = not_dones

        self.idx = 0
        self.last_save = 0
        self.full = True

    def sample(self, batch_size=1024):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=batch_size
        )

        obses = self.obs[idxs]

        next_obses = self.next_obs[idxs]
        obses = torch.as_tensor(obses, device=self.device).double()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(
            next_obses, device=self.device
        ).double()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        return obses, actions, rewards, next_obses, not_dones
