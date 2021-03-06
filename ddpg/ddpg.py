import copy
import logging
import os
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

experience = namedtuple('experience', ('S', 'A', 'r', 'dones', 'S_tp1'))

class PolicyNet(nn.Module):
    def __init__(self, obs_size, act_size):
        super(PolicyNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, act_size),
            nn.Tanh()
        )

    def forward(self, S):
        return self.net(S)

class QNet(nn.Module):
    def __init__(self, obs_size, act_size):
        super(QNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_size + act_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )

    def forward(self, S, A):
        return self.net(torch.cat([S, A], dim=1))

class TargetNet:
    """
    Wrapper around model which provides copy of it instead of trained weights
    """
    def __init__(self, model):
        self.model = model
        self.target_model = copy.deepcopy(model)

    def sync(self, alpha):
        """
        Blend params of target net with params from the model
        :param alpha:
        """
        assert isinstance(alpha, float)
        assert 0.0 < alpha <= 1.0
        state = self.model.state_dict()
        tgt_state = self.target_model.state_dict()
        for k, v in state.items():
            tgt_state[k] = tgt_state[k] * alpha + (1 - alpha) * v
        self.target_model.load_state_dict(tgt_state)
        
class TargetPi(TargetNet):
    def __call__(self, S):
        return self.target_model(S)
    
class TargetQ(TargetNet):
    def __call__(self, S, A):
        return self.target_model(S, A)

class Buffer:
    def __init__(self, size, device):
        assert isinstance(size, int)
        self.buffer = []
        self.capacity = size
        self.pos = 0
        self.device = device

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)

    def add(self, exp):
        if len(self.buffer) < self.capacity:
            self.buffer.append(exp)
        else:
            self.buffer[self.pos] = exp
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        """
        Get one random batch from experience replay
        :param batch_size:
        :return:
        """
        assert(len(self.buffer) >= batch_size)

        # Warning: replace=False makes random.choice O(n)
        keys = np.random.choice(len(self.buffer), batch_size, replace=True)
        S, A, r, dones, S_tp1 = [], [], [], [], []
        for key in keys:
            exp = self.buffer[key]
            S.append(exp.S)
            A.append(exp.A)
            r.append(exp.r)
            dones.append(exp.dones)
            S_tp1.append(exp.S_tp1)

        S_v = torch.tensor(np.array(S, dtype=np.float32)).to(self.device)
        A_v = torch.tensor(np.array(A, dtype=np.float32)).to(self.device)
        r_v = torch.tensor(np.array(r, dtype=np.float32)).to(self.device)
        S_tp1_v = torch.tensor(np.array(S_tp1, dtype=np.float32)).to(self.device)
        dones_v = torch.BoolTensor(dones).to(self.device)
        return S_v, A_v, r_v, dones_v, S_tp1_v

# based on coax utils implementations
class OUNoise:
    """


        Parameters
        ----------
        mu : float or ndarray, optional

            The mean  towards which the Ornstein-Uhlenbeck process should revert; must be
            broadcastable with the input actions.

        sigma : positive float or ndarray, optional

            The spread of the noise of the Ornstein-Uhlenbeck process; must be
            broadcastable with the input actions.

        theta : positive float or ndarray, optional

            The (element-wise) dissipation rate of the Ornstein-Uhlenbeck process; must
            be broadcastable with the input actions.

        min_value : float or ndarray, optional

            The lower bound used for clipping the output action; must be broadcastable with the input
            actions.

        max_value : float or ndarray, optional

            The upper bound used for clipping the output action; must be broadcastable with the input
            actions.

        random_seed : int, optional

            Sets the random state to get reproducible results.
    """
    def __init__(self, mu=0., sigma=1., theta=0.15, min_value=None,
                 max_value=None, random_seed=None):

        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.min_value = -1e15 if min_value is None else min_value
        self.max_value = 1e15 if max_value is None else max_value
        self.random_seed = random_seed
        self.rnd = np.random.RandomState(self.random_seed)
        self.reset()

    def reset(self):
        """Reset the Ornstein-Uhlenbeck process."""
        self._noise = None

    def __call__(self, a):
        """Add some Ornstein-Uhlenbeck to a continuous action. """
        a = np.asarray(a)
        if self._noise is None:
            self._noise = np.ones_like(a) * self.mu

        white_noise = np.asarray(self.rnd.randn(*a.shape), dtype=a.dtype)
        self._noise += self.theta * (self.mu - self._noise) + self.sigma * white_noise
        self._noise = np.clip(self._noise, self.min_value, self.max_value)
        return a + self._noise

def generate_gif(env, filepath, pi, max_episode_steps, device, resize_to=None, duration=32):
    """
    Store a gif from the episode frames.

    Parameters
    ----------
    env : gym environment
    filepath : str
    pi : nn.Module
    max_episode_steps : int
    resize_to : tuple of ints, optional
    duration : float, optional
    """
    
    # collect frames
    frames = []
    s = env.reset()
    for t in range(max_episode_steps):
        np_S = np.array(s, dtype=np.float32)
        a = pi(torch.tensor(np_S).to(device)).data.cpu().numpy()
        s_next, r, done, info = env.step(a)
        # store frame
        frame = env.render(mode='rgb_array')
        frame = Image.fromarray(frame)
        frame = frame.convert('P', palette=Image.ADAPTIVE)
        if resize_to is not None:
            if not (isinstance(resize_to, tuple) and len(resize_to) == 2):
                raise TypeError(
                    "expected a tuple of size 2, resize_to=(w, h)")
            frame = frame.resize(resize_to)

        frames.append(frame)

        if done:
            break

        s = s_next

    # store last frame
    frame = env.render(mode='rgb_array')
    frame = Image.fromarray(frame)
    frame = frame.convert('P', palette=Image.ADAPTIVE)
    if resize_to is not None:
        frame = frame.resize(resize_to)
    frames.append(frame)

    # generate gif
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    frames[0].save(
        fp=filepath, format='GIF', append_images=frames[1:], save_all=True,
        duration=duration, loop=0)