import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
from torch.nn.modules import loss
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from network import DQN, optimize
from utils import get_screen, ReplayMemory, Action_Selector, plot_durations
from vars import TARGET_UPDATE
import random

env = gym.make('CartPole-v0').unwrapped
env.reset()
action_selector = Action_Selector()

init_screen = get_screen(env)
_, _, screen_height, screen_width = init_screen.shape
n_actions = env.action_space.n

policy_net = DQN(screen_height, screen_width, n_actions)
target_net = DQN(screen_height, screen_width, n_actions)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
loss_func = nn.SmoothL1Loss()
memory = ReplayMemory(10000)
episode_durations = []

for i in range(50):
    env.reset()
    last_screen = get_screen(env)
    current_screen = get_screen(env)
    state = last_screen - current_screen
    for t in count():
        print(i, t)
        if state is not None:
            action = action_selector.select_action(policy_net, state)
            _, reward, done, _ = env.step(action.item())
            reward = torch.Tensor([reward])

            last_screen = current_screen
            current_screen = get_screen(env)
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None
                
            memory.push(state, action, next_state, reward)

            state = next_state

        optimize(memory, policy_net, target_net, optimizer, loss_func)
        if done:
            episode_durations.append(t + 1)
            plot_durations(episode_durations)
            break

    if i % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

