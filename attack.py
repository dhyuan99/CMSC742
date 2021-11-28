import gym
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim

from network import DQN, Attacker, optimize_attacker
from utils import get_screen, ReplayMemory, Action_Selector, plot_durations
from vars import TARGET_UPDATE, N_EPISODE

env = gym.make('CartPole-v0').unwrapped
env.reset()
action_selector = Action_Selector()

init_screen = get_screen(env)
_, _, screen_height, screen_width = init_screen.shape
n_actions = env.action_space.n

target_net = DQN(screen_height, screen_width, n_actions)
target_net.load_state_dict(torch.load('models/target_net.pth'))
target_net.eval()

attacker = Attacker(target_net, norm_bound=1)

optimizer = optim.RMSprop(Attacker.parameters(), lr=1e-3)
loss_func = nn.SmoothL1Loss()
memory = ReplayMemory(10000)
episode_durations = []

for i in range(N_EPISODE):
    env.reset()
    state = get_screen(env)
    for t in count():
        if state is not None:
            action = action_selector.select_action(target_net, state, attacker)
            _, reward, done, _ = env.step(action.item())
            reward = torch.Tensor([reward])
            
            if not done:
                next_state = get_screen(env)
            else:
                next_state = None
                
            memory.push(state, action, next_state, reward)

            state = next_state

        optimize_attacker(memory, target_net, attacker, optimizer, loss_func)
        
        if done:
            episode_durations.append(t + 1)
            plot_durations(episode_durations)
            break

    if i % TARGET_UPDATE == 0:
        torch.save(attacker.state_dict(), f'models/attacker.pth')