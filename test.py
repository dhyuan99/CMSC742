import torch
import gym
from vars import *
from Agent import Agent
from Attacker import Attacker

env = gym.make('CartPole-v0')
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

agent = Agent(seed=1423, layer_sizes=[input_dim, HIDDEN_LAYER_SIZE, output_dim], lr=999, sync_freq=SYNC_FREQ, exp_replay_size=EXP_REPLAY_SIZE, gamma=GAMMA)
agent.q_net.load_state_dict(torch.load('models/DQN.pth'))
attacker = Attacker(hidden_layer_size=HIDDEN_LAYER_SIZE, norm_bound=NORM_BOUND, num_layers=NUM_LAYERS)
attacker.load_state_dict(torch.load('models/attacker.pth'))

total_reward = 0
for i in range(1000):
    obs, done, r = env.reset(), False, 0
    while (not done):
        A =  agent.get_action(obs, env.action_space.n, epsilon=0, attacker=attacker)
        obs, _, done, _ = env.step(A.item())
        r += 1
    total_reward += r
    
print(f'average reward is {total_reward / 1000}')