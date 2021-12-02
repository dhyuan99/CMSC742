from Agent import Agent
from Attacker import Attacker
import gym
import torch
from vars import *
import matplotlib.pyplot as plt
import copy
from statsmodels.tsa.api import SimpleExpSmoothing

import argparse
import json
parser = argparse.ArgumentParser(description='Please provide the config file.')
parser.add_argument('-attack_config', type=str, help='the path to the attacker config file')
parser.add_argument('-agent_config', type=str, help='the path to the agent config file')
args = parser.parse_args()
with open(args.attack_config) as f:
    attack_cfg = json.load(f)
with open(args.agent_config) as f:
    agent_cfg = json.load(f)
    

env = gym.make('CartPole-v0')
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
agent = Agent(seed=1423,  
    layer_sizes=[input_dim, agent_cfg['hidden_layer_size'], output_dim], 
    lr=agent_cfg['lr'], 
    sync_freq=agent_cfg['sync_freq'], 
    exp_replay_size=agent_cfg['exp_replay_size'],
    gamma=agent_cfg['gamma'])
agent.q_net.load_state_dict(torch.load(agent_cfg['savepath']))
attacker = Attacker(hidden_layer_size=agent_cfg['hidden_layer_size'], norm_bound=attack_cfg['norm_bound'], num_layers=attack_cfg['num_layers'])
optimizer = torch.optim.Adam(attacker.parameters(), lr=attack_cfg['lr'])

durations = []
index = agent_cfg['exp_replay_size'] / 2
best_attacker = attacker

for i in range(attack_cfg['n_episodes']):
    obs, done, duration = env.reset(), False, 0
    while (not done):
        duration += 1 
        A = agent.get_action(obs, env.action_space.n, epsilon=0, attacker=attacker)
        obs_next, reward, done, _ = env.step(A.item())
        agent.collect_experience([obs, A.item(), reward, obs_next])
       
        obs = obs_next
        index += 1
        
        if(index > EXP_REPLAY_SIZE / 2):
            index = 0
            for _ in range(4):
                loss, _ = agent.compute_loss(batch_size=BATCH_SIZE)
                optimizer.zero_grad()
                (-loss).backward(retain_graph=True)
                optimizer.step()

    durations.append(duration)
    if duration <= min(durations):
        best_attacker = copy.deepcopy(attacker)

fit1 = SimpleExpSmoothing(durations, initialization_method="heuristic").fit(smoothing_level=0.05, optimized=False)
plt.plot(fit1.fittedvalues)
plt.savefig('hello.jpg')
plt.close()

env = gym.make('CartPole-v0')
r = 0
for i in range(1000):
    obs, done = env.reset(), False
    while (not done):
        A =  agent.get_action(obs, env.action_space.n, epsilon=0, attacker=best_attacker)
        obs, reward, done, _ = env.step(A.item())
        r += 1
print(f'Try 1000 episodes. Average reward is {r / 1000}.')

torch.save(best_attacker.state_dict(), 'models/attacker.pth')