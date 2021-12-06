import gym
import torch
from Agent import Agent
import matplotlib.pyplot as plt
import copy
from statsmodels.tsa.api import SimpleExpSmoothing
import argparse
import json

parser = argparse.ArgumentParser(description='Please provide the config file.')
parser.add_argument('-config', type=str, help='the path to the config file')
args = parser.parse_args()
with open(args.config) as f:
    config = json.load(f)

env = gym.make('CartPole-v0')
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
agent = Agent(seed=1423,
    layer_sizes=[input_dim, config['hidden_layer_size'], output_dim], 
    lr=config['lr'], 
    sync_freq=config['sync_freq'], 
    exp_replay_size=config['exp_replay_size'],
    gamma=config['gamma'])
agent.initialize(env)
if config['reg'] is not None:
    agent.q_net.load_state_dict(torch.load('models/agent_normal.pth'))
    agent.q_net.eval()

durations = []
index = config['exp_replay_size'] / 2
epsilon = config['eps_start']
best_agent = agent

for i in range(config['n_episodes']):
    obs, done, duration = env.reset(), False, 0
    while (not done):
        duration += 1 
        A = agent.get_action(obs, env.action_space.n, epsilon, attacker=None)
        obs_next, reward, done, _ = env.step(A.item())
        agent.collect_experience([obs, A.item(), reward, obs_next])
       
        obs = obs_next
        index += 1
        
        if(index > config['exp_replay_size'] / 2):
            index = 0
            for _ in range(4):
                q_loss, reg_loss = agent.compute_loss(batch_size=config['batch_size'], reg=config['reg'])
                loss = q_loss + reg_loss if reg_loss is not None else q_loss
                if config['reg'] is None:
                    agent.optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    agent.optimizer.step()
                else:
                    agent.optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    agent.optimizer.step()
                    

    if i % 100 == 0:         
        print(f'episode {i}, q_loss: {q_loss.item()}, reg_loss: {reg_loss.item() if reg_loss is not None else None}')

    if epsilon > config['eps_end']:
        epsilon -= (1 / config['eps_decay'])

    durations.append(duration)
    if duration >= max(durations):
        best_agent = copy.deepcopy(agent)

fit1 = SimpleExpSmoothing(durations, initialization_method="heuristic").fit(smoothing_level=0.05, optimized=False)
plt.plot(durations)
plt.plot(fit1.fittedvalues)
plt.xlabel("Episodes")
plt.ylabel("Duration")
plt.title("Performance of the Agent as Training.")
plt.savefig(config['figpath'])
plt.close()

env = gym.make('CartPole-v0')
r = 0
for i in range(1000):
    obs, done = env.reset(), False
    while (not done):
        A =  best_agent.get_action(obs, env.action_space.n, epsilon = 0)
        obs, reward, done, _ = env.step(A.item())
        r += 1

print(f"Try 1000 episodes. Average reward is {r / 1000}.")
agent.save(config['savepath'])