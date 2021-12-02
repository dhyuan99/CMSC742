from Agent import Agent
from Attacker import Attacker
import gym
import torch
from vars import *
import matplotlib.pyplot as plt
import copy
from statsmodels.tsa.api import SimpleExpSmoothing

env = gym.make('CartPole-v0')
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
agent = Agent(seed=1423, layer_sizes=[input_dim, HIDDEN_LAYER_SIZE, output_dim], lr=999, sync_freq=SYNC_FREQ, exp_replay_size=EXP_REPLAY_SIZE, gamma=GAMMA)
agent.q_net.load_state_dict(torch.load('models/DQN.pth'))
attacker = Attacker(hidden_layer_size=HIDDEN_LAYER_SIZE, norm_bound=NORM_BOUND, num_layers=NUM_LAYERS)
optimizer = torch.optim.Adam(attacker.parameters(), lr=LR)

durations = []
index = EXP_REPLAY_SIZE / 2
best_attacker = attacker

for i in range(N_EPISODES):
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
for i in range(10):
    obs, done, r = env.reset(), False, 0
    while (not done):
        A =  agent.get_action(obs, env.action_space.n, epsilon=0, attacker=best_attacker)
        obs, reward, done, _ = env.step(A.item())
        r += 1
    print(f'episode {i}, duration {r}.')

torch.save(best_attacker.state_dict(), 'models/attacker.pth')