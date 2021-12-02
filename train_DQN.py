import gym
from network import DQN_Agent
from vars import *
import matplotlib.pyplot as plt
import copy
from statsmodels.tsa.api import SimpleExpSmoothing

env = gym.make('CartPole-v0')
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
agent = DQN_Agent(seed=1423, layer_sizes=[input_dim, 64, output_dim], lr=LR, sync_freq=SYNC_FREQ, exp_replay_size=EXP_REPLAY_SIZE, gamma=GAMMA)
agent.initialize(env)
            
durations = []
index = EXP_REPLAY_SIZE / 2
epsilon = EPS_START
best_agent = agent

for i in range(N_EPISODES):
    obs, done, duration = env.reset(), False, 0
    while (not done):
        duration += 1 
        A = agent.get_action(obs, env.action_space.n, epsilon)
        obs_next, reward, done, _ = env.step(A.item())
        agent.collect_experience([obs, A.item(), reward, obs_next])
       
        obs = obs_next
        index += 1
        
        if(index > EXP_REPLAY_SIZE / 2):
            index = 0
            for _ in range(4):
                loss = agent.train(batch_size=BATCH_SIZE)

    if epsilon > EPS_END:
        epsilon -= (1 / EPS_DECAY)

    durations.append(duration)
    if duration >= max(durations):
        best_agent = copy.deepcopy(agent)

fit1 = SimpleExpSmoothing(durations, initialization_method="heuristic").fit(smoothing_level=0.05, optimized=False)
plt.plot(durations)
plt.plot(fit1.fittedvalues)
plt.savefig('hello.jpg')
plt.close()

env = gym.make('CartPole-v0')
for i in range(10):
    obs, done, r = env.reset(), False, 0
    while (not done):
        A =  best_agent.get_action(obs, env.action_space.n, epsilon = 0)
        obs, reward, done, _ = env.step(A.item())
        r += 1
    print(f'episode {i}, duration {r}.')