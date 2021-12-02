from torch import randint
import gym

r = 0
env = gym.make('CartPole-v0')
for i in range(1000):
    obs, done = env.reset(), False
    while (done != True) :
        A =  randint(0, env.action_space.n, (1,))
        obs, reward, done, info = env.step(A.item())
        r += 1
    
print(f"Try 1000 episodes. Average reward is {r / 1000}.")