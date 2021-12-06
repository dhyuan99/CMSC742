import torch
import torch.nn as nn
from collections import deque
import random
import copy

class Agent:
    
    def __init__(self, seed, layer_sizes, lr, sync_freq, exp_replay_size, gamma):
        torch.manual_seed(seed)
        self.q_net = self.build_nn(layer_sizes)
        self.target_net = copy.deepcopy(self.q_net)
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.optimizer_last_layer = torch.optim.Adam(self.q_net[-2].parameters(), lr=lr)
        
        self.network_sync_freq = sync_freq
        self.network_sync_counter = 0
        self.gamma = gamma
        self.exp_replay_size = exp_replay_size
        self.experience_replay = deque(maxlen=exp_replay_size)

        self.relu = nn.ReLU()
        self.icdf = torch.distributions.normal.Normal(0,1).icdf

    def initialize(self, env):
        index = 0
        for _ in range(self.exp_replay_size):
            obs = env.reset()
            done = False
            while (not done):
                A = self.get_action(obs, env.action_space.n, epsilon=1)
                obs_next, reward, done, _ = env.step(A.item())
                self.collect_experience([obs, A.item(), reward, obs_next])
                obs = obs_next
                index += 1
                if(index > self.exp_replay_size):
                    break
        
    def build_nn(self, layer_sizes):
        assert len(layer_sizes) > 1
        layers = []
        for index in range(len(layer_sizes)-1):
            linear = nn.Linear(layer_sizes[index], layer_sizes[index+1])
            act = nn.Tanh() if index < len(layer_sizes)-2 else nn.Identity()
            layers += (linear,act)
        return nn.Sequential(*layers)
    
    def get_action(self, state, action_space_len, epsilon, attacker=None):
        if attacker is None:
            with torch.no_grad():
                Qp = self.q_net(torch.from_numpy(state).float())
            _, A = torch.max(Qp, axis=0)
            A = A if torch.rand(1,).item() > epsilon else torch.randint(0, action_space_len, (1,))
            return A
        else:
            with torch.no_grad():
                x = torch.from_numpy(state).float()
                for i in range(len(self.q_net)-3):
                    x = self.q_net[i](x)
                x += attacker(x)
                Qp = self.q_net[-1](self.q_net[-2](self.q_net[-3](x)))
            _, A = torch.max(Qp, axis=0)
            A = A if torch.rand(1,).item() > epsilon else torch.randint(0, action_space_len, (1,))
            return A
                
    def get_q_next(self, state):
        with torch.no_grad():
            qp = self.target_net(state)
        q, _ = torch.max(qp, axis=1)    
        return q
    
    def collect_experience(self, experience):
        self.experience_replay.append(experience)
    
    def sample_from_experience(self, sample_size):
        if(len(self.experience_replay) < sample_size):
            sample_size = len(self.experience_replay)   
        sample = random.sample(self.experience_replay, sample_size)
        s = torch.tensor([exp[0] for exp in sample]).float()
        a = torch.tensor([exp[1] for exp in sample]).float()
        rn = torch.tensor([exp[2] for exp in sample]).float()
        sn = torch.tensor([exp[3] for exp in sample]).float()   
        return s, a, rn, sn
    
    def compute_loss(self, batch_size, reg=False):
        s, a, rn, sn = self.sample_from_experience( sample_size = batch_size)
        if(self.network_sync_counter == self.network_sync_freq):
            self.target_net.load_state_dict(self.q_net.state_dict())
            self.network_sync_counter = 0
        
        # predict expected return of current state using main network
        qp = self.q_net(s)
        pred_return, _ = torch.max(qp, axis=1)
        
        # get target return using target network
        q_next = self.get_q_next(sn)
        target_return = rn + self.gamma * q_next
        
        q_loss = self.loss_fn(pred_return, target_return)
        self.network_sync_counter += 1

        reg_loss = None
        if reg:
            K, sigma, gamma, lambd = reg['K'], reg['sigma'], reg['gamma'], reg['lambda']
            si = [s+torch.randn(size=s.shape) * sigma for _ in range(K)]
            # for i in range(len(self.q_net)-2):
            #     s = self.q_net[i](s)
            s = self.q_net(s)
            ai = [self.q_net(s) for s in si]
            ai = [torch.exp(a) / torch.sum(torch.exp(a), axis=1, keepdim=True) for a in ai]
            ai = torch.stack(ai, axis=2)
            eps = 1e-3
            ai = self.icdf((1-eps) * torch.mean(ai, axis=2) + eps / 2)
            reg_loss = lambd * torch.mean(self.relu(gamma - self.relu((ai[:,1] - ai[:,0]) * (2 * a - 1))))
        return q_loss, reg_loss

    def save(self, path):
        print(f'model saved at {path}')
        torch.save(self.q_net.state_dict(), path)