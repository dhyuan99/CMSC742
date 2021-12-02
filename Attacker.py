import torch.nn as nn

class Attacker(nn.Module):
    def __init__(self, hidden_layer_size, num_layers, norm_bound):
        super(Attacker, self).__init__()
        layers = []
        for _ in range(num_layers):
            linear = nn.Linear(hidden_layer_size, hidden_layer_size)
            act = nn.Tanh()
            layers += (linear, act)
        self.layers = nn.Sequential(*layers)
        self.norm_bound = norm_bound

    def forward(self, x):
        return self.layers(x) * self.norm_bound
        
