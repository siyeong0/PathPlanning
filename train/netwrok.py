import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class BasicNet(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        c, w, h = observation_space.shape
        self.convs = nn.Sequential(
        nn.Conv2d(c, 32, 8, stride=4, padding=0), nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=2, padding=0), nn.ReLU(),
        nn.Flatten()
     )
        with torch.no_grad():
            flat_dim = self.convs(torch.tensor(observation_space.sample()[None]).float()).shape[1]
        self.linear = nn.Sequential(nn.Linear(flat_dim, features_dim), nn.ReLU())

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight.data)
        
    def forward(self, x):
        x = self.convs(x)
        return self.linear(x)