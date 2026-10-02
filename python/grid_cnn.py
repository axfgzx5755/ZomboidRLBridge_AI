"""Small CNN for the existing row-major 5x5x5 navigation geometry."""
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class GridCNN(BaseFeaturesExtractor):
    """Keep spatial layout and append scalar state before the actor/critic MLPs."""
    def __init__(self, observation_space):
        size = observation_space.shape[0]
        if size not in (134, 142):
            raise ValueError('GridCNN requires navigation-v2 or door-v1')
        super().__init__(observation_space, features_dim=48 + size - 125)
        self.grid_encoder = nn.Sequential(
            nn.Conv2d(5, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(8 * 5 * 5, 48), nn.ReLU())

    @staticmethod
    def split(observations):
        grid = observations[:, 9:134].reshape(-1, 5, 5, 5).permute(0, 3, 1, 2).contiguous()
        scalars = torch.cat((observations[:, :9], observations[:, 134:]), dim=1)
        return grid, scalars

    def forward(self, observations):
        grid, scalars = self.split(observations)
        return torch.cat((self.grid_encoder(grid), scalars), dim=1)
