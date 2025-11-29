# ============================================
# File: shared/model.py
# ============================================

import torch
import torch.nn as nn


class ActorCriticNet(nn.Module):
    def __init__(self, obs_shape, n_actions, hidden_size=64, n_hidden_layers=1):
        super().__init__()
        c, h, w = obs_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            conv_out = self.conv(torch.zeros(1, c, h, w)).shape[1]

        layers = []
        # Input dim is conv_out + 1 (for prev_reward)
        in_dim = conv_out + 1 
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            in_dim = hidden_size

        self.mlp = nn.Sequential(*layers) if layers else nn.Identity()
        latent_dim = hidden_size if layers else in_dim

        self.actor = nn.Linear(latent_dim, n_actions)
        self.critic = nn.Linear(latent_dim, 1)

    def forward(self, x, prev_reward):
        # x: [B, C, H, W]
        # prev_reward: [B, 1]
        z_img = self.conv(x)
        
        # Fuse image features with previous reward
        z_combined = torch.cat([z_img, prev_reward], dim=1)
        
        z = self.mlp(z_combined)
        logits = self.actor(z)
        value = self.critic(z)
        return logits, value, z