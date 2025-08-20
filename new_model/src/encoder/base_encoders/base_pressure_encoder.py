# Import Modules
import torch
from torch import nn

# VAE Model
class LargeVariationalAutoEncoder(nn.Module):

    def __init__(self, input_dim=2520, l_dim=3072, h_dim=1024, z_dim=64): # Default h_dim updated based on train script
        super().__init__()
        # encoder
        self.img_2lhid = nn.Linear(input_dim, l_dim)
        self.lhid_2hid = nn.Linear(l_dim, h_dim)
        self.hid_2hid = nn.Linear(h_dim, h_dim)
        self.hid_2mu = nn.Linear(h_dim, z_dim)
        self.hid_2log_var = nn.Linear(h_dim, z_dim)

        # decoder
        self.z_2hid = nn.Linear(z_dim, h_dim)
        self.hid_2hid = nn.Linear(h_dim, h_dim)
        self.hid_2lhid = nn.Linear(h_dim, l_dim)
        self.lhid_2img = nn.Linear(l_dim, input_dim)

        self.relu = nn.ReLU()

    def encode(self, x):
        h = self.relu(self.img_2lhid(x))
        h = self.relu(self.lhid_2hid(h))
        h = self.relu(self.hid_2hid(h))
        mu, log_var = self.hid_2mu(h), self.hid_2log_var(h)
        return mu, log_var

    def decode(self, z):
        h = self.relu(self.z_2hid(z))
        h = self.relu(self.hid_2hid(h))
        h = self.relu(self.hid_2lhid(h))
        return self.lhid_2img(h)

    def forward(self, x):
        mu, log_var = self.encode(x)
        std = torch.exp(0.5*log_var)
        epsilon = torch.randn_like(std)
        z_reparametrized = mu + std*epsilon
        x_reconstructed = self.decode(z_reparametrized)
        return x_reconstructed, mu, log_var 