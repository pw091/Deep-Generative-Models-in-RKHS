import torch
import torch.nn as nn
import torch.nn.functional as F

class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()
        self.fc1 = nn.Linear(784 + 10, 400) #data and labels
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20 + 10, 400) #latent variable and labels
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        '''map to latent space'''
        h1 = F.relu(self.fc1(torch.cat((x, y), axis=1)))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        '''apply reparametarization trick'''
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        '''map from latent space'''
        h3 = F.relu(self.fc3(torch.cat((z, y), axis=1)))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        '''forward pass'''
        mu, logvar = self.encode(x.view(-1, 784), y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar