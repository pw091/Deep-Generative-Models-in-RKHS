import torch
import torch.nn.functional as F

def loss_function(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    '''evidence lower bound (ELBO)'''
    BCE: torch.Tensor = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum') #entropy (binary cross)
    KLD: torch.Tensor = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) #divergence (KL)
    return BCE + KLD