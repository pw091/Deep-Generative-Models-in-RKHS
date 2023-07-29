from typing import List
from torch.utils.data import DataLoader
import torch
from model.py import CVAE

def train(vae: CVAE, epochs: int, train_data: DataLoader, batch_size: int, optimizer, loss_function) -> List[float]:
    '''training loop for CVAE; results in trained model (1st arg) and returns mean loss at each epoch'''
    epoch_losses = []

    for epoch in range(1, epochs + 1):
        vae.train()
        train_loss = 0

        for batch_idx, (data, labels) in enumerate(train_data):
            data, labels = data, labels
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(data, labels)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        avg_loss = train_loss / len(train_data.dataset)
        epoch_losses.append(avg_loss)
        
    return epoch_losses