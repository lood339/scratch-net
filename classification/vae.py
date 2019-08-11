# modified from https://github.com/pytorch/examples/blob/master/vae/main.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(28*28, 400)
        self.fc21 = nn.Linear(400, 20)  # mu
        self.fc22 = nn.Linear(400, 20)  # logvar

        #decoder
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 28*28)

    def encoder(self, x):
        h1 = F.relu(self.fc1(x))
        # why return two?
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        """
        What is this?
        :param mu:
        :param logvar:
        :return:
        """
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)  #(0, 1) Gaussian distribution

        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encoder(x.view(-1, 28*28))  # --> 20
        z = self.reparameterize(mu, logvar)  # --> 20
        return self.decode(z), mu, logvar    # --> 28*28

