import sys
sys.path.append('../')

import torch
from torch import optim
from torchvision import datasets, transforms

from classification.vae import VAE

batch_size = 8
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
                    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True)

seed = 0
torch.manual_seed(seed)


device = torch.device("cuda" if args.cuda else "cpu")


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=le-3)

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x,
                                 x.view(-1, 28*28),
                                 reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

model.train()
train_loss = 0
for batch_idx, (data, _) in enumerate(train_loader):
    
