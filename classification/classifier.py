import torch
import torchvision
import torchvision.transforms as transforms
from resnet import *


train_transform=transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

test_transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])


batch_size = 128

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


import matplotlib.pyplot as plt
import numpy as np

#plt.interactive(False)

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

""""
dataiter = iter(trainloader)
images, labels = dataiter.next()
print(images[0].size())
imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
plt.show()
"""

import torch.nn as nn
import torch.nn.functional as F


device = 'cpu'
if torch.cuda.is_available():
    device = torch.device('cuda:0')

print(device)

net = resnet20()
net.to(device)

import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.1, momentum = 0.9, weight_decay=0.0001)


def training_accuracy(net, device, batch_size):
    class_correct = list(0.0 for i in range(10))
    class_total = list(0.0 for i in range(10))
    with torch.no_grad():
        for data in trainloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            c = (predicted == labels).squeeze()
            n = min(batch_size, len(labels))
            for i in range(n):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    #for i in range(10):
    #    print('Accuracy of %5s: %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
    print('Training Error: %.03f' %  (1.0 - sum(class_correct) / sum(class_total)))

def testing_accuracy(net, device, batch_size):
    class_correct = list(0.0 for i in range(10))
    class_total = list(0.0 for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            c = (predicted == labels).squeeze()
            n = min(batch_size, len(labels))
            for i in range(n):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    #for i in range(10):
    #    print('Accuracy of %5s: %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
    print('Testing Error: %.03f' %  (1.0 - sum(class_correct)/sum(class_total)))




for epoch in range(200):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        """
        if i%2000 == 1999:
            print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/2000))
            running_loss = 0.0
        if i%12000 == 11999:
            training_accuracy(net, device)
            testing_accuracy(net, device)
        """
    print('[Epoch: %d] loss: %.3f' % (epoch + 1, running_loss / 50000))
    running_loss = 0.0
    training_accuracy(net, device, batch_size)
    testing_accuracy(net, device, batch_size)

    if epoch == 50:
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    elif epoch == 100:
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)



print('Finished Training')


testing_accuracy(net, device, batch_size)

