import torch
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from resnet import *



train_transform=transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
    ])

test_transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
    ])


batch_size = 256

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


import torch.nn as nn


device = 'cpu'
if torch.cuda.is_available():
    device = torch.device('cuda:0')

print(device)

net = resnet20()
net.to(device)

import torch.optim as optim
criterion = nn.CrossEntropyLoss().cuda(device)
optimizer = optim.SGD(net.parameters(), lr = 0.1, momentum = 0.9, weight_decay=0.0001)
cudnn.benchmark = True


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
    for i in range(10):
        print('Accuracy of %5s: %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
    print('Testing Error: %.03f' %  (1.0 - sum(class_correct)/sum(class_total)))


def adjust_learning_rate(optimizer, epoch):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    if epoch < 80:
        lr = 0.1
    elif epoch < 120:
        lr = 0.1 * 0.1
    else:
        lr = 0.1 * 0.01

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



for epoch in range(160):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = net(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    if epoch == 0 or epoch%10 == 9:
        print('[Epoch: %d] loss: %.5f' % (epoch + 1, running_loss / 50000))
        training_accuracy(net, device, batch_size)
        testing_accuracy(net, device, 100)

    adjust_learning_rate(optimizer, epoch)



print('Finished Training')


testing_accuracy(net, device, 100)

