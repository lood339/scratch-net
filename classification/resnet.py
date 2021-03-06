import torch.nn as nn
import torch

__all__ = ['ResNet', 'ResNet20', 'ResNet32']

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out

    def extract_feature(self, x, features=[]):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        features.append(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        features.append(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(x)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(x)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 16
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # decrease space resolution and increase planes
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def extract_feature(self, x, features = []):
        x = self.conv1(x)
        x = self.bn1(x)
        features.append(x)
        x = self.relu(x)

        blocks = list(self.layer1)
        for b in blocks:
            x = b.extract_feature(x, features)

        blocks = list(self.layer2)
        for b in blocks:
            x = b.extract_feature(x, features)

        blocks = list(self.layer3)
        for b in blocks:
            x = b.extract_feature(x, features)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        features.append(x)
        return x

def ResNet20(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [3, 3, 3], **kwargs)
    return model

def ResNet32(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [5, 5, 5], **kwargs)
    return model


def ut_resnet20():
    resnet20 = ResNet(BasicBlock, [3, 3, 3], 10)

    input = torch.randn(1, 3, 32, 32)
    output = resnet20.forward(input)
    print(output.shape)

    # Estimate Size
    from pytorch_modelsize import SizeEstimator

    se = SizeEstimator(resnet20, input_size=(1, 3, 32, 32))
    print(se.estimate_size())


def ut_resnet32():
    resnet32 = ResNet(BasicBlock, [5, 5, 5], 10)

    input = torch.randn(2, 3, 32, 32)
    output = resnet32.forward(input)
    print(output.shape)

def ut_save_model():
    #resnet20 = ResNet(BasicBlock, [3, 3, 3], 10)

    #torch.save(resnet20.state_dict(), 'init.pth.tar')
    """
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        net = net.to(device)
    """

    device = torch.device('cpu')
    model = ResNet(BasicBlock, [3, 3, 3], 10)
    checkpoint = torch.load('resnet_0250_epoch.pth', map_location=device)
    model.load_state_dict(checkpoint['state_dict'])


    input = torch.randn(1, 3, 32, 32)
    features = []
    model.extract_feature(input, features)
    for feat in features:
        print(feat.shape)

    modules = list(model.children())[:-1]
    model = nn.Sequential(*modules)

    #print(model)

    test = 1




if __name__ == '__main__':
    ut_resnet20()

    #ut_resnet32()

    #ut_save_model()

