
# customize VGG network for image feature extraction

import torch.nn as nn
import torch.utils.model_zoo as model_zoon
import torch

__all__ = ['vgg16']

model_urls = {
    'vgg16' : 'https://download.pytorch.org/models/vgg16-397923af.pth',
}

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self.feature_pool = nn.AvgPool2d(kernel_size=7)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # N x C x H x W --> N x D, D is the feature dimension
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def feature_avg_pool(self, x):
        """
        average pool the last CNN layer as feature
        :param x:
        :return:
        """
        x = self.features(x)
        x = self.feature_pool(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)

    def analyze_weights(self):
        """
        @brief for debug
        :return:
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    print('Conv2d: \t', m.weight.shape, m.bias.shape)
                else:
                    print('Conv2d: \t', m.weight.shape)

            elif isinstance(m, nn.BatchNorm2d):
                print('BatchNorm2d: \t', m.weight, m.bias)
            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    print('Linear: \t', m.weight.shape, m.bias.shape)
                else:
                    print('Linear: \t', m.weight.shape)
            elif isinstance(m, nn.MaxPool2d):
                pass
                #print('MaxPool2d:', m.kernel_size)
            elif isinstance(m, nn.ReLU):
                pass
                #print('Relu: ')
            elif isinstance(m, nn.Dropout):
                pass
                #print('Dropout:')
            elif isinstance(m, nn.Sequential):
                print('Sequential:')
            else:
                print("warning: un-recognized layer types")

    def remove_half_parameters(self):
        """
        remove half of the parameters to make a small VGG
        It is only tested on VGG 16
        :return:
        """
        in_channels = 3
        for param in self.features.parameters():
            data = param.data
            n = data.size(0) // 2
            if len(data.size()) == 1:
                half_data = data[0:n]
            elif len(data.size()) == 4:
                c = data.size(1)
                if c != in_channels:  # skip the first input
                    c = c//2
                half_data = data[0:n, 0:c, :, :]
            param.data = half_data

        # only change the bottle-neck layer
        for param in self.classifier.parameters():
            data = param.data
            if len(data.size()) == 2:
                c = data.size(1)
                half_data = data[:, 0:c//2]
                param.data = half_data
                break


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    # Numbers are channel numbers. M is for max-pool
    'D':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}

def vgg16(pretrained=False, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoon.load_url(model_urls['vgg16']))
    return model



def ut_avg_pool_feature():
    import torchvision.transforms as transforms
    import cv2 as cv
    from PIL import Image
    import numpy as np
    from util import vis_images

    model = vgg16(True)
    #input = torch.randn(2, 3, 224, 224)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    im = cv.imread('/Users/jimmy/Desktop/roof_6_pool.jpg', 1) # sub_image
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    im = Image.fromarray(im)

    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    input = transform(im)
    print(input.shape)
    #vis = input.numpy().transpose(1, 2, 0)
    #vis = vis[:,:,[2,1,0]]
    #images = []
    #images.append(vis)
    #vis_images(images, 1)

    input = input.view(-1, 3, 224, 224)

    feature = model.feature_avg_pool(input)
    feature = feature.data
    feature = feature.numpy()
    feature = np.squeeze(feature, axis=(2,3))

    feature_norm = np.linalg.norm(feature, axis=1)
    feature = feature / feature_norm[:, np.newaxis]

    print(feature.shape)
    feature = feature**2
    print(np.sum(feature, axis=1))


def ut_pre_trained_vgg():
    import time
    model = vgg16(True)
    input = torch.randn(20, 3, 224, 224)
    t = time.time()
    output = model.forward(input)
    print(time.time() - t)
    print(output.shape)

    #model.analyze_weights()
    model.remove_half_parameters()
    #model.analyze_weights()
    t = time.time()
    output = model.forward(input)
    print(time.time() - t)
    print(output.shape)


if __name__ == '__main__':
    ut_avg_pool_feature()
    #ut_pre_trained_vgg()



