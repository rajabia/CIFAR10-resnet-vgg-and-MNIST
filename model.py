import torch.nn.functional as F
import torch.nn.init as init

import torch.nn as nn
import math
"""
UTILS CLASSES
"""




'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''

__all__ = ['mnist_model','VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19', 'ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202'
]
class VGG(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features, num_classes=10):
        super(VGG, self).__init__()
        self.features = features[0]
        self.block = features[1]
        self.num_classes = num_classes
        # import pdb;pdb.set_trace()

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True)
        )
        self.output = nn.Linear(512, self.num_classes)
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        # x = nn.Sequential(*self.features)(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.output(x)
        return x
        # function to extact the multiple features

    def feature_list(self, x):
        out_list = []

        for layer in self.block:

            x = nn.Sequential(*layer)(x)
            out_list.append(x)

        x = x.view(x.size(0), -1)

        x = self.classifier(x)
        y = self.output(x)
        return y, out_list

    def intermediate_forward(self, x, layer_index):
        for j in range(layer_index+1 ):
            x = nn.Sequential(*self.block[j])(x)
        return x

    def penultimate_forward(self, x):
        for layer in self.block:
            x = nn.Sequential(*layer)(x)


        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        penultimate = x
        x = self.output(x)
        return x, penultimate



def make_layers(cfg, batch_norm=False):
    layers = []
    sub_block = []
    blocks= []

    in_channels = 3
    for v in cfg:
        if v == 'M':
            mpool2d = nn.MaxPool2d(kernel_size=2, stride=2)
            layers += [mpool2d]
            sub_block +=[mpool2d]
            blocks.append(sub_block)
            sub_block = []
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            batchnorm2d = nn.BatchNorm2d(v)
            relu2d = nn.ReLU(inplace=True)

            if batch_norm:
                layers += [conv2d, batchnorm2d, relu2d]
                sub_block+= [conv2d, batchnorm2d, relu2d]
            else:
                layers += [conv2d, relu2d]
                # block.append([conv2d, relu2d])
                sub_block+=[conv2d, relu2d]
            in_channels = v
    return (nn.Sequential(*layers), blocks)

class MNIST(nn.Module):
    def __init__(self):
        super(MNIST,self).__init__()
        self.conv1=nn.Conv2d(1,64,3,1)
        self.conv2=nn.Conv2d(64,128,3,1)
        self.conv3=nn.Conv2d(128,200,3,1)
        self.conv4=nn.Conv2d(200,200,3,1)
        self.dropout1=nn.Dropout2d(0.25)
        self.dropout2=nn.Dropout2d(0.5)
        self.FC1=nn.Linear(3200,400)
        self.FC2=nn.Linear(400,256)
        self.FC3=nn.Linear(256,10)
    def forward(self,x):
        x=self.conv1(x)
        x=F.relu(x)
        x=self.conv2(x)
        x=F.relu(x)
        x=F.max_pool2d(x,2)
        
        x=self.conv3(x)
        x=F.relu(x)
        x=self.conv4(x)
        x=F.relu(x)
        x=F.max_pool2d(x,2)
        #x=self.dropout1(x)
        x=torch.flatten(x,1)
        x=self.FC1(x)
        x= F.relu(x)
        #x=self.dropout2(x)
        x=self.FC2(x)
        x= F.relu(x)
        output=self.FC3(x)
        return output


class cuda_conv(nn.Module):
    def __init__(self,num_class):
        super(cuda_conv, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, num_class)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def _weights_init(m):
    classname = m.__class__.__name__
    print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56(num_class):
    return ResNet(BasicBlock, [9, 9, 9], num_classes=num_class)


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}

def vgg11():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']))


def vgg11_bn():
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True))


def vgg13():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']))


def vgg13_bn(num_classes=10):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True),num_classes=num_classes)


def vgg16(num_class=10):
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']), num_classes=num_class)


def vgg16_bn(num_class=10):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True), num_classes=num_class)

def vgg19(num_classes=10):
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']),num_classes=num_classes)


def vgg19_bn(num_classes=10):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True),num_classes=num_classes)


def mnist_model():
    return MNIST()


