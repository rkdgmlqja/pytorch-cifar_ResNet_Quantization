'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch.nn.init as init
import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
import math
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable, Function
import math
from torch import linalg as LA


class Quant_Conv2d(nn.Conv2d):
    def __init__(self, in_planes, out_planes, kernel_size, padding=0, stride=1, bias=False):
        super(Quant_Conv2d, self).__init__(in_planes, out_planes, kernel_size, stride, padding, bias=False)
        self.weight_quant = Qweight.apply
        self.act_quant = Qact.apply
        self.bit = 4

    def forward(self, x, bit = 4):
        self.bit = bit
        q_w = self.weight_quant(self.weight, self.bit)
        q_a = self.act_quant(x, self.bit)
        y = F.conv2d(q_a, q_w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return y


class Quant_ReLU(nn.ReLU):
    def __init__(self, inplace = False):
        super(Quant_ReLU, self).__init__(inplace = False)
        self.act_quant = Qact.apply
        self.bit = 4
        self.RelU = nn.ReLU()

    def forward(self, x, bit = 4):
        self.bit = bit
        q_a = self.act_quant(x, self.bit)
        y = self.RelU(q_a)
        #y = y.detatch.numpy()
        return y



class Qweight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bit = 4):
        max_val = torch.max(x)
        min_val = torch.min(x)

        bits = math.pow(2, bit) - 1

        scale_factor = (max_val - min_val) / bits

        Quant_weight = torch.round(x / scale_factor)
        Quant_weight = Quant_weight * scale_factor

        return Quant_weight

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None  # Using STE


class Qact(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bit = 4):
        #print(x)
        max_val = torch.max(x)

        bits = math.pow(2, bit) - 1

        scale_factor = max_val / bits

        Quant_act = torch.round(x / scale_factor)  # Quantization
        Quant_act = Quant_act * scale_factor  # Dequantization

        return Quant_act#, None

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


############################
class Linearqt(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(Linearqt, self).__init__(in_features, out_features, bias)
        self.weight_quant = Qweight.apply
        self.act_quant = Qact.apply
        self.bit = 4

    def forward(self, x, bit = 4):
        self.bit = bit
        q_w = self.weight_quant(self.weight, self.bit)
        #print(self.in_features)
        q_a = self.act_quant(x, self.bit)
        return F.linear(q_a, q_w, self.bias)




#import linear_quantization as qt

class BasicBlockqt(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockqt, self).__init__()
        self.conv1 = Quant_Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Quant_Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.ReLUqt = Quant_ReLU()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                Quant_Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.ReLUqt(out)
        out = self.bn2(self.conv2(out))
        # print("original",x.shape)
        # print("shortcut",self.shortcut(x).shape)
        # print("conv2",out.shape)
        out += self.shortcut(x)
        out = self.ReLUqt(out)
        return out


class Bottleneckqt(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneckqt, self).__init__()
        self.conv1 = Quant_Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Quant_Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = Quant_Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.ReLUqt = Quant_ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Quant_Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.ReLUqt(out)
        out = self.bn2(self.conv2(out))
        out = self.ReLUqt(out)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.ReLUqt(out)
        return out


class ResNetqt(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetqt, self).__init__()
        self.in_planes = 64

        self.conv1 = Quant_Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = Linearqt(512*block.expansion, num_classes)
        self.ReLUqt = Quant_ReLU()
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.ReLUqt(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNetqt18():
    return ResNetqt(BasicBlockqt, [2, 2, 2, 2])


def ResNetqt34():
    return ResNetqt(BasicBlockqt, [3, 4, 6, 3])


def ResNetqt50():
    return ResNetqt(Bottleneckqt, [3, 4, 6, 3])


def ResNetqt101():
    return ResNetqt(Bottleneckqt, [3, 4, 23, 3])


def ResNetqt152():
    return ResNetqt(Bottleneckqt, [3, 8, 36, 3])


# def test():
#     net = ResNet18()
#     y = net(torch.randn(1, 3, 32, 32))
#     print(y.size())



# test()
'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        print("original", x.shape)
        print("shortcut", self.shortcut(x).shape)
        print("conv2", out.shape)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

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
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])
