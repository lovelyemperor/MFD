import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import CosineLinear_PEDCC



def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, shortcut=None):
        super(BasicBlock, self).__init__()
        self.layers = nn.Sequential(
            conv3x3(in_planes, planes, stride),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            conv3x3(planes, planes),
            nn.BatchNorm2d(planes),
        )
        self.shortcut = shortcut

    def forward(self, x):
        residual = x
        y = self.layers(x)
        if self.shortcut:
            residual = self.shortcut(x)
        y += residual
        y = F.relu(y)
        return y

class BasicBlock1(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, shortcut=None):
        super(BasicBlock1, self).__init__()
        self.layers = nn.Sequential(
            conv3x3(in_planes, planes, stride),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            conv3x3(planes, planes),
            nn.BatchNorm2d(planes),
        )
        self.shortcut = shortcut

    def forward(self, x):
        residual = x
        y = self.layers(x)
        if self.shortcut:
            residual = self.shortcut(x)
        y += residual
        y = F.relu(y)
        return y


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, shortcut=None):
        super(Bottleneck, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * 4),
        )
        self.shortcut = shortcut

    def forward(self, x):
        residual = x
        y = self.layers(x)
        if self.shortcut:
            residual = self.shortcut(x)
        y += residual
        y = F.relu(y)
        return y




class BasicBlock2(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, shortcut=None):
        super(BasicBlock2, self).__init__()
        self.layers = nn.Sequential(
            conv3x3(in_planes, planes, stride),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            conv3x3(planes, planes),
            nn.BatchNorm2d(planes),
        )
        self.shortcut = shortcut

    def forward(self, x):
        residual = x
        y = self.layers(x)
        if self.shortcut:
            residual = self.shortcut(x)
        y1 = residual + y
        y2 = torch.tanh(y1)
        return y2, y1



class Bottleneck1(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, shortcut=None):
        super(Bottleneck1, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * 4),
        )
        self.shortcut = shortcut

    def forward(self, x):
        residual = x
        y = self.layers(x)
        if self.shortcut:
            residual = self.shortcut(x)
        y1 = residual + y
        y2 = F.relu(y1)
        return y2, y


class Bottleneck2(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes,  feature_size=100, stride=1, shortcut=None):
        super(Bottleneck2, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            nn.Conv2d(planes, feature_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_size),
        )
        self.feature_size = feature_size

    def forward(self, x):
        residual = x
        y = self.layers(x)
        # if self.shortcut:
        residual = x[:, 0:self.feature_size, :, :]
        y = residual + y
        # y2 = torch.tanh(y1)
        return y, y

class Bottleneck3(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, shortcut=None):
        super(Bottleneck3, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * 4),
        )
        self.shortcut = shortcut

    def forward(self, x):
        residual = x
        y = self.layers(x)
        if self.shortcut:
            residual = self.shortcut(x)
        y += residual
        y = F.relu(y)
        return y

class ResNet(nn.Module):
    def __init__(self, block , nblocks, num_classes=100, feature_size=256):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.pre_layers = nn.Sequential(
            conv3x3(3, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.block1 = Bottleneck1(256, 64)
        # self.block2 = Bottleneck1(512, 128)
        # self.block3 = Bottleneck1(1024, 256)
        # self.block4 = Bottleneck2(2048, 512, feature_size=num_classes)
        # # self.block4_1 = Bottleneck2(2048, 512)
        # self.block4_2 = BasicBlock2(512, 512)
        self.layer1 = self._make_layer(block, 64, nblocks[0])
        self.layer2 = self._make_layer(block, 128, nblocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, nblocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, nblocks[3], stride=2)
        self.block11 = self.layer1[0]
        self.block12 = self.layer1[1]
        self.block13 = self.layer1[2]
        self.block21 = self.layer2[0]
        self.block22 = self.layer2[1]
        self.block23 = self.layer2[2]
        self.block24 = self.layer2[3]
        self.block31 = self.layer3[0]
        self.block32 = self.layer3[1]
        self.block33 = self.layer3[2]
        self.block34 = self.layer3[3]
        self.block35 = self.layer3[4]
        self.block36 = self.layer3[5]
        self.block41 = self.layer4[0]
        self.block42 = self.layer4[1]
        # self.layer4 = self._make_layer_1(block_1, 512, nblocks[3], stride=2)
        # self.avgpool = nn.AvgPool2d(4)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512*block.expansion, feature_size)
        # self.out = CosineLinear_PEDCC(feature_size, num_classes)
        self.fc1 = nn.Linear(512*block.expansion, num_classes)
        self.bn = nn.BatchNorm1d(num_classes)

    def l2_norm(self, input):          # According to amsoftmax, we have to normalize the feature, which is x here
        x_norm = torch.norm(input, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        xout = torch.div(input, x_norm)
        return xout

    def _make_layer(self, block, planes, nblocks, stride=1):
        shortcut = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            shortcut = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, shortcut))
        self.in_planes = planes * block.expansion
        for i in range(1, nblocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def _make_layer_1(self, block, planes, nblocks, stride=1):
        shortcut1 = nn.Sequential(
            nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )
        shortcut2 = nn.Sequential(
            nn.Conv2d(planes * block.expansion, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )
        shortcut3 = nn.Sequential(
            nn.Conv2d(planes * block.expansion, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )
        layers = []
        layers.append(block(self.in_planes, planes, stride, shortcut1))
        self.in_planes = planes * block.expansion
        layers.append(block(self.in_planes, planes, stride, shortcut2))
        layers.append(block(self.in_planes, planes, stride, shortcut3))
        return nn.Sequential(*layers)

    def forward(self, x_origin):

        x_0 = self.pre_layers(x_origin)
        # x_0_1 = self.maxpool(x_0)
        x_11 = self.block11(x_0)
        x_12 = self.block12(x_11)
        x_13 = self.block11(x_12)
        x_21 = self.block12(x_13)
        x_22 = self.block11(x_21)
        x_23 = self.block12(x_22)
        x_24 = self.block11(x_23)
        x_31 = self.block12(x_24)
        x_32 = self.block11(x_31)
        x_33 = self.block12(x_32)
        x_34 = self.block11(x_33)
        x_35 = self.block12(x_34)
        x_36 = self.block11(x_35)
        x_41 = self.block12(x_36)
        x_42 = self.block11(x_41)
        x_2048 = self.block4(x_42)
        # x_1 = self.layer1(x_0)
        # x_2 = self.layer2(x_1)
        # x_3 = self.layer3(x_2)
        # # x_2048 = self.layer4(x_3)
        # x_2048, layer4_output = self.block4(x_2048)
        # x_2048, layer4_output = self.block4_1(x_2048)
        # x_2048_1, layer4_output_1 = self.block4_1(x_2048)
        x1 = self.avgpool(x_2048)
        x1 = x1.view(x1.size(0), -1)
        # x = self.fc(x)
        # x = torch.tanh(x)
        # x_norm = self.l2_norm(x)
        # x_norm = self.fc(x_norm)
        # out = self.out(x_norm)
        out = self.fc1(x1)
        # out = self.bn(out)
        return x_0, x_11, x_12, x_13, x_21, x_22, x_23, x_24, x_31, x_32, x_33, x_34, x_35, x_36, x_41, x_42, x_2048, out
        # return x_0, x_0, x_1, x_2, x_3, x_2048, out
        # return x_origin, x, x_0, x_1, x_2
        # return out, layer1_output, layer2_output, layer3_output, layer4_output


def resnet18():
    return ResNet(BasicBlock, [2,2,2,2])

def resnet34():
    return ResNet(BasicBlock, [3,4,6,3])

def resnet50():
    return ResNet(Bottleneck, [3,4,6,3])

def resnet101():
    return ResNet(Bottleneck, [3,4,23,3])

def resnet152():
    return ResNet(Bottleneck, [3,8,36,3])

# cnn = ResNet(BasicBlock, [2,2,2,2])

