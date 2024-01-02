import torch
import torch.nn as nn
import torch.nn.functional as F
import random

random.seed(42)



class CombinedModel(nn.Module):
    def __init__(self, base_model, num_classes):
        super(CombinedModel, self).__init__()
        
        self.feature_extractor = base_model
        self.classifier = nn.Linear(512,num_classes)
    

    def forward(self, x):
        # Forward pass through the ResNet-based feature extractor
        x = self.feature_extractor(x)
        # Global average pooling (GAP)
        # x = torch.mean(x, dim=(2, 3))  # Assuming the output shape is (batch_size, 512, H, W)

        # Forward pass through the linear classification layer
        x = self.classifier(x)
        print('after class', x.shape)
        return x
    

class ImageNetModel(nn.Module):
    def __init__(self, base_model, in_channels = 512, num_classes = 2):
        super(ImageNetModel, self).__init__()
        
        self.feature_extractor = base_model
        self.classifier = nn.Linear(in_channels,num_classes)

    def forward(self, x):
        # Forward pass through the ResNet-based feature extractor
        x = self.feature_extractor(x)
        # print(x.shape)
        # Global average pooling (GAP)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        
        # x = x.view(x.size(0), -1)
        x = self.classifier(x)
        # x = F.softmax(x, dim=1)
        return x
    

class CLIPModel(nn.Module):
    def __init__(self, base_model, num_classes):
        super(CLIPModel, self).__init__()
        
        self.feature_extractor = base_model
        self.classifier = nn.Linear(512,num_classes)
    

    def forward(self, x):
        print(x.shape)
        # Forward pass through the ResNet-based feature extractor
        x = self.feature_extractor(x)
        print('after feature extraction', x.shape)
        # Global average pooling (GAP)
        x = torch.mean(x, dim=(2, 3))  # Assuming the output shape is (batch_size, 512, H, W)

        # Forward pass through the linear classification layer
        x = self.classifier(x)
        output = F.softmax(output, dim=1)
        print('after class', x.shape)
        return x
    


class CheXpertModel(nn.Module):
    def __init__(self, base_model, num_classes):
        super(CheXpertModel, self).__init__()
        
        self.feature_extractor = base_model.features
        self.classifier = nn.Linear(1024,num_classes)
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Forward pass through the ResNet-based feature extractor
        x = self.feature_extractor(x)
        # print(x.shape)
        # Global average pooling (GAP)
        x = self.global_avg_pooling(x)  # Assuming the output shape is (batch_size, 512, H, W)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # Forward pass through the linear classification layer
        x = self.classifier(x)
        return x
    


'''
Adapted from kuangliu/pytorch-cifar .
'''

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
    def __init__(self, block, num_blocks, in_channels=1, num_classes=2):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

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
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(in_channels, num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], in_channels=in_channels, num_classes=num_classes)


def ResNet50(in_channels, num_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3], in_channels=in_channels, num_classes=num_classes)



class ViTModel(nn.Module):
    def __init__(self, base_model, in_channels = 512, num_classes = 2):
        super(ViTModel, self).__init__()
        
        self.feature_extractor = base_model
        self.classifier = nn.Linear(in_channels,num_classes)

    def forward(self, x):
        # Forward pass through the ResNet-based feature extractor
        x = self.feature_extractor(x)
        print(x)
        # Global average pooling (GAP)
        # x = x.view(x.size(0), -1)
        # print(x.shape)
        
        # x = x.view(x.size(0), -1)
        x = self.classifier(x)
        # x = F.softmax(x, dim=1)
        return x