# -*- coding:utf-8 -*-

"""
University of Sherbrooke
Date:
Authors: Mamadou Mountagha BAH & Pierre-Marc Jodoin
License:
Other: Suggestions are welcome
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNNBlock(nn.Module):
    """
    this block is an example of a simple conv-relu-conv-relu block
    with 3x3 convolutions
    """

    def __init__(self, in_channels):
        super(SimpleCNNBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        output = F.relu(self.conv1(x))
        output = F.relu(self.conv2(output))
        return output


""" 
TODO

Suivant l'example ci-haut, vous devez rédiger les classes permettant de créer des :

1- Un bloc résiduel
2- Un bloc dense
3- Un bloc Bottleneck

Ces blocks seront utilisés dans le fichier YouNET.py
"""


class ResidualBlock(nn.Module):
    """
    this block is the building block of the residual network. it takes an 
    input with in_channels, applies some blocks of convolutional layers
    to reduce it to out_channels and sum it up to the original input,
    """
    
    # source : https://towardsdatascience.com/residual-network-implementing-resnet-a7da63c7b278

    def __init__(self, in_channels, out_channels, stride=1):
        """
        Inputs:
            in_channels - Number of input features
            out_channels - Number of output features. Note that this is only relevant if subsample is True, as otherwise, c_out = c_in
        """
        super(ResidualBlock, self).__init__()
        #TODO
        """
        self.in_channels, self.out_channels = in_channels, out_channels
        self.activation = 'relu'
        self.blocks = nn.Identity()
        self.activate = nn.ReLU(inplace=True)
        self.shortcut = nn.Identity()
        """
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),  
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.act_fn = nn.ReLU(inplace=True)
            
    def forward(self, x):

        
        z = self.net(x)
        """
        if self.downsample is not None:
            x = self.downsample(x)
            
        """
        out = z + x
        out = self.act_fn(out)
        return out


class DenseBlock(nn.Module):
    """
    This block is the building block of the Dense network. It takes an
    input with in_channels, applies some blocks of convolutional, batchnorm layers
    and then concatenate the output with the original input
    """
    
    #https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial5/Inception_ResNet_DenseNet.html#DenseNet

    def __init__(self, in_channels, num_layers, bn_size, growth_rate):
        """
        Inputs:
            in_channels - Number of input channels
            num_layers - Number of dense layers to apply in the block
            bn_size - Bottleneck size to use in the dense layers
            growth_rate - Growth rate to use in the dense layers
        """
        super(DenseBlock, self).__init__()
        act_fn = nn.ReLU(inplace=True)
        layers = []
        for layer_idx in range(num_layers):
            layers.append(
                DenseLayer(in_channels=in_channels + layer_idx * growth_rate, # Input channels are original plus the feature maps from previous layers
                           bn_size=bn_size,
                           growth_rate=growth_rate)
            )
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        return out
      

# lien :  https://github.com/pytorch/vision/blob/1aef87d01eec2c0989458387fa04baebcc86ea7b/torchvision/models/resnet.py#L75  
class BottleneckBlock(nn.Sequential):
    def __init__(self, in_planes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1):
        super(BottleneckBlock, self).__init__()
        self.expansion = 4
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = nn.Conv2d(in_planes, width, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=1,
                     padding=1, groups=1, bias=False, dilation=1)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width,  planes * self.expansion, kernel_size=1, stride=stride, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
      

class DenseLayer(nn.Module):

    def __init__(self, in_channels, bn_size, growth_rate):
        """
        Inputs:
            in_channels - Number of input channels
            bn_size - Bottleneck size (factor of growth rate) for the output of the 1x1 convolution. Typically between 2 and 4.
            growth_rate - Number of output channels of the 3x3 convolution
        """
        super().__init__()
        act_fn = nn.ReLU(inplace=True)
        self.net = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, bn_size * growth_rate, kernel_size=1, bias=False),
            nn.BatchNorm2d(bn_size * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        out = self.net(x)
        out = torch.cat([out, x], dim=1)
        return out
      
