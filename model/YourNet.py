# -*- coding:utf-8 -*- 

"""
University of Sherbrooke
Date:
Authors: Mamadou Mountagha BAH & Pierre-Marc Jodoin
License:
Other: Suggestions are welcome
Modified: 13/04/2023
By: Martin CABOTTE
For: IFT 714
"""

import torch.nn as nn
from model.CNNBaseModel import CNNBaseModel
from layers.CNNBlocks import ResidualBlock, DenseBlock, BottleneckBlock



class YourNet(CNNBaseModel):

    def __init__(self, num_classes=10, init_weights=True):
        super(YourNet, self).__init__()


        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

          	DenseBlock(16,1,2,48),

          
          	BottleneckBlock(64,16),
          	ResidualBlock(64,64),
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(65536, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """
        Forward pass of the model
        Args:
             x: Tensor
        """
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # reshape feature maps
        x = self.fc_layers(x)
        return x


'''
FIN DE VOTRE CODE
'''
