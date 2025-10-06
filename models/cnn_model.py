"""
CNN模型定义 / CNN Model Definition
用于图像分类任务（MNIST, Fashion-MNIST, CIFAR-10/100）
For image classification tasks (MNIST, Fashion-MNIST, CIFAR-10/100)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class SimpleCNN(nn.Module):
    """
    简单的CNN模型 / Simple CNN Model
    适用于MNIST和Fashion-MNIST数据集
    Suitable for MNIST and Fashion-MNIST datasets
    """
    
    def __init__(self, num_classes: int = 10, input_channels: int = 1):
        """
        初始化模型 / Initialize model
        
        Args:
            num_classes: 分类数量 / Number of classes
            input_channels: 输入通道数 / Number of input channels
        """
        super(SimpleCNN, self).__init__()
        
        # 卷积层 / Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # 批归一化层 / Batch normalization layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        
        # 池化层 / Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # 全连接层 / Fully connected layers
        # 对于28x28的输入，经过两次池化后变为7x7
        # For 28x28 input, after two pooling operations it becomes 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Dropout层 / Dropout layer
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播 / Forward pass
        
        Args:
            x: 输入张量 / Input tensor
            
        Returns:
            输出张量 / Output tensor
        """
        # 第一个卷积块 / First convolution block
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # 第二个卷积块 / Second convolution block
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # 展平 / Flatten
        x = x.view(x.size(0), -1)
        
        # 全连接层 / Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class CIFARCNN(nn.Module):
    """
    用于CIFAR数据集的CNN模型 / CNN Model for CIFAR datasets
    更深的网络结构以处理更复杂的图像
    Deeper network structure to handle more complex images
    """
    
    def __init__(self, num_classes: int = 10, input_channels: int = 3):
        """
        初始化模型 / Initialize model
        
        Args:
            num_classes: 分类数量 / Number of classes
            input_channels: 输入通道数 / Number of input channels
        """
        super(CIFARCNN, self).__init__()
        
        # 卷积块1 / Convolution block 1
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2