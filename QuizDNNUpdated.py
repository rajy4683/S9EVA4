"""
     All model class definitions for Quiz
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


class S9QuizDNN(nn.Module):
    def __init__(self, dropout=0.1):
        super(S9QuizDNN, self).__init__()
        self.dropout_val = dropout
        self.bias = False
        self.conv1 = nn.Conv2d(3, 64, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(64)
        self.conv7 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(64)
        self.pool8 = nn.MaxPool2d(2, 2) 

        self.conv9 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(64)

        self.conv10 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn10 = nn.BatchNorm2d(64)

        self.conv11 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn11 = nn.BatchNorm2d(64)

        self.gap_linear = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(64, 10, 1, bias=self.bias)
        )

    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x)))
        x3 = F.relu(self.bn3(self.conv3(x2+x1)))
        x3 = self.pool4(x3) ### 1st Pooled layer = x3 (same as x4) in the assignment

        x5 = F.relu(self.bn5(self.conv5(x3)))
        x6 = F.relu(self.bn6(self.conv6(x5+x3)))
        x7 = F.relu(self.bn7(self.conv7(x6+x5+x3)))
        x7 = self.pool8(x7)

        x9 = F.relu(self.bn9(self.conv9(x7)))
        x10 = F.relu(self.bn10(self.conv10(x7+x9)))
        x11 = F.relu(self.bn9(self.conv9(x7+x9+x10)))

        x12 = self.gap_linear(x11)
        x12 = x12.view(-1, 10)
        #x12 = F.log_softmax(x12, dim=1)
        #x = x.view(-1, 10)
        #x = F.log_softmax(x, dim=1)
        return x12









