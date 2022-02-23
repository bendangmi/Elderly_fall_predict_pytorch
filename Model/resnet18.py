"""
   File Name：     resnet18
   Description :
   Author :       本当迷
   date：          2022/2/22
"""


# 搭建神经网络
from torch import nn
from torchvision import models


class Resnet18(nn.Module):
    def __init__(self):
        super().__init__()
        model_ft = models.resnet18(pretrained=True)
        num_ft = model_ft.fc.in_features
        self.model = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ft, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.model(x)
        return x
