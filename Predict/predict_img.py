# -*- coding: utf-8 -*-

import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

class_names = os.listdir('../Data/train')  # 数据集路径
img_path = '../Text_data/img/fall.png'  # 测试图片路径
model_path = '../Train_model/weights.pth'  # 训练好的模型路径


def predict(image1, model, class_names1):
    t = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img = t(image1)
    img = torch.unsqueeze(img, 0)
    output = model(img)
    confidence1, prediction1 = torch.max(output, 1)
    return confidence1.item(), class_names1[int(prediction1)]


def plot_predict(img, confidence1, prediction1):
    plt.figure()
    plt.imshow(img)
    plt.axis('off')
    plt.title('Predict:{}, confidence:{:.4f}'.format(prediction1, confidence1))


# 建立模型
model_ft = models.resnet18(pretrained=False)
num_ft = model_ft.fc.in_features
model_ft.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ft, 2),
    nn.Softmax(dim=1)
)
model_ft.load_state_dict(torch.load(model_path))

image = Image.open(img_path)

confidence, prediction = predict(image, model_ft, class_names)
print('The input image is: {}, confidence is: {:.4f}'.format(prediction, confidence))
plot_predict(image, confidence, prediction)
