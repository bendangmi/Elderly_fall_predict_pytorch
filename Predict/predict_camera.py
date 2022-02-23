# -*- coding: utf-8 -*-

import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

class_names = os.listdir('../Data/train')  # 数据集路径
camera_path = '../Text_data/camera/1.mp4'  # 测试视频路径
model_path = '../Train_model/weights.pth'  # 训练好的模型路径

# 数据增强处理
t = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# 建立模型
model_ft = models.resnet18(pretrained=False)
num_ft = model_ft.fc.in_features
model_ft.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ft, 2),
    nn.Softmax(dim=1)
)
model_ft.load_state_dict(torch.load(model_path))

# 调用摄像头
# capture = cv2.VideoCapture(0)
# 导入本地视频
capture = cv2.VideoCapture(camera_path)
fps = 0.0

while True:
    t1 = time.time()
    ref, frame = capture.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(np.uint8(frame))  # 将array格式转化为Image格式

    plt.imshow(frame)
    frame_t = t(frame)
    frame_t = torch.unsqueeze(frame_t, 0)
    output = model_ft(frame_t)
    confidence, prediction1 = torch.max(output, 1)
    prediction_name = class_names[int(prediction1)]

    fps = (fps + (1. / (time.time() - t1))) / 2
    print(prediction_name, confidence.item())
    print('The prediction is :{}, confidence is {:.4f}'.format(prediction_name, confidence.item()))
    print("fps= %.2f" % fps)

    frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

    frame = cv2.putText(frame, 'The prediction is :{}, confidence is {:.4f}'.format(prediction_name, confidence.item()),
                        (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("video", frame)
    c = cv2.waitKey(30) & 0xff
    if c == 27:
        capture.release()
        break
