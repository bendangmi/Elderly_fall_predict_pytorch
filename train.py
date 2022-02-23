# -*- coding: utf-8 -*-

from __future__ import print_function, division

import copy
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tensorboardX import SummaryWriter

# 数据增强与数据预处理
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'Data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'val']}

data_loaders_ft = {x: DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=0)
                   for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes  # ['fall', 'stand']

# 可自动的判断系统是否支持GPU；若支持则用GPU，否则用cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("电脑支持GPU训练，训练方式为：GPU")
else:
    print("系统已自动为你选择训练方式为：cpu")
# 添加tensorboard
writer = SummaryWriter("Data_visualization/logs")


# -----------------------------
#   train_model(训练模型)
#  参数一：Train_model，表示我们选择的深度学习模型
#  参数二：data_loaders，表示我们的数据集
#  参数三：criterion，标准化
#  参数四：optimizer，优化器
#  num_epochs：训练次数
#  is_inception=False
# -----------------------------

######################################################################
def train_model(model, data_loaders, criterion, optimizer, num_epochs=25, is_inception=False):
    print("模型训练中~~~")
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 每个epoch都有一个训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 将 Train_model 设置为训练模式
            else:
                model.eval()  # 将 Train_model 设置为评估模式

            running_loss = 0.0
            running_corrects = 0

            # 迭代数据
            for inputs1, labels in data_loaders[phase]:
                inputs1 = inputs1.to(device)
                labels = labels.to(device)

                # 零参数梯度
                optimizer.zero_grad()

                # 前向
                # 如果只在训练时则跟踪轨迹
                with torch.set_grad_enabled(phase == 'train'):
                    # 获取模型输出并计算损失
                    # 开始的特殊情况，因为在训练中它有一个辅助输出。
                    # 在训练模式下，我们通过将最终输出和辅助输出相加来计算损耗
                    # 但在测试中我们只考虑最终输出。
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs1)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs1)
                        loss = criterion(outputs, labels)

                    _, prediction = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计
                running_loss += loss.item() * inputs1.size(0)
                running_corrects += torch.sum(prediction == labels.data)

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(data_loaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            writer.add_scalar(phase, epoch_loss, epoch_acc)  # 可视化记录训练与测试日志

            # 自动选择最优模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since  # 训练一共花费的总时间
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 加载最佳权值模型
    model.load_state_dict(best_model_wts)
    return model


######################################################################
# 创建神经网络模型
model_ft = models.resnet18(pretrained=True)
num_ft = model_ft.fc.in_features
# 这里每个输出样本的大小设置为 2。
# 或者，它可以推广到 nn.Linear(num_ft, len(class_names))。
model_ft.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ft, 2),
    nn.Softmax(dim=1)
)
model_ft = model_ft.to(device)

# 损失函数
criterion_ft = nn.CrossEntropyLoss()

# 优化器
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# 每 7 个 epoch 衰减 0.1 倍的 LR
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# 开始训练
model_ft = train_model(model=model_ft, data_loaders=data_loaders_ft, criterion=criterion_ft, optimizer=optimizer_ft,
                       num_epochs=2, is_inception=False)

# 保存模型
torch.save(model_ft.state_dict(), "Train_model/weights.pth")
print("模型训练完毕，模型已保存！")
