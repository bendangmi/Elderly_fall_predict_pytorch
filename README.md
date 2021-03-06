# Elderly_fall_recognition_pytorch

#### 介绍
通过pytorch预训练模型训练自己的分类网络，并分别用本地图片和摄像头预测老人摔倒

#### 环境
python 3.8
pytorch 1.10.2


#### 使用说明

1. Data文件夹存放训练数据集
2. Data_visualization文件夹存放数据可视化文件
3. Model文件夹放神经网络模型
4. Predict文件夹存放两种预测方式： </br>
   第一种：predict_img.py文件可以用来预测本地图片类别</br>
   第二种：predict_camera.py可以调用计算机摄像头预测摄像头拍摄到的物体类别
5. Train_model文件夹存放自己训练好的模型
6. Text_data文件夹存放测试数据，可提供视频and图片进行测试
7. train.py文件用来训练自己的模型

#### 预测结果
训练数据集共两种类别：人体的站立和摔倒，分别用本地图片和摄像头来验证模型训练效果
![本地图片预测结果](这里填预测图片直链)
![摄像头预测结果](这里填视频预测图片直链)</br>
*注：由于训练集较少，神经网络与各种参数并未选好，所以分类的准确率不是很高

#### 项目开源地址
https://github.com/bendangmi/Elderly_fall_predict_pytorch

