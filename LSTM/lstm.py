# -*- coding:utf-8 -*-
# File: lstm.py
# datetime: 2021/5/15 13:00
# software: PyCharm

import torch
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

"""
    LSTM训练过程：
    1.加载数据集
    2.切分数据，使其可迭代，每次迭代一个batch
    3.创建模型类
    4.初始化模型参数
    5.初始化损失类
    6.训练模型
"""

# 加载数据集
trainSet = datasets.MNIST(root='./data', train=True, download=True,
                          transform=transforms.ToTensor())
testSet = datasets.MNIST(root='./data', train=False, download=False,
                         transform=transforms.ToTensor())
# 2.2 定义超参数
BATCH_SIZE = 32  # 每批读取的数据大小
EPOCHS = 10

# 2.3 创建数据集的可迭代对象，即一个batch的读取数据
train_loader = torch.utils.data.DataLoader(dataset=trainSet, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=testSet, batch_size=BATCH_SIZE, shuffle=False)

# 2.4 查看一批batch的数据
images, labels = next(iter(test_loader))


# 显示一批数据
def imshow(inp, title=None):  # 不显示标题
    inp = inp.numpy().transpose((1, 2, 0))  # 数据变换为numpy格式,并调整顺利
    mean = np.array([0.485, 0.456, 0.406])  # 采用官网数据
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean  # 数据恢复
    inp = np.clip(inp, 0, 1)  # 像素值进行压缩
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


# 网格显示
out = torchvision.utils.make_grid(images)
imshow(out)


# 3 定义模型类
class LstmModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LstmModel, self).__init__()  # 初始化父类
        self.hidden_dim = hidden_dim  # 进行赋值
        self.layer_dim = layer_dim
        # 创建LSTM模型
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)  # batch_first=True把batchSize调到最前面
        # 全连接层，线性
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        #  初始隐藏层的状态设置为0
        #  (layer_dim, batch_size, hidden_dim)
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)  # 返回一个由标量值0填充的张量
        #  初始化cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        #  分离隐藏状态，避免梯度爆炸
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        #  只需要最后一层隐藏层的状态,因此用-1
        out = self.fc(out[:, -1, :])
        return out


# 4. 初始化模型
inputs_dim = 28  # 输入维度，图片是28*28
hiddens_dim = 100  # 隐藏层或神经元维度
layers_dim = 1  # 1个隐藏层
outputs_dim = 10  # 输出维度10:0-9
# 根据设备是否支持GPU来选择硬件
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = LstmModel(inputs_dim, hiddens_dim, layers_dim, outputs_dim)

# 查看网络参数
for i in range(len(list(model.parameters()))):
    print("参数: %d" %(i+1))
    print(list(model.parameters())[i].size())

# 5. 定义损失函数,交叉熵
criterion = nn.CrossEntropyLoss()

# 6. 初始化优化器
learning_rate = 0.1  # 太大会出现梯度爆炸
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 7. 模型训练
sequence_dim = 28  # 序列长度
loss_list = []  # 保存loss
accuracy_list = []
iteration_list = []
iter = 0

for epoch in range(EPOCHS):
    print(' Finish {} Epoch,'.format(epoch))
    for i, (images, labels) in enumerate(train_loader):  # enumerate函数可以返回枚举对象及对应序号
        model.train()  # 声明模型训练
        # 一个batch的数据转换为LSTM的输入维度
        images = images.view(-1, sequence_dim, inputs_dim).requires_grad_().to(device)
        labels = labels.to(device)
        # 梯度清0,否则会不断积累
        optimizer.zero_grad()  # optimizer数量可以更改
        # 前向传播
        outputs = model(images)
        # 计算损失
        loss = criterion(outputs, labels)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 计算器自增
        iter += 1
        # 模型验证,每500次验证一次
        if iter % 500 == 0:
            model.eval()  # 申明
            correct = 0.0
            total = 0.0
            # 迭代测试集
            for images, labels in test_loader:
                # 一个batch的数据转换为LSTM的输入维度
                images = images.view(-1, sequence_dim, inputs_dim).to(device)
                # 模型预测
                outputs = model(images)
                # 获得预测概率最大值的下标
                predict = torch.max(outputs.data, -1)[1]
                # 统计label的数量
                total += labels.size(0)  # labels.size(0)=32,一个batchSize的大小
                # 统计预测正确的数量
                if torch.cuda.is_available():
                    correct += (predict.gpu() == labels.gpu()).sum()
                else:
                    correct += (predict == labels).sum()
            # 计算accuracy
            accuracy = correct / total * 100
            # 保存loss,accuracy，iter
            loss_list.append(loss.data)
            accuracy_list.append(accuracy)
            iteration_list.append(iter)
            print('\tbatch: {} Loss: {} Accuracy: {}'.format(iter, loss.item(), accuracy))
            # loss可视化
            plt.plot(iteration_list, loss_list, color='darkorange')
            plt.xlabel('Number of Iteration')
            plt.ylabel('Loss')
            plt.title('LSTM')
            plt.savefig('LSTM_loss.png')
            plt.show()
            # accuracy可视化
            plt.plot(iteration_list, accuracy_list, color='r')
            plt.xlabel('Number of Iteration')
            plt.ylabel('Accuracy')
            plt.title('LSTM')
            plt.savefig('LSTM_accuracy.png')
            plt.show()
