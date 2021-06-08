# -*- coding:utf-8 -*-
# File: LeNet-5.py
# datetime: 20221/5/16 15:00
# software: PyCharm

import torch
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader


# 定义超参数
BATCH_SIZE = 32  # 每批读取的数据大小
EPOCHS = 10  # 训练的次数
learing_rate = 0.01  # 0.01


# 加载数据集
trainSet = datasets.MNIST(root='./data', train=True, download=True,
                          transform=transforms.ToTensor())
testSet = datasets.MNIST(root='./data', train=False, download=False,
                         transform=transforms.ToTensor())


# 创建数据集的可迭代对象，即一个batch的读取数据
train_loader = torch.utils.data.DataLoader(dataset=trainSet, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=testSet, batch_size=BATCH_SIZE, shuffle=False)


# 多维向量归一化为400*1的一维向量
class Flatten(nn.Module):
    def forward(self, input):
        return input.reshape(input.size(0), -1)


# 模型定义
model = nn.Sequential(
    # c1层
    nn.Conv2d(1, 6, (5, 5), 1, padding=2),  # 上下左右都填充
    nn.ReLU(),
    # s2层
    nn.AvgPool2d((2, 2)),
    # c3层
    nn.Conv2d(6, 16, (5, 5), 1),
    nn.ReLU(),
    # s4层
    nn.AvgPool2d((2, 2)),
    # c5层，全连接层
    Flatten(),  # 多维向量归一化为400*1
    nn.Linear(5*5*16, 120),
    nn.ReLU(),
    # f6层，全连接层
    nn.Linear(120, 84),
    nn.ReLU(),
    # 输出层，全连接层
    nn.Linear(84, 10),
)

# 定义损失函数、优化器
criterion = nn.CrossEntropyLoss()

# 定义优化器
optimizer = optim.SGD(model.parameters(), lr=learing_rate)

# 模型训练
iter = 0
loss_list = []  # 要定义在循环外面，不然出错，只有一个值
accuracy_list = []
iteration_list = []
for epoch in range (EPOCHS):
    print('Finish {} Epoch'.format(epoch))
    # 遍历整个训练集
    sum_loss = 0.0
    sum_acc = 0.0
    for i, (images, labels) in enumerate(train_loader, 1):
        # 一个batch的数据转换为LSTM的输入维度
        images = Variable(images)
        labels = Variable(labels)
        # 前向传播
        outputs = model(images)
        # 计算损失
        loss = criterion(outputs, labels)
        sum_loss += loss.item()*labels.size(0)
        # 计算accuracy
        # _, predict = torch.max(outputs, 1)
        predict = torch.max(outputs.data, -1)[1]
        accuracy = (predict == labels).float().mean()
        sum_acc += accuracy.item()
        # 梯度清零
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        iter += 1
    # 输出训练集的准确率和误差
    # print('\tTrain batch:{} Loss: {} Accuracy: {}'.format(iter, sum_loss / len(trainSet), (sum_acc / len(trainSet))*100))

        # 在测试集上验证
        # 模型验证,每500次验证一次
        if iter % 500 == 0:
            model.eval()  # 模型复原
            eval_loss = 0.0
            eval_acc = 0.0
            correct = 0.0
            total = 0.0
            for images, labels in test_loader:
                with torch.no_grad():
                    images = Variable(images)
                    labels = Variable(labels)
                outputs = model(images)
                # loss
                loss = criterion(outputs, labels)
                # eval_loss += loss.item()*labels.size(0)
                # accuracy
                # _, predict = torch.max(outputs, 1)
                predict = torch.max(outputs.data, -1)[1]
                # 统计label的数量
                total += labels.size(0)  # labels.size(0)=32,一个batchSize的大小
                correct += (predict == labels).sum()
                # eval_acc += accuracy.item()
            # 保存loss,accuracy，iter
            accuracy = correct / total * 100
            loss_list.append(loss)
            accuracy_list.append(accuracy)
            iteration_list.append(iter)

            # 输出测试集的准确度和误差
            print('\tTest batch:{} Loss: {} Accuracy: {}'.format(iter, loss.item(), accuracy))
            # loss可视化
            plt.plot(iteration_list, loss_list, color='darkorange')
            plt.xlabel('Number of Iteration')
            plt.ylabel('Loss')
            plt.title('LeNet-5')
            plt.savefig('LeNet5_loss.png')
            plt.show()
            # accuracy可视化
            plt.plot(iteration_list, accuracy_list, color='r')
            plt.xlabel('Number of Iteration')
            plt.ylabel('Accuracy')
            plt.title('LeNet-5')
            plt.savefig('LeNet5_accuracy.png')
            plt.show()
            # loss小于特定值退出训练
            # if sum_loss / (len(trainSet)) <= 0.005:
            #     break
        # 保存模型
        # torch.save(model.state_dict(), './cnn.pth')
        # torch.save(model, "cnn1.pk1")
