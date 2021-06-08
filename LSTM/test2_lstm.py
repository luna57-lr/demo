import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import numpy as np
#from logger import Logger

# 定义超参数
batch_size = 128        # 批的大小
learning_rate = 0.01     # 学习率
num_epoches = 10        # 遍历训练集的次数


# 下载训练集 MNIST 手写数字训练集
train_dataset = datasets.MNIST(
    root='./data', train=True, transform=transforms.ToTensor(), download=True)

test_dataset = datasets.MNIST(
    root='./data', train=False, transform=transforms.ToTensor())

# 数据集分批
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 多维向量归一化400*1一维向量
class Flatten(nn.Module):
    def forward(self, input):
        return input.reshape(input.size(0), -1)


# 模型定义
model = nn.Sequential(
    # C1
    nn.Conv2d(1, 6, (5, 5), 1, 2),
    nn.ReLU(),
    # S2
    nn.AvgPool2d((2, 2)),
    # C3
    nn.Conv2d(6, 16, (5, 5), 1),
    nn.ReLU(),
    # S4
    nn.AvgPool2d((2, 2)),
    # C5 # 多维向量归一化400*1一维向量
    Flatten(),
    nn.Linear(5*5*16, 120),
    # nn.Conv2d(16, 120, (5, 5), 1),
    nn.ReLU(),
    # nn.BatchNorm1d(120),# 批量准化
    # nn.Dropout(),
    # F6

    nn.Linear(120, 84),
    nn.ReLU(),
    # nn.BatchNorm1d(84),
    # nn.ReLU(),
    # OUTPUT
    # nn.Dropout(),
    nn.Linear(84, 10),
    # nn.Softmax(dim=1),

)


# 定义loss和optimizer
# 交叉熵损失函数
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # 优化器代替手动优化使用Adam 优化方法

# 开始训练
for epoch in range(num_epoches):
    print('epoch {}'.format(epoch + 1))      # .format为输出格式，formet括号里的即为左边花括号的输出
    print('*' * 10)
    running_loss = 0.0
    running_acc = 0.0

    # 遍历整个数据集
    for i, data in enumerate(train_loader, 1):
        img, label = data
        img = Variable(img)
        label = Variable(label)
        # print(img.data)
        # 向前传播
        out = model(img)
        # print(out.data.shape, img.shape, label.shape)
        # 计算loss
        loss = criterion(out, label)
        # 计算总的loss
        running_loss += loss.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        # 准确度
        accuracy = (pred == label).float().mean()
        running_acc += num_correct.item()
        # 向后传播
        optimizer.zero_grad()
        loss.backward()
        # 优化器执行
        optimizer.step()
    # 输出训练接的准确率和误差
    print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
        epoch + 1, running_loss / (len(train_dataset)), running_acc / (len(train_dataset))))
    # 以下未测试集
    # 模型复原
    model.eval()
    eval_loss = 0
    eval_acc = 0
    for data in test_loader:
        img, label = data
        # 测试集计算不再计算梯度
        with torch.no_grad():
            img = Variable(img)
            label = Variable(label)
        out = model(img)
        # 计算loss
        loss = criterion(out, label)
        eval_loss += loss.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.item()
        # 输出测试集准确度和误差
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        test_dataset)), eval_acc / (len(test_dataset))))
    print()
    # loss小于特定值则退出训练
    if running_loss / (len(train_dataset)) <= 0.005:
        break
# 保存模型
# torch.save(model.state_dict(), './cnn.pth')
# torch.save(model, "cnn1.pkl")
# model = torch.load("cnn1.pkl")

# 测试保存的模型
criterion = nn.CrossEntropyLoss()
eval_loss = 0
eval_acc = 0
count = 0
for data in test_loader:
    img, label = data
    with torch.no_grad():
        img = Variable(img)
        label = Variable(label)
    out = model(img)

    count += 1
    # print(out.shape, count)
    loss = criterion(out, label)
    # print(loss)
    #print(label.size(0))
    eval_loss += loss.item() * label.size(0)
    _, pred = torch.max(out, 1)
    num_correct = (pred == label).sum()
    eval_acc += num_correct.item()
print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
    test_dataset)), eval_acc / (len(test_dataset))))
print()