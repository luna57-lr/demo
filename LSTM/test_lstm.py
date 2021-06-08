import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torchvision import datasets, transforms    # datasets包含常用的数据集，transform 对图像进行预处理
import cv2


# training settings
batch_size = 64
train_dataset = datasets.MNIST(root='./data',train=True,transform=transforms.ToTensor(),download=True)
test_dataset = datasets.MNIST(root='./data',train=False,transform=transforms.ToTensor())
# Data Loader（Input Pipeline)是一个迭代器，torch.utils.data.DataLoader作用就是随机的在样本中选取数据组成一个小的batch。shuffle决定数据是否打乱
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

# 可视化数据图像
for i in range(1):
    plt.figure()
    print(train_loader.dataset.train_data[i].shape)
    plt.imshow(train_loader.dataset.train_data[i].numpy())


x = torch.randn(2, 2, 2)
# firstly change the data into diresed dimension, then reshape the tensor according to what I want
x.view(-1, 1, 4)

# 理解迭代器的深层含义，torch.utils.data.DataLoader的作用理解
for (data, target) in train_loader:
    for i in range(1):
        plt.figure()
        print("target:",target[i])
        print("data:",data.shape)
        plt.imshow(data[i].numpy()[0])
    break

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, 1, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = LeNet5()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
loss_func = torch.nn.CrossEntropyLoss()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_func(output, target)  # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) \n".format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)
    ))

for epoch in range(1, 2):
    train(model,  DEVICE, train_loader, optimizer, epoch)
    test(model, DEVICE, test_loader)
torch.save(model, 'LeNet5model.pth')





