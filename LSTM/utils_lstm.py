"""
    functions for lstm
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def imshow(inp, title=None):
    inp = inp.numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])  # pytorchvision.transforms进行数据处理求的均值
    std = np.array([0.229, 0.224, 0.225])  # 标准差
    inp = std * inp + mean  # 数据恢复
    inp = np.clip(inp, 0, 1)  # 把像素值压缩到[0，1]
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # 图片加载


