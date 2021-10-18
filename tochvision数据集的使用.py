import torchvision

# 下载cifar10的训练集和测试集
train_set = torchvision.datasets.CIFAR10(
    root='/Users/lixuecheng/Desktop/Python/Pytorch learning/Dataset',
    train=True, download=True)

test_set = torchvision.datasets.CIFAR10(
    root='/Users/lixuecheng/Desktop/Python/Pytorch learning/Dataset',
    train=True, download=True
)

# 查看数据集
print(train_set[0])
