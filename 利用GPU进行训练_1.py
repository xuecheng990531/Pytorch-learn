from numpy.lib.type_check import imag
from torch.nn.modules import module
import torchvision
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Conv2d,MaxPool2d,Flatten,Linear
from torch.utils.tensorboard import SummaryWriter



# 准备数据集
train_data=torchvision.datasets.CIFAR10(
    root='/Users/lixuecheng/Desktop/Pytorch learning copy/Dataset',train=True,
    transform=torchvision.transforms.ToTensor(),download=True
)
test_data=torchvision.datasets.CIFAR10(
    root='/Users/lixuecheng/Desktop/Pytorch learning copy/Dataset',train=False,
    transform=torchvision.transforms.ToTensor(),download=True
)


# 获取训练集的长度
train_data_size=len(train_data)
# 获取测试集的长度
test_data_size=len(test_data)

# Dataloader
train_dataloader=DataLoader(train_data,batch_size=64)
test_dataloader=DataLoader(test_data,batch_size=64)


# 我们首先定义好我们的网络，然后重开一个python，把网络直接复制过去，到时候这里直接引入import 就行
class Mymodel(nn.Module):
    def __init__(self):
        super(Mymodel,self).__init__()
        self.model1=nn.Sequential(
            Conv2d(3,32,5,stride=1,padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(32,32,5,stride=1,padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(32,64,5,stride=1,padding=2),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(1024,64),
            Linear(64,10)
        )
    def forward(self,x):
        x=self.model1(x)

        return x

# 实例化网络模型        
tudou=Mymodel()

# 判断是否支持cuda加速
if torch.cuda.is_available():
    tudou=tudou.cuda()

# 损失函数
loss_fn=nn.CrossEntropyLoss()

if torch.cuda.is_available():
    loss_fn=loss_fn.cuda()

# 优化器
optim=torch.optim.SGD(tudou.parameters(),lr=0.01)
# 网络训练
writer=SummaryWriter('cifar_model')

# 定义训练的步数
total_train_step=0

# 记录测试的次数
total_test_step=0

# 训练轮数
epoch=20


for i in range(epoch):
    print('---------------第 {} 轮训练开始---------------'.format(i+1))

    # 训练步骤开始
    for data in train_dataloader:
        images,targets=data

        if torch.cuda.is_available():
                images=images.cuda()
                targets=targets.cuda()

        output=tudou(images)

        loss=loss_fn(output,targets)

        # 优化器优化模型
        optim.zero_grad()
        loss.backward()
        optim.step()

        total_train_step=total_train_step+1
        accuracy=(output.argmax(1)==targets).sum()
        total_accuracy=0
        total_accuracy=int(total_accuracy+accuracy)

        if total_train_step%100==0:
            print('训练次数: {} ,Loss: {}'.format(total_train_step,loss))

    torch.save('Cifar10_{}_gpu.pth'.format(i))

    # 训练集跑完一遍，我们去测试集上看看效果
    total_loss=0
    # 表明没有了梯度
    with torch.no_grad():
        for data in test_dataloader:
            images,targets=data

            if torch.cuda.is_available():
                images=images.cuda()
                targets=targets.cuda()

        output=tudou(images)

        loss=loss_fn(output,targets)

        total_loss=total_loss+loss

        accuracy=((output.argmax(1)==targets).sum())

        total_accuracy=total_accuracy+accuracy

        print('准确率是: {} , Loss: {}'.format((total_accuracy/test_data_size),total_loss))



