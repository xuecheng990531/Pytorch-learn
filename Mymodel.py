# 引入必要的包
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import MaxPool2d,Conv2d,Linear,Flatten




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