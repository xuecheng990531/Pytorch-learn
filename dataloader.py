from os import write
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


test_data=torchvision.datasets.CIFAR10(
    root='/Users/lixuecheng/Desktop/Pytorch learning copy/Dataset',
    train=False,download=True,
    transform=torchvision.transforms.ToTensor()
)

test_dataloader=DataLoader(
    dataset=test_data,batch_size=64,
    shuffle=True,num_workers=0,
    drop_last=False
)


write=SummaryWriter('dataloader')

img,target=test_data[0]
print(img.shape)
print(type(img))
print(target)



step=1
for i in test_dataloader:
    imgs,targets=i
    # 一定要注意，之前用的是add_image,这里是images！！！！
    write.add_images('test_loader',imgs,step)
    step=step+1
write.close()