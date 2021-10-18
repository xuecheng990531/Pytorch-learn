import torch
import torchvision
from PIL import Image

# 当使用GPU训练后，本机不支持cuda加速需要在加载模型的后面加入map_location选项
model=torch.load('/Users/lixuecheng/Desktop/Pytorch learning copy/保存的模型/vgg16_method1',map_location=torch.device('cpu'))

image_path='/Users/lixuecheng/Desktop/Pytorch learning copy/test_images/R.jpeg'

img=Image.open(image_path)

img=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

print(type(img))

output=model(img)

print(output.argmax(1))

