import torch
import torchvision.transforms as transforms
from PIL import Image
from model import LeNet

# def main():
transform = transforms.Compose(                             # 数据预处理
    [transforms.Resize((32, 32)),                           # 首先需resize成跟训练集图像一样的大小
     transforms.ToTensor(),                                 #转化成tensor
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])#标准处理

# 预测
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 实例化网络，加载训练好的模型参数
net = LeNet()
net.load_state_dict(torch.load('Lenet.pth'))     #下载train训练出来的权重文件

#导入要测试的图像（自己找的，不在数据集中），放在源文件目录下
im = Image.open('1.jpg')
im = transform(im)  # [C, H, W]
im = torch.unsqueeze(im, dim=0)  # [N, C, H, W] # 对数据增加一个新维度，因为tensor的参数是[batch, channel, height, width]

with torch.no_grad():
    outputs = net(im)
    predict = torch.max(outputs, dim=1)[1].data.numpy()
print(classes[int(predict)])

# if __name__ == '__main__':
#     main()
