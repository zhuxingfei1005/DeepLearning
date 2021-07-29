import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
import time

# def main():
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])#对输入的图像数据做预处理

# 50000张训练图片
train_set = torchvision.datasets.CIFAR10(root='./data',      # 数据集存放目录
                                         train=True,         # 表示是数据集中的训练集
                                         download=False,     # 第一次运行时为True，下载数据集，下载完成后改为False
                                         transform=transform)# 预处理过程
#导入数据集：随机拿出batch_size=36，36张照片实验
train_loader = torch.utils.data.DataLoader(train_set,        # 导入的训练集
                                           batch_size=36,    # 每批训练的样本数
                                           shuffle=True,     # 是否打乱训练集
                                           num_workers=0)    # 使用线程数，在windows下设置为0

# 10000张验证图片  测试集
test_set = torchvision.datasets.CIFAR10(root='./data',
                                        train=False,          # 表示是数据集中的测试集
                                        download=False,
                                        transform=transform)
# 加载测试集
test_loader = torch.utils.data.DataLoader(test_set,
                                          batch_size=5000,    # 每批用于验证的样本数
                                          shuffle=False,
                                          num_workers=0)
test_data_iter = iter(test_loader)#通过迭代获取图像及对应的标签     # 获取测试集中的图像和标签，用于accuracy计算
test_image, test_label = test_data_iter.next()
#元组类型
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = LeNet()                                      #导入模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('GPU IS ',torch.cuda.is_available())
net.to(device)# 将网络分配到指定的device中
loss_function = nn.CrossEntropyLoss()              #定义损失函数（此函数包含了softmax函数，所以网络中没有使用softmax）
optimizer = optim.Adam(net.parameters(), lr=0.001) # 定义优化器（训练参数，学习率）
#训练开始
for epoch in range(5):          # loop over the dataset multiple times，迭代次数 # 一个epoch即对整个训练集进行一次训练
    running_loss = 0.0          #累加损失
    time_start = time.perf_counter()#开始计时
    for step, data in enumerate(train_loader, start=0):# 遍历训练集，step从0开始计算遍历训练集样本
        inputs, labels = data   # get the inputs; data is a list of [inputs, labels]
        optimizer.zero_grad()   #历史损失梯度清零

        # forward + backward + optimize
        outputs = net(inputs.to(device))                #将inputs分配到指定的device中，正向传播 得到的输入图片输入网络，得到网络输出
        loss = loss_function(outputs, labels.to(device))#将inputs分配到指定的device中，计算损失output预测值，labels真实值
        loss.backward()                      #反向传播
        optimizer.step()                     #优化，参数更新

        # print statistics# 打印耗时、损失、准确率等数据
        running_loss += loss.item()
        if step % 500 == 499:               # print every 500 mini-batches每隔500步打印一次信息
            with torch.no_grad():           #with上下文管理器,在以下步骤中（验证过程中）不用计算每个节点的损失梯度，防止内存占用
                outputs = net(test_image.to(device))   # [batch, 10]# 测试集传入网络（test_batch_size=5000），output维度为[5000,10]
                predict_y = torch.max(outputs, dim=1)[1]#寻找最大索引即最大概率分类1#以output中值最大位置对应的索引（标签）作为预测输出
                accuracy = torch.eq(predict_y, test_label.to(device)).sum().item() / test_label.size(0)# 打印epoch，step，loss，accuracy

                print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                      (epoch + 1, step + 1, running_loss / 500, accuracy))
                print('%f s' % (time.perf_counter() - time_start))
                running_loss = 0.0
print('Finished Training')

# 保存训练得到的参数
save_path = './Lenet.pth'
torch.save(net.state_dict(), save_path)


# if __name__ == '__main__':
#     main()
