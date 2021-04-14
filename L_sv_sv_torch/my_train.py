# coding  : utf-8
# @Author : wuhuaisj
# @time   : 2021/4/7 9:01

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
# from L_sv_sv_torch.model import resnet34 # 使用torchvisioin自有模块，不使用此定义
from torchvision.models.resnet import \
    resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d
from L_sv_sv_torch.process_data import ProcessData
import matplotlib.pyplot as plt

# 使用cpu或gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

# data_root = "数据融合实验"，路径比较复杂
# store_root = '数据融合实验/wuhuaisj/L_sv_sv_torch
data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
store_root = os.path.join(data_root, 'wuhuaisj', 'L_sv_sv_torch')

# batch_size
batch_size = 16
print('Using {} batch for training'.format(batch_size))

# 使用多少线程读入图片，为避免报错，且数据量适中，使用单个线程
# nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 4])
nw = 0
print('Using {} dataloader workers every process'.format(nw))

'''
transformer 调整此处:
原始图片 height,width：256x512

还可增加 resize

归一化方法：
transformer.ToTensor() 将PILImage或numpy数组转为 tensor
transformer.Normalize([means],[std])
[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]这一组平均值是从imagenet训练集中抽样算出来的
可计算自己数据集的均值和方差：
    https://blog.csdn.net/weixin_38533896/article/details/85951903
'''
data_transform = {
    "train": transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val": transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

# 加载自己的数据集，已经重写DataLoader函数，将数据集构造成 [anchor,positive,negative ]的样本
# anchor:锚点，positive：最近的5个图片中随机选取一个，negative:1/2~3/4 距离样本中随机选取一个
#   positive的选择是最近的5个，然后再计算多个的tripletloss并相加
#   并没有搞清楚，所以设定positive_size,便于修改，default:1
# train_dataset：训练集 validate_dataset: 测试集

image_path = os.path.join(data_root, "原始数据", "imgForTorch")
assert os.path.exists(image_path), "{} path does not exist.".format(image_path)  # assert 检查条件，符合则继续运行，否则停止

positive_size = 5
print("using {} positive sample for training".format(positive_size))
train_dataset = ProcessData(image_path, train=True,positive_size=5, transform=data_transform['train'])
train_num = len(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)

validate_dataset = ProcessData(image_path, train=False, transform=data_transform['val'])
val_num = len(validate_dataset)
validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, num_workers=nw)

print("using {} images for training, {} images for validation.".format(train_num, val_num))

# 加载已经别人已经构建好的模型 resnet 和训练好的权重
# 使用较大的网络效果较好
net_use = 'resnet101'
model_urls = {
    'resnet18': 'resnet18-f37072fd.pth',
    'resnet34': 'resnet34-b627a593.pth',
    'resnet50': 'resnet50-0676ba61.pth',
    'resnet101': 'resnet101-63fe2227.pth',
    'resnet152': 'resnet152-394f9c45.pth',
    'resnext50_32x4d': 'resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'wide_resnet101_2-32ee1156.pth',
}
net = None
if net_use == 'resnet18':
    net = resnet18()
elif net_use == 'resnet34':
    net = resnet34()
elif net_use == 'resnet50':
    net = resnet50()
elif net_use == 'resnet101':
    net = resnet101()
elif net_use == 'resnet152':
    net = resnet152()
elif net_use == 'resnext50_32x4d':
    net = resnext50_32x4d()
elif net_use == 'resnext101_32x8d':
    net = resnext101_32x8d()
else:
    raise Exception('please use correct resnt name')
model_weight_path = os.path.join(store_root, 'resnet', model_urls[net_use])
assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
net.load_state_dict(torch.load(model_weight_path, map_location=device))
net.to(device)

'''
定义loss函数 tripletmarginloss 
https://pytorch.org/docs/1.2.0/nn.html#tripletmarginloss
pytorch提供的loss函数，目前似乎不支持多个正样本
margin ？| a+p-n+margin | 设置多少我不知道，怎么调整我不知道 default:1
    如果 loss 稳定在 margin 左右，可能是网络陷入了局部极值，导致对所有样本的预测都是相同的
p:成对的距离标准 default 2
ps:后面的都可以忽略
swap: 论文中描述的距离交换
size_average: True:训练批次中损失的平均数 False:批量的损失将被忽略
reduce：不推荐使用
reduction:不推荐使用
'''
triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

'''
net.parameters() 获取网络参数
requires_grad=False 固定参数  =True 非固定参数

optimizer 优化器，用来保存当前的状态，并能够根据计算得到的梯度来更新参数
    给它一个可进行迭代优化的包含了所有参数列表params。 
    然后，您可以指定程序优化特定的选项，例如学习速率，权重衰减等
Adam算法：(Adaptive Moment Estimation)本质上是带有动量项的RMSprop，
    它利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率。
    它的优点主要在于经过偏置校正后，每一次迭代学习率都有个确定范围，使得参数比较平稳。
    param 参数列表
    lr 学习率，学习率可以动态调整，调整方法暂时未知

具体优化器的学习：见optimizer.txt
'''
params = [p for p in net.parameters() if p.requires_grad]
optimizer = optim.Adam(params, lr=0.0001)

# 对训练集的训练次数
epochs = 3
train_steps = len(train_loader)

# 存储和绘图相关
losses = []
uuid_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())


for epoch in range(epochs):
    # 根据程序开始时间和epochs构造存储路径
    # loss 存储路径 txt
    tmp_loss_name = '%s_epoch%d.txt' % (uuid_str, epoch)
    loss_path = os.path.join(store_root, 'loss_store', 'loss_txt') + "/" + tmp_loss_name
    # 训练的结果存储路径
    tmp_file_name = '%s_epoch%d.pth' % (uuid_str, epoch)
    save_path = os.path.join(store_root, 'result_store') + "/" + tmp_file_name
    # loss图片png存储路径，loss图片有三张，loss是合并在一起的，最后一张图片包含了epochs次训练的所有值
    tmp_pic_name = '%s_epoch%d.png' % (uuid_str, epoch)
    pic_path = os.path.join(store_root, 'loss_store', 'loss_png') + "/" + tmp_pic_name

    # 训练集
    net.train()
    running_loss = 0.0
    train_bar = tqdm(train_loader)  #进度条美化
    for step, image_sample in enumerate(train_bar):
        anchor_sample, positive_samples, negative_sample = image_sample
        optimizer.zero_grad() # 每个batch梯度置零
        anchor_logits = net(anchor_sample.to(device))
        negative_logits = net(negative_sample.to(device))
        loss = 0
        for positive_sample in positive_samples:
            positive_logits = net(positive_sample.to(device))
            loss = triplet_loss(anchor_logits, positive_logits, negative_logits)

        loss.backward() # 反向传播，计算当前梯度
        optimizer.step() # 根据梯度更新网络参数


        running_loss += loss.item()
        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)
        losses.append(loss.item())
        with open(loss_path, 'a') as f:
            f.write("train epoch[{}/{}] step {} loss:{:.3f}\n".format(epoch + 1, epochs, step + 1, loss))
            f.close()

    # 测试集
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        val_bar = tqdm(validate_loader)
        for val_data in val_bar:
            anchor_val, positive_val, negative_val = val_data
            anchor_outputs = net(anchor_val.to(device))
            positive_outputs = net(positive_val.to(device))
            negative_outputs = net(negative_val.to(device))
            loss = triplet_loss(anchor_outputs, positive_outputs, negative_outputs)
            # predict_y = torch.max(outputs, dim=1)[1]
            # acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

            val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)

    # val_accurate = acc / val_num
    # print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
    #       (epoch + 1, running_loss / train_steps, val_accurate))
    print('[epoch %d] train_loss: %.3f' % (epoch + 1, running_loss / train_steps))

    # if val_accurate > best_acc:
    #     best_acc = val_accurate
    torch.save(net.state_dict(), save_path)

    plt.plot(losses)
    plt.savefig(pic_path)

print('Finished Training')
