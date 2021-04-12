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
from L_sv_sv_torch.model import resnet34
from L_sv_sv_torch.process_data import ProcessData


# 使用cpu或gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

# data_root = "数据融合实验"
data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
image_path = os.path.join(data_root, "原始数据", "imgForTorch")
assert os.path.exists(image_path), "{} path does not exist.".format(image_path)  # assert 检查条件，符合则继续运行，否则停止


batch_size = 16
# nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 4])
nw =0
print('Using {} dataloader workers every process'.format(nw))

data_transform = {
    "train": transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val": transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
train_dataset = ProcessData(image_path, train=True, transform=data_transform['train'])
train_num = len(train_dataset)

# train_dataset 已经重写为符合三重损失的形式
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)

validate_dataset = ProcessData(image_path, train=False, transform=data_transform['val'])
val_num = len(validate_dataset)
validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, num_workers=nw)

print("using {} images for training, {} images for validation.".format(train_num, val_num))


net = resnet34()
model_weight_path = data_root +"\my_code\L_sv_sv_torch\\resnet34-333f7ec4.pth"
assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
net.load_state_dict(torch.load(model_weight_path, map_location=device))

net.to(device)

#define loss function
'''
DataLoader还需要重新写，写成anchor,positive,negative三个张量数组
[ [anchor,positive,negative],...,[anchor,positive,negative]]的tensor
'''
triplet_loss = nn.TripletMarginLoss(margin=1.0,p=2)

# 给每层网络设置学习率
params = [p for p in net.parameters() if p.requires_grad]
optimizer = optim.Adam(params, lr=0.0001)

epochs = 1
best_acc = 0.0

# 构造存储路径
uuid_str = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
tmp_loss_name = '%s.txt' % uuid_str
loss_path = os.path.join(data_root,'my_code','L_sv_sv_torch','loss_store') +"\\"+tmp_loss_name

tmp_file_name = '%s.pth' % uuid_str
save_path = os.path.join(data_root,'my_code','L_sv_sv_torch','result_store') +"\\"+tmp_file_name

losses = []
epochs = 1
best_acc = 0.0
train_steps = len(train_loader)

for epoch in range(epochs):
    net.train()
    running_loss = 0.0
    train_bar = tqdm(train_loader)
    for step,image_sample in enumerate(train_bar):
        anchor_sample,positive_sample,negative_sample = image_sample
        optimizer.zero_grad()
        anchor_logits = net(anchor_sample.to(device))
        positive_logits = net(positive_sample.to(device))
        negative_logits = net(negative_sample.to(device))
        loss = triplet_loss(anchor_logits,positive_logits,negative_logits)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,epochs,loss)

        # print(loss)

        losses.append(loss)
        with open(loss_path, 'a') as f:
            f.write("train epoch[{}/{}] step {} loss:{:.3f}\n".format(epoch + 1,epochs,step+1,loss))
            f.close()

    # 测试数据
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        val_bar = tqdm(validate_loader)
        for val_data in val_bar:
            anchor_val,positive_val,negative_val = val_data
            anchor_outputs = net(anchor_val.to(device))
            positive_outputs = net(positive_val.to(device))
            negative_outputs = net(negative_val.to(device))
            loss = triplet_loss(anchor_outputs,positive_outputs,negative_outputs)
            # predict_y = torch.max(outputs, dim=1)[1]
            # acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

            val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,epochs)

    # val_accurate = acc / val_num
    # print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
    #       (epoch + 1, running_loss / train_steps, val_accurate))
    print('[epoch %d] train_loss: %.3f' %(epoch + 1, running_loss / train_steps))

    # if val_accurate > best_acc:
    #     best_acc = val_accurate
    torch.save(net.state_dict(), save_path)

print('Finished Training')