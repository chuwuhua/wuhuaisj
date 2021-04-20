# coding  : utf-8
# @Author : wuhuaisj
# @time   : 2021/4/7 8:43
'''
重写 Dataset 方法，将数据集和位置读入为 map 式数据集
位置作为标签数据，后面如何通过标签读入？
'''
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from pyproj import Proj
from pyproj import transform as tfm
p1 = Proj(proj='merc', datum='WGS84')
p2 = Proj(proj='latlon', datum='WGS84')
class ProcessData(Dataset):
    def __init__(self,data_src,train=True,positive_size = 1,transform=None):
        if train:
            self.data_src = os.path.join(data_src, 'train')
        else:
            self.data_src = os.path.join(data_src, 'val')
        self.transform = transform
        self.positive_size = positive_size
        self.process()
    def __len__(self):
        return self.data_num


    def __getitem__(self, item):
        if self.transform:
            anchor_image = Image.open(os.path.join(self.data_src,self.triplets_sample[item][0]))
            anchor_image = self.transform(anchor_image)

            negative_image = Image.open(os.path.join(self.data_src,self.triplets_sample[item][2]))
            negative_image = self.transform(negative_image)

            positive_images=[]
            for p in self.triplets_sample[item][1]:
                positive_image = Image.open(os.path.join(self.data_src,p))
                positive_image = self.transform(positive_image)
                positive_images.append(positive_image)
            return anchor_image,positive_images,negative_image
        else:
            anchor_image = Image.open(os.path.join(self.data_src,self.triplets_sample[item][0]))
    
            negative_image = Image.open(os.path.join(self.data_src,self.triplets_sample[item][2]))
            
            positive_images=[]
            for p in self.triplets_sample[item][1]:
                positive_image = Image.open(os.path.join(self.data_src,p))
                positive_images.append(positive_image)
            return anchor_image,positive_images,negative_image
    # 构建锚点、正样本、负样本的数组
    # 图片名称包含经纬度，可快速查找得到结果
    def process(self):
        locations = []
        locations_wgs84 = []
        for image in os.listdir(self.data_src):
            [lat, lng] = (image[0:-4]).split('_')
            x, y = tfm(p2, p1, float(lng), float(lat))
            locations.append([x,y])
            locations_wgs84.append([lat,lng])
        # 计算距离矩阵
        self.data_num = len(locations_wgs84)
        dist = self.complete_distance_no_loops(locations)

        self.triplets_sample = []
        for anchor in range(self.data_num):
            positive_sample = dist[anchor].argsort()[1:self.positive_size+1]
            negative_sample = np.random.choice(dist[anchor].argsort()[self.data_num//2:(self.data_num*3)//4])
            
            positive_container = []
            for i in positive_sample:
                positive_container.append(
                    '_'.join(locations_wgs84[i]) + '.jpg'
                )
            
            self.triplets_sample.append([
                '_'.join(locations_wgs84[anchor])+'.jpg',
                positive_container,
                '_'.join(locations_wgs84[negative_sample]) + '.jpg'
            ])
    # 利用numpy快速计算距离矩阵
    def complete_distance_no_loops(self,X):
        X = np.asarray(X)
        num = X.shape[0]
        dists = np.zeros((num,num))
        dists = np.sqrt(
            -2 * np.dot(X, X.T) + np.sum(np.square(X), axis=1) + np.transpose([np.sum(np.square(X), axis=1)]))
        return dists


if __name__ == '__main__':
    # 测试代码，正确输出即可
    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "原始数据", "imgForTorch")
    train_dataset = ProcessData(image_path, train=True, transform=data_transform['train'])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    train_bar = tqdm(train_loader)
    for step, image_sample in enumerate(train_bar):
        anchor_sample,positive_sample,negative_sample = image_sample
        print(anchor_sample)
        break