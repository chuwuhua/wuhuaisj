# Author:wuhuaisj
# Date:2021/4/14 10:54
# Github:https://github.com/chuwuhua


'''
长沙市人流量数据存储在  原始数据/人流量/data_cs.h5  中
采用 h5py 读取和使用 data.data_cs.value
shape:1440,80,232
1440:时间 24*60
80：由北至南
232：由西至东
每分钟一组数据，每组数据都是由左上到右下的矩阵形状
给每组数据增加位置信息：
          minx       miny        maxx       maxy
185  111.890866  27.851024  114.256514  28.664368
x划分为232份
y划分为80份


'''
import os
import h5py
import matplotlib.pyplot as plt
import seaborn as sns


class PeopleCS():
    def __init__(self, data_path):
        self.data_path = data_path
        self.load()
        self.cut_ = False
        self.set_ = False

    # load data
    def load(self):
        file = h5py.File(self.data_path, 'r')
        self.data_cs = file['data_cs']

    # set time
    def set(self, time='0:00'):
        [h, m] = list(map(int, time.split(':')))
        self.time = time
        self.data_use = self.data_cs[h * 60 + m]
        self.set_ = True

    # 根据日期绘制热力图
    def show(self):
        if not self.set_:
            self.set()
        if not self.cut_:
            sns.heatmap(self.data_use, cmap='gray', square=True, xticklabels=False, yticklabels=False,
                        vmin=0, vmax=2000)
            plt.show()
        else:
            sns.heatmap(self.data_use_cut, cmap='gray', square=True, xticklabels=False, yticklabels=False,
                        vmin=0, vmax=2000)
            plt.show()

    # 切割出长沙市区部分
    def cut(self, num=4):
        shi = [111.890866, 27.851024, 114.256514, 28.664368]
        if num == 4:
            print('四区：开福区，芙蓉区，天心区，雨花区')
            qu = [112.903165, 27.914597, 113.182139, 28.392893]  # 四区
        else:
            print('六区：望城区，岳麓区，开福区，芙蓉区，天心区，雨花区')
            qu = [112.606134, 27.914597, 113.182139, 28.561062]  # 六区
        #         根据四区的经纬度范围取出对应的人流量信息
        shi_x = (shi[2] - shi[0]) / 232
        shi_y = (shi[3] - shi[1]) / 80
        arr = []
        for i in range(232):
            if shi[0] + shi_x * i <= qu[0] <= shi[0] + shi_x * (i + 1):
                print('较小经度{},粒度{}'.format(shi[0]+shi_x*i,shi_x))
                arr.append(i)
            if shi[0] + shi_x * i <= qu[2] <= shi[0] + shi_x * (i + 1):
                print('较大经度{},粒度{}'.format(shi[0] + shi_x * (i+1), shi_x))
                arr.append(i + 1)
        for i in range(80):
            if shi[3] - shi_y * i >= qu[3] >= shi[3] - shi_y * (i + 1):
                print('较大纬度{},粒度{}'.format(shi[3]-shi_y*i,shi_y))
                arr.append(i)
            if shi[3] - shi_y * i >= qu[1] >= shi[3] - shi_y * (i + 1):
                print('较小纬度{},粒度{}'.format(shi[3]-shi_y*(i+1),shi_y))
                arr.append(i + 1)
        print('行列范围:行：{}-{},列:{}-{}.（含左不含右）'.format(arr[0], arr[1], arr[2], arr[3]))
        self.data_use_cut = self.data_use[arr[2]:arr[3]]
        self.data_use_cut = self.data_use_cut[:, arr[0]:arr[1]]
        self.cut_ = True


if __name__ == '__main__':
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    data_path = os.path.join(data_root, '原始数据', '人流量', 'data_cs.h5')
    cs = PeopleCS(data_path)
    cs.set('7:30')
    cs.show()
    cs.cut(num=4)
    cs.show()
    cs.cut(num=6)
    cs.show()
    print('end')
