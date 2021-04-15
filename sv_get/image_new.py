# Author:wuhuaisj
# Date:2021/4/15 20:49
# Github:https://github.com/chuwuhua

# 重写为结构更好的断点续爬代码，健壮性更好一点
from sv_get.transform import LngLatTransfer
import pymysql

# 坐标转换
Trans = LngLatTransfer()
# 连接数据库（两个连接，分别有不同的任务）
myCon = pymysql.connect(host='127.0.0.1', port=3306, database='img', user='root', passwd='wuhuaisj')
cursor = myCon.cursor()

myCon1 = pymysql.connect(host='127.0.0.1', port=3306, database='img', user='root', passwd='wuhuaisj')
cursor1 = myCon1.cursor()


def getImage(start, end, city):
    pass

if __name__ == '__main__':
    # 定义数据字典 根据起始点坐标推算内容坐标
    # wgs84需要转为高德坐标
    cityJinweiArr = [
        {
            "左下": [112.90034510344827, 27.912024799999998],
            "右上": [113.18585434482759, 28.4000312],
            "city": "ChangSha"
        }
    ]
    for city in cityJinweiArr:
        cityName = city['city']
        start = city['左下']
        end = city['右上']
        getImage(start, end, cityName)
