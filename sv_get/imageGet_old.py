# coding=utf-8
import math
import json
import requests
import urllib
from urllib.request import urlopen
from optparse import OptionParser
# PIL Python Imaging Library 已经是Python平台事实上的图像处理标准库了。PIL功能非常强大，但API却非常简单易用
# 安装步骤 1.cmd 2.进入python的安装目录中的Scripts目录：3.输入命令：pip install pillow -i  http://pypi.douban.com/simple/ --trusted-host pypi.douban.com
from PIL import Image
from io import BytesIO
import numpy as np
import pymysql  # 自定义pymysql类
import time
# 连接数据库
myCon = pymysql.connect(host='127.0.0.1', port=3306, database='img', user='root', passwd='wuhuaisj')
cursor = myCon.cursor()

myCon1 = pymysql.connect(host='127.0.0.1', port=3306, database='img', user='root', passwd='wuhuaisj')
cursor1 = myCon1.cursor()

def getPanoBylocation_(location, img_name):
    # 将user_agent,referer写入头信息
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.120 Safari/537.36',
        'Referer': 'http://ditu.city8.com/gz/canyinfuwu/2716097_w3gd6'}
    lat = location.split(',')[0]
    lng = location.split(',')[1]
    url = 'https://sv.map.qq.com/xf?lat=' + lat + '&lng=' + lng + '&r=30&key=ba56844b2a30311fe2149b485ca2a031&output=jsonp&pf=jsapi&ref=jsapi&cb=qq.maps._svcb3.cbk3rdpp035'
    text = requests.get(url, headers=headers).text
    text = text.split("qq.maps._svcb3.cbk3rdpp035&&qq.maps._svcb3.cbk3rdpp035(")[1].split(")")[0]
    jsonMess = json.loads(text)

    if jsonMess['detail']:
        pano = str(jsonMess['detail']['svid'])  # 街景ID
        name = jsonMess['detail']['road_name']  # 道路名称
        print('查询成功%s' % (pano))
        # 查询记录是否存在，存在则无需再次下载
        sql = "SELECT * FROM streetimg WHERE id ='" + pano + "'"
        cursor1.execute(sql)
        resData = cursor1.fetchall()
        if len(resData) == 0:
            # 插入记录
            sql = "INSERT INTO streetimg (id,name,lat,lng,detail) VALUES (%s, %s, %s, %s, %s)"
            val = (pano, name, float(lat), float(lng), text)
            cursor.execute(sql, val)
            myCon.commit()
            # 全景图url
            url = "http://sv1.map.qq.com/thumb?svid=" + pano + "&x=0&y=0&from=web&level=0&size=0"
            # 分级瓦片
            download_(url, img_name, pano)
        else:
            print('图片已下载完成，无需再下载%s' % (pano))
    else:
        print('附近没有街景图片：%s,%s'%(lng,lat))


# 发送请求保存照片
def download_(url, name, pano):
    # 将user_agent,referer写入头信息
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.120 Safari/537.36',
        'Referer': 'http://ditu.city8.com/gz/canyinfuwu/2716097_w3gd6'}
    images = requests.get(url, headers=headers)

    if images.status_code == 200:
        try:  # 请求街景数据失败
            jsonMess = images.json()
            print('请求图片失败 原因：%s' % (jsonMess))
        except json.JSONDecodeError:  # json 编码异常捕获（街景请求成功）
            img = images.content
            print('图片: %s%s 正在下载..' % ('panoID ', pano))
            image = Image.open(BytesIO(img))
            image.save(r'' + name)


#

# wgs84转高德
def wgs84togcj02(lng, lat):
    PI = 3.1415926535897932384626
    ee = 0.00669342162296594323
    a = 6378245.0
    dlat = transformlat(lng - 105.0, lat - 35.0)
    dlng = transformlng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * PI
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * PI)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * PI)
    mglat = lat + dlat
    mglng = lng + dlng
    return [mglng, mglat]


# GCJ02/谷歌、高德 转换为 WGS84 gcj02towgs84
def gcj02towgs84(localStr):
    lng = float(localStr.split(',')[0])
    lat = float(localStr.split(',')[1])
    PI = 3.1415926535897932384626
    ee = 0.00669342162296594323
    a = 6378245.0
    dlat = transformlat(lng - 105.0, lat - 35.0)
    dlng = transformlng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * PI
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * PI)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * PI)
    mglat = lat + dlat
    mglng = lng + dlng
    return str(lng * 2 - mglng) + ',' + str(lat * 2 - mglat)


def transformlat(lng, lat):
    PI = 3.1415926535897932384626
    ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * \
          lat + 0.1 * lng * lat + 0.2 * math.sqrt(abs(lng))
    ret += (20.0 * math.sin(6.0 * lng * PI) + 20.0 *
            math.sin(2.0 * lng * PI)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lat * PI) + 40.0 *
            math.sin(lat / 3.0 * PI)) * 2.0 / 3.0
    ret += (160.0 * math.sin(lat / 12.0 * PI) + 320 *
            math.sin(lat * PI / 30.0)) * 2.0 / 3.0
    return ret


def transformlng(lng, lat):
    PI = 3.1415926535897932384626
    ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + \
          0.1 * lng * lat + 0.1 * math.sqrt(abs(lng))
    ret += (20.0 * math.sin(6.0 * lng * PI) + 20.0 *
            math.sin(2.0 * lng * PI)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lng * PI) + 40.0 *
            math.sin(lng / 3.0 * PI)) * 2.0 / 3.0
    ret += (150.0 * math.sin(lng / 12.0 * PI) + 300.0 *
            math.sin(lng / 30.0 * PI)) * 2.0 / 3.0
    return ret


# 获取经纬坐标
def getPoint(_points):
    point = _points.split(',')
    point_jin = point[0]
    point_wei = point[1]
    transOpints = wgs84togcj02(float(point_jin), float(point_wei))
    return transOpints


# 以txt文件格式存储
def saveText(filePath, str_):
    fo = open(filePath, "w+")  # 打开一个文件
    fo.write(str_);  # 内容写入
    fo.write('\n')
    fo.close()  # 关闭打开的文件


# 读取txt内容
def read_file(filePath):
    try:
        with open(filePath, 'r')as f:
            LIST = f.readlines()
            for line in LIST:
                print(line)
                return line.split('_')
            return []
    except FileNotFoundError:
        return []


# 输入左下以及右上角坐标 根据两点形成等差坐标组 进而获取图片
def getImage(start_point, end_point, cityName):
    # 取得起始坐标
    start_point_jin = start_point[0]
    start_point_wei = start_point[1]
    end_point_jin = end_point[0]
    end_point_wei = end_point[1]
    # 创建等差数组
    jins = np.arange(float(start_point_jin) * 1000*13, float(end_point_jin) * 1000*13, 10) / (1000*13)
    jins_num = len(jins)
    weis = np.arange(float(start_point_wei) * 1000*21, float(end_point_wei) * 1000*21, 10) / (1000*21)
    weis_num = len(weis)
    # 断点续爬 （初始化）
    filePath = 'logo.txt'
    indexArr = read_file(filePath)
    goFlag = 'flase'
    jinFlag = 'flase'
    weiFlag = 'flase'
    if len(indexArr) != 2:
        print('无需进行断点续爬')
    else:
        print('准备进行断点续爬！！！')
        goFlag = 'true'
        jins_c = 0
        weis_c = 0
    for jins_i in range(jins_num):
        if goFlag == 'true':
            if len(indexArr) == 2:
                jins_c = int(indexArr[0]) - jins_i  # 索引差值
            if jinFlag != 'true':
                jins_i = jins_c + jins_i  # 修正经度索引
            if jins_i > jins_num - jins_c:  # 保证索引不越界
                jinFlag = 'true'
                break
        jin = round(jins[jins_i], 7)
        for weis_i in range(weis_num):
            if goFlag == 'true':
                if len(indexArr) == 2:
                    weis_c = int(indexArr[1]) - weis_i  # 索引差值
                    indexArr = []  # 清空记录
                if weiFlag != 'true':
                    weis_i = weis_c + weis_i  # 修正纬度索引
                if weis_i > weis_num - weis_c:  # 保证索引不越界
                    weiFlag = 'true'
                    break
            wei = round(weis[weis_i], 7)
            # 断点续爬 （记录经纬度索引）
            saveText(filePath, str(jins_i) + '_' + str(weis_i))
            # 这里要注意下，对应的经纬度没有街景图的地方，输出的会是无效图片
            print(jin, wei)
            # D:\python\django\GeoModel
            img_name = "D:\\项目工作\\不完备数据\\数据融合实验\\原始数据\\长沙市四区街景地图" + cityName + "\\" + str(
                wei) + "_" + str(jin) + ".jpg"
            getPanoBylocation_(str(wei) + "," + str(jin), img_name)
            time.sleep(5)


# 112.94124111946027,28.086146863446505
# 113.06958759138627,28.27550641121654
if __name__ == '__main__':
    # 定义数据字典 根据起始点坐标推算内容坐标
    cityJinweiArr = [{"start": "112.90034510344827,27.912024799999998", "end": "113.18585434482759,28.4000312",
                      "city": "ChangSha"}]
    for city in cityJinweiArr:
        start_point = getPoint(city['start'])
        end_point = getPoint(city['end'])
        cityName = city['city']
        getImage(start_point, end_point, cityName)
