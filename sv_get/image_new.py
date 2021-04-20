# Author:wuhuaisj
# Date:2021/4/15 20:49
# Github:https://github.com/chuwuhua

# 重写为结构更好的断点续爬代码，健壮性更好一点
# 支持断点续爬，死机断网也没关系
from transform import LngLatTransfer
import pymysql
import redis
import requests
import json
from PIL import Image
from io import BytesIO
import sys
import time

# 坐标转换
Trans = LngLatTransfer()
# 连接数据库（两个连接，分别有不同的任务）
myCon = pymysql.connect(host='127.0.0.1', port=3306, database='img', user='root', passwd='root')
cursor = myCon.cursor()

myCon1 = pymysql.connect(host='127.0.0.1', port=3306, database='img', user='root', passwd='root')
cursor1 = myCon1.cursor()

r = redis.Redis(host='localhost', port=6379, decode_responses=True)
path = "/home/dc/CGW/原始数据/长沙市芙蓉区街景图片/"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.120 Safari/537.36',
    'Referer': 'http://ditu.city8.com/gz/canyinfuwu/2716097_w3gd6'}


def download(url, city, name, pano):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.120 Safari/537.36',
        'Referer': 'http://ditu.city8.com/gz/canyinfuwu/2716097_w3gd6'}
    images = requests.get(url, headers=headers)

    if images.status_code == 200:
        try:  # 请求街景数据失败
            jsonMess = images.json()
            print('请求图片失败 原因：%s' % (jsonMess))
            sys.exit(0)
        except json.JSONDecodeError:  # json 编码异常捕获（街景请求成功）
            img = images.content
            print('图片: %s%s 正在下载..' % ('panoID ', pano))
            image = Image.open(BytesIO(img))
            image.save(r'' + name)


def getImage(utm_x, utm_y, city):
    # utm转wgs84,wgw84转火星坐标系
    wgs84_x, wgs84_y = Trans.WebMercator_to_WGS84(utm_x, utm_y)
    gcj_x, gcj_y = Trans.WGS84_to_GCJ02(wgs84_x, wgs84_y)
    # 用位置构造url来请求json获取图片id
    # 请求的图片位置存在误差，但是不影响
    # 最后使用的还是需要用请求的 wgs84 坐标
    url = 'https://sv.map.qq.com/xf?lat=' \
          + str(gcj_x) + '&lng=' + str(gcj_y) \
          + '&r=20&key=ba56844b2a30311fe2149b485ca2a031&output=jsonp&pf=jsapi&ref=jsapi&cb=qq.maps._svcb3.cbk3rdpp035'
    try:
        text = requests.get(url, headers=headers).text
    except Exception as e:
        print('raise error:{}'.format(str(e)))
        sys.exit(0)
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
            val = (pano, name, float(wgs84_x), float(wgs84_y), text)
            cursor.execute(sql, val)
            myCon.commit()
            # 全景图url
            url = "http://sv1.map.qq.com/thumb?svid=" + pano + "&x=0&y=0&from=web&level=0&size=0"
            img_name = path +pano+ ".jpg"
            # 分级瓦片
            download(url,city, img_name, pano)
        else:
            print('图片已下载完成，无需再下载%s' % (pano))
    else:
        print('附近没有街景图片：%s,%s' % (wgs84_x, wgs84_y))
    time.sleep(5)
    r.set('position_x', str(utm_x))
    r.set('position_y', str(utm_y))


def processXY(start, end, city, granularity):
    # 先把经纬度转为utm投影，然后每granularity米搜索，这样来保证图片的密集程度
    # granularity=20 对于芙蓉区较为合适，大概请求36万次
    wgs84_x_start = start[0]
    wgs84_y_start = start[1]
    wgs84_x_end = end[0]
    wgs84_y_end = end[1]
    utm_x_start, utm_y_start = Trans.WGS84_to_WebMercator(lng=wgs84_x_start, lat=wgs84_y_start)
    utm_x_end, utm_y_end = Trans.WGS84_to_WebMercator(lng=wgs84_x_end, lat=wgs84_y_end)
   # 注意从此处开始的 utm都是整除了10的，但是不影响后面搜索的坐标
    utm_x_start = int(utm_x_start // granularity)*granularity
    utm_y_start = int(utm_y_start // granularity)*granularity
    utm_x_end = int(utm_x_end // granularity )*granularity
    utm_y_end = int(utm_y_end // granularity )*granularity
    # 用redis存储已经搜索到的位置,此处来获取搜索到的位置（索引位置）
    position_x = int(r.get('position_x'))
    position_y = int(r.get('position_y'))
    print('****************************')
    print('start process with location {},{}'.format(position_x, position_y))
    for y in range(utm_y_start, utm_y_end + granularity, granularity):
        if y < position_y:
            continue
        elif y == position_y:
            if position_x < utm_x_end+granularity:   
                for x in range(position_x + granularity, utm_x_end + granularity, granularity):
                    getImage(x, y, city)            
        else:
            for x in range(utm_x_start, utm_x_end + granularity, granularity):
                getImage(x, y, city)
    print('process Finish')


if __name__ == '__main__':
    # 定义数据字典 根据起始点坐标推算内容坐标
    # wgs84需要转为高德坐标
    cityJinweiArr = [
        {
            "左下": [112.96152565517241, 28.1661948],
            "右上": [113.10428027586207, 28.2373624],
            "city": "ChangSha_frq",
            "granularity": 20
        }
    ]
    for city in cityJinweiArr:
        cityName = city['city']
        start = city['左下']
        end = city['右上']
        granularity = city['granularity']
        processXY(start, end, cityName, granularity)
