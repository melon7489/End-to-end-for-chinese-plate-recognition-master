import os
import cv2
import json

# def get_json_data(jsonname):
#     # 获取json里面数据
#
#     with open(jsonname, 'rb') as f:
#         # 定义为只读模型，并定义名称为f
#
#         params = json.load(f)
#         # 加载json文件中的内容给params
#         for i in range(len(params)):
#             params[i]['score'] = params[i]['score'] * 10
#         # 修改内容
#
#         print("params", params)
#         # 打印
#
#         dict = params
#         # 将修改后的内容保存在dict中
#
#     f.close()
#     # 关闭json读模式
#
#     return dict
#     # 返回dict字典内容
#
#
# def write_json_data(dict):
#     # 写入json文件
#
#     with open('out', 'w') as r:
#         # 定义为写模式，名称定义为r
#
#         json.dump(dict, r)
#         # 将dict写入名称为r的文件中
#
#     r.close()
imgpath = 'D:/Users/DELL/Desktop/resize/'
jsonpath = 'D:/Users/DELL/Desktop/labelme/'
saveimgpath = 'D:/Users/DELL/Desktop/img/'

jsonlist = os.listdir(jsonpath)
jsonlist =  sorted(jsonlist, key=lambda x:int(x.split('.')[0]))   # 按照json文件夹的数字排序
print(jsonlist)
num = 0
for i,j in enumerate(jsonlist):
    num = num + 1
    print('正在处理：', j)
    newname = str(i + 233)
    img = cv2.imread(imgpath + j.split('.')[0] + '.jpg')
    cv2.imwrite(saveimgpath + newname + '.png', img)
    os.rename(jsonpath + j, jsonpath + newname + '.json')
print(num)