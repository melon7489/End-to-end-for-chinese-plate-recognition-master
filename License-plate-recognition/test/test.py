import os
import cv2
import numpy as np
from PIL import Image
from tensorflow import keras
from core import locate_and_correct
from Unet import unet_predict
from CNN import cnn_predict
import time



'''批量resize文件'''
# # 先把所有的图片存放到同一个文件夹
# pic_path = 'D:/Users/DELL/Desktop/chinese_license_plate_generator-master/multi_val/'  # 自己存图片的路径
# pic = os.listdir(pic_path)
# print(pic)
# num = 0
# for i in pic:
#     if i.endswith(('.jpg', '.png')):   # 看你的图片是什么格式，最好统一成jpg格式
#         num = num + 1
#         print('正在转换 %s'%i)
#         img = cv2.imdecode(np.fromfile(pic_path+str(i), dtype=np.uint8), -1)  # cv2.imread无法读取中文路径图片，改用此方式
#         # img = cv2.imread(pic_path+str(i))
#         print(img.shape)
#         img_resize = cv2.resize(img, dsize=(240, 80), interpolation=cv2.INTER_AREA)
#         print(img_resize.shape)
#         # if not os.path.exists(pic_path):
#         #     os.mkdir(pic_path)
#         # cv2.imwrite(pic_path+'111/'+str(i), img_resize)
#         cv2.imencode('.jpg', img_resize)[1].tofile('../cnndatasets/'+str(i))   # 中文路径
# print('共修改了 %d 张图片' % num)


'''
将制作好的标签文件，批量转化并提取
'''
# path = 'D:/Users/DELL/Desktop/zhoubao/'
# np.set_printoptions(threshold=1e6)
# jsonlist = []   #存放json文件的列表
# listdir = os.listdir(path + 'labelme/')
# for i in listdir:   # 提取后缀为.json的文件到jsonlist
#     if i.endswith('.json'):
#         jsonlist.append(i)
# print(jsonlist)
# jsonlist = sorted(jsonlist, key= lambda x:int(x.split('.')[0]))  # 按照文件名中的数字进行排序
# print(jsonlist)
#
# # 将json文件转化为文件夹
# if not os.path.exists('%sjson/' % path):
#     os.mkdir('%sjson/' % path)
# for i,json in enumerate(jsonlist):
#     print('正在处理    %s'%json)
#     os.system('labelme_json_to_dataset %slabelme/%s -o %sjson/json_%d' % (path, json, path, i+177))  #  往后排序更改这里的 i
# n = len(jsonlist)  # n为总共标注的图片数
# print('共计  %d  张图片' % n)
#
'''
将生成的json文件夹中的img.png和label.png进行处理
'''
# # dst_w=512
# # dst_h=512
# # dst_shape=(dst_w,dst_h,3)
# train_image = path + 'u_netdatasets/train_image/'
# if not os.path.exists(train_image):
#     os.makedirs(train_image)
# train_label = path + 'u_netdatasets/train_label/'
# if not os.path.exists(train_label):
#     os.makedirs(train_label)
# jsonpath = path + 'json/'
# dirlist = os.listdir(jsonpath)
# jsondirlist = []     # 存放生成的json文件夹的列表
# for i in dirlist:   # 提取后缀为.json的文件到jsonlist
#     if i.startswith('json_'):
#         jsondirlist.append(i)
# jsondirlist = sorted(jsondirlist, key=lambda x:int(x.split('_')[1]))   # 按照json文件夹的数字排序
# print(jsondirlist)
# for i in jsondirlist:
#     print('正在处理  %s' % i)
#     img = cv2.imread(path + 'json/%s/img.png' % i)
#     label = cv2.imread(path + 'json/%s/label.png' % i)
#     print(img.shape)
#     print(label.shape)
#     # 将label.png转化为只有 0 255 的二值图像
#     label = label / np.max(label[:, :, 2]) * 255
#     label[:, :, 0] = label[:, :, 1] = label[:, :, 2]
#     print(np.max(label[:, :, 2]))
#     print(set(label.ravel()))
#     cv2.imwrite(train_image + '%d.png' % int(i.split('_')[1]), img)
#     cv2.imwrite(train_label + '%d.png' % int(i.split('_')[1]), label)


'''测试集上的准确率'''
img_src_path = './ca/'
imgname = os.listdir(img_src_path)
unet = keras.models.load_model('../unet.h5')
cnn = keras.models.load_model('../cnn.h5')
accnum = 0
for name in imgname:
    start = time.time()
    y_true = name[0:2] + '·' + name[2:7]
    print('正在处理：', y_true)
    img_src = cv2.imdecode(np.fromfile(img_src_path + name, dtype=np.uint8), -1)  # 从中文路径读取时用
    img_src, img_mask = unet_predict(unet, img_src_path + name)
    img_src_copy, Lic_img = locate_and_correct(img_src, img_mask)  # 利用core.py中的locate_and_correct函数进行车牌定位和矫正
    Lic_pred = cnn_predict(cnn, Lic_img)  # 利用cnn进行车牌的识别预测,Lic_pred中存的是tuple(车牌图片,识别结果)
    if Lic_pred:
        for i, lic_pred in enumerate(Lic_pred):
            if lic_pred[1] == y_true:
                print('预测值：', lic_pred[1])
                accnum = accnum + 1
            else:
                print(name + '识别错误')
    else:
        print('未能识别')
    end = time.time()
    print('花费时间：', end - start)
print('测试集上的准确率：', accnum / len(imgname))



