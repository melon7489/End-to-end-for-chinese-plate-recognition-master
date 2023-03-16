# -*- coding:utf-8 -*-
from Unet import unet_predict
from core import locate_and_correct
from CNN import cnn_predict
from tensorflow.keras import models
import os
import cv2
import numpy as np
import skimage
from skimage import io, filters
from PIL import Image
from collections import Counter
import Augmentor
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂",
             "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

'''裁剪车牌制作cnn数据集'''
# unet = models.load_model('./test/unet.h5')
# imgpath = 'D:/py/DL/CCPD2019/ccpd_base/'
# savepath = 'D:/Users/DELL/Desktop/新建文件夹/'
# data = os.listdir(imgpath)
# num = 0
# a = 10000
# for img in data:
#     # img = 1 - 18_16 - 193 & 286_541 & 526 - 537 & 526_193 & 409_197 & 286_541 & 403 - 0_0_7_31_25_26_21 - 78 - 85
#     liclist = str(img).split('&')[6].split('-')[1].split('_')
#     if not liclist[0] == '0':  # 不是皖的话
#         src,mask = unet_predict(unet, imgpath+img)
#         img_src_copy, Lic_img = locate_and_correct(src, mask)
#         if not len(Lic_img) == 0:
#             no = num + 182  # 编号从几开始
#             num = num + 1
#             try:
#                 for i, j in enumerate(Lic_img):
#                     lic = provinces[int(liclist[0])]
#                     if not lic==0:
#                         print(type(lic),lic)
#                         for i in liclist[1:7]:
#                             lic = lic + ads[int(i)]
#                         print('正在处理 %s  第 %d 个' % (lic, num))
#                         cv2.imencode('.png', j)[1].tofile(savepath + lic + '_%d.png' % (no))  # 中文路径
#             except:
#                 print('未能识别！')
#         else:
#             print('未检测到车牌！')
# print('共计 %d' % num)

# # 0, 10, 11   皖，苏，浙  太多了,,云 吉 宁 新 桂 琼 甘 蒙 藏 贵 青 黑太少了
# imgpath = './cnndatasets/'
# savepath = 'D:/Users/DELL/Desktop/augment/'
# data = os.listdir(imgpath)
# num = 0
# P = []  # 存放省份，统计省份个数，以便于计算range
# for i in data:
#     province = i[0]
#     P.append(province)
# Ps = Counter(P)
# print(Ps)
# rang = 0
# for img in data:
#     def crarandom(range):
#         r = np.random.rand()
#         r = r / 1 * range
#         return r
#     # img = 1 - 18_16 - 193 & 286_541 & 526 - 537 & 526_193 & 409_197 & 286_541 & 403 - 0_0_7_31_25_26_21 - 78 - 85
#     province = img[0]
#     rang = Ps[province]
#     print(province, rang)
#     if 0 < rang < 5:
#         rang = 60
#     elif 5 <= rang < 10:
#         rang = 30
#     elif 10 <= rang <15:
#         rang = 20
#     elif 15 <= rang < 20:
#         rang = 15
#     elif 25 <= rang < 30:
#         rang = 10
#     elif 35 <= rang < 40:
#         rang = 8
#     elif 45 <= rang < 50:
#         rang = 6
#     else:
#         rang = 4
#     if province not in ['云', '吉', '宁', '新', '桂', '琼', '甘', '蒙', '藏', '贵', '青', '黑']:  # 不是皖，苏，浙   的话做高斯模糊进行数据增强
#         continue
#     print('正在处理{}'.format(img))
#     im = io.imread(imgpath + img)
#     for a in range(rang):
#         num = num + 1
#         ran = np.random.rand()
#         if 0 < ran < 0.4:  # 高斯滤波
#             print('高斯滤波')
#             im = filters.gaussian(im, sigma=crarandom(0.3))  # sigma=0.4
#         elif ran < 0.7:# 椒盐
#             im = skimage.util.random_noise(im, mode='s&p', amount=crarandom(0.01))
#             print('椒盐')
#         else:  # 高斯噪声
#             im = skimage.util.random_noise(im, mode='gaussian', var=crarandom(0.0001))
#             print('高斯噪声')
#         # cv2.imencode('.png', im)[1].tofile(savepath + img[0:7] + '_%d' % (num + 15000))  # 中文路径
#         io.imsave(savepath + img[0:7] + '_%d.png' % (num + 20000), im)
# print('共计 {}'.format(num))



# from collections import Counter
# import os
# from matplotlib import pyplot as plt
#
# img = os.listdir('D:/Users/DELL/Desktop/新建文件夹/')
# P = []
# ads = []
# for i in img:
#     P.append(i[0])
#     for j in i[1:7]:
#         ads.append(j)
# a = Counter(P)
# b = Counter(ads)
# img = os.listdir('./cnndatasets/')
# P = []
# ads = []
# for i in img:
#     P.append(i[0])
#     for j in i[1:7]:
#         ads.append(j)
# c = Counter(P)
# d = Counter(ads)
#
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.figure(figsize=(12, 7))
# plt.subplot(221)
# plt.bar(a.keys(), a.values())
# plt.subplot(222)
# plt.bar(b.keys(), b.values())
# plt.subplot(223)
# plt.bar(c.keys(), c.values())
# plt.subplot(224)
# plt.bar(d.keys(), d.values())
# plt.savefig('./cnndata.png')
# plt.show()

'''对CNN数据集进行数据增强(亮度、颜色、对比度)'''
# img_path = 'D:/Users/DELL/Desktop/augment/' # image directory
# p = Augmentor.Pipeline(img_path)
# p.random_brightness(probability=1, min_factor=0.7, max_factor=1.2)
# p.random_color(probability=1, min_factor=0.0, max_factor=1)
# p.random_contrast(probability=1, min_factor=0.7, max_factor=1.2)
# p.sample(len(os.listdir(img_path))-1)
# name = os.listdir(img_path + 'output/')
# print(name)
# for i in name:
#     img = cv2.imdecode(np.fromfile(img_path + 'output/' + i, dtype=np.uint8), -1)  # 从中文路径读取时用
#     cv2.imencode('.png', img)[1].tofile(img_path + '新建文件夹/' + i.split('_')[2] + '_'+ i.split('_')[3]) # 中文路径保存

'''对CNN数据集进行数据增强(亮度、颜色、对比度)'''
# img_src = './u_netdatasets/train_image/'  # image directory
# img_mask = './u_netdatasets/train_label/'
#
# p = Augmentor.Pipeline(img_src)
# p.ground_truth(img_mask)
#
# p.rotate(probability=1, max_left_rotation=20, max_right_rotation=20)
# p.sample(177)
# # p.random_brightness(probability=1, min_factor=0.8, max_factor=1.2)
# # p.sample((len(os.listdir(img_src))-1)//2)
# # p.random_contrast(probability=1, min_factor=0.8, max_factor=1.2)
# # p.sample((len(os.listdir(img_src))-1)//2)
# train = os.listdir(img_src + 'output/')
# src = []
# mask = []
# for i in train:
#     if i.startswith('_groundtruth_'):
#         mask.append(i)
#     else:
#         src.append(i)
# np.set_printoptions(linewidth=1000)
# print(src)
# print(mask)
# for a in range(len(src)):
#     print('正在处理：{}'.format(src[a]))
#     print('正在处理：{}'.format(mask[a]))
#     imgsr = cv2.imread(img_src + 'output/' + src[a])
#     imgma = cv2.imread(img_src + 'output/' + mask[a])  # mask处理之后不在是二值（0,255），出现其他类别，进行二值化
#     ret, dst = cv2.threshold(imgma[:, :, 2], 128, 255, cv2.THRESH_BINARY)
#     imgma[:, :, 0] = imgma[:, :, 1] = imgma[:, :, 2] = dst
#     cv2.imwrite(img_src + '{}.png'.format(a+823), imgsr)
#     cv2.imwrite(img_mask + '{}.png'.format(a + 823), imgma)
#
# for i in range(len(os.listdir('./u_netdatasets/train_label/'))):
#     im1 = cv2.imread(img_mask + '/{}.png'.format(i+823))
#     print(set(im1.ravel()))
#     print(i)


