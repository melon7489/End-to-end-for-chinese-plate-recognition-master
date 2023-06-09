# -*- coding:utf-8 -*-
import numpy as np
import os
import cv2
from tensorflow.keras.optimizers import Adam  # 自适应动量
from tensorflow.keras import layers, losses, models
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model
from PIL import Image
import matplotlib.pyplot as plt
def unet_train():
    # 数据的预处理
    # 输入图像为512 * 512,resize之后再进行训练
    height = 512
    width = 512
    # 数据训练集路径
    path = './u_netdatasets/'
    input_name = os.listdir(path + 'train_image')  # 返回指定路径下的文件和文件夹列表
    n = len(input_name)
    print("数据集共 %d 数据" % n)
    X_train, y_train = [], []  # 存放训练集的输入和标签
    for i in range(n):  # 切记i是从0开始的，图片也要从0开始，否则会出错
        print("正在读取第%d张图片" % i)
        img = cv2.imread(path + 'train_image/%d.png' % i)
        label = cv2.imread(path + 'train_label/%d.png' % i)
        X_train.append(img)
        y_train.append(label)
    X_train = np.array(X_train)
    y_train = np.array(y_train)


    # 卷积+BN+激活 param：输入，输出维度，核大小，步长，边界是否填充
    def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same'):
        x = layers.Conv2D(nb_filter, kernel_size, strides=strides, padding=padding)(x)  # 参数个数 = kernel_size**2*通道数*nb_filter+nb_filter
        x = layers.BatchNormalization(axis=3)(x)  #BN
        x = layers.LeakyReLU(alpha=0.1)(x)  # 激活
        return x

    # 卷积+BN+激活 param：输入，输出维度，核大小，步长，边界是否填充
    def Conv2dT_BN(x, filters, kernel_size, strides=(2, 2), padding='same'):
        x = layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)(x)  # 反卷积层
        x = layers.BatchNormalization(axis=3)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        return x
    # 输入层512*512*3
    inpt = layers.Input(shape=(512, 512, 3))
    # 卷积层 输入，输出维度，核大小
    conv1 = Conv2d_BN(inpt, 8, (3, 3))
    conv1 = Conv2d_BN(conv1, 8, (3, 3))
    pool1 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)

    conv2 = Conv2d_BN(pool1, 16, (3, 3))
    conv2 = Conv2d_BN(conv2, 16, (3, 3))
    pool2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)

    conv3 = Conv2d_BN(pool2, 32, (3, 3))
    conv3 = Conv2d_BN(conv3, 32, (3, 3))
    pool3 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)

    conv4 = Conv2d_BN(pool3, 64, (3, 3))
    conv4 = Conv2d_BN(conv4, 64, (3, 3))
    pool4 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv4)

    conv5 = Conv2d_BN(pool4, 128, (3, 3))
    conv5 = layers.Dropout(0.5)(conv5)
    conv5 = Conv2d_BN(conv5, 128, (3, 3))
    conv5 = layers.Dropout(0.5)(conv5)

    convt1 = Conv2dT_BN(conv5, 64, (3, 3))
    concat1 = layers.concatenate([conv4, convt1], axis=3)  # 有效特征层在通道上的堆叠
    concat1 = layers.Dropout(0.5)(concat1)
    conv6 = Conv2d_BN(concat1, 64, (3, 3))
    conv6 = Conv2d_BN(conv6, 64, (3, 3))

    convt2 = Conv2dT_BN(conv6, 32, (3, 3))
    concat2 = layers.concatenate([conv3, convt2], axis=3)
    concat2 = layers.Dropout(0.5)(concat2)
    conv7 = Conv2d_BN(concat2, 32, (3, 3))
    conv7 = Conv2d_BN(conv7, 32, (3, 3))

    convt3 = Conv2dT_BN(conv7, 16, (3, 3))
    concat3 = layers.concatenate([conv2, convt3], axis=3)
    concat3 = layers.Dropout(0.5)(concat3)
    conv8 = Conv2d_BN(concat3, 16, (3, 3))
    conv8 = Conv2d_BN(conv8, 16, (3, 3))

    convt4 = Conv2dT_BN(conv8, 8, (3, 3))
    concat4 = layers.concatenate([conv1, convt4], axis=3)
    concat4 = layers.Dropout(0.5)(concat4)
    conv9 = Conv2d_BN(concat4, 8, (3, 3))
    conv9 = Conv2d_BN(conv9, 8, (3, 3))
    conv9 = layers.Dropout(0.5)(conv9)
    outpt = layers.Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv9)

    model = models.Model(inpt, outpt)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    model.summary()

    plot_model(model, to_file="./Unet.png", show_shapes=True, rankdir='TB')

    # 回调函数（系统函数，调用我的函数）————保存模型，防止掉电时候全没了
    checkpoint = ModelCheckpoint('uuuunet.h5', save_best_only=True)
    print("开始训练u-net")
    log = model.fit(X_train, y_train, epochs=30, batch_size=3, callbacks=[checkpoint], validation_split=0.1)  # batch_size不要过大，否则内存容易溢出
    # print(log)
    print(log.history)
    model.save('unet.h5')
    print('unet.h5保存成功!!!')


def unet_predict(unet, img_src_path):
    img_src = cv2.imdecode(np.fromfile(img_src_path, dtype=np.uint8), -1)  # 从中文路径读取时用
    # img_src=cv2.imread(img_src_path)
    if img_src.shape != (512, 512, 3):
        img_src = cv2.resize(img_src, dsize=(512, 512), interpolation=cv2.INTER_AREA)[:, :, :3]  # dsize=(宽度,高度),[:,:,:3]是防止图片为4通道图片，后续无法reshape
    img_src = img_src.reshape(1, 512, 512, 3)  # 预测图片shape为(1,512,512,3)

    img_mask = unet.predict(img_src)  # 归一化除以255后进行预测
    np.set_printoptions(suppress=True, precision=8, linewidth=1e6, threshold=1e6)
    img_src = img_src.reshape(512, 512, 3)  # 将原图reshape为3维
    img_mask = img_mask.reshape(512, 512, 3)  # 将预测后图片reshape为3维
    img_mask = img_mask / np.max(img_mask) * 255  # 归一化后乘以255
    img_mask[:, :, 2] = img_mask[:, :, 1] = img_mask[:, :, 0]  # 三个通道保持相同
    img_mask = img_mask.astype(np.uint8)  # 将img_mask类型转为int型

    return img_src, img_mask
if __name__ == '__main__':
    unet_train()

