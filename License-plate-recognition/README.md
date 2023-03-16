## 1. 程序运行

```markdown
1. 选择UI.py运行main函数即可进入主程序运行界面,选择图片后，点击识别车牌即可。
2. 训练模型时，运行train.py或独立运行Unet.py和CNN.py
```

## 2. 数据集

```markdown
1. u_netdatasets文件夹下存放u_net模型训练的数据集，包含train_image和train_label两个子文件夹。
2. cnndatasets文件夹下存放cnn模型训练的数据集。
3. test文件夹下的ca文件夹存放整个项目的测试集图像。
```

## 3. 模型文件

```markdown
1. unet.h5和cnn.h5为训练好的模型文件。
2. uuuunet.h5和cccccn.h5为回调函数临时保存的模型文件。
```

## 4. 其他

```markdown
1. test下的test.py和cnndatasets.py为进行测试或生成数据时的临时性测试程序。
2. Unet.png和CNN.pn为模型的可视化结构图像。
3. core.py主要用来裁剪和矫正车牌区域。
```







