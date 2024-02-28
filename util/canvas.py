    # heatmap 函数：将灰度图像转换为热图。它使用matplotlib的jet颜色映射来实现这一转换。输入是一个灰度图像，输出是一个颜色映射后的图像，其中灰度值被转换成对应的颜色表示。

    # loss_ploy 函数：绘制损失函数的变化曲线。这对于观察训练过程中损失如何随着迭代次数减少而减少非常有用。它接收损失列表（每个周期的平均损失），步骤总数，和周期长度，生成损失值随训练步骤的变化图，并保存为PNG格式。

    # plt_ploys 函数：绘制多个策略或数据集的损失曲线。它允许一次比较多个不同的损失曲线，用于比较不同模型或参数配置的表现。函数接受一个包含多个损失列表的字典，每个键对应一组损失值，并将它们绘制在同一个图表上，每条曲线用不同的颜色表示。

#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = '古溪'

import numpy as np
import random
import matplotlib.pyplot as plt


def heatmap(im_gray):
    cmap = plt.get_cmap('jet')
    rgba_img = cmap(255 - im_gray)
    Hmap = np.delete(rgba_img, 3, 2)
    # print(Hmap.shape, Hmap.max(), Hmap.min())
    # cv2.imshow("heat_img", Hmap)
    # cv2.waitKey(0)
    return Hmap


def loss_ploy(loss_list, steps, period, name=""):
    fig1, ax1 = plt.subplots(figsize=(16, 9))
    ax1.plot(range(steps // period), loss_list)
    ax1.set_title("Average loss vs step*{}".format(period))
    ax1.set_xlabel("step*{}".format(period))
    ax1.set_ylabel("Current loss")
    plt.savefig('{}@loss_vs_step*{}.png'.format(name,period))
    plt.clf()


def plt_ploys(ploys, period, name=""):
    fig1, ax1 = plt.subplots(figsize=(16, 9))
    cnames = ['aliceblue','antiquewhite','aqua','aquamarine','azure',
               'blanchedalmond','blue','blueviolet','brown','burlywood',
               'coral','cornflowerblue','cornsilk','crimson','cyan',
               'darkblue','deeppink','deepskyblue','dodgerblue','forestgreen',
               'gold','goldenrod','green','greenyellow','honeydew','hotpink',
               'lawngreen','lightblue','lightgreen','lightpink','lightsalmon',
               'lightseagreen','lightsteelblue','lightyellow','lime','limegreen',
               'mediumseagreen','mediumspringgreen','midnightblue','orange','orangered',
               'pink','red','royalblue','seagreen','skyblue','springgreen','steelblue',
               'tan','teal','thistle','yellow','yellowgreen']

    color = random.sample(cnames, len(ploys.keys()))
    for ii, key in enumerate(ploys.keys()):
        ax1.plot(range(1, len(ploys[key])+1), ploys[key],color=color[ii], label=key)
    ax1.set_title("Loss Carve line")
    ax1.set_xlabel("step*{}".format(period))
    ax1.set_ylabel("Current loss")
    plt.legend(ploys.keys())
    plt.savefig('{}@loss_vs_step*{}.png'.format(name, period))
    plt.clf()

if __name__ == '__main__':
    # TODO ADD CODE
    pass
