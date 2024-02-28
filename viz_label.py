# 这个脚本是一个简单的测试用例，旨在演示如何从一个定制的数据集 Meter 中加载数据，并可视化加载的数据。这种类型的脚本对于检查数据加载是否正确以及数据增强是否按预期工作特别有用。以下是代码的详细解释：
# 数据准备和变换设置：

#     设定均值 means 和标准差 stds，这些是针对图像数据进行正规化所需的统计数据。
#     初始化 Augmentation 类的实例 transform，设置图像的目标尺寸以及均值和标准差用于图像的标准化处理。

# 数据集加载：

#     创建 Meter 类的实例 trainset，并将前面定义的图像变换 transform 传递给它。

# 数据可视化：

#     遍历数据集的每个元素，对于每个索引 idx：
#         从 trainset 获取图像 img，指针掩膜 pointer_mask，表盘掩膜 dail_mask，文本掩膜 text_mask，训练掩膜 train_mask，边框 bboxs 和文本 transcripts。
#         将图像 img 从张量转换为可视化的格式：首先交换通道（从CHW到HWC），然后反标准化（使用预定的均值和标准差），最后将数据类型转换为无符号8位整型。
#         打印出 transcripts，即文本标签。
#         使用 OpenCV 的 imshow 函数显示图像和各种掩膜，其中掩膜通过自定义的 heatmap 函数转换为热图以便于可视化。
#         调用 cv2.waitKey(0) 来暂停，等待用户按任意键继续或关闭窗口。

# 注意：脚本中使用了 canvas 模块别名 cav，这个别名需要确保在你的工作环境中是可用的，它应该是与可视化相关的自定义工具。在实际使用中，可能需要根据实际情况调整路径和模块导入。

# 这个脚本非常有用于数据集的调试和检查，确保在开始训练深度学习模型之前，输入数据的质量和格式是正确的。

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import os
import numpy as np
# from util import strs
from dataset.data_util import pil_load_img
from dataset.dataload import TextDataset, TextInstance
# from util.io import read_lines
# from util.misc import norm2
import json
import os
import cv2
from util.augmentation import Augmentation
from util import canvas as cav
import time
from dataset.meter_data import Meter


if __name__ == '__main__':


    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)

    transform = Augmentation(
        size=640, mean=means, std=stds
    )

    trainset = Meter(transform=transform)


    for idx in range(0, len(trainset)):
        t0 = time.time()
        print("idx",idx)


        img,  pointer_mask, dail_mask, text_mask, train_mask,bboxs,transcripts =trainset[idx]

        img = img.transpose(1, 2, 0)
        img = ((img * stds + means) * 255).astype(np.uint8)

        print("trans",transcripts)
        cv2.imshow('imgs', img)
        cv2.imshow("pointer_mask", cav.heatmap(np.array(pointer_mask * 255 / np.max(pointer_mask), dtype=np.uint8)))
        cv2.imshow("dail_mask", cav.heatmap(np.array(dail_mask * 255 / np.max(dail_mask), dtype=np.uint8)))
        cv2.imshow("text_mask", cav.heatmap(np.array(text_mask * 255 / np.max(text_mask), dtype=np.uint8)))

        cv2.waitKey(0)
