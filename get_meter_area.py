# 这段脚本定义了一个 Detector 类，它使用预训练的 YOLOv5 模型进行对象检测，尤其是针对仪表和数字的检测。以下是代码的详细解释：
# 类 Detector 初始化：

#     初始化模型参数，包括图像尺寸、阈值、最大帧数等。
#     在 init_model 方法中，加载预训练的 YOLOv5 模型权重，并将模型置于评估模式。
#     为每个识别类别分配随机颜色。

# 图像预处理：

#     在 preprocess 方法中，对输入图像进行大小调整和归一化处理，以适配模型输入要求。

# 绘制边界框：

#     plot_bboxes 方法在图像上绘制检测到的边界框和类别标签。

# 检测方法：

#     detect 方法处理图像，进行对象检测，并返回检测后的图像、检测信息、识别的数字列表和仪表列表。
#     使用非最大抑制（NMS）来过滤重叠的检测框。
#     根据检测结果，提取仪表和数字的图像区域。

# 主程序：

#     实例化 Detector 类。
#     从指定路径读取图像列表，并对每张图像执行检测。
#     计算并打印处理每帧图像所需的平均时间和帧率（FPS）。

# 这个脚本适用于监控和分析诸如电表或水表之类的仪表读数，可以在自动化监控系统或远程仪表读取应用中使用。通过调整阈值和模型参数，可以优化检测的准确性和处理速度。

import torch
import numpy as np
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.augmentations import letterbox
from yolov5.utils.torch_utils import select_device
import cv2
from random import randint
import os
import time


class Detector(object):

    def __init__(self):
        self.img_size = 640
        self.threshold = 0.6
        self.max_frame = 160
        self.init_model()

    def init_model(self):

        self.weights = 'yolov5/best.pt'
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        self.device = select_device(self.device)
        model = attempt_load(self.weights, map_location=self.device)
        model.to(self.device).eval()
        model.half()
        # torch.save(model, 'test.pt')
        self.m = model
        self.names = model.module.names if hasattr(
            model, 'module') else model.names
        self.colors = [
            (randint(0, 255), randint(0, 255), randint(0, 255)) for _ in self.names
        ]

    def preprocess(self, img):

        img0 = img.copy()
        img = letterbox(img, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half()  # 半精度
        img /= 255.0  # 图像归一化
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img0, img

    def plot_bboxes(self, image, bboxes, line_thickness=None):
        tl = line_thickness or round(
            0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
        for (x1, y1, x2, y2, cls_id, conf) in bboxes:
            color = self.colors[self.names.index(cls_id)]
            c1, c2 = (x1, y1), (x2, y2)
            cv2.rectangle(image, c1, c2, color,
                          thickness=tl, lineType=cv2.LINE_AA)
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(
                cls_id, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(image, '{} ID-{:.2f}'.format(cls_id, conf), (c1[0], c1[1] - 2), 0, tl / 3,
                        [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        return image

    def detect(self, im,i):


        im0, img = self.preprocess(im)

        pred = self.m(img, augment=False)[0]
        pred = pred.float()
        pred = non_max_suppression(pred, self.threshold, 0.3)

        pred_boxes = []
        image_info = {}
        count = 0

        digital_list,meter_list=[],[]
        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                for *x, conf, cls_id in det:
                    lbl = self.names[int(cls_id)]


                    x1, y1 = int(x[0]), int(x[1])
                    x2, y2 = int(x[2]), int(x[3])

                    region=im0[y1:y2,x1:x2]
                    if lbl =="meter":
                        meter_list.append(region)
                    else:
                        digital_list.append(region)

                    pred_boxes.append(
                        (x1, y1, x2, y2, lbl, conf))
                    count += 1
                    key = '{}-{:02}'.format(lbl, count)
                    image_info[key] = ['{}×{}'.format(
                        x2-x1, y2-y1), np.round(float(conf), 3)]

        im = self.plot_bboxes(im, pred_boxes)
        return im, image_info, digital_list, meter_list


if __name__=="__main__":
    det=Detector()
    path='/home/sy/ocr/datasets/all_meter_image/'
    img_list=os.listdir(path)
    total_time=0
    num=0
    for i in img_list:
        img=cv2.imread(path+i)
        start_time=time.time()
        image,image_info,digital_list, meter_list=det.detect(img,i)
        end_time=time.time()
        total_time += end_time - start_time
        fps = (num + 1) / total_time
        num+=1
        print("fps",fps)

