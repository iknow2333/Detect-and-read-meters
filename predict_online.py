# 这段Python代码是一个软件应用程序的一部分，设计用于使用图像处理和光学字符识别（OCR）技术从仪表（如水表、煤气表或电表）读取值。让我们逐步了解代码的每个部分：

#     导入库和模块：脚本开始时导入必要的Python库和模块。这包括os用于与文件系统交互，cv2用于使用OpenCV进行图像处理，numpy用于数值操作，torch用于PyTorch功能，以及几个实用模块，如BaseTransform、config、BaseOptions等，这些可能是为这个特定应用程序设计的自定义模块。

#     配置和初始化：
#         它通过BaseOptions()初始化选项和配置设置，并根据命令行参数或默认设置更新这些配置。
#         它将当前配置打印到控制台。

#     模型准备：
#         它设置了一个名为TextNet的模型，这可能是一个为图像中的文本检测和识别设计的神经网络。模型被初始化，加载预训练的权重，并设置到适当的设备（CPU或GPU）。
#         它还初始化了一个StringLabelConverter，这可能用于在OCR过程中编码和解码文本标签。

#     检测和OCR准备：
#         脚本设置了一个检测器（Detector）和一个专门针对遮罩区域的文本检测器（TextDetector_mask），可能是专为检测仪表区域和这些区域内的文本区域量身定做的。
#         初始化了MeterReader，用于从图像中读取仪表值，以及准备了BaseTransform用于图像预处理。

#     处理图像：
#         它列出了demo/目录中的所有图像，并在循环中处理每张图像。
#         对于每张图像，它使用OpenCV函数打开并显示图像。
#         然后使用Detector检测图像中的仪表区域。如果没有检测到仪表，它将继续处理下一张图像。
#         如果检测到仪表，每个仪表图像将单独处理：它被转换（可能是调整大小和标准化），转换成PyTorch张量，并移动到适当的设备。
#         处理后的图像随后通过文本检测器进行预测，以获取仪表的指针、刻度盘、文本和标准点的预测结果。

#     解码和显示结果：
#         脚本解码OCR预测为人类可读的文本，并处理显示和解释检测到的仪表读数。
#         对于每个检测到的仪表，它从其张量表示重建原始图像，应用后处理，并使用MeterReader从检测到的指针、刻度盘、文本和标准点解释仪表读数。

# 总而言之，这段代码是一个系统的一部分，设计用于自动从图像中读取和解释仪表读数。它涉及图像处理、对象检测和文本识别，以从公用事业仪表的照片中提取有用信息。
import os
import cv2
import numpy as np
from util.augmentation import BaseTransform
from util.config import config as cfg, update_config, print_config
from util.option import BaseOptions
from network.textnet import TextNet
from util.detection_mask import TextDetector as TextDetector_mask
import torch
from util.misc import to_device
from util.read_meter import MeterReader
from util.converter import keys,StringLabelConverter
from get_meter_area import  Detector

# parse arguments

option = BaseOptions()
args = option.initialize()

update_config(cfg, args)
print_config(cfg)

predict_dir='demo/'

model = TextNet(is_training=False, backbone=cfg.net)
model_path = os.path.join(cfg.save_dir, cfg.exp_name,
                          'textgraph_{}_{}.pth'.format(model.backbone_name, cfg.checkepoch))
model.load_model(model_path)
model = model.to(cfg.device)
converter=StringLabelConverter(keys)

det=Detector()
detector = TextDetector_mask(model)
meter = MeterReader()
transform=BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds)


image_list=os.listdir(predict_dir)



for index in image_list:
    print("**************",index)
    image = cv2.imread(predict_dir+index)
    
    cv2.imshow("det1",image)
    cv2.waitKey(0)

    
    # detect meter area
    image, image_info, digital_list, meter_list = det.detect(image, index)

    if len(meter_list)==0:
        print("no detected meter")
        continue
    else:
        for i in meter_list:
            cv2.imshow("det",i)
            cv2.waitKey(0)

            image,_=transform(i)
            image = image.transpose(2, 0, 1)
            image=torch.from_numpy(image).unsqueeze(0)
            image=to_device(image)

            output = detector.detect1(image)

            pointer_pred, dail_pred, text_pred, preds, std_points = output['pointer'], output['dail'], output['text'], output['reco'], output['std']

            # decode predicted text
            pred, preds_size = preds
            if pred != None:
                _, pred = pred.max(2)
                pred = pred.transpose(1, 0).contiguous().view(-1)
                pred_transcripts = converter.decode(pred.data, preds_size.data, raw=False)
                pred_transcripts = [pred_transcripts] if isinstance(pred_transcripts, str) else pred_transcripts
                # print("preds", pred_transcripts)
            else:
                pred_transcripts = None


            img_show = image[0].permute(1, 2, 0).cpu().numpy()
            img_show = ((img_show * cfg.stds + cfg.means) * 255).astype(np.uint8)


            result = meter(img_show, pointer_pred, dail_pred, text_pred, pred_transcripts, std_points)

