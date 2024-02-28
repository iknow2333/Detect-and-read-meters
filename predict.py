# 这段代码展示了一个使用深度学习模型来读取和识别场景中计量器（如电表或水表）读数的过程。以下是代码的详细解释：

#     初始化和配置：
#         使用 BaseOptions 类解析命令行参数。
#         更新配置 cfg 根据解析的参数。
#         打印配置，以便验证。

#     模型和工具初始化：
#         初始化 TextNet，一个用于文本检测和识别的深度学习模型。
#         从配置文件中加载模型权重。
#         初始化 StringLabelConverter，用于将文本标签转换为训练模型时使用的编码，以及反向转换。
#         初始化 TextDetector_mask 和 MeterReader 类，用于检测图像中的文本和仪表读数。

#     图像预处理和检测：
#         读取指定目录下的所有图像。
#         对每张图像进行预处理，包括缩放和标准化。
#         使用 TextDetector_mask 对预处理后的图像进行文本和仪表检测。
#         使用 StringLabelConverter 解码预测出的文本结果。

#     结果处理和展示：
#         将检测到的指针、表盘、文本区域及其识别结果用于读取和解释仪表读数。
#         打印解码后的文本预测结果。

# 代码中还包含了一些特定的操作，如使用自定义的 MeterReader_distro 来处理图像中的特定区域，这可能是为了处理图像中的分布情况或特定类型的仪表。

# 整个流程模拟了实际应用中的场景，即使用深度学习模型自动识别和读取实际环境中的仪表读数，这对于自动化监测和数据收集系统非常有用。
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
from meter_distro.ditro import MeterReader_distro


# parse arguments
option = BaseOptions()
args = option.initialize()

update_config(cfg, args)
print_config(cfg)
predict_dir='scene_image_data/detected_meter/pointer/'
# predict_dir='dem o_data/'

model = TextNet(is_training=False, backbone=cfg.net)
model_path = os.path.join(cfg.save_dir, cfg.exp_name,
                          'textgraph_{}_{}.pth'.format(model.backbone_name, cfg.checkepoch))
model.load_model(model_path)
model = model.to(cfg.device)
converter=StringLabelConverter(keys)

detector = TextDetector_mask(model)
meter = MeterReader()
meter_distro=MeterReader_distro()

image_list=os.listdir(predict_dir)
# print(image_list)
transform=BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds)

for i in image_list:
    print("**********",i)
    image = cv2.imread(predict_dir+i)
    restro_image, circle_center = meter_distro(image, i)

    image,_=transform(restro_image)
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
        print("preds", pred_transcripts)
    else:
        pred_transcripts = None


    img_show = image[0].permute(1, 2, 0).cpu().numpy()
    img_show = ((img_show * cfg.stds + cfg.means) * 255).astype(np.uint8)


    result = meter(img_show, pointer_pred, dail_pred, text_pred, pred_transcripts, std_points)

