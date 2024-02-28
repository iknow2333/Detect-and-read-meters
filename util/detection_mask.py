# 这段代码定义了一个名为 TextDetector 的类，用于执行文本检测任务。这个类封装了模型的使用，使得可以直接通过图像数据来检测文本。下面是对这个类和它的方法的详细解释：
# 初始化（init）:

#     接收一个预先训练好的模型作为参数。
#     将这个模型赋值给类的内部变量 self.model。
#     将模型设置为评估模式，这意味着在这个模式下，模型的行为会与训练时不同（例如，禁用dropout等）。

# 检测（detect1）:

#     这个方法接受一个图像张量 image 作为输入。
#     使用 torch.no_grad() 上下文管理器来禁用梯度计算，这是因为在推理（或评估）阶段不需要反向传播，禁用梯度计算可以减少内存使用量并提高计算速度。
#     调用模型的 forward_test 方法来获取预测结果。这个方法可能是为了区分训练和测试阶段而特别定义的，它返回五个输出：
#         pointer_pred：指针区域的预测。
#         dail_pred：表盘区域的预测。
#         text_pred：文本区域的预测。
#         pred_recog：识别的文本或字符。
#         std_points：可能是用于标准化或校准的点。
#     将第一个图像张量从GPU（如果使用的是CUDA）转移到CPU，并转换为NumPy数组，这样它就可以用于后续的图像处理或显示。
#     构建一个字典 output，包含模型的所有输出和原始图像，然后返回这个字典。

# 这个 TextDetector 类使得文本检测过程简化为创建一个实例并调用一个方法，这对于将文本检测模型集成到应用程序中非常有用。它隐藏了模型推理的细节，提供了一个简单的接口来获取文本检测的结果。
import numpy as np
import cv2
from util.config import config as cfg
import torch


class TextDetector(object):

    def __init__(self, model):
        self.model = model

        # evaluation mode
        model.eval()

    def detect1(self, image):

        with torch.no_grad():
            # get model output
            pointer_pred,dail_pred,text_pred,pred_recog, std_points= self.model.forward_test(image)


        image = image[0].data.cpu().numpy()


        output = {
            'image': image,
            'pointer': pointer_pred,
            'dail': dail_pred,
            'text': text_pred,
            'reco':pred_recog,
            'std':std_points
        }
        return output




















