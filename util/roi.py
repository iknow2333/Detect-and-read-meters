    # roi_transform 函数：
    #     输入参数包括特征图（feature），一个代表ROI的边界框（box），以及输出图像的大小（size）。
    #     该函数首先根据给定的ROI坐标（边界框）和目标大小，计算从原图到目标图像大小的仿射变换矩阵。
    #     使用 cv2.getAffineTransform 计算仿射变换矩阵，然后通过 param2theta 函数转换该矩阵，使其适用于PyTorch的仿射网格函数。
    #     应用这个变换矩阵，使用 torch.nn.functional.affine_grid 和 torch.nn.functional.grid_sample 对原始特征图进行变换，得到旋转并裁剪后的特征图。
    #     最后，将变换后的特征图转换为灰度图。

    # param2theta 函数：
    #     这个函数用于将OpenCV的仿射变换矩阵转换成PyTorch的theta参数，PyTorch用这些参数在仿射变换中生成网格。
    #     它首先将仿射变换矩阵扩展为3x3矩阵，然后求逆（因为在PyTorch中使用的是反向映射）。
    #     接着，它根据原图和目标图像的尺寸调整变换矩阵的元素。

    # batch_roi_transform 函数：
    #     这个函数对一批图像和对应的ROI边界框进行处理。
    #     它遍历每个图像和相应的边界框，使用 roi_transform 函数对每个ROI进行变换，并收集所有处理后的ROI。
    #     最后，将所有ROI堆叠成一个新的批次张量。

    # rgb_to_grayscale 函数：
    #     这个函数将RGB图像转换为灰度图像，使用的是ITU-R 601-2 luma转换公式。
    #     它检查输入图像是否为3通道，然后应用转换公式。
import cv2
import torch
import numpy as np


def roi_transform(feature, box, size=(32, 180)):
    resize_h, resize_w = size
    x1, y1, x2, y2, x3, y3, x4, y4 = box
    # rotated_rect = cv2.minAreaRect(np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]]))
    # box_w, box_h = rotated_rect[1][0], rotated_rect[1][1]

    width = feature.shape[2]
    height = feature.shape[1]

    mapped_x1, mapped_y1 = (0, 0)
    mapped_x4, mapped_y4 = (0, resize_h)

    mapped_x2, mapped_y2 = (resize_w, 0)

    src_pts = np.float32([(x1, y1), (x2, y2), (x4, y4)])
    dst_pts = np.float32([
        (mapped_x1, mapped_y1), (mapped_x2, mapped_y2), (mapped_x4, mapped_y4)
    ])

    affine_matrix = cv2.getAffineTransform(src_pts.astype(np.float32), dst_pts.astype(np.float32))
    affine_matrix = param2theta(affine_matrix, width, height)

    affine_matrix *= 1e20  # cancel the error when type conversion
    affine_matrix = torch.tensor(affine_matrix, device=feature.device, dtype=torch.float)
    affine_matrix /= 1e20


    grid = torch.nn.functional.affine_grid(affine_matrix.unsqueeze(0), feature.unsqueeze(0).size())
    feature_rotated = torch.nn.functional.grid_sample(feature.unsqueeze(0), grid)
    feature_rotated = feature_rotated[:, :, 0:resize_h, 0:resize_w]

    feature_rotated = feature_rotated.squeeze(0)
    gray_scale_img = rgb_to_grayscale(feature_rotated).unsqueeze(0)

    return gray_scale_img


def param2theta(param, w, h):
    param = np.vstack([param, [0, 0, 1]])
    param = np.linalg.inv(param)

    theta = np.zeros([2, 3])
    theta[0, 0] = param[0, 0]
    theta[0, 1] = param[0, 1] * h / w
    theta[0, 2] = param[0, 2] * 2 / w + theta[0, 0] + theta[0, 1] - 1
    theta[1, 0] = param[1, 0] * w / h
    theta[1, 1] = param[1, 1]
    theta[1, 2] = param[1, 2] * 2 / h + theta[1, 0] + theta[1, 1] - 1
    return theta


def batch_roi_transform(feature_map, boxes, mapping, size=(32, 180)):
    rois = []
    for img_index, box in zip(mapping, boxes):
        feature = feature_map[img_index]
        rois.append(roi_transform(feature, box, size))
    rois = torch.stack(rois, dim=0)
    return rois


def rgb_to_grayscale(img):
    # type: (Tensor) -> Tensor
    """Convert the given RGB Image Tensor to Grayscale.
    For RGB to Grayscale conversion, ITU-R 601-2 luma transform is performed which
    is L = R * 0.2989 + G * 0.5870 + B * 0.1140
    Args:
        img (Tensor): Image to be converted to Grayscale in the form [C, H, W].
    Returns:
        Tensor: Grayscale image.
    """
    if img.shape[0] != 3:
        raise TypeError('Input Image does not contain 3 Channels')

    return (0.2989 * img[0] + 0.5870 * img[1] + 0.1140 * img[2]).to(img.dtype)
