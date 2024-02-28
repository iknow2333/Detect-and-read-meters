# 这段代码定义了一个用于深度学习项目的配置管理。这里使用 EasyDict 来创建一个易于访问的配置字典 config，并提供了默认设置。接下来，我将详细解释代码的关键部分：

#     配置参数设置：
#         config.batch_size: 设置批处理大小。
#         config.max_epoch: 设置训练的最大轮次。
#         config.start_epoch: 设置训练开始的轮次，默认为0。
#         config.lr: 设置学习率。
#         config.cuda: 指示是否使用CUDA（GPU加速）。
#         config.input_size: 设置输入图像的大小。
#         config.max_annotation 和 config.max_roi: 设置每个图像的最大多边形标注和区域感兴趣的数量。
#         config.max_points: 每个多边形的最大点数。
#         config.use_hard: 是否使用难以识别的样本。
#         config.tr_thresh 和 config.tcl_thresh: 设置文本区域和文本连通区域的阈值。
#         config.expend: 设置后处理中的扩展比率。
#         config.k_at_hop 和 config.active_connection: 用于定义图结构的参数。
#         config.graph_link 和 config.link_thresh: 用于定义是否使用图连接和图连接的阈值。

#     函数 update_config：
#         此函数用于将额外的配置（通常是从命令行或外部文件读取的）更新到现有配置中。
#         它遍历额外配置中的所有项目，并将其添加或更新到主配置中。

#     函数 print_config：
#         此函数用于打印当前的配置设置。
#         遍历配置字典并打印每个键值对。

# 这种配置管理方式在机器学习和深度学习项目中非常常见，因为它提供了一种灵活的方式来维护和更新实验设置。通过将配置参数集中在一个地方，可以轻松调整实验参数，同时确保代码的整洁和可维护性。此外，使用 EasyDict 允许通过属性访问而不是字典键访问，这使得代码更加直观和易于编写。
from easydict import EasyDict
import torch
import os

config = EasyDict()

#config.gpu = "0,1"

# dataloader jobs number
#config.num_workers = 8

# batch_size
config.batch_size = 1

# training epoch number
config.max_epoch = 200

config.start_epoch = 0

# learning rate
config.lr = 1e-4

# using GPU
config.cuda = True

config.k_at_hop1 = 10

config.output_dir = 'output'

config.input_size = 640

# max polygon per image
config.max_annotation = 200

# max polygon per image
# synText, total-text:600; CTW1500: 1200; icdar: ; MLT: ; TD500: .
config.max_roi = 600

# max point per polygon
config.max_points = 20

# use hard examples (annotated as '#')
config.use_hard = True

# demo tr threshold
config.tr_thresh = 0.9  #0.9

# demo tcl threshold
config.tcl_thresh = 0.4
#0.9 0.5 81.14

# expand ratio in post processing
config.expend = -0.05 #0.15

# k-n graph
config.k_at_hop = [8, 8]

# unn connect
config.active_connection = 3

config.graph_link = False  ### for Total-text icdar15 graph_link = False; forTD500 and CTW1500, graph_link = True
config.link_thresh = 0.85 #0.9

def update_config(config, extra_config):
    for k, v in vars(extra_config).items():
        config[k] = v
    # print(config.gpu)
    config.device = torch.device('cuda') if config.cuda else torch.device('cpu')


def print_config(config):
    print('==========Options============')
    for k, v in config.items():
        print('{}: {}'.format(k, v))
    print('=============End=============')
