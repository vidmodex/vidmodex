"""
Inspired by
https://github.com/PeizeSun/SparseR-CNN/blob/dff4c43a9526a6d0d2480abc833e78a7c29ddb1a/detectron2/config/defaults.py
"""
from fvcore.common.config import CfgNode as CN


def fill_SE_config(conf, input_channels,
                   out_channels,
                   expanded_channels,
                   kernel_size,
                   stride,
                   padding,
                   padding_avg,
                   ):
    conf.expanded_channels = expanded_channels
    conf.padding_avg = padding_avg
    fill_conv(conf, input_channels,
              out_channels,
              kernel_size,
              stride,
              padding,
              )


def fill_conv(conf, input_channels,
              out_channels,
              kernel_size,
              stride,
              padding, ):
    conf.input_channels = input_channels
    conf.out_channels = out_channels
    conf.kernel_size = kernel_size
    conf.stride = stride
    conf.padding = padding


_C = CN()

_C.MODEL = CN()

###################
#### MoViNetA2 ####
###################

_C.MODEL.MoViNetA2 = CN()
_C.MODEL.MoViNetA2.name = "A2"
_C.MODEL.MoViNetA2.weights = "https://github.com/Atze00/MoViNet-pytorch/blob/main/weights/modelA2_statedict_v3?raw=true"
_C.MODEL.MoViNetA2.stream_weights = "https://github.com/Atze00/MoViNet-pytorch/blob/main/weights/modelA2_stream_statedict_v3?raw=true"

_C.MODEL.MoViNetA2.conv1 = CN()
fill_conv(_C.MODEL.MoViNetA2.conv1, 3, 16, (1, 3, 3), (1, 2, 2), (0, 1, 1))

_C.MODEL.MoViNetA2.blocks = [[CN() for _ in range(3)],
                             [CN() for _ in range(5)],
                             [CN() for _ in range(5)],
                             [CN() for _ in range(6)],
                             [CN() for _ in range(7)]]

# Block2
fill_SE_config(_C.MODEL.MoViNetA2.blocks[0][0], 16, 16, 40, (1, 5, 5), (1, 2, 2), (0, 2, 2), (0, 1, 1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[0][1], 16, 16, 40, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[0][2], 16, 16, 64, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1))

# block 3
fill_SE_config(_C.MODEL.MoViNetA2.blocks[1][0], 16, 40, 96, (3, 3, 3), (1, 2, 2), (1, 1, 1), (0, 1, 1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[1][1], 40, 40, 120, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[1][2], 40, 40, 96, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[1][3], 40, 40, 96, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[1][4], 40, 40, 120, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1))

# block 4
fill_SE_config(_C.MODEL.MoViNetA2.blocks[2][0], 40, 72, 240, (5, 3, 3), (1, 2, 2), (2, 1, 1), (0, 1, 1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[2][1], 72, 72, 160, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[2][2], 72, 72, 240, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[2][3], 72, 72, 192, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[2][4], 72, 72, 240, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1))

# block 5
fill_SE_config(_C.MODEL.MoViNetA2.blocks[3][0], 72, 72, 240, (5, 3, 3), (1, 1, 1), (2, 1, 1), (0, 1, 1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[3][1], 72, 72, 240, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[3][2], 72, 72, 240, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[3][3], 72, 72, 240, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[3][4], 72, 72, 144, (1, 5, 5), (1, 1, 1), (0, 2, 2), (0, 1, 1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[3][5], 72, 72, 240, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1))

# block 6
fill_SE_config(_C.MODEL.MoViNetA2.blocks[4][0], 72, 144, 480, (5, 3, 3), (1, 2, 2), (2, 1, 1), (0, 1, 1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[4][1], 144, 144, 384, (1, 5, 5), (1, 1, 1), (0, 2, 2), (0, 1, 1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[4][2], 144, 144, 384, (1, 5, 5), (1, 1, 1), (0, 2, 2), (0, 1, 1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[4][3], 144, 144, 480, (1, 5, 5), (1, 1, 1), (0, 2, 2), (0, 1, 1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[4][4], 144, 144, 480, (1, 5, 5), (1, 1, 1), (0, 2, 2), (0, 1, 1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[4][5], 144, 144, 480, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[4][6], 144, 144, 576, (1, 3, 3), (1, 1, 1), (0, 1, 1), (0, 1, 1))

_C.MODEL.MoViNetA2.conv7 = CN()
fill_conv(_C.MODEL.MoViNetA2.conv7, 144, 640, (1, 1, 1), (1, 1, 1), (0, 0, 0))

_C.MODEL.MoViNetA2.dense9 = CN()
_C.MODEL.MoViNetA2.dense9.hidden_dim = 2048


###################
#### MoViNetA5 ####
###################


_C.MODEL.MoViNetA5 = CN()
_C.MODEL.MoViNetA5.name = "A5"
_C.MODEL.MoViNetA5.weights = "https://github.com/Atze00/MoViNet-pytorch/blob/main/weights/modelA5_statedict_v3?raw=true"
_C.MODEL.MoViNetA5.conv1 = CN()
fill_conv(_C.MODEL.MoViNetA5.conv1, 3,24,(1,3,3),(1,2,2),(0,1,1))


_C.MODEL.MoViNetA5.blocks = [ [CN() for _ in range(6)],
                              [CN() for _ in range(11)],
                              [CN() for _ in range(13)],
                              [CN() for _ in range(11)],
                              [CN() for _ in range(18)]]

#Block2
fill_SE_config(_C.MODEL.MoViNetA5.blocks[0][0], 24, 24, 64, (1,5,5), (1,2,2), (0,2,2), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA5.blocks[0][1], 24, 24, 64, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA5.blocks[0][2], 24, 24, 96, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA5.blocks[0][3], 24, 24, 64, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA5.blocks[0][4], 24, 24, 96, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA5.blocks[0][5], 24, 24, 64, (3,3,3), (1,1,1), (1,1,1), (0,1,1))

#block 3
fill_SE_config(_C.MODEL.MoViNetA5.blocks[1][0], 24, 64, 192, (5,3,3), (1,2,2), (2,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA5.blocks[1][1], 64, 64, 152, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA5.blocks[1][2], 64, 64, 152, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA5.blocks[1][3], 64, 64, 152, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA5.blocks[1][4], 64, 64, 192, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA5.blocks[1][5], 64, 64, 192, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA5.blocks[1][6], 64, 64, 192, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA5.blocks[1][7], 64, 64, 152, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA5.blocks[1][8], 64, 64, 152, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA5.blocks[1][9], 64, 64, 192, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA5.blocks[1][10], 64, 64, 192, (3,3,3), (1,1,1), (1,1,1), (0,1,1))

#block 4
fill_SE_config(_C.MODEL.MoViNetA5.blocks[2][0], 64, 112, 376, (5,3,3), (1,2,2), (2,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA5.blocks[2][1], 112, 112, 224, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA5.blocks[2][2], 112, 112, 376, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA5.blocks[2][3], 112, 112, 376, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA5.blocks[2][4], 112, 112, 296, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA5.blocks[2][5], 112, 112, 376, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA5.blocks[2][6], 112, 112, 224, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA5.blocks[2][7], 112, 112, 376, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA5.blocks[2][8], 112, 112, 376, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA5.blocks[2][9], 112, 112, 296, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA5.blocks[2][10], 112, 112, 376, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA5.blocks[2][11], 112, 112, 376, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA5.blocks[2][12], 112, 112, 376, (3,3,3), (1,1,1), (1,1,1), (0,1,1))

#block 5
fill_SE_config(_C.MODEL.MoViNetA5.blocks[3][0], 112, 120, 376, (5,3,3), (1,1,1), (2,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA5.blocks[3][1], 120, 120, 376, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA5.blocks[3][2], 120, 120, 376, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA5.blocks[3][3], 120, 120, 376, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA5.blocks[3][4], 120, 120, 224, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA5.blocks[3][5], 120, 120, 376, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA5.blocks[3][6], 120, 120, 376, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA5.blocks[3][7], 120, 120, 224, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA5.blocks[3][8], 120, 120, 376, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA5.blocks[3][9], 120, 120, 376, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA5.blocks[3][10], 120, 120, 376, (3,3,3), (1,1,1), (1,1,1), (0,1,1))

#block 6
fill_SE_config(_C.MODEL.MoViNetA5.blocks[4][0], 120 , 224, 744, (5,3,3), (1,2,2), (2,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA5.blocks[4][1], 224, 224, 744, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA5.blocks[4][2], 224, 224, 600, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA5.blocks[4][3], 224, 224, 600, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA5.blocks[4][4], 224, 224, 744, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA5.blocks[4][5], 224, 224, 744, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA5.blocks[4][6], 224, 224, 744, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA5.blocks[4][7], 224, 224, 896, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA5.blocks[4][8], 224, 224, 600, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA5.blocks[4][9], 224, 224, 600, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA5.blocks[4][10], 224, 224, 896, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA5.blocks[4][11], 224, 224, 744, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA5.blocks[4][12], 224, 224, 744, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA5.blocks[4][13], 224, 224, 896, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA5.blocks[4][14], 224, 224, 600, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA5.blocks[4][15], 224, 224, 600, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA5.blocks[4][16], 224, 224, 744, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA5.blocks[4][17], 224, 224, 744, (3,3,3), (1,1,1), (1,1,1), (0,1,1))

_C.MODEL.MoViNetA5.conv7= CN()
fill_conv(_C.MODEL.MoViNetA5.conv7, 224,992,(1,1,1),(1,1,1),(0,0,0))

_C.MODEL.MoViNetA5.dense9= CN()
_C.MODEL.MoViNetA5.dense9.hidden_dim = 2048