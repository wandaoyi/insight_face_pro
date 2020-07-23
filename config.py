#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/06/23 16:57
# @Author   : WanDaoYi
# @FileName : config.py
# ============================================

import os
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by: from config import cfg

cfg = __C

# 公共配置文件
__C.COMMON = edict()

# 相对路径 当前路径
__C.COMMON.RELATIVE_PATH = "./"

__C.COMMON.DATA_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "data")

__C.COMMON.DETECT_FACE_MODEL_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "./models/detect_model")
__C.COMMON.DETECT_MODEL_NAME_LIST = ["det1.npy", "det2.npy", "det3.npy"]

# 文件后缀名
__C.COMMON.JSON_SUFFIX = ".json"
__C.COMMON.PNG_SUFFIX = ".png"
__C.COMMON.TXT_SUFFIX = ".txt"
__C.COMMON.PARAMS_SUFFIX = ".params"
__C.COMMON.IDX_SUFFIX = ".idx"

__C.COMMON.IMAGE_SUFFIX_LIST = [".png", ".jpg", ".jpeg"]

# True 为彩色; False 为 灰度图
__C.COMMON.COLOR_MODE_FLAG = True

# model-symbol.json 文件名
__C.COMMON.MODEL_SYMBOL_FILE = "model-symbol.json"

# 人脸图像的 [h, w, c] 尺度。解压作者的数据集中 有 property 文件
# 如 property 文件中的信息: 85164,112,112
# 第一个数值为 identity 值，第二、第三个数值为图像的 image_size, 即这里的 face_shape
__C.COMMON.FACE_SHAPE = [112, 112, 3]

# 输出特征向量的维度 embedding_feature
__C.COMMON.EMBEDDING_SIZE = 128

# gpu 设备号
__C.COMMON.GPU_ID = 0

# 结构 字典，结构内容有点多，用到什么调什么。
__C.COMMON.STRUCTURE_DICT = {"network": {"r100": {"net_name": "f_res_net",
                                                  "per_batch_size": 32,
                                                  "num_layers": 100,
                                                  "net_output": "E",
                                                  "net_unit": 3},
                                         "r100fc": {"net_name": "f_res_net",
                                                    "per_batch_size": 32,
                                                    "num_layers": 100,
                                                    "net_output": "FC",
                                                    "net_unit": 3},
                                         "r50": {"net_name": "f_res_net",
                                                 "per_batch_size": 32,
                                                 "num_layers": 50,
                                                 "net_output": "E",
                                                 "net_unit": 3},
                                         "r50v1": {"net_name": "f_res_net",
                                                   "per_batch_size": 32,
                                                   "num_layers": 50,
                                                   "net_output": "E",
                                                   "net_unit": 1},
                                         "d169": {"net_name": "f_dense_net",
                                                  "per_batch_size": 32,
                                                  "num_layers": 169,
                                                  "net_output": "E",
                                                  "dense_net_dropout": 0.0,
                                                  "net_unit": 3},
                                         "d201": {"net_name": "f_dense_net",
                                                  "per_batch_size": 32,
                                                  "num_layers": 201,
                                                  "net_output": "E",
                                                  "dense_net_dropout": 0.0,
                                                  "net_unit": 3},
                                         "y1": {"net_name": "f_mobile_face_net",
                                                "per_batch_size": 32,
                                                "emb_size": 128,
                                                "net_output": "GDC",
                                                "net_unit": 3},
                                         "y2": {"net_name": "f_mobile_face_net",
                                                "per_batch_size": 32,
                                                "emb_size": 256,
                                                "net_output": "GDC",
                                                "net_blocks": [2, 8, 16, 4],
                                                "net_unit": 3},
                                         "m1": {"net_name": "f_mobile_net",
                                                "per_batch_size": 32,
                                                "emb_size": 256,
                                                "net_output": "GDC",
                                                "net_multiplier": 1.0,
                                                "net_unit": 3},
                                         "m05": {"net_name": "f_mobile_net",
                                                 "per_batch_size": 32,
                                                 "emb_size": 256,
                                                 "net_output": "GDC",
                                                 "net_multiplier": 0.5,
                                                 "net_unit": 3},
                                         "m_nas": {"net_name": "f_m_nas_net",
                                                   "per_batch_size": 32,
                                                   "emb_size": 256,
                                                   "net_output": "GDC",
                                                   "net_multiplier": 1.0,
                                                   "net_unit": 3},
                                         "m_nas05": {"net_name": "f_m_nas_net",
                                                     "per_batch_size": 32,
                                                     "emb_size": 256,
                                                     "net_output": "GDC",
                                                     "net_multiplier": 0.5,
                                                     "net_unit": 3},
                                         "m_nas025": {"net_name": "fm_nas_net",
                                                      "per_batch_size": 32,
                                                      "emb_size": 256,
                                                      "net_output": "GDC",
                                                      "net_multiplier": 0.25,
                                                      "net_unit": 3}
                                         },
                             "data_set": {"e_more": {"data_set": "e_more",
                                                     "rec_data_path": "./data/train_data/train.rec",
                                                     "idx_data_path": "./data/train_data/train.idx",
                                                     "bin_data_file_path": "./data/val_data",
                                                     # class_num 来自 property，为人脸id数目，为了能够较好的拟合数据
                                                     "class_num": 7,
                                                     # "class_num": 85164,
                                                     "val_targets": ["val_lfw.bin"]},
                                          "retina": {"data_set": "e_more",
                                                     "rec_data_path": "J:/face_recognize/faces_ms1m_112x112/train.rec",
                                                     "idx_data_path": "J:/face_recognize/faces_ms1m_112x112/train.idx",
                                                     "bin_data_file_path": "J:/face_recognize/faces_ms1m_112x112",
                                                     "class_num": 502,
                                                     "val_targets": ["lfw.bin", "cfp_fp.bin"]}
                                          },
                             "loss": {"softmax": {"loss_name": "softmax"},
                                      "nsoftmax": {"loss_name": "margin_softmax",
                                                   "loss_s": 64.0,
                                                   "loss_m1": 1.0,
                                                   "loss_m2": 0.0,
                                                   "loss_m3": 0.0},
                                      "arcface": {"loss_name": "margin_softmax",
                                                  "loss_s": 64.0,
                                                  "loss_m1": 1.0,
                                                  "loss_m2": 0.5,
                                                  "loss_m3": 0.0},
                                      "cosface": {"loss_name": "margin_softmax",
                                                  "loss_s": 64.0,
                                                  "loss_m1": 1.0,
                                                  "loss_m2": 0.0,
                                                  "loss_m3": 0.35},
                                      "combined": {"loss_name": "margin_softmax",
                                                   "loss_s": 64.0,
                                                   "loss_m1": 1.0,
                                                   "loss_m2": 0.3,
                                                   "loss_m3": 0.2},
                                      "triplet": {"loss_name": "triplet",
                                                  "images_per_identity": 5,
                                                  "triplet_alpha": 0.3,
                                                  "triplet_bag_size": 7200,
                                                  "triplet_max_ap": 0.0,
                                                  "lr": 0.05},
                                      "a_triplet": {"loss_name": "a_triplet",
                                                    "images_per_identity": 5,
                                                    "triplet_alpha": 0.35,
                                                    "triplet_bag_size": 7200,
                                                    "triplet_max_ap": 0.0,
                                                    "lr": 0.05}
                                      }
                             }

# ################################################
# ################################################

# 数据蒸馏 和 生成网络数据的 配置文件
__C.DATA_SET = edict()

# 人脸检测与矫正路径
__C.DATA_SET.INPUT_ALIGN_IMAGE_FILE_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "data/align_input")
__C.DATA_SET.OUTPUT_ALIGN_IMAGE_FILE_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "data/align_output")

# 人脸分类路径
__C.DATA_SET.OUTPUT_FACE_CLASSIFY_FOLD_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "data/face_classify")
__C.DATA_SET.OUTPUT_INFO_FACE_CLASSIFY_FOLD_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "info/face_classify")
__C.DATA_SET.OUTPUT_NOT_RECOGNIZE_FACE_FOLD_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "output/not_recognize")

# 人脸图像聚类路径
__C.DATA_SET.OUTPUT_CLUSTER_NOISE_FACE_FOLD_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "output/cluster_noise_face")
__C.DATA_SET.OUTPUT_SAME_FACE_FOLD_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "output/same_image")
__C.DATA_SET.OUTPUT_INFO_NOISE_FACE_FOLD_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "info/cluster_noise")
__C.DATA_SET.OUTPUT_INFO_SAME_FACE_FOLD_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "info/same_image")

# 数据划分文件路径
__C.DATA_SET.TRAIN_DATA_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "info/train_data.txt")
__C.DATA_SET.VAL_DATA_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "info/val_data.txt")
__C.DATA_SET.PAIR_DATA_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "info/pair.txt")
__C.DATA_SET.VAL_BIN_FOLD_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "data/val_data")
__C.DATA_SET.TRAIN_FOLD_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "data/train_data")
__C.DATA_SET.IDENTITY_PROPERTY_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "data/train_data/property")

__C.DATA_SET.BIN_NAME = "val_lfw.bin"
__C.DATA_SET.REC_NAME = "train.rec"
__C.DATA_SET.IDX_NAME = "train.idx"

# 人脸识别的阈值 (min_threshold_value, max_threshold_value)
# 当 cos_value > max_threshold_value 为同一个人
# 当 cos_value < min_threshold_value 为不同的人
# 当 min_threshold_value < cos_value < max_threshold_value 为识别不出来
# 这个值，是根据模型来设置的，调到更适应模型来对人脸进行分类
__C.DATA_SET.RECOGNIZE_THRESHOLD_VALUE = (0.67, 0.75)

# 判断两张人脸图像是否相同，大于阈值的则为相同的图像
__C.DATA_SET.SAME_THRESHOLD_VALUE = 0.999
# 是否递归查看文件夹，True 为递归
__C.DATA_SET.RECURSIVE_FLAG = True

# 复制目标图像后，是否删除原图像，True 为删除
__C.DATA_SET.DELETE_FLAG = True

# 训练集和验证集占数据的百分比
__C.DATA_SET.TRAIN_PERCENT = 0.7
__C.DATA_SET.VAL_PERCENT = 0.3

# 正样本数量
__C.DATA_SET.POSITIVE_SAMPLE_NUM = 30

# 负样本为正样本的 几倍
__C.DATA_SET.NEGATIVE_TIMES_POSITIVE = 2

# 方法运行模式:
# 0 对人物图像进行单独检测和矫正人脸
# 1 单独对检测和矫正后的人脸进行分类(前提是要有检测和矫正后 112 x 112 大小的人脸)
# 2 单独对分类后的人脸进行聚类处理(前提是需要有分类后的 112 x 112 的人脸)
# 3 对人物图像进行人脸检测和矫正后，再对人脸进行分类
# 4 对检测和矫正后的人脸进行分类，再对人脸进行聚类
# 5 对人物图像进行人脸检测和矫正，之后对人脸进行分类，再之后对人脸进行聚类
__C.DATA_SET.FUN_MODE = 1

# 训练配置文件
__C.TRAIN = edict()

__C.TRAIN.INPUT_DATA_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "data/train_data/train.rec")

__C.TRAIN.MODEL_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "models/model-y1/model-0000.params")

__C.TRAIN.SAVE_MODEL_FILE_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "save_train_model")

# model name prefix
__C.TRAIN.MODEL_NAME_PREFIX = "model"

# 网络结构选择
__C.TRAIN.MODEL_NET = "y1"

__C.TRAIN.DATA_SET = "e_more"

# 计算 loss 的方式
__C.TRAIN.FACE_LOSS = "arcface"

__C.TRAIN.NET_INPUT = 1
__C.TRAIN.NET_OUTPUT = "E"

# 输出层，链接层的类型，如"GDC"也是其中一种，
# 具体查看 ./utils/model_utils.py
__C.TRAIN.NET_BLOCKS = [1, 4, 6, 2]

# 键值存储
__C.TRAIN.KV_STORE = "device"

# 网络的激活函数
__C.TRAIN.NET_ACT = "prelu"

# 学习率的倍数
__C.TRAIN.FC7_LR_MULTIPLE = 1.0

# 权重刷衰减的倍数
__C.TRAIN.FC7_WD_MULTIPLE = 1.0

# fc7 是否有 bias 层
__C.TRAIN.FC7_NO_BIAS = False

# Focal loss，一种改进的交叉损失熵
__C.TRAIN.CE_LOSS = True

# 是否计算一个网络占用的浮点数内存
__C.TRAIN.COUNT_FLOPS_FLAG = True

__C.TRAIN.ME_MONGER_FLAG = False

# 学习率
__C.TRAIN.LEARNING_RATE = 0.000001

# 权值衰减值 Weight decay
# 权值衰减以惩罚大权值的正则化项来增强目标函数。惩罚与每个重量的大小的平方成比例。
# float 值，如果不适用，则 default=0.0
__C.TRAIN.WEIGHT_DECAY = 0.0005

# 动量因子 momentum
__C.TRAIN.MOMENTUM = 0.9

# 训练的 epochs
__C.TRAIN.EPOCHS = 100000

# 训练 这么多次就打印一下日志信息
__C.TRAIN.PRINT_STEP = 20
# 训练这么多次验证一次
__C.TRAIN.VAL_STEP = 200

# 每达到步数，学习率变为原来的百分之十
__C.TRAIN.LEARNING_RATE_STEP_LIST = [20000, 50000, 100000]

# 0: discard saving. 1: save when necessary. 2: always save
__C.TRAIN.SAVE_MODEL_NUM = 0

# 训练的最大步骤，如果大于 0 的时候，并且训练步骤大于 max_steps 时，结束训练
__C.TRAIN.MAX_STEPS = 0

# GPU 设备号
__C.TRAIN.GPU_NUM = "0"

# mxnet 需要的缓冲空间
__C.TRAIN.WORKSPACE = 256

__C.TRAIN.NET_SE = 0

# 数据是否进行随机进行镜像翻转
__C.TRAIN.DATA_RAND_MIRROR_FLAG = True

# 数据是否进行随机裁剪
__C.TRAIN.DATA_CROP_FLAG = False

# 是否检测输出的特征向量
__C.TRAIN.CHECK_FEATURE_FLAG = False

# 数据进行彩色增强
__C.TRAIN.DATA_COLOR_AUG = 0

# 表示每个人的图像数目要大于该值才进行训练
__C.TRAIN.DATA_IMAGES_FILTER = 0

# ################################################
# ################################################

# 验证配置文件
__C.VAL = edict()

# bin data file data path
__C.VAL.STANDARD_BIN_DATA_FILE_PATH = os.path.join(__C.COMMON.DATA_PATH, "standard_bin")

# 如果没指定数据 list=[], 则使用该文件目录下的所以数据
# __C.VAL.BIN_NAME_LIST = ["cfp_ff.bin", "cfp_fp.bin", "lfw.bin"]
__C.VAL.BIN_NAME_LIST = ["lfw.bin"]

# model file path
__C.VAL.MODEL_FILE_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "models/model-y1")

__C.VAL.BAD_CASE_OUTPUT_FILE_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "output/bad_case")
__C.VAL.BIN_OUTPUT_FILE_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "output/bin")
__C.VAL.OUTPUT_BIN_NAME = "temp.bin"

# 如果指定模型 list=[], 则使用该目录下所有的模型
# 模型名 list, for example: ["model-0000.params", "model-0001.params", "model-0005.params"]
__C.VAL.MODEL_NAME_LIST = ["model-0000.params"]

# 人脸身份 identity, 解压作者的数据集中 有 property 文件
# 如 property 文件中的信息: 85164,112,112
# 第一个数值为 identity 值，第二、第三个数值为图像的 image_size
__C.VAL.FACE_IDENTITY = 85164

__C.VAL.BATCH_SIZE = 8

# gap
__C.VAL.GAP = 10

# gap image shape
__C.VAL.IMAGE_SHAPE = [112, 224, 3]

# 模式,
# 0: 表示对数据集正常验证;
# 1: 表示查看较差的验证数据，FN, FP等; bad case
# 0 或 1 之外: 表示保存混合数据
__C.VAL.MODE = 0

# k 折 交叉验证
__C.VAL.K_FOLDS = 10

# 是否使用 flip 数据进行 acc 计算, True 为使用
__C.VAL.FLIP_FLAG = True

# 测试配置文件
__C.TEST = edict()

__C.TEST.MODEL_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "models/model-y1/model-0000.params")
