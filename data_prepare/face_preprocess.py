#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/07/14 13:47
# @Author   : WanDaoYi
# @FileName : face_preprocess.py
# ============================================

import cv2
import numpy as np
from utils import data_utils
from skimage import transform as trans


def preprocess(image, bounding_box=None, landmark=None, **kwargs):
    """
        图像预处理
    :param image: image or image path
    :param bounding_box:
    :param landmark:
    :param kwargs:
    :return:
    """

    # 如果 image 为 image_path, 则读取图像
    if isinstance(image, str):
        image = data_utils.read_image(image, **kwargs)
        pass

    tran_param = None
    # image_shape = [112, 112, 3]
    image_shape = kwargs.get("image_shape", [])
    if len(image_shape) > 0:
        if len(image_shape) == 1:
            image_shape = [image_shape[0], image_shape[0]]
            pass

        assert image_shape[0] == 112 or image_shape[1] == 96
        pass

    if landmark is not None:
        assert len(image_shape) >= 2, "image shape is not more than 2"
        # Source coordinates; 人脸关键点的源坐标(共 5 个点): 2 眼睛点; 1 鼻子点; 2 嘴角点
        src = np.array([[30.2946, 51.6963],
                        [65.5318, 51.5014],
                        [48.0252, 71.7366],
                        [33.5493, 92.3655],
                        [62.7299, 92.2041]], dtype=np.float32)

        if image_shape[1] == 112:
            src[:, 0] += 8.0
            pass

        # Destination coordinates
        # 将检测到的目标关键点转为 坐标点;
        # 前面 5 个值为 x_1 ~ x_5; 后面 5 个值为 y_1 ~ y_5; 例如:
        # [250.46382, 331.79272,            [[250.46382, 180.89687],
        #  296.87363, 228.75145,             [331.79272, 199.45087],
        #  288.48895, 180.89687,   --->      [296.87363, 259.50702],
        #  199.45087, 259.50702,             [228.75145, 288.98694],
        #  288.98694, 301.3608]              [288.48895, 301.3608]]
        dst = landmark.astype(np.float32)
        # 构建 E 矩阵
        # [[1., 0., 0.],
        #  [0., 1., 0.],
        #  [0., 0., 1.]]
        tran_sim = trans.SimilarityTransform()
        # 对目标关键点检测结果与源坐标点 计算 仿射 矩阵, 例如:
        # [[0.36391862, 0.1036921, -71.11934277],
        #  [-0.1036921, 0.36391862, 11.32058247],
        #  [0., 0., 1.]]
        tran_sim.estimate(dst, src)
        # 获取放射变换的 [[a_11, a_12, a_13], [a_21, a_22, a_23]]
        # 例如: [[0.36391862, 0.1036921, -71.11934277], [-0.1036921, 0.36391862, 11.32058247]]
        tran_param = tran_sim.params[0: 2, :]
        pass

    if tran_param is None:
        if bounding_box is None:
            det = np.zeros(4, dtype=np.int32)
            det[0] = int(image.shape[1] * 0.0625)
            det[1] = int(image.shape[0] * 0.0625)
            det[2] = image.shape[1] - det[0]
            det[3] = image.shape[0] - det[1]
            pass
        else:
            det = bounding_box
            pass

        margin = kwargs.get('margin', 44)
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, image.shape[1])
        bb[3] = np.minimum(det[3] + margin / 2, image.shape[0])
        ret = image[bb[1]: bb[3], bb[0]: bb[2], :]

        if len(image_shape) > 0:
            ret = cv2.resize(ret, (image_shape[1], image_shape[0]))
            pass
        return ret
        pass
    else:
        assert len(image_shape) >= 2, "image shape is not more than 2"
        # 仿射变换: 可同时对图片做裁剪、旋转、转换、模式调整等多重操作
        # x = a_11 * x_0 + a_12 * y_0 + a_13
        # y = a_21 * x_0 + a_22 * y_0 + a_23
        warped = cv2.warpAffine(image, tran_param, (image_shape[1], image_shape[0]), borderValue=0.0)
        return warped
        pass

    pass
