#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/07/13 14:21
# @Author   : WanDaoYi
# @FileName : align_lfw.py
# ============================================

import os
import cv2
import numpy as np
from scipy import misc
import tensorflow as tf
from utils import data_utils
from face_detector import detect_face
from data_prepare import face_preprocess


def do_align_lfw(input_file_path, output_file_path, model_file_path, model_name_list,
                 image_shape=(112, 112, 3), recursive=False, suffix_info_list=()):
    """

    :param input_file_path: 输入文件夹
    :param output_file_path: 输出文件夹
    :param model_file_path: 人脸检测模型文件夹
    :param model_name_list: 人脸检测模型文件名 list
    :param image_shape:
    :param recursive: 是否递归查看文件夹
    :param suffix_info_list: 后缀名 list, example: [".png", ".jpg"]
    :return:
    """

    print("detect_align_lfw_data.....")
    # 判断输出文件夹
    data_utils.make_file(output_file_path, remove_flag=True)

    minsize = 60
    threshold = [0.6, 0.85, 0.8]
    factor = 0.85

    # image_info_list_generator[(0, 'cosmos/000001.png', 0), (1, 'cosmos/000002.png', 0)]
    # image_info = (0, 'cosmos/000001.png', 0) ---> image_info[0] 为第几张图像
    # image_info[1] 为 input_file_path 目录下的该图像的相对路径
    # image_info[2] 为 input_file_path 目录下的文件夹的量化号
    image_info_list_generator = data_utils.get_file_path_list(input_file_path, recursive, suffix_info_list)
    # print("image_path_list: {}".format(image_path_list))

    with tf.device('/gpu:0'):
        with tf.Graph().as_default():
            # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
            # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            sess = tf.Session()
            with sess.as_default():
                p_net, r_net, o_net = detect_face.create_mt_cnn(sess, model_file_path, model_name_list)
                pass
            pass
        pass

    n_rof_images_total = 0
    n_rof = np.zeros((5,), dtype=np.int32)
    face_count = 0

    for image_info in image_info_list_generator:
        print("-" * 100)
        # print("image_info: {}".format(image_info))
        image_path = os.path.join(input_file_path, image_info[1])
        print("image_path: {}".format(image_path))

        if n_rof_images_total % 100 == 0:
            print("Processing {}, {}".format(n_rof_images_total, n_rof))
            pass

        n_rof_images_total += 1

        if not os.path.exists(image_path):
            print("image path: {} is not found".format(image_path))
            continue
            pass

        try:
            image = misc.imread(image_path)
            pass
        except (IOError, ValueError, IndexError) as e:
            print("{}: {}".format(image_path, e))
            pass
        else:
            image_channel = image.ndim
            if image_channel < 2:
                print("Unable to align {}, image dim error".format(image_path))
                continue
                pass

            if image_channel == 2:
                image = data_utils.gray_to_rgb(image)
                pass

            image = image[:, :, 0: 3]

            image_fold_and_file_list = image_info[1].replace("\\", "/").split("/")

            image_fold_and_file_list_len = len(image_fold_and_file_list)

            if image_fold_and_file_list_len <= 1:
                output_file_fold_path = output_file_path
                pass
            else:
                fold_path = ""
                for fold_name in image_fold_and_file_list[: -1]:
                    fold_path = os.path.join(fold_path, fold_name)
                    pass

                output_file_fold_path = os.path.join(output_file_path, fold_path)
                pass

            if not os.path.exists(output_file_fold_path):
                os.makedirs(output_file_fold_path)
                pass

            _minsize = minsize
            _landmark = None

            # 人脸检测
            bounding_boxes, points = detect_face.detect(image, _minsize, p_net, r_net, o_net, threshold, factor)

            # print("image_shape: {}".format(image.shape))
            # detect_box = bounding_boxes[0]
            # detect_box = np.array(detect_box)
            # target_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # # image_shape: h, w, c
            # detect_image = target_image[int(detect_box[1]): int(detect_box[3]),
            #                             int(detect_box[0]): int(detect_box[2])]
            # print("detect_image_shape: {}".format(detect_image.shape))
            # cv2.imshow("detect_image", detect_image)
            # cv2.waitKey(0)

            if len(points) == 0:
                print("image_path: {}".format(image_path))
            else:
                _landmark = points.T
                pass

            # print("_landmark: {}".format(_landmark))
            # print(_landmark.shape)
            faces_sum_num = bounding_boxes.shape[0]
            # 人脸矫正和 归一化尺寸为 112 x 112
            for num in range(faces_sum_num):
                warped = face_preprocess.preprocess(image, bbox=bounding_boxes[num],
                                                    landmark=_landmark[num].reshape([2, 5]).T,
                                                    image_shape=image_shape)

                bgr = warped[..., ::-1]
                # cv2.imshow(str(num), bgr)
                # cv2.waitKey(0)

                file_name = "{:04d}.png".format(face_count)
                output_path = os.path.join(output_file_fold_path, file_name)
                print("output_path: {}".format(output_path))
                # print(target_file)
                cv2.imwrite(output_path, bgr)
                face_count += 1
                pass
            pass
        pass
    pass


if __name__ == "__main__":
    from datetime import datetime
    from config import cfg

    # 代码开始时间
    start_time = datetime.now()
    print("开始时间: {}".format(start_time))

    input_data_path = "../data/align_input"
    output_data_path = "../data/align_output"
    model_file = "../models/detect_model"
    model_name = cfg.COMMON.DETECT_MODEL_NAME_LIST
    img_shape = [112, 112, 3]
    recursive_flag = False
    suffix_list = [".png", ".jpg", ".jpeg"]

    do_align_lfw(input_data_path, output_data_path, model_file, model_name,
                 img_shape, recursive_flag, suffix_list)

    # 代码结束时间
    end_time = datetime.now()
    print("结束时间: {}, 训练模型耗时: {}".format(end_time, end_time - start_time))
