#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/07/17 19:47
# @Author   : WanDaoYi
# @FileName : face_cluster.py
# ============================================

import os
import numpy as np
from utils import get_data_utils, data_utils
from config import cfg


def cluster(input_class_fold_path, noise_fold_path, same_image_fold_path, info_noise_fold_path,
            info_same_fold_path, face_model, threshold_value=0.999, recursive=True,
            image_suffix_list=cfg.COMMON.IMAGE_SUFFIX_LIST, delete_flag=False):

    # 输入文件夹中的人脸图像数据
    image_info_list_generator = data_utils.get_file_path_list(input_class_fold_path, recursive, image_suffix_list)
    image_info_list = list(image_info_list_generator)
    print("image_info_list: {}".format(image_info_list))

    identity_list = data_utils.generate_target_identity(image_info_list)
    print("identity_list: {}".format(identity_list))

    for one_identity_list in identity_list:
        one_identity_embedding_list = get_data_utils.generate_identity_embedding(input_class_fold_path,
                                                                                 one_identity_list,
                                                                                 face_model,
                                                                                 color_mode_flag=True)

        one_image_path_list = []
        one_embedding_list = []
        for identity_embedding_info in one_identity_embedding_list:
            one_image_path = identity_embedding_info[0]
            one_embedding = identity_embedding_info[-1]
            one_image_path_list.append(one_image_path)
            one_embedding_list.append(one_embedding)
            pass

        one_embedding_list = np.array(one_embedding_list)

        # 获取噪声图像的下标
        image_index_info_dict = get_data_utils.get_noise_image_name_info(one_embedding_list)
        if len(image_index_info_dict) > 0:

            if not os.path.exists(info_noise_fold_path):
                os.makedirs(info_noise_fold_path)
                pass

            # 移除噪声图像
            data_utils.deal_noise_image(info_noise_fold_path, noise_fold_path, one_image_path_list,
                                        image_index_info_dict, delete_flag=delete_flag)

            pass

        # 删除相同的图像
        get_data_utils.delete_same_image(same_image_fold_path, info_same_fold_path,
                                         one_image_path_list, one_embedding_list,
                                         image_index_info_dict, threshold_value,
                                         delete_flag)
        pass
    pass


if __name__ == "__main__":
    from datetime import datetime

    # 代码开始时间
    start_time = datetime.now()
    print("开始时间: {}".format(start_time))

    model_path = "../models/model-y1/model-0000.params"
    face_recognize_model = get_data_utils.loading_face_recognize_model(model_path)

    input_class_fold = "../data/face_class"
    noise_fold = "../output/cluster_noise"
    same_image_fold = "../output/same_image"
    info_noise = "../info/cluster_noise"
    info_same_image = "../info/same_image"

    threshold = 0.999
    delete = True

    cluster(input_class_fold, noise_fold, same_image_fold, info_noise, info_same_image,
            face_recognize_model, threshold_value=threshold, delete_flag=delete)

    # 代码结束时间
    end_time = datetime.now()
    print("结束时间: {}, 训练模型耗时: {}".format(end_time, end_time - start_time))
    pass
