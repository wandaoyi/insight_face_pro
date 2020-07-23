#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/07/15 23:36
# @Author   : WanDaoYi
# @FileName : face_classify.py
# ============================================

import os
import shutil
import numpy as np
from utils import get_data_utils, data_utils
from config import cfg


def merge_identity(output_path, info_path, not_recognize_path, input_identity_embedding_list,
                   target_identity_embedding_list, fold_len=8, file_len=4,
                   threshold_value=(0.66, 0.75), delete_flag=True):
    if not os.path.exists(info_path):
        os.makedirs(info_path)
        pass

    file_in_storage = open(os.path.join(info_path, 'in_storage.txt'), 'w')
    file_not_recognize = open(os.path.join(info_path, 'not_recognize.txt'), 'w')
    file_create_lib = open(os.path.join(info_path, 'create_lib.txt'), 'w')

    # 对 target fold 和 file 长度的定义
    if len(target_identity_embedding_list) == 0:
        fold_len_str = "{:0" + str(fold_len) + "d}"
        file_len_str = "{:0" + str(file_len) + "d}{}"
        pass
    else:
        target_identity_embedding_0 = target_identity_embedding_list[0][0]
        target_image_path_0 = target_identity_embedding_0[0].replace("\\", "/")
        fold_path_list_0 = target_image_path_0.split("/")
        fold_len = len(fold_path_list_0[-2])
        file_len = len(os.path.splitext(fold_path_list_0[-1])[0])

        fold_len_str = "{:0" + str(fold_len) + "d}"
        file_len_str = "{:0" + str(file_len) + "d}{}"
        pass

    identity_count = len(target_identity_embedding_list)
    not_recognize_num = 0
    for input_embedding_info in input_identity_embedding_list:

        input_image_path = input_embedding_info[0]
        input_embedding = input_embedding_info[1]

        # 获取输入图像的后缀名
        image_suffix = os.path.splitext(input_image_path)[-1]

        # 如果输出目标文件夹为空，则第一个图像无需计算相似度，直接入库
        if len(target_identity_embedding_list) == 0:
            target_identity_fold_path = os.path.join(output_path, fold_len_str.format(identity_count))
            os.makedirs(target_identity_fold_path)
            new_identity_path = os.path.join(target_identity_fold_path, file_len_str.format(0, image_suffix))

            file_create_lib.write(
                input_image_path + "\t" + new_identity_path + "\t" + str([0, 1]) + "\n\n")

            target_identity_embedding_list.append([(new_identity_path, input_embedding)])

            shutil.copy(input_image_path, new_identity_path)

            if delete_flag:
                os.remove(input_image_path)
                pass

            identity_count += 1
            pass
        # 如果输出目标文件夹不为空，则计算相似度之后再处理
        else:

            identity_similar_list = []
            for target_embedding_info in target_identity_embedding_list:

                similar_value_list = []
                for target_identity_embedding_info in target_embedding_info:
                    target_embedding = target_identity_embedding_info[1]
                    similar_value = get_data_utils.get_cos_similar(input_embedding, target_embedding)
                    similar_value_list.append(similar_value[0])
                    pass
                similar_value_list.sort(reverse=True)
                # print("similar_value_list: {}".format(similar_value_list))
                identity_max_similar_value = similar_value_list[0]
                identity_similar_list.append(identity_max_similar_value)
                pass

            identity_similar_list = np.array(identity_similar_list)
            identity_index = np.argmax(identity_similar_list)
            max_similar_value = identity_similar_list[identity_index]

            # 最大概率的前五名 identity_index, 如果 identity_index_list_len 小于 5, 则前几名。
            five_max_identity_index_list = identity_similar_list.argsort()[-5:][::-1]
            five_max_identity_value_list = identity_similar_list[five_max_identity_index_list]

            # five_max_identity_info_list = [[max_similar_value_identity, max_similar_value],
            #                                [second_max_similar_value_identity, second_max_similar_value]]
            five_max_identity_info_list = []
            for index in range(len(five_max_identity_index_list)):
                five_max_identity_info_list.append([five_max_identity_index_list[index],
                                                    five_max_identity_value_list[index]])
                pass

            # 与 identity_index 人脸库的人脸相似, 将人脸图像添加到 identity_index 中
            if max_similar_value > threshold_value[-1]:
                target_identity_fold_path = os.path.join(output_path, fold_len_str.format(identity_index))
                file_name_list = os.listdir(target_identity_fold_path)
                file_num = len(file_name_list)
                target_identity_path = os.path.join(target_identity_fold_path, file_len_str.format(file_num,
                                                                                                   image_suffix))

                target_identity_embedding_list[identity_index].append((target_identity_path, input_embedding))

                file_in_storage.write(
                    input_image_path + "\t" + target_identity_path + "\t" + str(five_max_identity_info_list) + "\n\n")

                shutil.copyfile(input_image_path, target_identity_path)

                if delete_flag:
                    os.remove(input_image_path)
                    pass
                pass
            # threshold_value[0] < threshold_value < threshold_value[0]
            # 为识别不出人类
            elif max_similar_value > threshold_value[0]:
                if not os.path.exists(not_recognize_path):
                    os.makedirs(not_recognize_path)
                    pass
                not_recognize_image_path = os.path.join(not_recognize_path, file_len_str.format(not_recognize_num,
                                                                                                image_suffix))

                file_not_recognize.write(
                    input_image_path + "\t" + not_recognize_image_path + "\t" + str(
                        five_max_identity_info_list) + "\n\n")

                shutil.copyfile(input_image_path, not_recognize_image_path)
                if delete_flag:
                    os.remove(input_image_path)
                    pass
                not_recognize_num += 1
                pass
            # threshold_value < threshold_value[0]
            # 为人类库不存在的人脸，需要对该人脸进行入库
            else:
                new_identity_fold_path = os.path.join(output_path, fold_len_str.format(identity_count))
                if not os.path.exists(new_identity_fold_path):
                    os.makedirs(new_identity_fold_path)
                    pass

                new_identity_path = os.path.join(new_identity_fold_path, file_len_str.format(0, image_suffix))

                target_identity_embedding_list.append([(new_identity_path, input_embedding)])

                file_create_lib.write(
                    input_image_path + "\t" + new_identity_path + "\t" + str(five_max_identity_info_list) + "\n\n")

                shutil.copyfile(input_image_path, new_identity_path)
                if delete_flag:
                    os.remove(input_image_path)
                    pass

                identity_count += 1
                pass
            pass
        pass

    file_in_storage.close()
    file_not_recognize.close()
    file_create_lib.close()
    pass


def classifier(input_file_path, output_file_path, info_file_path, not_recognize_file_path, face_model,
               recursive=True, image_suffix_list=cfg.COMMON.IMAGE_SUFFIX_LIST, threshold_value=(0.66, 0.75),
               delete_flag=True):
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)
        pass

    # 输入文件夹中的人脸图像数据
    image_info_list_generator = data_utils.get_file_path_list(input_file_path, recursive, image_suffix_list)
    image_info_list = list(image_info_list_generator)

    # 获取已经 align face 好的 image path
    image_file_list = []
    for image_info in image_info_list:
        image_file_list.append(image_info[1])
        pass

    print("image_file_list: {}".format(image_file_list))

    # 整理输出文件夹 identity 内的图像顺序
    output_fold_name_list = os.listdir(output_file_path)
    for fold_name in output_fold_name_list:
        output_fold_path = os.path.join(output_file_path, fold_name)
        data_utils.file_rename(output_fold_path, name_len=8)
        data_utils.file_rename(output_fold_path, name_len=4)
        pass

    # 获取输出文件夹中已有的 人脸身份类别
    target_list_generator = data_utils.get_file_path_list(output_file_path, recursive, image_suffix_list)
    target_list = list(target_list_generator)

    identity_list = data_utils.generate_target_identity(target_list)
    print("identity_list: {}".format(identity_list))

    input_identity_embedding_list = get_data_utils.generate_identity_embedding(input_file_path,
                                                                               image_file_list,
                                                                               face_model,
                                                                               color_mode_flag=True)

    target_identity_embedding_list = []
    for one_identity_list in identity_list:
        one_identity_embedding_list = get_data_utils.generate_identity_embedding(output_file_path,
                                                                                 one_identity_list,
                                                                                 face_model,
                                                                                 color_mode_flag=True)
        target_identity_embedding_list.append(one_identity_embedding_list)
        pass

    merge_identity(output_file_path, info_file_path, not_recognize_file_path,
                   input_identity_embedding_list, target_identity_embedding_list,
                   fold_len=8, file_len=4, threshold_value=threshold_value,
                   delete_flag=delete_flag)

    # print("target_identity_embedding_list: {}".format(target_identity_embedding_list))
    pass


if __name__ == "__main__":
    from datetime import datetime

    # 代码开始时间
    start_time = datetime.now()
    print("开始时间: {}".format(start_time))

    model_path = "../models/model-y1/model-0000.params"

    face_recognize_model = get_data_utils.loading_face_recognize_model(model_path)

    input_file = "../data/align_output"
    output_file = "../data/face_class"
    info_file = "../info/face_classify"
    not_recognize_file = "../output/not_recognize"
    threshold = (0.67, 0.75)
    delete = False
    classifier(input_file, output_file, info_file, not_recognize_file, face_recognize_model,
               threshold_value=threshold, delete_flag=delete)

    # 代码结束时间
    end_time = datetime.now()
    print("结束时间: {}, 训练模型耗时: {}".format(end_time, end_time - start_time))
    pass
