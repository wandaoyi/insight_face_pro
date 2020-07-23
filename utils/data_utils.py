#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/06/29 16:59
# @Author   : WanDaoYi
# @FileName : data_utils.py
# ============================================

import os
import cv2
import uuid
import json
import shutil
import numpy as np


def make_file(file_path, remove_flag=False):
    """
        创建文件夹 在 shutil.rmtree(file_path) 不好用的情况下使用
    :param file_path: 文件夹路径
    :param remove_flag: 是否删除原来已有的文件夹及文件
    :return:
    """
    if remove_flag:
        if os.path.exists(file_path):
            # shutil.rmtree(file_path)
            file_name_list = os.listdir(file_path)
            for file_name in file_name_list:
                path_info = os.path.join(file_path, file_name)
                os.remove(path_info)
                pass
            pass
        else:
            os.makedirs(file_path)
            pass
        pass
    else:
        if not os.path.exists(file_path):
            os.makedirs(file_path)
            pass
        pass
    pass


def get_file_path_list(input_file_path, recursive=False, suffix_info_list=()):
    """
        获取文件路径list
    :param input_file_path:
    :param recursive: 是否使用递归去遍历文件夹
    :param suffix_info_list: 文件的后缀名
    :return:
    """
    i = 0
    if recursive:
        cat = {}
        for path, dirs, file_list in os.walk(input_file_path, followlinks=True):
            dirs.sort()
            file_list.sort()

            for file_name in file_list:
                file_path = os.path.join(path, file_name)
                suffix = os.path.splitext(file_name)[1].lower()

                if os.path.isfile(file_path) and (suffix in suffix_info_list):
                    if path not in cat:
                        cat[path] = len(cat)
                        pass

                    yield (i, os.path.relpath(file_path, input_file_path), cat[path])
                    i += 1
                    pass
                pass
            pass

        for k, v in sorted(cat.items(), key=lambda x: x[1]):
            print(os.path.relpath(k, input_file_path), v)
            pass
        pass
    else:
        for file_name in sorted(os.listdir(input_file_path)):
            file_path = os.path.join(input_file_path, file_name)
            suffix = os.path.splitext(file_name)[1].lower()
            if os.path.isfile(file_path) and (suffix in suffix_info_list):
                yield (i, os.path.relpath(file_path, input_file_path), 0)
                i += 1
                pass
            pass
        pass
    pass


def file_rename(fold_path, name_len=4):
    """
        文件重命名
    :param fold_path: 文件夹路径
    :param name_len: 名字的长度
    :return:
    """
    name_len_str = "{:0" + str(name_len) + "d}{}"
    name_list = os.listdir(fold_path)
    name_count = 0
    for name in name_list:
        name_suffix = os.path.splitext(name)[-1]
        file_path = os.path.join(fold_path, name)
        new_file_path = os.path.join(fold_path, name_len_str.format(name_count, name_suffix))
        os.rename(file_path, new_file_path)
        name_count += 1
        pass
    pass


def fold_rename(fold_path, fold_len=8):
    """
        文件夹重命名
    :param fold_path:
    :param fold_len:
    :return:
    """
    name_len_str = "{:0" + str(fold_len) + "d}"
    name_list = os.listdir(fold_path)
    name_count = 0
    for fold_name in name_list:
        ori_fold_path = os.path.join(fold_path, fold_name)
        new_fold_path = os.path.join(fold_path, name_len_str.format(name_count))
        os.rename(ori_fold_path, new_fold_path)
        name_count += 1
        pass
    pass


def data_file_rename(data_fold_path, fold_len_value=(8, 10), file_len_value=(4, 6)):
    fold_rename(data_fold_path, fold_len_value[-1])
    fold_rename(data_fold_path, fold_len_value[0])

    identity_fold_name_list = os.listdir(data_fold_path)
    for identity_fold_name in identity_fold_name_list:
        identity_fold_path = os.path.join(data_fold_path, identity_fold_name)
        file_rename(identity_fold_path, file_len_value[-1])
        file_rename(identity_fold_path, file_len_value[0])
        pass
    pass


def load_json(data_path):
    with open(data_path, encoding="utf-8") as file:
        json_data = json.load(file)

        return json_data
        pass
    pass


def dump_json(data_dic, data_path):
    with open(data_path, "w", encoding="utf-8") as file:
        json.dump(data_dic, file, indent=2)
        pass
    pass


def get_uuid():
    uid = uuid.uuid1()
    return uid.hex
    pass


def resize_image(image, new_size):
    # @UndefinedVariable
    image_resize = cv2.resize(image, (new_size[1], new_size[0]), interpolation=cv2.INTER_AREA)
    return image_resize
    pass


def generate_bounding_box(image, reg, scale, threshold):
    """
        use heat_map to generate bounding boxes
    :param image: heat_map
    :param reg:
    :param scale:
    :param threshold:
    :return:
    """

    stride = 2
    cell_size = 12

    image = np.transpose(image)
    dx_1 = np.transpose(reg[:, :, 0])
    dy_1 = np.transpose(reg[:, :, 1])
    dx_2 = np.transpose(reg[:, :, 2])
    dy_2 = np.transpose(reg[:, :, 3])
    y, x = np.where(image >= threshold)

    if y.shape[0] == 1:
        dx_1 = np.flipud(dx_1)
        dy_1 = np.flipud(dy_1)
        dx_2 = np.flipud(dx_2)
        dy_2 = np.flipud(dy_2)
        pass

    score = image[(y, x)]
    reg = np.transpose(np.vstack([dx_1[(y, x)], dy_1[(y, x)], dx_2[(y, x)], dy_2[(y, x)]]))

    if reg.size == 0:
        reg = np.empty((0, 3))
        pass

    bb = np.transpose(np.vstack([y, x]))
    q1 = np.fix((stride * bb + 1) / scale)
    q2 = np.fix((stride * bb + cell_size - 1 + 1) / scale)
    bounding_box = np.hstack([q1, q2, np.expand_dims(score, 1), reg])
    return bounding_box, reg
    pass


def gray_to_rgb(image):
    """
        灰度图转 RGB
    :param image:
    :return:
    """
    w, h = image.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = image
    ret[:, :, 1] = image
    ret[:, :, 2] = image

    return ret
    pass


def nms(boxes, threshold, method):
    if boxes.size == 0:
        return np.empty((0, 3))
    x_1 = boxes[:, 0]
    y_1 = boxes[:, 1]
    x_2 = boxes[:, 2]
    y_2 = boxes[:, 3]
    score = boxes[:, 4]

    area = (x_2 - x_1 + 1) * (y_2 - y_1 + 1)

    # score 升序排列的 index 值
    score_index_list = np.argsort(score)
    # 构建一个 score.size 的零值的 list
    pick = np.zeros_like(score, dtype=np.int16)
    counter = 0

    while score_index_list.size > 0:
        max_score_index = score_index_list[-1]
        pick[counter] = max_score_index
        counter += 1
        last_indexes_list = score_index_list[0: -1]
        xx_1 = np.maximum(x_1[max_score_index], x_1[last_indexes_list])
        yy_1 = np.maximum(y_1[max_score_index], y_1[last_indexes_list])
        xx_2 = np.minimum(x_2[max_score_index], x_2[last_indexes_list])
        yy_2 = np.minimum(y_2[max_score_index], y_2[last_indexes_list])

        w = np.maximum(0.0, xx_2 - xx_1 + 1)
        h = np.maximum(0.0, yy_2 - yy_1 + 1)
        inter = w * h

        if method == "Min":
            boxes_iou = inter / np.minimum(area[max_score_index], area[last_indexes_list])
            pass
        else:
            boxes_iou = inter / (area[max_score_index] + area[last_indexes_list] - inter)
            pass
        score_index_list = score_index_list[np.where(boxes_iou <= threshold)]
        pass

    pick = pick[0: counter]

    return pick
    pass


def re_rec(bounding_box):
    w = bounding_box[:, 2] - bounding_box[:, 0]
    h = bounding_box[:, 3] - bounding_box[:, 1]
    max_length = np.maximum(w, h)

    bounding_box[:, 0] = bounding_box[:, 0] + w * 0.5 - max_length * 0.5
    bounding_box[:, 1] = bounding_box[:, 1] + h * 0.5 - max_length * 0.5
    bounding_box[:, 2: 4] = bounding_box[:, 0: 2] + np.transpose(np.tile(max_length, (2, 1)))

    return bounding_box
    pass


def bounding_box_reg(bounding_box, reg):
    """
        calibrate bounding boxes
    :param bounding_box:
    :param reg:
    :return:
    """
    if reg.shape[1] == 1:
        reg = np.reshape(reg, (reg.shape[2], reg.shape[3]))

    w = bounding_box[:, 2] - bounding_box[:, 0] + 1
    h = bounding_box[:, 3] - bounding_box[:, 1] + 1
    b_1 = bounding_box[:, 0] + reg[:, 0] * w
    b_2 = bounding_box[:, 1] + reg[:, 1] * h
    b_3 = bounding_box[:, 2] + reg[:, 2] * w
    b_4 = bounding_box[:, 3] + reg[:, 3] * h
    bounding_box[:, 0:4] = np.transpose(np.vstack([b_1, b_2, b_3, b_4]))
    return bounding_box
    pass


def pad(boxes, w, h):
    tmp_w = (boxes[:, 2] - boxes[:, 0] + 1).astype(np.int32)
    tmp_h = (boxes[:, 3] - boxes[:, 1] + 1).astype(np.int32)
    num_box = boxes.shape[0]

    dx = np.ones((num_box, ), dtype=np.int32)
    dy = np.ones((num_box, ), dtype=np.int32)
    edx = tmp_w.copy().astype(np.int32)
    edy = tmp_h.copy().astype(np.int32)

    x = boxes[:, 0].copy().astype(np.int32)
    y = boxes[:, 1].copy().astype(np.int32)
    ex = boxes[:, 2].copy().astype(np.int32)
    ey = boxes[:, 3].copy().astype(np.int32)

    tmp = np.where(ex > w)
    edx.flat[tmp] = np.expand_dims(-ex[tmp] + w + tmp_w[tmp], 1)
    ex[tmp] = w

    tmp = np.where(ey > h)
    edy.flat[tmp] = np.expand_dims(-ey[tmp] + h + tmp_h[tmp], 1)
    ey[tmp] = h

    tmp = np.where(x < 1)
    dx.flat[tmp] = np.expand_dims(2 - x[tmp], 1)
    x[tmp] = 1

    tmp = np.where(y < 1)
    dy.flat[tmp] = np.expand_dims(2 - y[tmp], 1)
    y[tmp] = 1

    return dy, edy, dx, edx, y, ey, x, ex, tmp_w, tmp_h
    pass


def read_image(image_path, **kwargs):
    mode = kwargs.get("mode", "rgb")
    layout = kwargs.get("layout", "HWC")

    if mode == "gray":
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        pass
    else:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if mode == "rgb":
            image = image[..., ::-1]
            pass

        if layout == "CHW":
            image = np.transpose(image, (2, 0, 1))
            pass
        pass

    return image
    pass


def generate_target_identity(image_info_list):
    """
        生成目标身份 list
    :param image_info_list:
    :return:
    """

    image_identity_and_path = {}
    for image_info in image_info_list:
        image_path = image_info[1]
        identity_index = image_info[2]
        if identity_index not in image_identity_and_path:
            image_identity_and_path.update({identity_index: [image_path]})
            pass
        else:
            image_identity_and_path[identity_index].append(image_path)
            pass
        pass

    identity_count_list = []
    image_identity_and_path_key = [key_name for key_name in image_identity_and_path]
    image_identity_and_path_key.sort()
    for key_name in image_identity_and_path_key:
        identity_count_list.append(image_identity_and_path[key_name])
        pass

    return identity_count_list
    pass


def deal_noise_image(info_noise_fold, noise_file_path, one_image_path_list,
                     image_index_info_dict, delete_flag=False):

    input_image_fold_path = os.path.split(one_image_path_list[0])[0]

    fold_name = input_image_fold_path.replace("\\", "/").split("/")[-1]

    output_fold_path = os.path.join(noise_file_path, fold_name)

    one_info_noise_file_path = os.path.join(info_noise_fold, fold_name) + ".txt"
    file_noise = open(one_info_noise_file_path, "w")

    for key_value in image_index_info_dict:

        image_index_info_list = image_index_info_dict[key_value]

        one_output_fold_path = os.path.join(output_fold_path, "{:05d}".format(key_value))
        if not os.path.exists(one_output_fold_path):
            os.makedirs(one_output_fold_path)
            pass

        for image_index in image_index_info_list:
            image_path = one_image_path_list[image_index]
            image_name = os.path.split(image_path)[-1]

            one_noise_image_path = os.path.join(one_output_fold_path, image_name)

            file_noise.write(image_path + "\t" + one_noise_image_path + "\n\n")

            shutil.copyfile(image_path, one_noise_image_path)

            if delete_flag:
                os.remove(image_path)
                pass
            pass
        pass

    file_noise.close()
    pass

