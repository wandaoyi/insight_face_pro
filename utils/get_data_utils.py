#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/06/23 19:00
# @Author   : WanDaoYi
# @FileName : get_data_utils.py
# ============================================

import os
import cv2
import pickle
import shutil
import numpy as np
import mxnet as mx
from mxnet import ndarray as nd
from sklearn import preprocessing
from sklearn.cluster import DBSCAN


def load_bin(bin_data_file_path, bin_name_list, image_shape=(112, 112, 3)):
    """
        加载 .bin data
    :param bin_data_file_path:
    :param bin_name_list:
    :param image_shape:
    :return:
    """

    # 保存加载.bin,其中保存则data_set，data_set有两个元素，
    # 一个表示两张图片像素，一个表示两张图片是否相同
    ver_list = []

    for bin_name in bin_name_list:
        bin_data_path = os.path.join(bin_data_file_path, bin_name)
        if os.path.exists(bin_data_path):
            print("loading {} data.....".format(bin_name))
            with open(bin_data_path, "rb") as file:
                bins, is_same_list = pickle.load(file, encoding="bytes")
                pass
            print("bins_len: {}, is_same_list_len: {}".format(len(bins), len(is_same_list)))

            is_same_list_len = len(is_same_list)
            # 使用 data_len 避免 data 为单数的时候，没对应有 is_same_list 信息
            data_len = is_same_list_len * 2
            data_list = []
            for _ in [0, 1]:
                data = nd.empty((data_len, image_shape[2], image_shape[0], image_shape[1]))
                data_list.append(data)
                pass

            for i in range(data_len):
                bin_data = bins[i]
                image = mx.image.imdecode(bin_data)

                if image.shape[1] != image_shape[0]:
                    image = mx.image.resize_short(image, image_shape[0])
                    pass
                image = nd.transpose(image, axes=(2, 0, 1))

                for flip in [0, 1]:
                    if flip == 1:
                        image = mx.ndarray.flip(data=image, axis=2)
                        pass
                    data_list[flip][i][:] = image
                    pass
                if i % 1000 == 0:
                    print("loading {} bin....".format(i))
                    pass

                # break
                pass
            print("{} {}".format(bin_name, data_list[0].shape))
            ver_list.append((data_list, is_same_list))
            pass

        # break
        pass

    return ver_list
    pass


def load_model(model_path):
    # 模型是否存在，如果存在则加载模型，如果不存在则抛异常
    assert os.path.exists(model_path), "model path is not exists: {}".format(model_path)

    path_info, file_name = os.path.split(model_path)
    name_info, suffix_info = os.path.splitext(file_name)
    name_prefix, name_epoch = name_info.split("-")
    model_path_prefix = os.path.join(path_info, name_prefix)
    epoch_num = int(name_epoch)

    symbol, arg_params, aux_params = mx.model.load_checkpoint(model_path_prefix, epoch_num)

    return symbol, arg_params, aux_params
    pass


def loading_face_recognize_model(model_path, image_shape=(112, 112, 3)):
    symbol, arg_params, aux_params = load_model(model_path)
    model = mx.mod.Module(symbol=symbol, context=mx.gpu())
    model.bind(data_shapes=[("data", (1, image_shape[2], image_shape[0], image_shape[1]))])
    model.set_params(arg_params, aux_params)

    return model
    pass


def predict_embedding_feature(model, image):
    image = nd.array(image)
    image = nd.transpose(image, axes=(2, 0, 1)).astype("float32")
    image = nd.expand_dims(image, axis=0)

    data_batch = mx.io.DataBatch(data=(image,))

    model.forward(data_batch, is_train=False)
    net_out = model.get_outputs()
    embedding_feature = net_out[0].asnumpy()
    norm_embedding_feature = preprocessing.normalize(embedding_feature, axis=1).flatten()

    return norm_embedding_feature


def generate_identity_embedding(file_path, image_file_list, face_model, color_mode_flag=True):
    """
        生成预测 identity_embedding
    :param file_path: 人脸文件夹路径
    :param image_file_list: image path 等信息
    :param face_model: 人脸识别模型
    :param color_mode_flag:
    :return:
    """
    identity_embedding_list = []

    for image_file in image_file_list:
        image_path = os.path.join(file_path, image_file)
        if color_mode_flag:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            pass
        else:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            pass
        # 对一张图片进行特征向量预测
        embedding_feature = predict_embedding_feature(face_model, image)

        identity_embedding_list.append([image_path, embedding_feature])
        pass

    return identity_embedding_list
    pass


def get_cos_similar(input_embedding, target_embedding):
    vector_product = np.matmul(input_embedding, target_embedding.T).reshape(-1)
    # embedding_norm = np.linalg.norm(input_embedding, axis=1) * np.linalg.norm(target_embedding, axis=1)
    embedding_norm = np.linalg.norm(input_embedding) * np.linalg.norm(target_embedding)
    cos_value = vector_product / embedding_norm.reshape(-1)
    # smooth_L_1 = 0.5 + 0.5x
    similar_value = 0.5 + 0.5 * cos_value
    return similar_value
    pass


def get_noise_image_name_info(one_embedding_list):
    """
        通过聚类获取文件夹内的噪声图像
    :param one_embedding_list: 一个文件夹内所有图像的预测特征 np.array()
    :return:
    """
    y_predict = DBSCAN(eps=1.0, min_samples=1).fit_predict(one_embedding_list)

    one_embedding_list_len = one_embedding_list.shape[0]
    result_map = {}
    for index in range(one_embedding_list_len):
        label = int(y_predict[index])
        if label not in result_map:
            result_map[label] = [index]
            pass
        else:
            result_map[label].append(index)
            pass
        pass

    max_value = [0, 0]
    result_map_key_list = [key for key in result_map]
    for key_value in result_map_key_list:
        one_result_list = result_map[key_value]
        one_result_list_len = len(one_result_list)
        if one_result_list_len > max_value[1]:
            max_value[0] = key_value
            max_value[1] = one_result_list_len
            pass
        pass
    if max_value[1] > 0:
        result_map.pop(max_value[0])
        pass

    # result_map: {1: [19], 2: [20]}
    # print("result_map: {}".format(result_map))

    return result_map
    pass


def delete_same_image(same_image_fold_path, info_same_fold_path, one_image_path_list,
                      one_embedding_list, image_index_info_dict, threshold_value, delete_flag=False):

    input_image_fold_path = os.path.split(one_image_path_list[0])[0]

    # 移除前面噪声图像的 image_path and embedding_feature
    remove_index_list = []
    for key_value in image_index_info_dict:
        image_index_info_list = image_index_info_dict[key_value]

        for image_index in image_index_info_list:
            remove_index_list.append(image_index)
            pass
        pass
    if len(remove_index_list) > 0:
        remove_index_list.sort(reverse=True)
        one_embedding_list = list(one_embedding_list)
        for remove_index in remove_index_list:
            one_image_path_list.pop(remove_index)
            one_embedding_list.pop(remove_index)
            pass
        pass

    same_image_index_list = []
    last_image_index_list = []
    one_embedding_list_len = len(one_embedding_list)
    for i in range(0, one_embedding_list_len - 1):

        if i in same_image_index_list:
            continue
            pass

        front_embedding = one_embedding_list[i]

        for j in range(i + 1, one_embedding_list_len):
            later_embedding = one_embedding_list[j]
            similar_value = get_cos_similar(front_embedding, later_embedding)
            # 如果 预测值非常高，接近于 1，则表明两张图像非常相似，极可能是相同的图像。
            # 具体，可以通过调阈值来控制。
            if similar_value[0] > threshold_value:
                last_image_index_list.append(i)
                same_image_index_list.append(j)
                pass
            pass
        pass

    same_image_index_list_len = len(same_image_index_list)

    if same_image_index_list_len > 0:

        fold_name = input_image_fold_path.replace("\\", "/").split("/")[-1]
        same_image_fold = os.path.join(same_image_fold_path, fold_name)
        if not os.path.exists(same_image_fold):
            os.makedirs(same_image_fold)
            pass

        if not os.path.exists(info_same_fold_path):
            os.makedirs(info_same_fold_path)
            pass

        info_same_file_path = os.path.join(info_same_fold_path, fold_name) + ".txt"
        file_same = open(info_same_file_path, "w")

        for index in range(same_image_index_list_len):
            last_index_value = last_image_index_list[index]
            same_index_value = same_image_index_list[index]
            last_image_path = one_image_path_list[last_index_value]
            same_image_path = one_image_path_list[same_index_value]

            image_name = os.path.split(same_image_path)[-1]

            remove_image_path = os.path.join(same_image_fold, image_name)

            file_same.write(last_image_path + "\t" + remove_image_path + "\n\n")

            shutil.copyfile(same_image_path, remove_image_path)

            if delete_flag:
                os.remove(same_image_path)
                pass
            pass

        file_same.close()
        pass
    pass
