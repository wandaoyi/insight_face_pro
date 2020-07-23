#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/06/24 13:21
# @Author   : WanDaoYi
# @FileName : verification_model.py
# ============================================

from datetime import datetime
import os
import cv2
import math
import pickle
import sklearn
import mxnet as mx
import numpy as np
from mxnet import ndarray as nd
from utils import calculate_utils, data_utils, get_data_utils

from config import cfg


class VerificationModel(object):

    def __init__(self):

        # bin data file data path
        self.bin_data_file_path = cfg.VAL.STANDARD_BIN_DATA_FILE_PATH
        self.bin_name_list = cfg.VAL.BIN_NAME_LIST

        # bad case output file path
        self.bad_case_output_file_path = cfg.VAL.BAD_CASE_OUTPUT_FILE_PATH

        self.bin_output_file_path = cfg.VAL.BIN_OUTPUT_FILE_PATH
        self.output_bin_name = cfg.VAL.OUTPUT_BIN_NAME

        self.gap = cfg.VAL.GAP
        # deal bad case
        self.image_shape = cfg.VAL.IMAGE_SHAPE

        # 模型文件夹路径
        self.model_file_path = cfg.VAL.MODEL_FILE_PATH
        # 模型文件名 list
        self.model_name_list = cfg.VAL.MODEL_NAME_LIST
        # face shape [h, w]
        self.face_shape = cfg.COMMON.FACE_SHAPE

        self.batch_size = cfg.VAL.BATCH_SIZE

        self.gpu_id = cfg.COMMON.GPU_ID
        self.ctx = mx.gpu(self.gpu_id)

        self.params_suffix = cfg.COMMON.PARAMS_SUFFIX
        self.json_suffix = cfg.COMMON.JSON_SUFFIX

        # model-symbol.json 文件名
        self.model_symbol_file = cfg.COMMON.MODEL_SYMBOL_FILE

        # 模式
        self.mode = cfg.VAL.MODE

        # n 折交叉验证
        self.k_folds = cfg.VAL.K_FOLDS

        # 是否使用 flip 数据进行 acc 计算, True 为使用
        self.flip_flag = cfg.VAL.FLIP_FLAG
        pass

    def load_model_list(self):
        print("loading model.....")

        epoch_list = []
        # 如果 self.model_name_list = [], 则遍历文件夹内的模型
        if len(self.model_name_list) == 0:
            file_name_list = os.listdir(self.model_file_path)
            for file_name in file_name_list:
                if not file_name.endswith(self.params_suffix):
                    continue
                    pass

                name_info = os.path.splitext(file_name)[0]
                epoch_info = name_info.split('-')[1]
                epoch = int(epoch_info)
                epoch_list.append(epoch)
                pass
            pass
        else:
            for file_name in self.model_name_list:
                if not file_name.endswith(self.params_suffix):
                    continue
                    pass
                epoch = int(file_name.split('.')[0].split('-')[1])
                epoch_list.append(epoch)
                pass
            pass

        # 将 list 降序排列，这样验证的时候就会先验证后面的模型
        epoch_list = sorted(epoch_list, reverse=True)

        # 构造模型路径
        symbol = self.model_symbol_file.split("-")[0]
        model_file_path = os.path.join(self.model_file_path, symbol)

        model_net_list = []
        for epoch_num in epoch_list:
            print("loading {}, {}".format(model_file_path, epoch_num))
            sym, arg_params, aux_params = mx.model.load_checkpoint(model_file_path, epoch_num)

            # 提取fc1_output层
            all_layers = sym.get_internals()
            sym = all_layers['fc1_output']
            model = mx.mod.Module(symbol=sym, context=mx.gpu(), label_names=[])

            # 对输入进行绑定
            model.bind(data_shapes=[('data', (self.batch_size,
                                              self.face_shape[2],
                                              self.face_shape[0],
                                              self.face_shape[1]))])
            # 对参数进行设定
            model.set_params(arg_params, aux_params)
            model_net_list.append(model)
            pass

        # 完成所有模型的加载，所有模型都保存在 model_net_list
        # 注意这里的所有模型是同一网络的模型，只是后缀epoch不相同而已

        print("loading model over!")

        return model_net_list
        pass

    @staticmethod
    def evaluate(embedding_feature, is_same_list, k_folds):
        """

        :param embedding_feature:
        :param is_same_list:
        :param k_folds: k 折验证
        :return:
        """

        # 取所有的第一张图片的向量
        embedding_feature_1 = embedding_feature[0::2]
        # 取所有的第二张图片的向量
        embedding_feature_2 = embedding_feature[1::2]

        is_same_np_as_list = np.asarray(is_same_list)

        # Calculate evaluation metrics
        # 余弦阈值
        thresholds_cos = np.arange(0, 1, 0.001)
        # 欧氏阈值
        thresholds_euclidean = np.arange(0, 4, 0.001)

        # tpr:正类预测正确（召回率）
        # fpr:正类预测错误
        # acc:准确率

        # 余弦相似度 acc
        tpr, fpr, acc_fold_list = calculate_utils.calculate_cos(embedding_feature_1,
                                                                embedding_feature_2,
                                                                is_same_np_as_list,
                                                                k_folds,
                                                                thresholds_cos,
                                                                pca=0)

        # 欧氏距离 acc
        # tpr, fpr, acc_fold_list = calculate_utils.calculate_roc(embedding_feature_1,
        #                                                         embedding_feature_2,
        #                                                         is_same_np_as_list,
        #                                                         k_folds,
        #                                                         thresholds_euclidean,
        #                                                         pca=0)

        far_target = 1e-3
        # 欧氏距离获取 tpr, fpr 等信息
        val, val_std, far = calculate_utils.calculate_val(embedding_feature_1,
                                                          embedding_feature_2,
                                                          is_same_np_as_list,
                                                          k_folds,
                                                          thresholds_euclidean,
                                                          far_target)

        print("val: {}".format(val))
        print("val_std: {}".format(val_std))
        print("far: {}".format(far))

        return tpr, fpr, acc_fold_list, val, val_std, far
        pass

    @staticmethod
    def get_embedding_feature(data_list, model_net, batch_size, data_extra, label_shape):
        """
            获取特征
        :param data_list: 数据集
        :param model_net: 模型
        :param batch_size:
        :param data_extra: 额外数据
        :param label_shape: label 的形状
        :return:
        """

        # 如果存在额外数据
        if data_extra is None:
            data_extra_info = []
            pass
        else:
            data_extra_info = nd.array(data_extra)
            pass

        # 如果标签的形状没有设定
        if label_shape is None:
            label_info = nd.ones((batch_size,))
            pass
        else:
            label_info = nd.ones(label_shape)
            pass

        data_list_len = len(data_list)

        # 输出特征向量
        embedding_feature_list = []

        for i in range(data_list_len):

            data_info = data_list[i]

            embedding_feature = None

            # data_info_shape: (data_batch, 3, 112, 112)
            data_info_batch = data_info.shape[0]
            # 一个batch size 起始的下标号
            one_batch_start_index = 0
            while one_batch_start_index < data_info_batch:
                # 一个batch size 结束的下标号
                one_batch_end_index = min(one_batch_start_index + batch_size, data_info_batch)

                count = one_batch_end_index - one_batch_start_index

                # 切割数据，得到一个batch_size的数据
                one_batch_data_info = nd.slice_axis(data_info,
                                                    axis=0,
                                                    begin=one_batch_end_index - batch_size,
                                                    end=one_batch_end_index)

                # 如果有额外测试数据 data_extra_info，则进行添加
                if data_extra is None:
                    data_batch = mx.io.DataBatch(data=(one_batch_data_info,), label=(label_info,))
                    pass
                else:
                    data_batch = mx.io.DataBatch(data=(one_batch_data_info, data_extra_info), label=(label_info,))
                    pass

                # 传入数据，进行前向传播
                model_net.forward(data_batch, is_train=False)
                # 获得输出
                output_info = model_net.get_outputs()

                # 获得输出的向量 shape: (batch_size, 128)
                embeddings = output_info[0].asnumpy()

                if embedding_feature is None:
                    # shape: (data_batch, 128)
                    embedding_feature = np.zeros((data_info.shape[0], embeddings.shape[1]))
                    pass
                embedding_feature[one_batch_start_index: one_batch_end_index, :] = embeddings[(batch_size - count):, :]

                one_batch_start_index = one_batch_end_index
                pass

            embedding_feature_list.append(embedding_feature)
            pass

        return embedding_feature_list
        pass

    def test_details(self, data_set, model_net, batch_size, k_folds=8,
                     data_extra=None, label_shape=None):
        """
            对数据进行测试
        :param data_set: 数据集
        :param model_net: 模型
        :param batch_size:
        :param k_folds: K 折验证
        :param data_extra:
        :param label_shape: 标签数据的形状
        :return:
        """
        # 两张图片的像素
        data_list = data_set[0]

        # 标签
        is_same_list = data_set[1]

        # 输出特征向量
        # embedding_feature_list = []

        embedding_feature_list = self.get_embedding_feature(data_list,
                                                            model_net,
                                                            batch_size,
                                                            data_extra,
                                                            label_shape)

        norm_info = 0.0
        norm_count = 0

        # 对每个测试数据测试出来的结果进行评估
        for feature in embedding_feature_list:

            feature_batch = feature.shape[0]
            for i in range(feature_batch):
                feature_info = feature[i]

                # 求向量的范数：
                norm_value = np.linalg.norm(feature_info)
                norm_info += norm_value
                norm_count += 1
                pass
            pass

        feature_norm = norm_info / norm_count

        # 使用 flip 处理后的图像
        if self.flip_flag:
            embedding_info = sklearn.preprocessing.normalize(embedding_feature_list[0] + embedding_feature_list[1])
            pass
        else:
            embedding_info = sklearn.preprocessing.normalize(embedding_feature_list[0])
            pass

        print("embedding_info_shape: {}".format(embedding_info.shape))

        print("is_same_list_len: {}".format(len(is_same_list)))
        tpr, fpr, acc_fold_list, val, val_std, far = self.evaluate(embedding_info, is_same_list, k_folds)

        acc = np.mean(acc_fold_list)
        std = np.std(acc_fold_list)

        return acc, std, feature_norm, embedding_feature_list
        pass

    def test_bad_case(self, data_set, model_net, batch_size, k_folds=8,
                      name_info="", data_extra=None, label_shape=None):
        """

        :param data_set: 数据集
        :param model_net: 模型
        :param batch_size:
        :param k_folds:
        :param name_info:
        :param data_extra: 额外数据
        :param label_shape:
        :return:
        """

        data_utils.make_file(self.bad_case_output_file_path, remove_flag=True)

        # 两张图片的像素
        data_list = data_set[0]

        # 标签
        is_same_list = data_set[1]

        # 输出特征向量
        embedding_feature_list = self.get_embedding_feature(data_list,
                                                            model_net,
                                                            batch_size,
                                                            data_extra,
                                                            label_shape)

        embedding_feature = embedding_feature_list[0] + embedding_feature_list[1]
        embedding_feature = sklearn.preprocessing.normalize(embedding_feature)

        # 取所有的第一张图片的向量
        embedding_feature_1 = embedding_feature[0::2]
        # 取所有的第二张图片的向量
        embedding_feature_2 = embedding_feature[1::2]

        # Calculate evaluation metrics
        thresholds = np.arange(0, 4, 0.01)
        is_same_np_as_list = np.asarray(is_same_list)

        pos_output, neg_output = calculate_utils.get_pos_neg_sample(data_list,
                                                                    embedding_feature_1,
                                                                    embedding_feature_2,
                                                                    is_same_np_as_list,
                                                                    k_folds,
                                                                    thresholds)

        pos_output = sorted(pos_output, key=lambda x: x[2], reverse=True)
        neg_output = sorted(neg_output, key=lambda x: x[2], reverse=False)

        neg_output_len = len(neg_output)
        if neg_output_len > 0:
            threshold = neg_output[0][3]
            pass
        else:
            threshold = neg_output[-1][3]
            pass

        print("threshold: {}".format(threshold))

        font = cv2.FONT_HERSHEY_SIMPLEX

        # false_negative: 漏判
        # false_positive: 误判
        for item in [(pos_output, 'positive(false_negative).png'), (neg_output, 'negative(false_positive).png')]:

            # positive or negative output data
            out_data = item[0]
            if len(out_data) == 0:
                continue
                pass

            # 列
            cols = 4
            # 行
            rows = 8000
            rows_num = int(math.ceil(len(out_data) / cols))
            rows = min(rows, rows_num)

            image = np.zeros((self.image_shape[0] * rows + 20,
                              self.image_shape[1] * cols + (cols - 1) * self.gap, 3),
                             dtype=np.uint8)
            # 将背景设为白色
            image[:, :, :] = 255
            text_color = (153, 255, 51)

            for out_index, out_info in enumerate(out_data):
                row_index = out_index // cols
                col_index = out_index % cols
                if row_index == rows:
                    break
                    pass
                image_a = out_info[0]
                image_b = out_info[1]

                dist = out_info[2]

                image_concat = np.concatenate((image_a, image_b), axis=1)

                # 确保 image_concat 为连续数组
                image_concat = np.ascontiguousarray(image_concat)

                dist_value = "{:.4f}".format(dist)

                cv2.putText(image_concat, dist_value, (80, self.image_shape[0] // 2 + 7), font, 0.6, text_color, 2)

                row_start_index = row_index * self.image_shape[0]
                row_end_index = (row_index + 1) * self.image_shape[0]
                col_start_index = col_index * self.image_shape[1] + self.gap * col_index
                col_end_index = (col_index + 1) * self.image_shape[1] + self.gap * col_index

                image[row_start_index: row_end_index, col_start_index: col_end_index, :] = image_concat
                pass

            threshold_value = "threshold: {:.4f}".format(threshold)
            cv2.putText(image, threshold_value, (image.shape[1] // 2 - 70, image.shape[0] - 5), font, 0.6, text_color,
                        2)

            file_name = item[1]

            if len(name_info) > 0:
                file_name = name_info + "_" + file_name
                pass
            file_name = os.path.join(self.bad_case_output_file_path, file_name)
            cv2.imwrite(file_name, image)
            pass
        print("It's over!")
        pass

    def dump_r(self, data_set, model_net, batch_size, data_extra=None, label_shape=None):
        """

        :param data_set: 数据集
        :param model_net: 模型
        :param batch_size:
        :param data_extra: 额外数据
        :param label_shape:
        :return:
        """

        data_utils.make_file(self.bin_output_file_path, remove_flag=True)

        # 两张图片的像素
        data_list = data_set[0]

        # 标签
        is_same_list = data_set[1]

        embedding_feature_list = self.get_embedding_feature(data_list,
                                                            model_net,
                                                            batch_size,
                                                            data_extra,
                                                            label_shape)

        embedding_feature = embedding_feature_list[0] + embedding_feature_list[1]
        embedding_feature = sklearn.preprocessing.normalize(embedding_feature)

        is_same_np_as_list = np.asarray(is_same_list)

        output_file_path = os.path.join(self.bin_output_file_path, self.output_bin_name)
        # "wb" 不支持 encoding="utf-8" 参数, 会报错
        with open(output_file_path, "wb") as file:
            pickle.dump((embedding_feature, is_same_np_as_list), file, protocol=pickle.HIGHEST_PROTOCOL)
            pass
        pass

    def do_verification(self):
        # 完成所有模型的加载，所有模型都保存在 model_net_list
        # 注意这里的所有模型是同一网络的模型，只是后缀epoch不相同而已
        model_net_list = self.load_model_list()

        # 保存加载 .bin, 其中保存则 data_set，data_set有两个元素，
        # 一个表示两张图片像素，一个表示两张图片是否相同
        ver_list = get_data_utils.load_bin(bin_data_file_path=self.bin_data_file_path,
                                           bin_name_list=self.bin_name_list,
                                           image_shape=self.face_shape)

        if self.mode == 0:
            # 对每个数据集进行测试
            ver_list_len = len(ver_list)
            for i in range(ver_list_len):

                data_name = self.bin_name_list[i]
                name_info = os.path.splitext(data_name)[0]

                result_list = []
                # 对每个模型进行测试
                for model_net in model_net_list:
                    acc, std, feature_norm, embedding_feature_list = self.test_details(ver_list[i],
                                                                                       model_net,
                                                                                       self.batch_size,
                                                                                       self.k_folds)
                    if self.flip_flag:
                        print("acc_flip({}): {:.5f}+-{:.5f}".format(name_info, acc, std))
                        pass
                    else:
                        print("acc({}): {:.5f}+-{:.5f}".format(name_info, acc, std))
                        pass

                    result_list.append(acc)
                    pass
                print("max acc info of {} is {:.5f}".format(name_info, np.max(result_list)))
                pass
            pass
        elif self.mode == 1:
            model_net = model_net_list[0]
            data_bin = ver_list[0]
            bin_name = self.bin_name_list[0]
            name_info = os.path.splitext(bin_name)[0]
            self.test_bad_case(data_bin, model_net, self.batch_size, self.k_folds, name_info)
            pass
        else:
            model_net = model_net_list[0]
            data_bin = ver_list[0]
            self.dump_r(data_bin, model_net, self.batch_size)
            pass
        pass


if __name__ == "__main__":
    # 代码开始时间
    start_time = datetime.now()
    print("开始时间: {}".format(start_time))

    demo = VerificationModel()
    demo.do_verification()

    # 代码结束时间
    end_time = datetime.now()
    print("结束时间: {}, 训练模型耗时: {}".format(end_time, end_time - start_time))
