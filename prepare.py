#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/06/30 01:04
# @Author   : WanDaoYi
# @FileName : prepare.py
# ============================================

import os
import cv2
import pickle
import random
import mxnet as mx
import numpy as np
from utils import data_utils
from config import cfg


class Prepare(object):

    def __init__(self):
        self.face_classify_fold_path = cfg.DATA_SET.OUTPUT_FACE_CLASSIFY_FOLD_PATH

        # 各成分数据保存路径
        self.train_data_path = cfg.DATA_SET.TRAIN_DATA_PATH
        self.val_data_path = cfg.DATA_SET.VAL_DATA_PATH

        self.train_fold_path = cfg.DATA_SET.TRAIN_FOLD_PATH
        self.val_bin_fold_path = cfg.DATA_SET.VAL_BIN_FOLD_PATH
        self.identity_property_path = cfg.DATA_SET.IDENTITY_PROPERTY_PATH
        self.pair_file_path = cfg.DATA_SET.PAIR_DATA_PATH

        self.bin_name = cfg.DATA_SET.BIN_NAME
        self.rec_name = cfg.DATA_SET.REC_NAME
        self.idx_name = cfg.DATA_SET.IDX_NAME

        self.positive_sample_num = cfg.DATA_SET.POSITIVE_SAMPLE_NUM
        self.negative_times = cfg.DATA_SET.NEGATIVE_TIMES_POSITIVE

        self.face_shape = cfg.COMMON.FACE_SHAPE
        self.image_suffix_list = cfg.COMMON.IMAGE_SUFFIX_LIST
        self.recursive_flag = cfg.DATA_SET.RECURSIVE_FLAG

        # 训练数据集的百分比
        self.train_percent = cfg.DATA_SET.TRAIN_PERCENT
        self.val_percent = cfg.DATA_SET.VAL_PERCENT
        pass

    @staticmethod
    def data_divide(image_path_list, train_percent, face_classify_fold_path,
                    train_data_path, val_data_path):
        image_path_list_len = len(image_path_list)
        n_train_sample = int(image_path_list_len * train_percent)
        train_image_path_list = random.sample(image_path_list, n_train_sample)
        print("n_train_sample: {}".format(n_train_sample))

        train_file = open(train_data_path, "w")
        val_file = open(val_data_path, "w")

        train_sample_dict = {}
        val_sample_dict = {}

        for path_info in image_path_list:

            fold_name = os.path.split(path_info)[0]

            image_path = os.path.join(face_classify_fold_path, path_info)

            if path_info in train_image_path_list:
                train_file.write(image_path + "\n")

                if fold_name not in train_sample_dict:
                    train_sample_dict.update({fold_name: [path_info]})
                    pass
                else:
                    train_sample_dict[fold_name].append(path_info)
                    pass

                continue
                pass

            val_file.write(image_path + "\n")

            if fold_name not in val_sample_dict:
                val_sample_dict.update({fold_name: [path_info]})
                pass
            else:
                val_sample_dict[fold_name].append(path_info)
                pass

        train_file.close()
        val_file.close()

        print("train_sample_dict: {}".format(train_sample_dict))
        print("val_sample_dict: {}".format(val_sample_dict))

        return train_sample_dict, val_sample_dict
        pass

    @staticmethod
    def generate_bin_data(image_data_path, output_bin_path, bin_name, pair_file_path,
                          sample_dict, positive_sample_num=2000, negative_times=2):
        """
            生成 bin 数据集
        :param image_data_path: 样本路径
        :param output_bin_path: output .bin data fold path
        :param bin_name: example val.bin
        :param pair_file_path: 配对数据的文件
        :param sample_dict: 样本集，如 {"00000000": ["00000000/0001.png", "00000000/0004.png"],
                                       "00000001": ["00000001/0000.png", "00000001/0003.png", "00000001/0005.png"]}
        :param positive_sample_num: 正样本数
        :param negative_times: 负样本数为正样本数的 negative_times 倍
        :return:
        """
        if not os.path.exists(output_bin_path):
            os.makedirs(output_bin_path)
            pass
        # 获取所有的正例
        positive_sample_list = []
        for identity in sample_dict:
            identity_sample_list = sample_dict[identity]
            identity_sample_list_len = len(identity_sample_list)
            if identity_sample_list_len > 1:
                for front_sample_index in range(identity_sample_list_len - 1):
                    front_identity_sample = identity_sample_list[front_sample_index]
                    for last_sample_index in range(front_sample_index + 1, identity_sample_list_len):
                        last_identity_sample = identity_sample_list[last_sample_index]

                        positive_sample_list.append([front_identity_sample, last_identity_sample, True])
                        pass
                    pass
                pass
            pass

        positive_sample_list_len = len(positive_sample_list)
        # 获取目标正例
        if positive_sample_list_len > positive_sample_num:
            target_positive_sample_list = random.sample(positive_sample_list, positive_sample_num)
            pass
        else:
            count_num = 1
            target_positive_sample_list = []
            while True:
                target_index = np.random.randint(0, positive_sample_list_len)
                target_positive_sample = positive_sample_list[target_index]

                target_positive_sample_list.append(target_positive_sample)

                if count_num > positive_sample_num:
                    break
                    pass
                count_num += 1
                pass
            pass

        # 释放资源
        positive_sample_list.clear()

        # 负样本数为正样本数的 negative_times 倍
        negative_sample_len = positive_sample_num * negative_times
        sample_dict_identity_list = [key for key in sample_dict]
        sample_dict_identity_list_len = len(sample_dict_identity_list)
        negative_sample_list = []
        for i in range(negative_sample_len):
            front_random_identity_index = np.random.randint(0, sample_dict_identity_list_len)
            front_random_identity = sample_dict_identity_list[front_random_identity_index]
            front_identity_sample_list = sample_dict[front_random_identity]
            front_identity_sample_list_len = len(front_identity_sample_list)
            front_random_sample_index = np.random.randint(0, front_identity_sample_list_len)
            front_random_sample = front_identity_sample_list[front_random_sample_index]

            while True:
                last_random_identity_index = np.random.randint(0, sample_dict_identity_list_len)

                if front_random_identity_index != last_random_identity_index:
                    last_random_identity = sample_dict_identity_list[last_random_identity_index]
                    last_identity_sample_list = sample_dict[last_random_identity]
                    last_identity_sample_list_len = len(last_identity_sample_list)
                    last_random_sample_index = np.random.randint(0, last_identity_sample_list_len)
                    last_random_sample = last_identity_sample_list[last_random_sample_index]

                    negative_sample_list.append([front_random_sample, last_random_sample, False])

                    break
                    pass
                pass
            pass

        # 获得总样本
        all_sample_list = target_positive_sample_list + negative_sample_list
        np.random.shuffle(all_sample_list)

        # 释放资源
        negative_sample_list.clear()
        target_positive_sample_list.clear()

        # 保存 配对样本 用于参考
        with open(pair_file_path, "w") as file:
            for sample_info in all_sample_list:
                file.write(",".join([str(info) for info in sample_info]) + '\n')
                pass
            pass

        image_bin_list = []
        same_flag_list = []
        print("读取样本并保存为 .bin 数据......")
        for sample_info in all_sample_list:
            front_image_path = os.path.join(image_data_path, sample_info[0])
            last_image_path = os.path.join(image_data_path, sample_info[1])
            if os.path.exists(front_image_path) and os.path.exists(last_image_path):
                with open(front_image_path, "rb") as file:
                    image_bin = file.read()
                    image_bin_list.append(image_bin)
                    pass
                with open(last_image_path, "rb") as file:
                    image_bin = file.read()
                    image_bin_list.append(image_bin)
                    pass
                same_flag_list.append(sample_info[2])
                pass
            pass

        print("all_sample_list: {}".format(all_sample_list))
        all_sample_list.clear()

        bin_data_path = os.path.join(output_bin_path, bin_name)
        # 如果 .bin 文件存在，则移除
        if os.path.exists(bin_data_path):
            os.remove(bin_data_path)
            pass
        with open(bin_data_path, "wb") as file:
            pickle.dump((image_bin_list, same_flag_list), file, protocol=pickle.HIGHEST_PROTOCOL)
            pass

        image_bin_list.clear()
        same_flag_list.clear()
        pass

    @staticmethod
    def generate_rec_idx_data(sample_dict, image_fold_path, output_rec_path, output_idx_path):
        """
            生成 .rec 和 .idx 数据
        :param sample_dict: 样本集，如 {"00000000": ["00000000/0001.png", "00000000/0004.png"],
                                       "00000001": ["00000001/0000.png", "00000001/0003.png", "00000001/0005.png"]}
        :param image_fold_path: 样本路径
        :param output_rec_path:
        :param output_idx_path:
        :return:
        """

        # 删除已有文件
        if os.path.exists(output_rec_path):
            os.remove(output_rec_path)
            pass

        if os.path.exists(output_idx_path):
            os.remove(output_idx_path)
            pass

        # 获取 id list
        identity_list = [key for key in sample_dict]
        # 对 id list 升序操作
        identity_list.sort()
        # 对 sample 进行排序
        image_count = 1
        sample_info_list = []
        identity_sample_num_list = []
        for identity_num in identity_list:
            image_path_list = sample_dict[identity_num]
            image_path_list_len = len(image_path_list)
            identity_index = identity_list.index(identity_num)

            identity_sample_num_list.append([identity_index, image_path_list_len])

            for image_info in image_path_list:
                image_path = os.path.join(image_fold_path, image_info)
                sample_info_list.append([image_count, image_path, identity_index])
                image_count += 1
                pass
            pass

        print("sample_info_list: {}".format(sample_info_list))

        sample_info_list_len = len(sample_info_list)
        identity_list_len = len(identity_list)

        # 保存 property 文件
        output_fold_path = os.path.split(output_rec_path)[0]
        identity_property_path = os.path.join(output_fold_path, "property")
        sample_info_0 = sample_info_list[0]
        image_cv2 = cv2.imread(sample_info_0[1])
        h, w, c = image_cv2.shape
        with open(identity_property_path, "w") as file:
            train_identity_num = str(identity_list_len) + "," + str(h) + "," + str(w)
            file.write(train_identity_num)
            pass

        print("读取样本并保存为 .rec and .idx 数据......")
        record = mx.recordio.MXIndexedRecordIO(output_idx_path, output_rec_path, "w")

        # 空字节
        null_byte = b''

        # 设置 header_0 信息
        identity_sample_start_index = sample_info_list_len + 1
        identity_sample_end_index = sample_info_list_len + identity_list_len + 1
        label_num = np.array([identity_sample_start_index, identity_sample_end_index])
        header_0 = mx.recordio.IRHeader(0, label_num, 0, 0)
        image_record_0 = mx.recordio.pack(header_0, null_byte)
        record.write_idx(0, image_record_0)

        for sample_info in sample_info_list:
            header = mx.recordio.IRHeader(0, sample_info[2], sample_info[0], 0)
            with open(sample_info[1], "rb") as file:
                image = file.read()

                image_record = mx.recordio.pack(header, image)

                record.write_idx(sample_info[0], image_record)
                pass
            pass

        header_count = 1
        count_index = identity_sample_start_index
        for identity_sample_num in identity_sample_num_list:
            sample_num = identity_sample_num[1]

            label_num = np.array([header_count, header_count + sample_num])
            header = mx.recordio.IRHeader(0, label_num, count_index, 0)
            image_record = mx.recordio.pack(header, null_byte)
            record.write_idx(count_index, image_record)

            header_count += sample_num
            count_index += 1
            pass

        pass

    def get_train_data_lfw(self):
        data_utils.data_file_rename(self.face_classify_fold_path)

        image_info_path_list_generator = data_utils.get_file_path_list(input_file_path=self.face_classify_fold_path,
                                                                       recursive=self.recursive_flag,
                                                                       suffix_info_list=self.image_suffix_list)
        image_info_path_list = list(image_info_path_list_generator)

        image_path_list = []
        for image_info_path in image_info_path_list:
            image_path = image_info_path[1]
            image_path_list.append(image_path)
            pass
        print("image_path_list: {}".format(image_path_list))

        train_sample_dict, val_sample_dict = self.data_divide(image_path_list,
                                                              self.train_percent,
                                                              self.face_classify_fold_path,
                                                              self.train_data_path,
                                                              self.val_data_path)

        self.generate_bin_data(image_data_path=self.face_classify_fold_path,
                               output_bin_path=self.val_bin_fold_path,
                               bin_name=self.bin_name,
                               pair_file_path=self.pair_file_path,
                               sample_dict=val_sample_dict,
                               positive_sample_num=self.positive_sample_num,
                               negative_times=self.negative_times)

        rec_path = os.path.join(self.train_fold_path, self.rec_name)
        idx_path = os.path.join(self.train_fold_path, self.idx_name)

        if not os.path.exists(self.train_fold_path):
            os.makedirs(self.train_fold_path)
            pass
        self.generate_rec_idx_data(sample_dict=train_sample_dict,
                                   image_fold_path=self.face_classify_fold_path,
                                   output_rec_path=rec_path,
                                   output_idx_path=idx_path)

        pass


if __name__ == "__main__":
    from datetime import datetime

    # 代码开始时间
    start_time = datetime.now()
    print("开始时间: {}".format(start_time))

    demo = Prepare()
    demo.get_train_data_lfw()

    # 代码结束时间
    end_time = datetime.now()
    print("结束时间: {}, 训练模型耗时: {}".format(end_time, end_time - start_time))
    pass
