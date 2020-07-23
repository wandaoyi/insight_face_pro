#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/07/09 23:08
# @Author   : WanDaoYi
# @FileName : triplet_image_iter.py
# ============================================

import os
import copy
import numbers
import sklearn
import numpy as np

import mxnet as mx
from mxnet import ndarray as nd
from mxnet import io
from mxnet import recordio
from annoy import AnnoyIndex


class TripletFaceImageIter(io.DataIter):

    def __init__(self, rec_data_path=None, idx_data_path=None, batch_size=16,
                 data_shape=(3, 112, 112), shuffle_flag=False, rand_mirror=False,
                 cutoff=0, ctx_num=0, images_per_identity=0, triplet_params=None,
                 mx_model=None, data_name="data", label_name="softmax_label"):

        super(TripletFaceImageIter, self).__init__()

        # 检查 data_shape 格式
        self.check_data_shape(data_shape)

        assert rec_data_path, "rec_data_path is None or empty!"

        if idx_data_path is None or idx_data_path == "":
            path_info = os.path.splitext(rec_data_path)[0]
            idx_data_path = path_info + ".idx"
            pass

        print("loading record data: {} and {} ....".format(rec_data_path, idx_data_path))

        self.image_rec = recordio.MXIndexedRecordIO(idx_data_path, rec_data_path, "r")

        assert self.image_rec is not None, "image_rec is None...."

        self.header_0, self.image_idx, self.identity_2_range_dict = self.get_image_idx()

        self.seq_image_idx = self.image_idx
        self.ori_seq_image_idx = copy.deepcopy(self.image_idx)
        print("seq_image_idx_len: {}".format(self.seq_image_idx))

        self.data_shape = data_shape
        self.batch_size = batch_size
        self.shuffle_flag = shuffle_flag
        self.rand_mirror = rand_mirror
        self.cutoff = cutoff
        # gpu 设备数
        self.ctx_num = ctx_num
        # 每个 gpu 上, 每个人脸身份的图像数
        self.images_per_identity = images_per_identity
        self.mx_model = mx_model
        self.triplet_params = triplet_params
        self.triplet_mode = False

        self.provide_data = [(data_name, (self.batch_size,) + self.data_shape)]
        self.provide_label = [(label_name, (self.batch_size,))]

        # 每个 gpu 的 batch size
        self.per_batch_size = int(self.batch_size / self.ctx_num)

        assert self.images_per_identity > 0, "images_per_identity is not more than zero..."

        # 获取迭代次数 和 每个 gpu 的人脸身份数
        self.repeat, self.per_identities = self.get_repeat()

        assert self.triplet_params is not None, "triplet_params is None..."
        assert self.mx_model is not None, "mx_model is None..."

        self.triplet_bag_size = self.triplet_params[0]
        self.triplet_alpha = self.triplet_params[1]
        self.triplet_max_ap = self.triplet_params[2]

        self.seq_min_size = self.batch_size * 2
        self.triplet_mode = True
        self.hard_mining = False
        self.triplet_cur = 0
        self.triplet_seq = []
        self.triplet_reset()

        self.cur = 0
        self.n_batch = 0
        self.is_init = False

        pass

    def triplet_reset(self):
        self.triplet_cur = 0
        self.triplet_seq = []

        identity_list = []
        for identity in self.identity_2_range_dict:
            identity_list.append(identity)
            pass

        # 洗牌
        np.random.shuffle(identity_list)

        for identity in identity_list:
            # 一个 tuple 的 label value: (start_label, end_label)
            label_value = self.identity_2_range_dict[identity]
            label_list = range(*label_value)
            # 洗牌
            np.random.shuffle(label_list)
            if len(label_list) > self.images_per_identity:
                label_list = label_list[0: self.images_per_identity]
                pass

            self.triplet_seq += label_list
            pass

        print("triplet_seq_len: {}".format(len(self.triplet_seq)))
        assert len(self.triplet_seq) >= self.triplet_bag_size
        pass

    def select_triplets(self):
        self.seq_image_idx = []
        while len(self.seq_image_idx) < self.seq_min_size:

            embedding_feature = None

            bag_size = self.triplet_bag_size
            batch_size = self.batch_size
            tag = []

            if self.triplet_cur + bag_size > len(self.triplet_seq):
                self.triplet_reset()
                pass

            data = nd.zeros(self.provide_data[0][1])
            if self.provide_label is not None:
                label = nd.zeros(self.provide_label[0][1])
                pass
            else:
                label = None
                pass

            # 一个batch size 起始的下标号
            one_batch_start_index = 0
            while True:
                # 一个batch size 结束的下标号
                one_batch_end_index = min(one_batch_start_index + batch_size, bag_size)

                if one_batch_start_index >= one_batch_end_index:
                    break
                    pass

                for batch_index in range(one_batch_start_index, one_batch_end_index):
                    index = batch_index + self.triplet_cur
                    identity = self.triplet_seq[index]
                    image_rec_idx = self.image_rec.read_idx(identity)
                    header, image_str = recordio.unpack(image_rec_idx)

                    image_arr = mx.image.imdecode(image_str)

                    data_index = batch_index - one_batch_start_index
                    data[data_index][:] = self.post_process_data(image_arr)

                    label_info = header.label
                    if not isinstance(label_info, numbers.Number):
                        label_info = label_info[0]
                        pass
                    if label is not None:
                        label[data_index][:] = label_info
                        pass
                    tag.append((int(label_info), identity))
                    pass

                data_batch = mx.io.DataBatch(data=(data,))
                self.mx_model.forward(data_batch, is_train=False)
                net_out = self.mx_model.get_outputs()
                net_out = net_out[0].asnumpy()

                if embedding_feature is None:
                    embedding_feature = np.zeros((bag_size, net_out.shape[1]))
                    pass

                embedding_feature[one_batch_start_index: one_batch_end_index, :] = net_out

                one_batch_start_index = one_batch_end_index
                pass

            assert len(tag) == bag_size, "len(tag) neq bag_size...."
            self.triplet_cur += bag_size

            norm_embedding_feature = sklearn.preprocessing.normalize(embedding_feature)

            images_per_identity = [1]
            for bag_index in range(1, bag_size):
                if tag[bag_index][0] == tag[bag_index - 1][0]:
                    images_per_identity[-1] += 1
                    pass
                else:
                    images_per_identity.append(1)
                    pass
                pass

            # shape=(T, 3)
            triplet_list = self.pick_triplets(norm_embedding_feature, images_per_identity)

            # 一个batch size 起始的下标号
            one_batch_start_index = 0
            while True:
                # 一个batch size 结束的下标号
                one_batch_end_index = one_batch_start_index + self.per_batch_size // 3

                if one_batch_end_index > len(triplet_list):
                    break
                    pass

                triplet_info = triplet_list[one_batch_start_index: one_batch_end_index]

                for i in range(3):
                    for triplet in triplet_info:
                        pos_info = triplet[i]
                        idx_info = tag[pos_info][1]
                        self.seq_image_idx.append(idx_info)
                        pass
                    pass

                one_batch_start_index = one_batch_end_index
            pass
            pass
        pass

    def pick_triplets(self, embedding_feature, images_per_identity):

        feature_start_identity = 0
        triplet_list = []
        identity_per_batch = len(images_per_identity)

        dist_list = self.pairwise_dists(embedding_feature)

        for identity in range(identity_per_batch):
            # 每个身份人脸的图像数量
            images_num = int(images_per_identity[identity])
            for image_index in range(1, images_num):
                one_identity = feature_start_identity + image_index - 1
                neg_dists_square = dist_list[one_identity]

                for pair in range(image_index, images_num):
                    pair_identity = feature_start_identity + pair
                    feature_square = np.square(embedding_feature[one_identity] - embedding_feature[pair_identity])
                    pos_dist_square = np.sum(feature_square)

                    neg_dists_square[feature_start_identity: feature_start_identity + images_num] = np.NaN

                    if self.triplet_max_ap > 0.0:
                        if pos_dist_square > self.triplet_max_ap:
                            continue
                            pass
                        pass

                    # FaceNet selection
                    all_neg = np.where(np.logical_and(neg_dists_square - pos_dist_square < self.triplet_alpha,
                                                      pos_dist_square < neg_dists_square))[0]

                    negs_identity_num = all_neg.shape[0]
                    if negs_identity_num > 0:
                        random_negs_identity_index = np.random.randint(negs_identity_num)
                        neg_identity = all_neg[random_negs_identity_index]
                        triplet_list.append((one_identity, pair_identity, neg_identity))
                        pass
                    pass
                pass

            feature_start_identity += images_num
            pass

        np.random.shuffle(triplet_list)

        return triplet_list
        pass

    def pairwise_dists(self, embedding_feature):
        nd_feature_list = []
        nd_dist_list = []
        dist_list = []
        for ctx_index in range(self.ctx_num):
            nd_feature = mx.nd.array(embedding_feature, mx.gpu(ctx_index))
            nd_feature_list.append(nd_feature)
            pass

        for identity in range(embedding_feature.shape[0]):
            feature_identity = identity % self.ctx_num
            nd_feature = nd_feature_list[feature_identity]
            feature = nd_feature[identity]
            body = mx.nd.broadcast_sub(feature, nd_feature)
            body = body * body
            body = mx.nd.sum_axis(body, axis=1)
            nd_dist_list.append(body)

            if len(nd_dist_list) == self.ctx_num or identity == embedding_feature.shape[0] - 1:
                for dist in nd_dist_list:
                    dist_list.append(dist.asnumpy())
                    pass
                nd_dist_list = []
                pass
            pass
        return dist_list
        pass

    def get_repeat(self, iter_image_num=3000000.0):
        """
            获取迭代次数 和 人脸身份数
        :param iter_image_num: 迭代图像的数量, float
        :return:
        """
        print("iter_image_num: {}".format(iter_image_num))
        # 获取每个 gpu 人脸身份数
        per_identities = int(self.per_batch_size / self.images_per_identity)
        all_images_of_identities = self.images_per_identity * len(self.identity_2_range_dict)
        repeat = int(iter_image_num / all_images_of_identities)

        return repeat, per_identities
        pass

    def get_image_idx(self):
        image_rec_idx = self.image_rec.read_idx(0)
        header, _ = recordio.unpack(image_rec_idx)

        assert header.flag > 0, "header.flag is not more than zero....header: {}".format(header)

        header_0 = (int(header.label[0]), int(header.label[1]))

        image_idx = range(1, int(header.label[0]))
        identity_2_range_dict = {}

        seq_identity = range(int(header.label[0]), int(header.label[1]))

        for identity in seq_identity:
            image_rec_idx = self.image_rec.read_idx(identity)
            header, _ = recordio.unpack(image_rec_idx)
            start_label = int(header.label[0])
            end_label = int(header.label[1])
            identity_2_range_dict[identity] = (start_label, end_label)
            pass

        return header_0, image_idx, identity_2_range_dict
        pass

    def hard_mining_reset(self):
        data = nd.zeros(self.provide_data[0][1])
        label = nd.zeros(self.provide_label[0][1])

        embedding_feature = None
        # 一个batch size 起始的下标号
        one_batch_start_index = 0
        batch_num = 0

        ori_seq_image_idx_len = len(self.ori_seq_image_idx)
        while one_batch_start_index < ori_seq_image_idx_len:
            batch_num += 1
            if batch_num % 10 == 0:
                print("loading batch_num: {}, one_batch_start_index: {}".format(batch_num, one_batch_start_index))
                pass

            # 一个batch size 结束的下标号
            one_batch_end_index = min(one_batch_start_index + self.batch_size, ori_seq_image_idx_len)

            count = one_batch_end_index - one_batch_start_index

            for batch_index in range(count):
                identity_index = batch_index + one_batch_start_index
                identity = self.ori_seq_image_idx[identity_index]
                image_rec_idx = self.image_rec.read_idx(identity)
                header, image_str = recordio.unpack(image_rec_idx)

                image_arr = mx.image.imdecode(image_str)
                data[batch_index][:] = self.post_process_data(image_arr)
                label[batch_index][:] = header.label
                pass

            data_batch = mx.io.DataBatch(data=(data,), label=(label,))
            self.mx_model.forward(data_batch, is_train=False)
            net_out = self.mx_model.get_outputs()
            feature = net_out[0].asnumpy()
            norm_feature = sklearn.preprocessing.normalize(feature)

            if count < self.batch_size:
                norm_feature = norm_feature[0: count, :]
                pass

            if embedding_feature is None:
                # shape: (data_batch, 128)
                embedding_feature = np.zeros((len(self.identity_2_range_dict), norm_feature.shape[1]), dtype=np.float32)
                pass

            np_label = label.asnumpy()

            for batch_index in range(count):
                label_index = int(np_label[batch_index])
                embedding_feature[label_index] += norm_feature[batch_index]
                pass

            one_batch_start_index = one_batch_end_index
            pass

        norm_embedding_feature = sklearn.preprocessing.normalize(embedding_feature)

        thresh_value = AnnoyIndex(norm_embedding_feature.shape[1], metric="euclidean")

        for identity in range(norm_embedding_feature.shape[0]):
            thresh_value.add_item(identity, norm_embedding_feature[identity])
            pass

        thresh_value.build(20)

        k = self.per_identities

        self.seq_image_idx = []

        for identity in range(norm_embedding_feature.shape[0]):
            nn_list = thresh_value.get_nns_by_item(identity, k)

            assert nn_list[0] == identity

            for label_info in nn_list:

                assert label_info < len(self.identity_2_range_dict)

                identity_index = self.header_0[0] + label_info
                label_value = self.identity_2_range_dict[identity_index]
                label_list = range(*label_value)

                if len(label_list) < self.images_per_identity:
                    np.random.shuffle(label_list)
                    pass
                else:
                    label_list = np.random.choice(label_list, self.images_per_identity, replace=False)
                    pass

                for image_index in range(self.images_per_identity):
                    identity_value = label_list[identity % len(label_list)]
                    self.seq_image_idx.append(identity_value)
                    pass
                pass
            pass
        pass

    def reset(self):
        self.cur = 0

        if self.triplet_mode:
            self.select_triplets()
            pass
        elif not self.hard_mining:
            self.seq_image_idx = []
            identity_list = []
            for identity in self.identity_2_range_dict:
                label_value = self.identity_2_range_dict[identity]
                identity_list.append((identity, range(*label_value)))
                pass

            for repeat_index in range(self.repeat):
                if repeat_index % 10 == 0:
                    print("repeat_index: {}".format(repeat_index))
                    pass

                if self.shuffle_flag:
                    np.random.shuffle(identity_list)
                    pass

                for identity_and_label in identity_list:
                    # identity = identity_and_label[0]
                    label_range = identity_and_label[1]

                    if len(label_range) < self.images_per_identity:
                        np.random.shuffle(label_range)
                        pass
                    else:
                        # 从 label_range 随机选出 self.images_per_identity 个 label 组成一个 list
                        label_range = np.random.choice(label_range, self.images_per_identity, replace=False)
                        pass

                    label_range_len = len(label_range)
                    for image_index in range(self.images_per_identity):
                        label_info = label_range[image_index % label_range_len]
                        self.seq_image_idx.append(label_info)
                        pass
                    pass
                pass
            pass
        else:
            self.hard_mining_reset()
            pass

        if self.seq_image_idx is None and self.image_rec is not None:
            self.image_rec.reset()
            pass
        pass

    def next_sample(self):
        while True:
            if self.cur >= len(self.seq_image_idx):
                raise StopIteration
                pass

            identity = self.seq_image_idx[self.cur]
            self.cur += 1
            image_rec_idx = self.image_rec.read_idx(identity)
            header, image_str = recordio.unpack(image_rec_idx)
            label = header.label
            if not isinstance(label, numbers.Number):
                label = label[0]
                pass

            return label, image_str
            pass
        pass

    def next(self):
        if not self.is_init:
            self.reset()
            self.is_init = True
            pass

        self.n_batch += 1
        batch_size = self.batch_size
        c, h, w = self.data_shape
        batch_data = nd.empty((batch_size, c, h, w))

        if self.provide_label is not None:
            batch_label = nd.empty(self.provide_label[0][1])
            pass
        else:
            batch_label = None
            pass

        batch_num = 0

        try:
            while batch_num < batch_size:
                label, image_str = self.next_sample()
                image_arr = mx.image.imdecode(image_str)

                if self.rand_mirror:
                    random_num = np.random.randint(0, 2)
                    if random_num == 1:
                        image_arr = mx.ndarray.flip(data=image_arr, axis=1)
                        pass
                    pass

                if self.cutoff > 0:
                    center_h = np.random.randint(0, image_arr.shape[0])
                    center_w = np.random.randint(0, image_arr.shape[1])
                    half = self.cutoff // 2
                    start_h = max(0, center_h - half)
                    end_h = min(image_arr.shape[0], center_h + half)
                    start_w = max(0, center_w - half)
                    end_w = min(image_arr.shape[1], center_w + half)

                    image_arr = image_arr.astype("float32")
                    image_arr[start_h: end_h, start_w: end_w, :] = 127.5
                    pass

                image_data = [image_arr]

                try:
                    self.check_valid_image(image_data)
                    pass
                except RuntimeError as e:
                    print("Invalid image, skipping: {}".format(str(e)))
                    pass

                for image in image_data:
                    assert batch_num < batch_size, "Batch size must be multiples of augmenter output length"
                    batch_data[batch_num][:] = self.post_process_data(image)

                    if self.provide_label is not None:
                        batch_label[batch_num][:] = label
                        pass

                    batch_num += 1
                    pass
                pass
            pass
        except StopIteration:
            if batch_num < batch_size:
                raise StopIteration
                pass
            pass

        label_list = None
        if self.provide_label is not None:
            label_list = [batch_size]
            pass

        return io.DataBatch([batch_data], label_list, batch_size - batch_num)
        pass

    @staticmethod
    def check_valid_image(data):
        if len(data[0].shape) == 0:
            raise RuntimeError('Data shape is wrong')
            pass
        pass

    @staticmethod
    def post_process_data(image_arr):
        """
            [height, width, channel] ---> [channel, height, width]
        :param image_arr:
        :return:
        """
        return nd.transpose(image_arr, axes=(2, 0, 1))
        pass

    @staticmethod
    def check_data_shape(data_shape):
        """
            检测 data shape 格式是否合规范
        :param data_shape: [channel, height, width] ---> [3, 112, 112]
        :return:
        """
        if not len(data_shape) == 3:
            raise ValueError('data_shape should have length 3, with dimensions CxHxW')
        if not data_shape[0] == 3:
            raise ValueError('This iterator expects inputs to have 3 channels.')
        pass

    pass


class TripletFaceImageIterList(io.DataIter):
    def __init__(self, iter_list):
        super(TripletFaceImageIterList, self).__init__()
        assert len(iter_list) > 0
        self.provide_data = iter_list[0].provide_data
        self.provide_label = iter_list[0].provide_label
        self.iter_list = iter_list
        self.cur_iter = None

    def reset(self):
        self.cur_iter.reset()

    def next(self):
        self.cur_iter = np.random.choice(self.iter_list)
        while True:
            try:
                ret = self.cur_iter.next()
            except StopIteration:
                self.cur_iter.reset()
                continue
            return ret
