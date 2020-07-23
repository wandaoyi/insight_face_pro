#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/07/09 12:41
# @Author   : WanDaoYi
# @FileName : image_iter.py
# ============================================

import os
import numbers
import numpy as np
from PIL import Image
from io import BytesIO

import mxnet as mx
from mxnet import ndarray as nd
from mxnet import io
from mxnet import recordio


class FaceImageIter(io.DataIter):

    def __init__(self, rec_data_path=None, idx_data_path=None, batch_size=16,
                 data_shape=(3, 112, 112), shuffle_flag=False, mean=None,
                 rand_mirror=False, cutoff=0, color_jitter=0, images_filter=0,
                 data_name="data", label_name="softmax_label"):
        super(FaceImageIter, self).__init__()

        assert rec_data_path, "rec_data_path is None or empty!"
        if idx_data_path is None or idx_data_path == "":
            path_info = os.path.splitext(rec_data_path)[0]
            idx_data_path = path_info + ".idx"
            pass

        print("loading record data: {} and {} ....".format(rec_data_path, idx_data_path))

        self.image_rec = recordio.MXIndexedRecordIO(idx_data_path, rec_data_path, "r")

        assert self.image_rec is not None, "image_rec is None...."

        # images_filter: 表示每个人的图像数目要大于该值才进行训练
        self.image_idx = self.get_image_idx(images_filter)

        if shuffle_flag:
            self.seq_image_idx = self.image_idx
            self.ori_seq_image_idx = self.image_idx.copy()
            pass
        else:
            self.seq_image_idx = None
            pass

        self.mean, self.nd_mean = self.get_mean(mean)

        # 检查 data_shape 格式
        self.check_data_shape(data_shape)

        self.data_shape = tuple(data_shape)
        self.batch_size = batch_size
        self.shuffle_flag = shuffle_flag
        self.rand_mirror = rand_mirror
        self.cutoff = cutoff
        self.color_jitter = color_jitter
        # 用于色彩增强
        self.color_jitter_aug = mx.image.ColorJitterAug(0.125, 0.125, 0.125)

        self.provide_data = [(data_name, (self.batch_size,) + self.data_shape)]
        self.provide_label = [(label_name, (self.batch_size,))]

        self.cur = 0
        self.n_batch = 0
        self.is_init = False

        pass

    def get_image_idx(self, images_filter):
        image_rec_idx = self.image_rec.read_idx(0)
        header, _ = recordio.unpack(image_rec_idx)
        if header.flag > 0:
            image_idx = []
            seq_identity = range(int(header.label[0]), int(header.label[1]))
            for identity in seq_identity:
                image_rec_idx = self.image_rec.read_idx(identity)
                header, _ = recordio.unpack(image_rec_idx)
                start_label = int(header.label[0])
                end_label = int(header.label[1])
                count = end_label - start_label
                if count < images_filter:
                    continue
                    pass
                image_idx += range(start_label, end_label)
                pass
            pass
        else:
            image_idx = list(self.image_rec.keys)
            pass

        return image_idx
        pass

    @staticmethod
    def get_mean(mean):
        if mean:
            mean_info = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
            nd_mean_info = mx.nd.array(mean_info).reshape((1, 1, 3))
            pass
        else:
            mean_info = mean
            nd_mean_info = None
            pass

        return mean_info, nd_mean_info
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

    def reset(self):
        """Resets the iterator to the beginning of the data."""
        print('call reset()')
        self.cur = 0
        if self.shuffle_flag:
            np.random.shuffle(self.seq_image_idx)
            pass
        if self.seq_image_idx is None and self.image_rec is not None:
            self.image_rec.reset()
            pass
        pass

    def next_sample(self):
        """
            reading in next sample
        :return:
        """
        if self.seq_image_idx is not None:
            while True:
                if self.cur >= len(self.seq_image_idx):
                    raise StopIteration
                    pass

                index = self.seq_image_idx[self.cur]
                self.cur += 1

                image_rec_idx = self.image_rec.read_idx(index)
                header, image_str = recordio.unpack(image_rec_idx)
                label = header.label
                if not isinstance(label, numbers.Number):
                    label = label[0]
                return label, image_str
                pass
            pass
        else:
            image_rec_idx = self.image_rec.read()
            if image_rec_idx is None:
                raise StopIteration
                pass
            header, image_str = recordio.unpack(image_rec_idx)
            label = header.label
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
        batch_label = nd.empty(self.provide_label[0][1])

        i = 0

        try:
            while i < batch_size:
                label, image_str = self.next_sample()
                # image_str ---> NDArray 112x112x3
                # 可以 image_arr.asnumpy() ---> numpy array type
                # plt.imshow() 来展示
                image_arr = mx.image.imdecode(image_str)

                if image_arr.shape[0] != image_arr.shape[1]:
                    image_arr = mx.image.resize_short(image_arr, self.data_shape[1])
                    pass

                # 镜像翻转
                if self.rand_mirror:
                    rand_int = np.random.randint(0, 2)
                    if rand_int == 1:
                        image_arr = mx.ndarray.flip(data=image_arr, axis=1)
                        pass
                    pass

                if self.color_jitter > 0:
                    if self.color_jitter > 1:
                        rand_int = np.random.randint(0, 2)
                        if rand_int == 1:
                            # 精简增强
                            image_arr = self.compress_aug(image_arr)
                            pass
                        pass

                    # 将 像素转为 float32
                    image_arr = image_arr.astype("float32", copy=False)
                    # 颜色增强
                    image_arr = self.color_jitter_aug(image_arr)
                    pass

                if self.nd_mean is not None:
                    image_arr = image_arr.astype('float32', copy=False)
                    image_arr -= self.nd_mean
                    image_arr *= 0.0078125
                    pass

                # 随机裁剪
                if self.cutoff > 0:
                    rand_int = np.random.randint(0, 2)
                    if rand_int == 1:
                        center_h = np.random.randint(0, image_arr.shape[0])
                        center_w = np.random.randint(0, image_arr.shape[1])
                        half = self.cutoff // 2
                        start_h = max(0, center_h - half)
                        end_h = min(image_arr.shape[0], center_h + half)
                        start_w = max(0, center_w - half)
                        end_w = min(image_arr.shape[1], center_w + half)
                        image_arr[start_h: end_h, start_w: end_w, :] = 128
                        pass
                    pass

                image_data = [image_arr]

                try:
                    # 检测图像数据
                    self.check_valid_image(image_data)
                    pass
                except RuntimeError as e:
                    print("Invalid image, skipping: {}".format(e))
                    continue
                    pass

                for image_info in image_data:
                    assert i < batch_size, 'Batch size must be multiples of augmenter output length'

                    # [height, width, channel] ---> [channel, height, width]
                    batch_data[i][:] = self.post_process_data(image_info)
                    batch_label[i][:] = label
                    i += 1
                    pass
                pass
            pass
        except StopIteration:
            if i < batch_size:
                raise StopIteration
            pass

        return io.DataBatch([batch_data], [batch_label], batch_size - i)
        pass

    @staticmethod
    def compress_aug(image):
        buf = BytesIO()
        image = Image.fromarray(image.asnumpy(), "RGB")
        rand_int = np.random.randint(2, 21)
        image.save(buf, format="JPEG", quality=rand_int)
        buf = buf.getvalue()
        image = Image.open(BytesIO(buf))

        return nd.array(np.asarray(image, "float32"))
        pass

    @staticmethod
    def check_valid_image(image_data):
        """
            检查图像数据
        :param image_data:
        :return:
        """
        if len(image_data[0].shape) == 0:
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


class FaceImageIterList(io.DataIter):
    def __init__(self, iter_list):
        super(FaceImageIterList, self).__init__()
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
