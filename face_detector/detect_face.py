#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/07/13 15:11
# @Author   : WanDaoYi
# @FileName : detect_face.py
# ============================================

import os
import numpy as np
import tensorflow as tf
from utils import data_utils
from six import string_types, iteritems
from config import cfg


def layer(op):
    """
        Decorator for composable network layers
    :param op:
    :return:
    """

    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.terminals) == 0:
            raise RuntimeError("No input variables found for layer {}".format(name))
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated
    pass


class Network(object):

    def __init__(self, inputs, trainable=True):
        # The input nodes for this network
        self.inputs = inputs
        # The current list of terminal nodes
        self.terminals = []
        # Mapping from layer names to layers
        self.layers = dict(inputs)
        # If true, the resulting variables are set as trainable
        self.trainable = trainable

        self.setup()

        pass

    def setup(self):
        """
            Construct the network
        :return:
        """
        raise NotImplementedError('Must be implemented by the subclass.')
        pass

    @staticmethod
    def load(data_path, session, ignore_missing=False):
        """

        :param data_path:
        :param session:
        :param ignore_missing:
        :return:
        """
        data_dict = np.load(data_path, encoding='latin1', allow_pickle=True).item()  # pylint: disable=no-member

        for op_name in data_dict:
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in iteritems(data_dict[op_name]):
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                        pass
                    except ValueError:
                        if not ignore_missing:
                            raise
                        pass
                    pass
                pass
            pass
        pass

    def feed(self, *args):
        """

        :param args:
        :return:
        """
        assert len(args) != 0, "len(args) is zero..."
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, string_types):
                try:
                    fed_layer = self.layers[fed_layer]
                    pass
                except KeyError:
                    raise KeyError("Unknown layer name fed: {}".format(fed_layer))
                pass
            self.terminals.append(fed_layer)
            pass
        return self
        pass

    def get_unique_name(self, prefix):
        identity = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1

        return "{}_{}".format(prefix, identity)
        pass

    def make_var(self, name, shape):
        """

        :param name:
        :param shape:
        :return:
        """
        return tf.get_variable(name, shape, trainable=self.trainable)
        pass

    @staticmethod
    def validate_padding(padding):
        assert padding in ('SAME', 'VALID')
        pass

    @layer
    def conv(self, inp, k_h, k_w, c_o, s_h, s_w, name,
             relu=True, padding='SAME', group=1, biased=True):

        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        c_i = int(inp.get_shape()[-1])
        # Verify that the grouping parameter is valid
        assert c_i % group == 0
        assert c_o % group == 0
        # Convolution for a given input and kernel
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i // group, c_o])
            # This is the common-case. Convolve the input without any further complications.
            output = convolve(inp, kernel)
            # Add the biases
            if biased:
                biases = self.make_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)
                pass

            if relu:
                # ReLU non-linearity
                output = tf.nn.relu(output, name=scope.name)
                pass
            return output
        pass

    @layer
    def prelu(self, inp, name):
        with tf.variable_scope(name):
            i = int(inp.get_shape()[-1])
            alpha = self.make_var('alpha', shape=(i,))
            output = tf.nn.relu(inp) + tf.multiply(alpha, -tf.nn.relu(-inp))
        return output

    @layer
    def max_pool(self, inp, k_h, k_w, s_h, s_w, name, padding='SAME'):
        self.validate_padding(padding)
        return tf.nn.max_pool(inp,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def fc(self, inp, num_out, name, relu=True):
        with tf.variable_scope(name):
            input_shape = inp.get_shape()
            if input_shape.ndims == 4:
                # The input is spatial. Vectorize it first.
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= int(d)
                feed_in = tf.reshape(inp, [-1, dim])
            else:
                feed_in, dim = (inp, input_shape[-1].value)
            weights = self.make_var('weights', shape=[dim, num_out])
            biases = self.make_var('biases', [num_out])
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=name)
            return fc

        pass

    @layer
    def softmax(self, target, axis, name=None):
        max_axis = tf.reduce_max(target, axis, keepdims=True)
        target_exp = tf.exp(target - max_axis)
        normalize = tf.reduce_sum(target_exp, axis, keepdims=True)
        softmax = tf.div(target_exp, normalize, name)
        return softmax


class PNet(Network):
    def setup(self):
        (self.feed('data')  # pylint: disable=no-value-for-parameter, no-member
         .conv(3, 3, 10, 1, 1, padding='VALID', relu=False, name='conv1')
         .prelu(name='PReLU1')
         .max_pool(2, 2, 2, 2, name='pool1')
         .conv(3, 3, 16, 1, 1, padding='VALID', relu=False, name='conv2')
         .prelu(name='PReLU2')
         .conv(3, 3, 32, 1, 1, padding='VALID', relu=False, name='conv3')
         .prelu(name='PReLU3')
         .conv(1, 1, 2, 1, 1, relu=False, name='conv4-1')
         .softmax(3, name='prob1'))

        (self.feed('PReLU3')  # pylint: disable=no-value-for-parameter
         .conv(1, 1, 4, 1, 1, relu=False, name='conv4-2'))


class RNet(Network):
    def setup(self):
        (self.feed('data')  # pylint: disable=no-value-for-parameter, no-member
         .conv(3, 3, 28, 1, 1, padding='VALID', relu=False, name='conv1')
         .prelu(name='prelu1')
         .max_pool(3, 3, 2, 2, name='pool1')
         .conv(3, 3, 48, 1, 1, padding='VALID', relu=False, name='conv2')
         .prelu(name='prelu2')
         .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
         .conv(2, 2, 64, 1, 1, padding='VALID', relu=False, name='conv3')
         .prelu(name='prelu3')
         .fc(128, relu=False, name='conv4')
         .prelu(name='prelu4')
         .fc(2, relu=False, name='conv5-1')
         .softmax(1, name='prob1'))

        (self.feed('prelu4')  # pylint: disable=no-value-for-parameter
         .fc(4, relu=False, name='conv5-2'))


class ONet(Network):
    def setup(self):
        (self.feed('data')  # pylint: disable=no-value-for-parameter, no-member
         .conv(3, 3, 32, 1, 1, padding='VALID', relu=False, name='conv1')
         .prelu(name='prelu1')
         .max_pool(3, 3, 2, 2, name='pool1')
         .conv(3, 3, 64, 1, 1, padding='VALID', relu=False, name='conv2')
         .prelu(name='prelu2')
         .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
         .conv(3, 3, 64, 1, 1, padding='VALID', relu=False, name='conv3')
         .prelu(name='prelu3')
         .max_pool(2, 2, 2, 2, name='pool3')
         .conv(2, 2, 128, 1, 1, padding='VALID', relu=False, name='conv4')
         .prelu(name='prelu4')
         .fc(256, relu=False, name='conv5')
         .prelu(name='prelu5')
         .fc(2, relu=False, name='conv6-1')
         .softmax(1, name='prob1'))

        (self.feed('prelu5')  # pylint: disable=no-value-for-parameter
         .fc(4, relu=False, name='conv6-2'))

        (self.feed('prelu5')  # pylint: disable=no-value-for-parameter
         .fc(10, relu=False, name='conv6-3'))


def create_mt_cnn(sess, model_file_path=cfg.COMMON.DETECT_FACE_MODEL_PATH,
                  model_name_list=cfg.COMMON.DETECT_MODEL_NAME_LIST):
    print("model_file_path: {}".format(model_file_path))
    print("model_name_list: {}".format(model_name_list))
    det_model_path_1 = os.path.join(model_file_path, model_name_list[0])
    det_model_path_2 = os.path.join(model_file_path, model_name_list[1])
    det_model_path_3 = os.path.join(model_file_path, model_name_list[2])

    with tf.variable_scope('pnet'):
        data = tf.placeholder(tf.float32, (None, None, None, 3), 'input')
        p_net = PNet({'data': data})
        p_net.load(det_model_path_1, sess)
        pass

    with tf.variable_scope('rnet'):
        data = tf.placeholder(tf.float32, (None, 24, 24, 3), 'input')
        r_net = RNet({'data': data})
        r_net.load(det_model_path_2, sess)
        pass

    # 用于做关键点检测
    with tf.variable_scope('onet'):
        data = tf.placeholder(tf.float32, (None, 48, 48, 3), 'input')
        o_net = ONet({'data': data})
        o_net.load(det_model_path_3, sess)
        pass

    p_net_fun = lambda image: sess.run(('pnet/conv4-2/BiasAdd:0', 'pnet/prob1:0'), feed_dict={'pnet/input:0': image})
    r_net_fun = lambda image: sess.run(('rnet/conv5-2/conv5-2:0', 'rnet/prob1:0'), feed_dict={'rnet/input:0': image})
    o_net_fun = lambda image: sess.run(('onet/conv6-2/conv6-2:0', 'onet/conv6-3/conv6-3:0', 'onet/prob1:0'),
                                       feed_dict={'onet/input:0': image})

    return p_net_fun, r_net_fun, o_net_fun
    pass


def detect(image, min_size, p_net, r_net, o_net, threshold, factor):
    """
        检测人脸
    :param image:
    :param min_size:
    :param p_net:
    :param r_net:
    :param o_net:
    :param threshold:
    :param factor:
    :return:
    """
    factor_count = 0
    total_boxes = np.empty((0, 9))
    points = []
    h = image.shape[0]
    w = image.shape[1]
    # 获取较短的边长
    min_length = np.amin([h, w])
    # min_size = 60
    m = 12.0 / min_size
    min_length = min_length * m
    # create scale pyramid
    scales = []

    # 将人脸检测的最小边设置为 12, 即最小可以检测到 12 x 12 大小的人脸
    while min_length >= 12:
        scales += [m * np.power(factor, factor_count)]
        min_length = min_length * factor
        factor_count += 1
        pass

    # first stage
    for j in range(len(scales)):
        # 多尺度预测: 通过图像金字塔来进行 bounding box 预测
        scale = scales[j]
        hs = int(np.ceil(h * scale))
        ws = int(np.ceil(w * scale))
        image_resize = data_utils.resize_image(image, (hs, ws))
        # print("image_resize_shape: {}".format(image_resize.shape))
        image_data = (image_resize - 127.5) * 0.0078125
        image_x = np.expand_dims(image_data, 0)
        image_y = np.transpose(image_x, (0, 2, 1, 3))
        out = p_net(image_y)
        out_0 = np.transpose(out[0], (0, 2, 1, 3))
        out_1 = np.transpose(out[1], (0, 2, 1, 3))

        boxes, _ = data_utils.generate_bounding_box(out_1[0, :, :, 1].copy(),
                                                    out_0[0, :, :, :].copy(),
                                                    scale,
                                                    threshold[0])

        # inter-scale nms
        pick = data_utils.nms(boxes.copy(), 0.5, 'Union')
        if boxes.size > 0 and pick.size > 0:
            boxes = boxes[pick, :]
            total_boxes = np.append(total_boxes, boxes, axis=0)
            pass

        pass

    num_box = total_boxes.shape[0]
    if num_box > 0:
        pick = data_utils.nms(total_boxes.copy(), 0.7, 'Union')
        total_boxes = total_boxes[pick, :]
        # 计算 bounding box regression
        reg_w = total_boxes[:, 2] - total_boxes[:, 0]
        reg_h = total_boxes[:, 3] - total_boxes[:, 1]
        qq1 = total_boxes[:, 0] + total_boxes[:, 5] * reg_w
        qq2 = total_boxes[:, 1] + total_boxes[:, 6] * reg_h
        qq3 = total_boxes[:, 2] + total_boxes[:, 7] * reg_w
        qq4 = total_boxes[:, 3] + total_boxes[:, 8] * reg_h
        total_boxes = np.transpose(np.vstack([qq1, qq2, qq3, qq4, total_boxes[:, 4]]))
        total_boxes = data_utils.re_rec(total_boxes.copy())
        total_boxes[:, 0:4] = np.fix(total_boxes[:, 0:4]).astype(np.int32)
        dy, edy, dx, edx, y, ey, x, ex, tmp_w, tmp_h = data_utils.pad(total_boxes.copy(), w, h)
        pass
    else:
        dy = None
        edy = None
        dx = None
        edx = None
        y = None
        ey = None
        x = None
        ex = None
        tmp_w = None
        tmp_h = None
        pass

    num_box = total_boxes.shape[0]
    if num_box > 0:
        # second stage
        temp_image = np.zeros((24, 24, 3, num_box))

        for k in range(0, num_box):
            tmp = np.zeros((int(tmp_h[k]), int(tmp_w[k]), 3))
            tmp[dy[k] - 1:edy[k], dx[k] - 1:edx[k], :] = image[y[k] - 1: ey[k], x[k] - 1: ex[k], :]
            if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:
                temp_image[:, :, :, k] = data_utils.resize_image(tmp, (24, 24))
            else:
                return np.empty((0, 9))
            pass

        temp_image = (temp_image - 127.5) * 0.0078125
        temp_image_1 = np.transpose(temp_image, (3, 1, 0, 2))
        out = r_net(temp_image_1)
        # 计算 bounding box regression 的参数
        out_0 = np.transpose(out[0])
        # 人脸分类的概率值
        out_1 = np.transpose(out[1])
        # 人脸检测人脸目标的概率值
        score = out_1[1, :]
        # 阈值 threshold = [0.6, 0.85, 0.8]
        pass_index = np.where(score > threshold[1])
        total_boxes = np.hstack([total_boxes[pass_index[0], 0: 4].copy(), np.expand_dims(score[pass_index].copy(), 1)])
        mv = out_0[:, pass_index[0]]
        if total_boxes.shape[0] > 0:
            pick = data_utils.nms(total_boxes, 0.7, 'Union')
            total_boxes = total_boxes[pick, :]
            total_boxes = data_utils.bounding_box_reg(total_boxes.copy(), np.transpose(mv[:, pick]))
            total_boxes = data_utils.re_rec(total_boxes.copy())
            pass
        pass

    num_box = total_boxes.shape[0]
    if num_box > 0:
        # third stage
        total_boxes = np.fix(total_boxes).astype(np.int32)
        dy, edy, dx, edx, y, ey, x, ex, tmp_w, tmp_h = data_utils.pad(total_boxes.copy(), w, h)

        temp_image = np.zeros((48, 48, 3, num_box))

        for k in range(0, num_box):
            tmp = np.zeros((int(tmp_h[k]), int(tmp_w[k]), 3))
            tmp[dy[k] - 1:edy[k], dx[k] - 1:edx[k], :] = image[y[k] - 1:ey[k], x[k] - 1:ex[k], :]
            if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:
                temp_image[:, :, :, k] = data_utils.resize_image(tmp, (48, 48))
            else:
                return np.empty((0, 9))
            pass

        temp_image = (temp_image - 127.5) * 0.0078125
        temp_image_1 = np.transpose(temp_image, (3, 1, 0, 2))
        out = o_net(temp_image_1)
        # 计算 bounding box regression 的参数
        out0 = np.transpose(out[0])
        # 人脸关键点坐标概率信息
        out1 = np.transpose(out[1])
        # 人脸分类概率信息
        out2 = np.transpose(out[2])
        # 人脸检测人脸目标的概率值
        score = out2[1, :]
        # 获取人脸关键点矩阵
        points = out1
        # threshold = [0.6, 0.85, 0.8]
        # 获取 得分大于阈值的 数据下标
        pass_index = np.where(score > threshold[2])
        # 根据下标获取目标的人脸关键点矩阵
        points = points[:, pass_index[0]]
        # 水平方向 concat, 将 坐标值 与 概率值 concat 到一起.
        # 例如: np.hstack([x1, y1, x2, y2], [0.997]) ---> [x1, y1, x2, y2, 0.997]
        total_boxes = np.hstack([total_boxes[pass_index[0], 0: 4].copy(), np.expand_dims(score[pass_index].copy(), 1)])
        mv = out0[:, pass_index[0]]

        w = total_boxes[:, 2] - total_boxes[:, 0] + 1
        h = total_boxes[:, 3] - total_boxes[:, 1] + 1
        # np.tile(w, (5, 1)) 为 将 w 复制为 5 倍的行 信息
        # 例如: w = [253., 270., 264., 270., 283., 261., 267., 265.]
        # 复制后为: [[253., 270., 264., 270., 283., 261., 267., 265.],
        #           [253., 270., 264., 270., 283., 261., 267., 265.],
        #           [253., 270., 264., 270., 283., 261., 267., 265.],
        #           [253., 270., 264., 270., 283., 261., 267., 265.]]
        # 将 points 的概率值 转为 坐标值
        points[0: 5, :] = np.tile(w, (5, 1)) * points[0: 5, :] + np.tile(total_boxes[:, 0], (5, 1)) - 1
        points[5: 10, :] = np.tile(h, (5, 1)) * points[5: 10, :] + np.tile(total_boxes[:, 1], (5, 1)) - 1
        if total_boxes.shape[0] > 0:
            # 计算 bounding box regression
            total_boxes = data_utils.bounding_box_reg(total_boxes.copy(), np.transpose(mv))
            # 通过 nms 找出最大概率的目标
            pick = data_utils.nms(total_boxes.copy(), 0.7, 'Min')
            total_boxes = total_boxes[pick, :]
            points = points[:, pick]
            pass

    # total_boxes 为目标的 bounding boxes, points 为关键点的坐标
    return total_boxes, points
    pass
