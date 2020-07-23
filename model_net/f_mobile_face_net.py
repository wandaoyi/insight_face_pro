#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/07/02 18:01
# @Author   : WanDaoYi
# @FileName : f_mobile_face_net.py
# ============================================

import mxnet as mx
from utils import model_utils
from config import cfg


def act_function(data, act_type, name):
    # ignore param act_type, set it in this function
    if act_type == 'prelu':
        body = mx.sym.LeakyReLU(data=data, act_type='prelu', name=name)
    else:
        body = mx.sym.Activation(data=data, act_type=act_type, name=name)
    return body
    pass


def conv_feature(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                 bn_mom=cfg.TRAIN.MOMENTUM, net_act=cfg.TRAIN.NET_ACT,
                 num_group=1, name=None, suffix=""):
    """

    :param data:
    :param num_filter:
    :param kernel:
    :param stride:
    :param pad:
    :param bn_mom:
    :param net_act:
    :param num_group:
    :param name:
    :param suffix:
    :return:
    """
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride,
                              pad=pad, no_bias=True, name='%s%s_conv2d' % (name, suffix))

    bn = mx.sym.BatchNorm(data=conv, name='%s%s_batchnorm' % (name, suffix), fix_gamma=False, momentum=bn_mom)
    act = act_function(data=bn, act_type=net_act, name='%s%s_relu' % (name, suffix))
    return act
    pass


def linear(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
           num_group=1, bn_mom=cfg.TRAIN.MOMENTUM, name=None, suffix=""):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride,
                              pad=pad, no_bias=True, name='%s%s_conv2d' % (name, suffix))
    bn = mx.sym.BatchNorm(data=conv, name='%s%s_batchnorm' % (name, suffix), fix_gamma=False, momentum=bn_mom)
    return bn
    pass


def conv_only(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix=''):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride,
                              pad=pad, no_bias=True, name='%s%s_conv2d' % (name, suffix))
    return conv
    pass


def d_residual(data, num_out=1, kernel=(3, 3), stride=(2, 2), pad=(1, 1), bn_mom=cfg.TRAIN.MOMENTUM,
               net_act=cfg.TRAIN.NET_ACT, num_group=1, name=None, suffix=''):
    conv = conv_feature(data=data, num_filter=num_group, kernel=(1, 1), pad=(0, 0), stride=(1, 1),
                        net_act=net_act, name='%s%s_conv_sep' % (name, suffix))

    conv_dw = conv_feature(data=conv, num_filter=num_group, num_group=num_group, kernel=kernel,
                           pad=pad, stride=stride, net_act=net_act, name='%s%s_conv_dw' % (name, suffix))

    proj = linear(data=conv_dw, num_filter=num_out, kernel=(1, 1), pad=(0, 0), stride=(1, 1),
                  bn_mom=bn_mom, name='%s%s_conv_proj' % (name, suffix))
    return proj


def residual(data, num_block=1, num_out=1, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
             bn_mom=cfg.TRAIN.MOMENTUM, net_act=cfg.TRAIN.NET_ACT, num_group=1, name=None, suffix=''):
    identity = data
    for i in range(num_block):
        shortcut = identity
        conv = d_residual(data=identity, num_out=num_out, kernel=kernel, stride=stride, pad=pad,
                          bn_mom=bn_mom, net_act=net_act, num_group=num_group,
                          name='%s%s_block' % (name, suffix), suffix='%d' % i)
        identity = conv + shortcut
    return identity


def get_symbol():

    data = mx.symbol.Variable(name="data")
    data = data - 127.5
    data = data * 0.0078125

    # 输出特征维度
    embedding_size = cfg.COMMON.EMBEDDING_SIZE
    fc_type = cfg.TRAIN.NET_OUTPUT
    blocks = cfg.TRAIN.NET_BLOCKS
    bn_mom = cfg.TRAIN.MOMENTUM
    net_act = cfg.TRAIN.NET_ACT

    conv_1 = conv_feature(data, num_filter=64, kernel=(3, 3), pad=(1, 1), stride=(2, 2), bn_mom=bn_mom, name="conv_1")

    if blocks[0] == 1:
        conv_2_dw = conv_feature(conv_1, num_group=64, num_filter=64, kernel=(3, 3), pad=(1, 1), stride=(1, 1),
                                 bn_mom=bn_mom, name="conv_2_dw")
        pass

    else:
        conv_2_dw = residual(conv_1, num_block=blocks[0], num_out=64, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                             bn_mom=bn_mom, net_act=net_act, num_group=64, name="res_2")
        pass

    conv_23 = d_residual(conv_2_dw, num_out=64, kernel=(3, 3), stride=(2, 2), pad=(1, 1),
                         bn_mom=bn_mom, net_act=net_act, num_group=128, name="dconv_23")

    conv_3 = residual(conv_23, num_block=blocks[1], num_out=64, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                      bn_mom=bn_mom, net_act=net_act, num_group=128, name="res_3")

    conv_34 = d_residual(conv_3, num_out=128, kernel=(3, 3), stride=(2, 2), pad=(1, 1),
                         bn_mom=bn_mom, net_act=net_act, num_group=256, name="dconv_34")

    conv_4 = residual(conv_34, num_block=blocks[2], num_out=128, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                      bn_mom=bn_mom, net_act=net_act, num_group=256, name="res_4")
    conv_45 = d_residual(conv_4, num_out=128, kernel=(3, 3), stride=(2, 2), pad=(1, 1),
                         bn_mom=bn_mom, net_act=net_act, num_group=512, name="dconv_45")

    conv_5 = residual(conv_45, num_block=blocks[3], num_out=128, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                      bn_mom=bn_mom, net_act=net_act, num_group=256, name="res_5")

    conv_6_sep = conv_feature(conv_5, num_filter=512, kernel=(1, 1), pad=(0, 0), stride=(1, 1),
                              bn_mom=bn_mom, name="conv_6sep")

    # embedding_size 为输出 128 维
    fc1 = model_utils.get_fc1(conv_6_sep, embedding_size, fc_type, bn_mom=bn_mom)
    return fc1
