#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/07/20 21:47
# @Author   : WanDaoYi
# @FileName : f_res_net.py
# ============================================

import mxnet as mx
from utils import model_utils, me_monger_utils
from config import cfg


def residual_unit_v1(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs):
    """
        Return ResNet Unit symbol for building ResNet
    :param data: str; Input data
    :param num_filter: int; Number of output channels
    :param stride: tuple; Stride used in convolution
    :param dim_match: Boolean; True means channel number between input and output is the same,
                                otherwise means differ
    :param name: str; Base name of the operators
    :param bottle_neck: Boolean;
    :param kwargs:
    :return:
    """

    use_se = kwargs.get('version_se', 1)
    bn_mom = kwargs.get('bn_mom', 0.9)
    workspace = kwargs.get('workspace', 256)
    me_monger_flag = kwargs.get('me_monger_flag', False)
    act_type = kwargs.get('version_act', 'prelu')
    # print('in unit1')
    if bottle_neck:
        conv1 = model_utils.conv_feature(data=data, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=stride,
                                         pad=(0, 0), no_bias=True, workspace=workspace, name=name + '_conv1')
        bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        act1 = model_utils.act_function(data=bn1, act_type=act_type, name=name + '_relu1')
        conv2 = model_utils.conv_feature(data=act1, num_filter=int(num_filter * 0.25), kernel=(3, 3), stride=(1, 1),
                                         pad=(1, 1), no_bias=True, workspace=workspace, name=name + '_conv2')
        bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act2 = model_utils.act_function(data=bn2, act_type=act_type, name=name + '_relu2')
        conv3 = model_utils.conv_feature(data=act2, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                         no_bias=True, workspace=workspace, name=name + '_conv3')
        bn3 = mx.sym.BatchNorm(data=conv3, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')

        if use_se:
            # se begin
            body = mx.sym.Pooling(data=bn3, global_pool=True, kernel=(7, 7), pool_type='avg', name=name + '_se_pool1')
            body = model_utils.conv_feature(data=body, num_filter=num_filter // 16, kernel=(1, 1), stride=(1, 1),
                                            pad=(0, 0), name=name + "_se_conv1", workspace=workspace)
            body = model_utils.act_function(data=body, act_type=act_type, name=name + '_se_relu1')
            body = model_utils.conv_feature(data=body, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                            name=name + "_se_conv2", workspace=workspace)
            body = mx.symbol.Activation(data=body, act_type='sigmoid', name=name + "_se_sigmoid")
            bn3 = mx.symbol.broadcast_mul(bn3, body)
            # se end

        if dim_match:
            shortcut = data
        else:
            conv1sc = model_utils.conv_feature(data=data, num_filter=num_filter, kernel=(1, 1), stride=stride,
                                               no_bias=True, workspace=workspace, name=name + '_conv1sc')
            shortcut = mx.sym.BatchNorm(data=conv1sc, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_sc')
        if me_monger_flag:
            shortcut._set_attr(mirror_stage='True')
            pass
        return model_utils.act_function(data=bn3 + shortcut, act_type=act_type, name=name + '_relu3')
    else:
        conv1 = model_utils.conv_feature(data=data, num_filter=num_filter, kernel=(3, 3), stride=stride, pad=(1, 1),
                                         no_bias=True, workspace=workspace, name=name + '_conv1')
        bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
        act1 = model_utils.act_function(data=bn1, act_type=act_type, name=name + '_relu1')
        conv2 = model_utils.conv_feature(data=act1, num_filter=num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                                         no_bias=True, workspace=workspace, name=name + '_conv2')
        bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')
        if use_se:
            # se begin
            body = mx.sym.Pooling(data=bn2, global_pool=True, kernel=(7, 7), pool_type='avg', name=name + '_se_pool1')
            body = model_utils.conv_feature(data=body, num_filter=num_filter // 16, kernel=(1, 1), stride=(1, 1),
                                            pad=(0, 0), name=name + "_se_conv1", workspace=workspace)
            body = model_utils.act_function(data=body, act_type=act_type, name=name + '_se_relu1')
            body = model_utils.conv_feature(data=body, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                            name=name + "_se_conv2", workspace=workspace)
            body = mx.symbol.Activation(data=body, act_type='sigmoid', name=name + "_se_sigmoid")
            bn2 = mx.symbol.broadcast_mul(bn2, body)
            # se end

        if dim_match:
            shortcut = data
        else:
            conv1sc = model_utils.conv_feature(data=data, num_filter=num_filter, kernel=(1, 1), stride=stride,
                                               no_bias=True, workspace=workspace, name=name + '_conv1sc')
            shortcut = mx.sym.BatchNorm(data=conv1sc, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_sc')
        if me_monger_flag:
            shortcut._set_attr(mirror_stage='True')
            pass
        return model_utils.act_function(data=bn2 + shortcut, act_type=act_type, name=name + '_relu3')
    pass


def residual_unit_v1_l(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs):
    """
        Return ResNet Unit symbol for building ResNet
    :param data: str; Input data
    :param num_filter: int; Number of output channels
    :param stride: tuple; Stride used in convolution
    :param dim_match: Boolean; True means channel number between input and output is the same,
                                otherwise means differ
    :param name: str; Base name of the operators
    :param bottle_neck: Boolean;
    :param kwargs:
    :return:
    """

    use_se = kwargs.get('version_se', 1)
    bn_mom = kwargs.get('bn_mom', 0.9)
    workspace = kwargs.get('workspace', 256)
    me_monger_flag = kwargs.get('me_monger_flag', False)
    act_type = kwargs.get('version_act', 'prelu')
    # print('in unit1')
    if bottle_neck:
        conv1 = model_utils.conv_feature(data=data, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=(1, 1),
                                         pad=(0, 0), no_bias=True, workspace=workspace, name=name + '_conv1')
        bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        act1 = model_utils.act_function(data=bn1, act_type=act_type, name=name + '_relu1')
        conv2 = model_utils.conv_feature(data=act1, num_filter=int(num_filter * 0.25), kernel=(3, 3), stride=(1, 1),
                                         pad=(1, 1), no_bias=True, workspace=workspace, name=name + '_conv2')
        bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act2 = model_utils.act_function(data=bn2, act_type=act_type, name=name + '_relu2')
        conv3 = model_utils.conv_feature(data=act2, num_filter=num_filter, kernel=(1, 1), stride=stride, pad=(0, 0),
                                         no_bias=True, workspace=workspace, name=name + '_conv3')
        bn3 = mx.sym.BatchNorm(data=conv3, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')

        if use_se:
            # se begin
            body = mx.sym.Pooling(data=bn3, global_pool=True, kernel=(7, 7), pool_type='avg', name=name + '_se_pool1')
            body = model_utils.conv_feature(data=body, num_filter=num_filter // 16, kernel=(1, 1), stride=(1, 1),
                                            pad=(0, 0), name=name + "_se_conv1", workspace=workspace)
            body = model_utils.act_function(data=body, act_type=act_type, name=name + '_se_relu1')
            body = model_utils.conv_feature(data=body, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                            name=name + "_se_conv2", workspace=workspace)
            body = mx.symbol.Activation(data=body, act_type='sigmoid', name=name + "_se_sigmoid")
            bn3 = mx.symbol.broadcast_mul(bn3, body)
            # se end

        if dim_match:
            shortcut = data
        else:
            conv1sc = model_utils.conv_feature(data=data, num_filter=num_filter, kernel=(1, 1), stride=stride,
                                               no_bias=True, workspace=workspace, name=name + '_conv1sc')
            shortcut = mx.sym.BatchNorm(data=conv1sc, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_sc')
        if me_monger_flag:
            shortcut._set_attr(mirror_stage='True')
            pass
        return model_utils.act_function(data=bn3 + shortcut, act_type=act_type, name=name + '_relu3')
    else:
        conv1 = model_utils.conv_feature(data=data, num_filter=num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                                         no_bias=True, workspace=workspace, name=name + '_conv1')
        bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
        act1 = model_utils.act_function(data=bn1, act_type=act_type, name=name + '_relu1')
        conv2 = model_utils.conv_feature(data=act1, num_filter=num_filter, kernel=(3, 3), stride=stride, pad=(1, 1),
                                         no_bias=True, workspace=workspace, name=name + '_conv2')
        bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')
        if use_se:
            # se begin
            body = mx.sym.Pooling(data=bn2, global_pool=True, kernel=(7, 7), pool_type='avg', name=name + '_se_pool1')
            body = model_utils.conv_feature(data=body, num_filter=num_filter // 16, kernel=(1, 1), stride=(1, 1),
                                            pad=(0, 0), name=name + "_se_conv1", workspace=workspace)
            body = model_utils.act_function(data=body, act_type=act_type, name=name + '_se_relu1')
            body = model_utils.conv_feature(data=body, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                            name=name + "_se_conv2", workspace=workspace)
            body = mx.symbol.Activation(data=body, act_type='sigmoid', name=name + "_se_sigmoid")
            bn2 = mx.symbol.broadcast_mul(bn2, body)
            # se end

        if dim_match:
            shortcut = data
        else:
            conv1sc = model_utils.conv_feature(data=data, num_filter=num_filter, kernel=(1, 1), stride=stride,
                                               no_bias=True, workspace=workspace, name=name + '_conv1sc')
            shortcut = mx.sym.BatchNorm(data=conv1sc, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_sc')
        if me_monger_flag:
            shortcut._set_attr(mirror_stage='True')
            pass
        return model_utils.act_function(data=bn2 + shortcut, act_type=act_type, name=name + '_relu3')
    pass


def residual_unit_v2(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs):

    """
        Return ResNet Unit symbol for building ResNet
    :param data: str; Input data
    :param num_filter: int; Number of output channels
    :param stride: tuple; Stride used in convolution
    :param dim_match: Boolean; True means channel number between input and output is the same,
                                otherwise means differ
    :param name: str; Base name of the operators
    :param bottle_neck: Boolean;
    :param kwargs:
    :return:
    """

    use_se = kwargs.get('version_se', 1)
    bn_mom = kwargs.get('bn_mom', 0.9)
    workspace = kwargs.get('workspace', 256)
    me_monger_flag = kwargs.get('me_monger_flag', False)
    act_type = kwargs.get('version_act', 'prelu')
    # print('in unit2')
    if bottle_neck:
        # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        act1 = model_utils.act_function(data=bn1, act_type=act_type, name=name + '_relu1')
        conv1 = model_utils.conv_feature(data=act1, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=(1, 1),
                                         pad=(0, 0), no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act2 = model_utils.act_function(data=bn2, act_type=act_type, name=name + '_relu2')
        conv2 = model_utils.conv_feature(data=act2, num_filter=int(num_filter * 0.25), kernel=(3, 3), stride=stride,
                                         pad=(1, 1), no_bias=True, workspace=workspace, name=name + '_conv2')
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
        act3 = model_utils.act_function(data=bn3, act_type=act_type, name=name + '_relu3')
        conv3 = model_utils.conv_feature(data=act3, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                         no_bias=True, workspace=workspace, name=name + '_conv3')
        if use_se:
            # se begin
            body = mx.sym.Pooling(data=conv3, global_pool=True, kernel=(7, 7), pool_type='avg', name=name + '_se_pool1')
            body = model_utils.conv_feature(data=body, num_filter=num_filter // 16, kernel=(1, 1), stride=(1, 1),
                                            pad=(0, 0), name=name + "_se_conv1", workspace=workspace)
            body = model_utils.act_function(data=body, act_type=act_type, name=name + '_se_relu1')
            body = model_utils.conv_feature(data=body, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                            name=name + "_se_conv2", workspace=workspace)
            body = mx.symbol.Activation(data=body, act_type='sigmoid', name=name + "_se_sigmoid")
            conv3 = mx.symbol.broadcast_mul(conv3, body)
        if dim_match:
            shortcut = data
        else:
            shortcut = model_utils.conv_feature(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride,
                                                no_bias=True, workspace=workspace, name=name + '_sc')
        if me_monger_flag:
            shortcut._set_attr(mirror_stage='True')
            pass
        return conv3 + shortcut
    else:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
        act1 = model_utils.act_function(data=bn1, act_type=act_type, name=name + '_relu1')
        conv1 = model_utils.conv_feature(data=act1, num_filter=num_filter, kernel=(3, 3), stride=stride, pad=(1, 1),
                                         no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')
        act2 = model_utils.act_function(data=bn2, act_type=act_type, name=name + '_relu2')
        conv2 = model_utils.conv_feature(data=act2, num_filter=num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                                         no_bias=True, workspace=workspace, name=name + '_conv2')
        if use_se:
            # se begin
            body = mx.sym.Pooling(data=conv2, global_pool=True, kernel=(7, 7), pool_type='avg', name=name + '_se_pool1')
            body = model_utils.conv_feature(data=body, num_filter=num_filter // 16, kernel=(1, 1), stride=(1, 1),
                                            pad=(0, 0), name=name + "_se_conv1", workspace=workspace)
            body = model_utils.act_function(data=body, act_type=act_type, name=name + '_se_relu1')
            body = model_utils.conv_feature(data=body, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                            name=name + "_se_conv2", workspace=workspace)
            body = mx.symbol.Activation(data=body, act_type='sigmoid', name=name + "_se_sigmoid")
            conv2 = mx.symbol.broadcast_mul(conv2, body)
        if dim_match:
            shortcut = data
        else:
            shortcut = model_utils.conv_feature(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride,
                                                no_bias=True, workspace=workspace, name=name + '_sc')
        if me_monger_flag:
            shortcut._set_attr(mirror_stage='True')
            pass
        return conv2 + shortcut
    pass


def residual_unit_v3(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs):
    """
        Return ResNet Unit symbol for building ResNet
    :param data: str; Input data
    :param num_filter: int; Number of output channels
    :param stride: tuple; Stride used in convolution
    :param dim_match: Boolean; True means channel number between input and output is the same,
                                otherwise means differ
    :param name: str; Base name of the operators
    :param bottle_neck: Boolean;
    :param kwargs:
    :return:
    """

    use_se = kwargs.get('version_se', 1)
    bn_mom = kwargs.get('bn_mom', 0.9)
    workspace = kwargs.get('workspace', 256)
    me_monger_flag = kwargs.get('me_monger_flag', False)
    act_type = kwargs.get('version_act', 'prelu')
    # print('in unit3')
    if bottle_neck:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        conv1 = model_utils.conv_feature(data=bn1, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=(1, 1),
                                         pad=(0, 0), no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act1 = model_utils.act_function(data=bn2, act_type=act_type, name=name + '_relu1')
        conv2 = model_utils.conv_feature(data=act1, num_filter=int(num_filter * 0.25), kernel=(3, 3), stride=(1, 1),
                                         pad=(1, 1), no_bias=True, workspace=workspace, name=name + '_conv2')
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
        act2 = model_utils.act_function(data=bn3, act_type=act_type, name=name + '_relu2')
        conv3 = model_utils.conv_feature(data=act2, num_filter=num_filter, kernel=(1, 1), stride=stride, pad=(0, 0),
                                         no_bias=True, workspace=workspace, name=name + '_conv3')
        bn4 = mx.sym.BatchNorm(data=conv3, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn4')

        if use_se:
            # se begin
            body = mx.sym.Pooling(data=bn4, global_pool=True, kernel=(7, 7), pool_type='avg', name=name + '_se_pool1')
            body = model_utils.conv_feature(data=body, num_filter=num_filter // 16, kernel=(1, 1), stride=(1, 1),
                                            pad=(0, 0), name=name + "_se_conv1", workspace=workspace)
            body = model_utils.act_function(data=body, act_type=act_type, name=name + '_se_relu1')
            body = model_utils.conv_feature(data=body, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                            name=name + "_se_conv2", workspace=workspace)
            body = mx.symbol.Activation(data=body, act_type='sigmoid', name=name + "_se_sigmoid")
            bn4 = mx.symbol.broadcast_mul(bn4, body)
            # se end

        if dim_match:
            shortcut = data
        else:
            conv1sc = model_utils.conv_feature(data=data, num_filter=num_filter, kernel=(1, 1), stride=stride,
                                               no_bias=True, workspace=workspace, name=name + '_conv1sc')
            shortcut = mx.sym.BatchNorm(data=conv1sc, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_sc')
        if me_monger_flag:
            shortcut._set_attr(mirror_stage='True')
            pass
        return bn4 + shortcut
    else:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        conv1 = model_utils.conv_feature(data=bn1, num_filter=num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                                         no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act1 = model_utils.act_function(data=bn2, act_type=act_type, name=name + '_relu1')
        conv2 = model_utils.conv_feature(data=act1, num_filter=num_filter, kernel=(3, 3), stride=stride, pad=(1, 1),
                                         no_bias=True, workspace=workspace, name=name + '_conv2')
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
        if use_se:
            # se begin
            body = mx.sym.Pooling(data=bn3, global_pool=True, kernel=(7, 7), pool_type='avg', name=name + '_se_pool1')
            body = model_utils.conv_feature(data=body, num_filter=num_filter // 16, kernel=(1, 1), stride=(1, 1),
                                            pad=(0, 0), name=name + "_se_conv1", workspace=workspace)
            body = model_utils.act_function(data=body, act_type=act_type, name=name + '_se_relu1')
            body = model_utils.conv_feature(data=body, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                            name=name + "_se_conv2", workspace=workspace)
            body = mx.symbol.Activation(data=body, act_type='sigmoid', name=name + "_se_sigmoid")
            bn3 = mx.symbol.broadcast_mul(bn3, body)
            # se end

        if dim_match:
            shortcut = data
        else:
            conv1sc = model_utils.conv_feature(data=data, num_filter=num_filter, kernel=(1, 1), stride=stride,
                                               no_bias=True, workspace=workspace, name=name + '_conv1sc')
            shortcut = mx.sym.BatchNorm(data=conv1sc, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_sc')
        if me_monger_flag:
            shortcut._set_attr(mirror_stage='True')
            pass
        return bn3 + shortcut
    pass


def residual_unit(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs):
    """
        Return ResNet Unit symbol for building ResNet
    :param data: str; Input data
    :param num_filter: int; Number of output channels
    :param stride: tuple; Stride used in convolution
    :param dim_match: Boolean; True means channel number between input and output is the same,
                                otherwise means differ
    :param name: str; Base name of the operators
    :param bottle_neck: Boolean;
    :param kwargs:
    :return:
    """

    uv = kwargs.get('version_unit', 3)
    version_input = kwargs.get('version_input', 1)
    if uv == 1:
        if version_input == 0:
            return residual_unit_v1(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs)
        else:
            return residual_unit_v1_l(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs)
    elif uv == 2:
        return residual_unit_v2(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs)
    else:
        return residual_unit_v3(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs)


def res_net(units, num_stages, filter_list, num_classes, bottle_neck, **kwargs):
    version_input = kwargs.get("version_input", 1)

    assert version_input >= 0, "version_input is not more than zero!"

    version_output = kwargs.get("version_output", "E")
    fc_type = version_output
    act_type = kwargs.get("version_act", "prelu")
    workspace = kwargs.get("workspace", 256)
    bn_mom = kwargs.get("bn_mom", 0.9)

    data = mx.sym.Variable(name='data')
    if version_input == 0:
        # data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
        data = mx.sym.identity(data=data, name='id')
        data = data - 127.5
        data = data * 0.0078125
        body = model_utils.conv_feature(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2, 2), pad=(3, 3),
                                        no_bias=True, name="conv0", workspace=workspace)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
        body = model_utils.act_function(data=body, act_type=act_type, name='relu0')
        # body = mx.sym.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')
        pass
    elif version_input == 2:
        data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
        body = model_utils.conv_feature(data=data, num_filter=filter_list[0], kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                                        no_bias=True, name="conv0", workspace=workspace)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
        body = model_utils.act_function(data=body, act_type=act_type, name='relu0')
        pass
    else:
        data = mx.sym.identity(data=data, name='id')
        data = data - 127.5
        data = data * 0.0078125
        body = data
        body = model_utils.conv_feature(data=body, num_filter=filter_list[0], kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                                        no_bias=True, name="conv0", workspace=workspace)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
        body = model_utils.act_function(data=body, act_type=act_type, name='relu0')
        pass

    for i in range(num_stages):
        body = residual_unit(body, filter_list[i + 1], (2, 2), False,
                             name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, **kwargs)
        for j in range(units[i] - 1):
            body = residual_unit(body, filter_list[i + 1], (1, 1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                 bottle_neck=bottle_neck, **kwargs)

    if bottle_neck:
        body = model_utils.conv_feature(data=body, num_filter=512, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                        no_bias=True, name="convd", workspace=workspace)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bnd')
        body = model_utils.act_function(data=body, act_type=act_type, name='relud')

    fc1 = model_utils.get_fc1(body, num_classes, fc_type)

    return fc1
    pass


def get_symbol():
    # 输出特征维度
    embedding_size = cfg.COMMON.EMBEDDING_SIZE

    bn_mom = cfg.TRAIN.MOMENTUM
    version_input = cfg.TRAIN.NET_INPUT
    workspace = cfg.TRAIN.WORKSPACE
    net_se = cfg.TRAIN.NET_SE
    net_act = cfg.TRAIN.NET_ACT
    me_monger_flag = cfg.TRAIN.ME_MONGER_FLAG
    image_shape = cfg.COMMON.FACE_SHAPE

    structure_dict = cfg.COMMON.STRUCTURE_DICT
    model_net = structure_dict["network"]
    model_net_name = model_net[cfg.TRAIN.MODEL_NET]
    num_layers = model_net_name["num_layers"]
    net_output = model_net_name["net_output"]
    net_unit = model_net_name["net_unit"]
    per_batch_size = model_net_name["per_batch_size"]

    kwargs = {"version_se": net_se, "version_input": version_input,
              "version_output": net_output, "version_unit": net_unit,
              "version_act": net_act, "bn_mom": bn_mom,
              "workspace": workspace, "me_monger_flag": me_monger_flag}

    if num_layers >= 500:
        filter_list = [64, 256, 512, 1024, 2048]
        bottle_neck = True
    else:
        filter_list = [64, 64, 128, 256, 512]
        bottle_neck = False
    num_stages = 4
    if num_layers == 18:
        units = [2, 2, 2, 2]
    elif num_layers == 34:
        units = [3, 4, 6, 3]
    elif num_layers == 49:
        units = [3, 4, 14, 3]
    elif num_layers == 50:
        units = [3, 4, 14, 3]
    elif num_layers == 74:
        units = [3, 6, 24, 3]
    elif num_layers == 90:
        units = [3, 8, 30, 3]
    elif num_layers == 98:
        units = [3, 4, 38, 3]
    elif num_layers == 99:
        units = [3, 8, 35, 3]
    elif num_layers == 100:
        units = [3, 13, 30, 3]
    elif num_layers == 134:
        units = [3, 10, 50, 3]
    elif num_layers == 136:
        units = [3, 13, 48, 3]
    elif num_layers == 140:
        units = [3, 15, 48, 3]
    elif num_layers == 124:
        units = [3, 13, 40, 5]
    elif num_layers == 160:
        units = [3, 24, 49, 3]
    elif num_layers == 101:
        units = [3, 4, 23, 3]
    elif num_layers == 152:
        units = [3, 8, 36, 3]
    elif num_layers == 200:
        units = [3, 24, 36, 3]
    elif num_layers == 269:
        units = [3, 30, 48, 8]
    else:
        raise ValueError("no experiments done on num_layers {}, you can do it yourself".format(num_layers))

    net = res_net(units=units,
                  num_stages=num_stages,
                  filter_list=filter_list,
                  num_classes=embedding_size,
                  bottle_neck=bottle_neck, **kwargs)

    if me_monger_flag:
        data_shape = (per_batch_size, image_shape[2], image_shape[0], image_shape[1])
        net_mem_planned = me_monger_utils.search_plan(net, data=data_shape)
        old_cost = me_monger_utils.get_cost(net, data=data_shape)
        new_cost = me_monger_utils.get_cost(net_mem_planned, data=data_shape)

        print('Old feature map cost=%d MB' % old_cost)
        print('New feature map cost=%d MB' % new_cost)
        net = net_mem_planned
        pass
    return net
    pass
