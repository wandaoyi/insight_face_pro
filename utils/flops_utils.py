#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/07/08 18:07
# @Author   : WanDaoYi
# @FileName : flops_utils.py
# ============================================

import json


def is_no_bias(attr):
    flag = False
    if 'no_bias' in attr and (attr['no_bias'] is True or attr['no_bias'] == 'True'):
        flag = True
    return flag


def count_conv_flops(input_shape, output_shape, attr):
    kernel = attr['kernel'][1:-1].split(',')
    kernel = [int(x) for x in kernel]

    # print('kernel', kernel)
    if is_no_bias(attr):
        ret = (2 * input_shape[1] * kernel[0] * kernel[1] - 1) * output_shape[2] * output_shape[3] * output_shape[1]
    else:
        ret = 2 * input_shape[1] * kernel[0] * kernel[1] * output_shape[2] * output_shape[3] * output_shape[1]
    num_group = 1
    if 'num_group' in attr:
        num_group = int(attr['num_group'])
    ret /= num_group
    return int(ret)


def count_fc_flops(input_filter, output_filter, attr):
    # print(input_filter, output_filter ,attr)
    ret = 2 * input_filter * output_filter
    if is_no_bias(attr):
        ret -= output_filter
    return int(ret)


def count_flops(sym, **data_shapes):
    all_layers = sym.get_internals()
    # print(all_layers)
    arg_shapes, out_shapes, aux_shapes = all_layers.infer_shape(**data_shapes)
    out_shape_dict = dict(zip(all_layers.list_outputs(), out_shapes))

    nodes = json.loads(sym.tojson())['nodes']
    node_id_shape = {}
    for node_id, node in enumerate(nodes):
        name = node['name']
        layer_name = name + "_output"
        if layer_name in out_shape_dict:
            node_id_shape[node_id] = out_shape_dict[layer_name]
    # print(node_id_shape)
    flops_info = 0
    for node_id, node in enumerate(nodes):
        flops = 0
        if node['op'] == 'Convolution':
            output_shape = node_id_shape[node_id]
            attr = node['attrs']
            input_node_id = node['inputs'][0][0]
            input_shape = node_id_shape[input_node_id]
            flops = count_conv_flops(input_shape, output_shape, attr)
        elif node['op'] == 'FullyConnected':
            attr = node['attrs']
            output_shape = node_id_shape[node_id]
            input_node_id = node['inputs'][0][0]
            input_shape = node_id_shape[input_node_id]
            output_filter = output_shape[1]
            input_filter = input_shape[1] * input_shape[2] * input_shape[3]
            # assert len(input_shape)==4 and input_shape[2]==1 and input_shape[3]==1
            flops = count_fc_flops(input_filter, output_filter, attr)
        # print(node, flops)
        flops_info += flops

    return flops_info


def flops_str(flops_info):
    preset = [(1e12, 'T'), (1e9, 'G'), (1e6, 'M'), (1e3, 'K')]

    for p in preset:
        if flops_info // p[0] > 0:
            n = flops_info / p[0]
            ret = "{:.1f}{}".format(n, p[1])
            return ret
    ret = "{:.1f}".format(flops_info)
    return ret


if __name__ == "__main__":

    import os
    import mxnet as mx
    from datetime import datetime

    # 代码开始时间
    start_time = datetime.now()
    print("开始时间: {}".format(start_time))

    # 预训练模型文件夹路径
    model_path = "../models/model-y1/model-0000.params"

    path_info, file_name = os.path.split(model_path)
    name_info, suffix_info = os.path.splitext(file_name)
    name_prefix, name_epoch = name_info.split("-")
    model_path_prefix = os.path.join(path_info, name_prefix)
    epoch_num = int(name_epoch)
    symbol, arg_params, aux_params = mx.model.load_checkpoint(model_path_prefix, epoch_num)

    gt_layers = symbol.get_internals()
    symbol = gt_layers['fc1_output']
    flop_info = count_flops(symbol, data=(1, 3, 112, 112))
    print("flop_info: {}".format(flop_info))

    flop_info_str = flops_str(flop_info)
    print("flop_info_str: {}".format(flop_info_str))

    # 代码结束时间
    end_time = datetime.now()
    print("结束时间: {}, 训练模型耗时: {}".format(end_time, end_time - start_time))
