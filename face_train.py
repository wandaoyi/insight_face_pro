#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/07/01 19:00
# @Author   : WanDaoYi
# @FileName : face_train.py
# ============================================

from datetime import datetime
import os
import sys
import logging
import mxnet.optimizer as optimizer

from utils.metric_utils import *
from verification_model import VerificationModel
from utils import data_utils, flops_utils, get_data_utils
from data_prepare.image_iter import FaceImageIter
from data_prepare.triplet_image_iter import TripletFaceImageIter

# 给动态方法: eval(net_name).get_symbol() 使用
from model_net import f_mobile_face_net

from config import cfg

os.environ['CUDA_VISIBLE_DEVICES'] = cfg.TRAIN.GPU_NUM


# 用于打印训练日志:
# INFO:root:Epoch[0] Batch [20-40]	Speed: 19.72 samples/sec	acc=0.000000	loss_value=55.078926
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class FaceTrain(object):

    def __init__(self):

        self.verification_model = VerificationModel()

        # 预训练模型文件夹路径
        self.model_path = cfg.TRAIN.MODEL_PATH

        self.learning_rate = cfg.TRAIN.LEARNING_RATE
        # 权值衰减值
        self.weight_decay = cfg.TRAIN.WEIGHT_DECAY
        # 训练多少步就打印一次日志
        self.print_step = cfg.TRAIN.PRINT_STEP

        self.val_step = cfg.TRAIN.VAL_STEP

        self.save_model_num = cfg.TRAIN.SAVE_MODEL_NUM

        # 是否检测输出的特征向量
        self.check_feature_flag = cfg.TRAIN.CHECK_FEATURE_FLAG

        # 每达到步数，学习率变为原来的百分之十
        self.learning_rate_step_list = cfg.TRAIN.LEARNING_RATE_STEP_LIST

        # 人脸图像的 shape [112, 112, 3]
        self.face_shape = cfg.COMMON.FACE_SHAPE

        # loss 的计算方式
        self.face_loss = cfg.TRAIN.FACE_LOSS

        # 模型信息 字典
        self.model_dict = cfg.COMMON.STRUCTURE_DICT

        # net info
        self.net_dict = self.model_dict["network"]
        self.model_net = cfg.TRAIN.MODEL_NET
        self.net_info = self.net_dict[self.model_net]
        print("net_info: {}".format(self.net_info))

        # data info
        self.data_dict = self.model_dict["data_set"]
        self.data_set = cfg.TRAIN.DATA_SET
        self.data_info = self.data_dict[self.data_set]
        print("data_info: {}".format(self.data_info))

        # loss info
        self.loss_dict = self.model_dict["loss"]
        self.loss_info = self.loss_dict[cfg.TRAIN.FACE_LOSS]
        print("loss_info: {}".format(self.loss_info))

        # 模型保存文件夹路径
        self.save_model_file_path = cfg.TRAIN.SAVE_MODEL_FILE_PATH
        # 保存模型名的前缀
        self.save_model_prefix = "{}_{}_{}".format(self.model_net, self.face_loss, self.data_set)
        self.save_model_path = os.path.join(self.save_model_file_path, self.save_model_prefix)
        data_utils.make_file(self.save_model_path)
        self.save_model_prefix_path = self.save_model_path + cfg.TRAIN.MODEL_NAME_PREFIX

        self.max_steps = cfg.TRAIN.MAX_STEPS

        self.kv_store = cfg.TRAIN.KV_STORE

        # 学习率的倍数
        self.fc7_lr_multiple = cfg.TRAIN.FC7_LR_MULTIPLE
        # 权重刷衰减的倍数
        self.fc7_wd_multiple = cfg.TRAIN.FC7_WD_MULTIPLE
        # fc7 是否有 bias 层
        self.fc7_no_bias = cfg.TRAIN.FC7_NO_BIAS

        # Focal loss，一种改进的交叉损失熵
        self.ce_loss = cfg.TRAIN.CE_LOSS

        # 是否计算一个网络占用的浮点数内存
        self.count_flops_flag = cfg.TRAIN.COUNT_FLOPS_FLAG

        # 输出特征维度: 128
        self.embedding_size = cfg.COMMON.EMBEDDING_SIZE

        # .idx 后缀
        self.idx_suffix = cfg.COMMON.IDX_SUFFIX

        self.data_rand_mirror_flag = cfg.TRAIN.DATA_RAND_MIRROR_FLAG
        self.data_crop_flag = cfg.TRAIN.DATA_CROP_FLAG
        self.data_color_aug = cfg.TRAIN.DATA_COLOR_AUG
        self.data_image_filter = cfg.TRAIN.DATA_IMAGES_FILTER
        pass

    def get_symbol(self):

        net_name = self.net_info["net_name"]
        # 输出特征向量的维度, 默认输出 128 维
        emb_size = self.net_info["emb_size"] if "emb_size" in self.net_info else self.embedding_size
        # 每个 GPU 的 batch size
        per_batch_size = self.net_info["per_batch_size"]
        loss_name = self.loss_info["loss_name"]
        class_num = self.data_info["class_num"]

        print("class_num: {}".format(class_num))

        # 获得一个特征向量
        # 动态方法: eval(net_name) 为将字符串名转为方法名,
        # 例如: 字符串: "f_mobile_face_net" ---> 方法: f_mobile_face_net
        # eval("f_mobile_face_net").get_symbol() ---> f_mobile_face_net.get_symbol()
        embedding_feature = eval(net_name).get_symbol()

        # print("embedding_feature: {}".format(embedding_feature))

        out_list = [mx.symbol.BlockGrad(embedding_feature)]

        # 定义一个标签的占位符，用来存放标签
        gt_label = mx.symbol.Variable('softmax_label')

        if loss_name == "softmax":
            # 定义一个全连接层的权重，使用全局池化代替全链接层
            fc_weight_init = mx.symbol.Variable("fc7_weight",
                                                shape=(class_num, emb_size),
                                                lr_mult=self.fc7_lr_multiple,
                                                wd_mult=self.fc7_wd_multiple,
                                                init=mx.init.Normal(0.01))

            # 如果不设置bias，使用全局池化代替全链接层，得到每个id的概率值
            if self.fc7_no_bias:
                fc7 = mx.sym.FullyConnected(data=embedding_feature,
                                            weight=fc_weight_init,
                                            no_bias=True,
                                            num_hidden=class_num,
                                            name='fc7')
                pass
            # 如果设置_bias，使用全局池化代替全链接层，得到每个id的cos_t
            else:
                fc_bias = mx.symbol.Variable('fc7_bias', lr_mult=2.0, wd_mult=0.0)
                fc7 = mx.sym.FullyConnected(data=embedding_feature,
                                            weight=fc_weight_init,
                                            bias=fc_bias,
                                            num_hidden=class_num,
                                            name='fc7')
                pass

            softmax = mx.symbol.SoftmaxOutput(data=fc7,
                                              label=gt_label,
                                              name='softmax',
                                              normalization='valid')
            out_list.append(softmax)

            if self.ce_loss:
                # ce_loss = mx.symbol.softmax_cross_entropy(data=fc7,
                #                                           label=gt_label,
                #                                           name='ce_loss')/per_batch_size

                body = mx.symbol.SoftmaxActivation(data=fc7)
                body = mx.symbol.log(body)
                one_hot_label = mx.sym.one_hot(gt_label, depth=class_num, on_value=-1.0, off_value=0.0)
                body = body * one_hot_label

                ce_loss = mx.symbol.sum(body) / per_batch_size
                out_list.append(mx.symbol.BlockGrad(ce_loss))
                pass

            pass
        elif loss_name == "margin_softmax":
            # 定义一个全连接层的权重，使用全局池化代替全链接层
            fc_weight_init = mx.symbol.Variable("fc7_weight",
                                                shape=(class_num, emb_size),
                                                lr_mult=self.fc7_lr_multiple,
                                                wd_mult=self.fc7_wd_multiple,
                                                init=mx.init.Normal(0.01))

            # 获得loss中m的缩放系数, L4 loss 公式中的:
            # e^{s[cos(m_1 * theta_y_i + m_2) - m_3]}
            s = self.loss_info["loss_s"]
            m_1 = self.loss_info["loss_m1"]
            m_2 = self.loss_info["loss_m2"]
            m_3 = self.loss_info["loss_m3"]

            # 先进行L2正则化，然后进行全链接
            l2_norm_fc_weight_init = mx.symbol.L2Normalization(fc_weight_init, mode='instance')
            l2_norm_embedding = mx.symbol.L2Normalization(embedding_feature, mode='instance', name='fc1n') * s

            # with ag.pause(train_mode=True):
            # 使用全局池化代替全链接层，得到每个id的角度*64
            fc7 = mx.sym.FullyConnected(data=l2_norm_embedding, weight=l2_norm_fc_weight_init,
                                        no_bias=True, num_hidden=class_num, name='fc7')

            # 查看 fc7_out_shape
            in_shape, out_shape, uax_shape = fc7.infer_shape(data=(2, 3, 112, 112))
            print("fc7_out_shape: {}".format(out_shape))

            # 其存在m1,m2,m3是为了把算法整合在一起，
            # arcface cosface combined
            if m_1 != 1.0 or m_2 != 0.0 or m_3 != 0.0:
                # cosface loss
                if m_1 == 1.0 and m_2 == 0.0:
                    s_m = s * m_3
                    gt_one_hot = mx.sym.one_hot(gt_label, depth=class_num, on_value=s_m, off_value=0.0)
                    fc7 = fc7 - gt_one_hot
                    pass
                # arcface combined
                else:
                    # fc7每一行找出 gt_label 对应的值,即 角度 * s
                    zy = mx.sym.pick(fc7, gt_label, axis=1)

                    # in_shape, out_shape, uax_shape = zy.infer_shape(data=(2, 3, 112, 112), softmax_label=(2,))
                    # # print('zy', out_shape)

                    # 进行复原，前面乘以了 s，cos_t 为 -1 到 1 之间
                    cos_t = zy / s

                    # t为0-3.14之间
                    # 该 arccos 是为了让后续的cos单调递增
                    t = mx.sym.arccos(cos_t)

                    # m1  sphereface
                    if m_1 != 1.0:
                        t = t * m_1

                    # arcface 或者 combined
                    if m_2 > 0.0:
                        t = t + m_2

                    #  t为 0-3.14 之间，单调递增
                    body = mx.sym.cos(t)

                    # combined 或者 arcface
                    if m_3 > 0.0:
                        body = body - m_3

                    new_zy = body * s

                    # 得到差值
                    diff = new_zy - zy

                    # 扩展一个维度
                    diff = mx.sym.expand_dims(diff, 1)

                    # 把标签转化为 one_hot 编码
                    gt_one_hot = mx.sym.one_hot(gt_label, depth=class_num, on_value=1.0, off_value=0.0)

                    # 进行更新
                    body = mx.sym.broadcast_mul(gt_one_hot, diff)
                    fc7 = fc7 + body
                    pass
                pass

            softmax = mx.symbol.SoftmaxOutput(data=fc7,
                                              label=gt_label,
                                              name='softmax',
                                              normalization='valid')
            out_list.append(softmax)

            if self.ce_loss:
                # ce_loss = mx.symbol.softmax_cross_entropy(data=fc7,
                #                                           label=gt_label,
                #                                           name='ce_loss')/per_batch_size

                body = mx.symbol.SoftmaxActivation(data=fc7)
                body = mx.symbol.log(body)
                one_hot_label = mx.sym.one_hot(gt_label, depth=class_num, on_value=-1.0, off_value=0.0)
                body = body * one_hot_label

                ce_loss = mx.symbol.sum(body) / per_batch_size
                out_list.append(mx.symbol.BlockGrad(ce_loss))
                pass

            pass
        elif loss_name.find("triplet") >= 0:

            triplet_alpha = self.loss_info["triplet_alpha"]

            l2_norm_embedding = mx.symbol.L2Normalization(embedding_feature, mode='instance', name='fc1n')
            anchor = mx.symbol.slice_axis(l2_norm_embedding, axis=0,
                                          begin=0,
                                          end=per_batch_size // 3)
            # 正样本
            positive = mx.symbol.slice_axis(l2_norm_embedding, axis=0,
                                            begin=per_batch_size // 3,
                                            end=2 * per_batch_size // 3)
            # 负样本
            negative = mx.symbol.slice_axis(l2_norm_embedding, axis=0,
                                            begin=2 * per_batch_size // 3,
                                            end=per_batch_size)
            if loss_name == "triplet":
                ap = anchor - positive
                an = anchor - negative
                ap = ap * ap
                an = an * an
                # (T,1)
                ap = mx.symbol.sum(ap, axis=1, keepdims=1)
                # (T,1)
                an = mx.symbol.sum(an, axis=1, keepdims=1)
                triplet_loss = mx.symbol.Activation(data=(ap - an + triplet_alpha), act_type='relu')
                triplet_loss = mx.symbol.mean(triplet_loss)
                pass
            else:
                ap = anchor * positive
                an = anchor * negative
                # (T,1)
                ap = mx.symbol.sum(ap, axis=1, keepdims=1)
                # (T,1)
                an = mx.symbol.sum(an, axis=1, keepdims=1)
                ap = mx.sym.arccos(ap)
                an = mx.sym.arccos(an)
                triplet_loss = mx.symbol.Activation(data=(ap - an + triplet_alpha), act_type='relu')
                triplet_loss = mx.symbol.mean(triplet_loss)
                pass

            triplet_loss = mx.symbol.MakeLoss(triplet_loss)
            out_list.append(mx.sym.BlockGrad(gt_label))
            out_list.append(triplet_loss)
            pass

        # 聚集所有的符号
        out = mx.symbol.Group(out_list)
        print("out: {}".format(out))

        return out
        pass

    def do_train(self):

        # 判断使用GPU还是CPU
        ctx = []
        cvd = os.environ['CUDA_VISIBLE_DEVICES'].strip()

        if len(cvd) > 0:
            gpu_num_list = cvd.split(",")
            gpu_num_list_len = len(gpu_num_list)
            for i in range(gpu_num_list_len):
                ctx.append(mx.gpu(i))
                pass
            pass

        if len(ctx) == 0:
            ctx = [mx.cpu()]
            print("use cpu")
            pass
        else:
            print("gpu num: {}".format(ctx))
            pass

        assert self.face_shape[0] == self.face_shape[1], "face_shape[0] neq face_shape[1]"

        # 每个 GPU 的 batch size
        per_batch_size = self.net_info["per_batch_size"]

        ctx_count = len(ctx)

        batch_size = per_batch_size * ctx_count

        data_shape = [self.face_shape[2], self.face_shape[0], self.face_shape[1]]

        data_shape = tuple(data_shape)

        # 模型是否存在，如果存在则加载模型，如果不存在则初始化权重
        if os.path.exists(self.model_path):
            path_info, file_name = os.path.split(self.model_path)
            name_info, suffix_info = os.path.splitext(file_name)
            name_prefix, name_epoch = name_info.split("-")
            model_path_prefix = os.path.join(path_info, name_prefix)
            epoch_num = int(name_epoch)

            symbol, arg_params, aux_params = mx.model.load_checkpoint(model_path_prefix, epoch_num)

            # 模型构建
            symbol_info = self.get_symbol()
            pass
        else:
            arg_params = None
            aux_params = None
            # 模型构建
            symbol_info = self.get_symbol()
            pass

        # 计算模型的缓存空间
        if self.count_flops_flag:
            all_layers = symbol_info.get_internals()
            fc1_sym = all_layers['fc1_output']
            flops_info = flops_utils.count_flops(fc1_sym, data=(1, data_shape[0], data_shape[1], data_shape[2]))
            flops_info_str = flops_utils.flops_str(flops_info)
            print("Network flops_info_str: {}".format(flops_info_str))
            pass

        model = mx.mod.Module(context=mx.gpu(),
                              # context=ctx,
                              symbol=symbol_info)

        loss_name = self.loss_info["loss_name"]

        rec_data_path = self.data_info["rec_data_path"]
        idx_data_path = self.data_info["idx_data_path"]

        bin_data_file_path = self.data_info["bin_data_file_path"]
        val_targets_list = self.data_info["val_targets"]

        # 加载 .bin data 验证数据
        bin_data_list = get_data_utils.load_bin(bin_data_file_path=bin_data_file_path,
                                                bin_name_list=val_targets_list,
                                                image_shape=self.face_shape)

        # 主要获取数据的迭代器，triplet 与 sfotmax 输入数据的迭代器是不一样的，
        # 具体哪里不一样，后续章节为大家分析
        if loss_name.find("triplet") >= 0:
            triplet_bag_size = self.loss_info["triplet_bag_size"]
            triplet_alpha = self.loss_info["triplet_alpha"]
            triplet_max_ap = self.loss_info["triplet_max_ap"]
            images_per_identity = self.loss_info["images_per_identity"]
            triplet_params = [triplet_bag_size, triplet_alpha, triplet_max_ap]

            train_data_iter = TripletFaceImageIter(rec_data_path=rec_data_path,
                                                   idx_data_path=idx_data_path,
                                                   batch_size=batch_size,
                                                   data_shape=data_shape,
                                                   shuffle_flag=True,
                                                   rand_mirror=self.data_rand_mirror_flag,
                                                   cutoff=self.data_crop_flag,
                                                   ctx_num=ctx_count,
                                                   images_per_identity=images_per_identity,
                                                   triplet_params=triplet_params,
                                                   mx_model=model
                                                   )

            metric2 = LossValueMetric()
            eval_metrics = [mx.metric.create(metric2)]
            pass
        else:
            train_data_iter = FaceImageIter(rec_data_path=rec_data_path,
                                            idx_data_path=idx_data_path,
                                            batch_size=batch_size,
                                            data_shape=data_shape,
                                            shuffle_flag=True,
                                            rand_mirror=self.data_rand_mirror_flag,
                                            cutoff=self.data_crop_flag,
                                            color_jitter=self.data_color_aug,
                                            images_filter=self.data_image_filter
                                            )

            metric1 = AccMetric()
            eval_metrics = [mx.metric.create(metric1)]

            # Focal loss，一种改进的交叉损失熵
            if self.ce_loss:
                metric2 = LossValueMetric()
                eval_metrics.append(mx.metric.create(metric2))
                pass
            pass

        # 把 train_data_iter 转化为 mx.io.PrefetchingIter 迭代器
        train_data_iter = mx.io.PrefetchingIter(train_data_iter)

        net_name = self.net_info["net_name"]

        if net_name == "f_res_net" or net_name == "f_mobile_face_net":
            # resNet style
            initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="out", magnitude=2)
            pass
        else:
            initializer = mx.init.Xavier(rnd_type='uniform', factor_type="in", magnitude=2)
            pass

        re_scale = 1.0 / ctx_count

        optimize = optimizer.Adam(learning_rate=self.learning_rate, wd=self.weight_decay, rescale_grad=re_scale)
        callback_speed = mx.callback.Speedometer(batch_size, self.print_step)

        # 最高的准曲率 lfw and target
        highest_acc = [0.0, 0.0]

        global_step = [0]
        save_step = [0]

        print("learning_rate_step_list: {}".format(self.learning_rate_step_list))

        def batch_callback_fun(param):
            global_step[0] += 1
            m_batch = global_step[0]

            for step in self.learning_rate_step_list:
                if m_batch == step:
                    optimize.lr *= 0.1
                    print("learning rate change to: {}".format(optimize.lr))
                    break
                    pass
                pass

            # print(param)
            callback_speed(param)

            # 每1000批次进行一次打印
            if m_batch % 1000 == 0:
                print("learning_rate: {}\nbatch: {}\n epoch: {}".format(optimize.lr, param.nbatch, param.epoch))
                pass

            if m_batch >= 0 and m_batch % self.val_step == 0:
                acc_list = self.ver_test(bin_data_list=bin_data_list,
                                         val_targets_list=val_targets_list,
                                         model_net=model,
                                         batch_size=batch_size,
                                         n_batch=m_batch)

                save_step[0] += 1
                m_save = save_step[0]
                do_save_flag = False
                is_highest_flag = False

                print("-" * 100)
                # 如果存在评估集
                print("acc_list: {}".format(acc_list))
                if len(acc_list) > 0:
                    score = sum(acc_list)
                    if acc_list[-1] >= highest_acc[-1]:
                        if acc_list[-1] > highest_acc[-1]:
                            is_highest_flag = True
                            pass
                        else:
                            if score >= highest_acc[0]:
                                is_highest_flag = True
                                highest_acc[0] = score
                                pass
                            pass

                        highest_acc[-1] = acc_list[-1]
                        pass

                    pass

                # 判断是否保存模型
                if is_highest_flag:
                    do_save_flag = True
                    pass
                if self.save_model_num == 0:
                    do_save_flag = False
                    pass
                elif self.save_model_num == 2:
                    do_save_flag = True
                    pass
                elif self.save_model_num == 3:
                    m_save = 1
                    pass

                if do_save_flag:
                    print("m_save: {}".format(m_save))

                    arg, aux = model.get_params()

                    if self.check_feature_flag:
                        all_layers = model.symbol.get_internals()
                        fc1_sym = all_layers["fc1_output"]

                        arg_base = {}
                        for key in arg:
                            if not key.startswith("fc7"):
                                arg_base[key] = arg[key]
                                pass
                            pass
                        mx.model.save_checkpoint(self.save_model_prefix_path, m_save, fc1_sym, arg_base, aux)
                        pass
                    else:
                        mx.model.save_checkpoint(self.save_model_prefix_path, m_save, model.symbol, arg, aux)
                        pass
                    pass

                print("highest_acc[m_batch: {}]: {:.5f}".format(m_batch, highest_acc[-1]))
                pass

            # 如果最大步骤大于 0, 且 训练步骤 > 最大步骤, 则退出程序
            if self.max_steps > 0 and m_batch > self.max_steps:
                sys.exit(0)
                pass
            pass

        begin_epoch = 0

        # 训练模型
        model.fit(train_data=train_data_iter,
                  begin_epoch=begin_epoch,
                  num_epoch=999999,
                  eval_metric=eval_metrics,
                  kvstore=self.kv_store,
                  optimizer=optimize,
                  initializer=initializer,
                  arg_params=arg_params,
                  aux_params=aux_params,
                  allow_missing=True,
                  batch_end_callback=batch_callback_fun
                  )

        pass

    def ver_test(self, bin_data_list, val_targets_list, model_net, batch_size, n_batch, k_folds=8):
        acc_result_list = []
        bin_data_list_len = len(bin_data_list)
        for bin_data_index in range(bin_data_list_len):
            acc, std, feature_norm, feature_list = self.verification_model.test_details(bin_data_list[bin_data_index],
                                                                                        model_net=model_net,
                                                                                        batch_size=batch_size,
                                                                                        k_folds=k_folds)

            bin_name_info = os.path.splitext(val_targets_list[bin_data_index])[0]
            print("[{}][{}] XNorm: {:.5f}".format(bin_name_info, n_batch, feature_norm))
            print("[{}]acc: {:.5f}+-{:.5f}".format(bin_name_info, acc, std))
            acc_result_list.append(acc)
            pass

        return acc_result_list
        pass


if __name__ == "__main__":
    # 代码开始时间
    start_time = datetime.now()
    print("开始时间: {}".format(start_time))

    demo = FaceTrain()
    demo.do_train()

    # 代码结束时间
    end_time = datetime.now()
    print("结束时间: {}, 训练模型耗时: {}".format(end_time, end_time - start_time))
