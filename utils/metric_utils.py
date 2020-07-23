#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/07/13 13:43
# @Author   : WanDaoYi
# @FileName : metric_utils.py
# ============================================

# 训练日志信息打印
import mxnet as mx


class AccMetric(mx.metric.EvalMetric):
    def __init__(self):
        self.axis = 1
        super(AccMetric, self).__init__(
            'acc', axis=self.axis,
            output_names=None, label_names=None)
        self.losses = []
        self.count = 0

    def update(self, labels, predicts):
        self.count += 1
        label = labels[0]
        predict_label = predicts[1]
        # print('ACC', label.shape, pred_label.shape)
        if predict_label.shape != label.shape:
            predict_label = mx.ndarray.argmax(predict_label, axis=self.axis)
            pass
        predict_label = predict_label.asnumpy().astype('int32').flatten()
        label = label.asnumpy()
        if label.ndim == 2:
            label = label[:, 0]
            pass
        label = label.astype('int32').flatten()
        assert label.shape == predict_label.shape
        self.sum_metric += (predict_label.flat == label.flat).sum()
        self.num_inst += len(predict_label.flat)


class LossValueMetric(mx.metric.EvalMetric):
    def __init__(self):
        self.axis = 1
        super(LossValueMetric, self).__init__('loss_value',
                                              axis=self.axis,
                                              output_names=None,
                                              label_names=None)
        self.losses = []

    def update(self, labels, predicts):
        # label = labels[0].asnumpy()
        predict = predicts[-1].asnumpy()
        # print('in loss', pred.shape)
        # print(pred)
        loss = predict[0]
        self.sum_metric += loss
        self.num_inst += 1.0
        # gt_label = preds[-2].asnumpy()
        # print(gt_label)






