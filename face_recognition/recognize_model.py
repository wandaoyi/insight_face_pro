#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/07/14 18:33
# @Author   : WanDaoYi
# @FileName : recognize_model.py
# ============================================

import sklearn
import mxnet as mx
from mxnet import ndarray as nd
from utils import get_data_utils
from config import cfg


class RecognizeModel(object):

    def __init__(self, model_path=cfg.TEST.MODEL_PATH):

        self.symbol, self.arg_params, self.aux_params = get_data_utils.load_model(model_path)
        self.model = mx.mod.Module(symbol=self.symbol, context=mx.gpu())
        self.image_shape = cfg.COMMON.FACE_SHAPE
        self.model.bind(data_shapes=[('data', (1, self.image_shape[2], self.image_shape[0], self.image_shape[1]))])
        self.model.set_params(self.arg_params, self.aux_params)
        pass

    # 得到输入图像的特征向量
    def predict(self, image):
        image = nd.array(image)
        image = nd.transpose(image, axes=(2, 0, 1)).astype("float32")
        image = nd.expand_dims(image, axis=0)

        data_batch = mx.io.DataBatch(data=(image,))

        self.model.forward(data_batch, is_train=False)
        net_out = self.model.get_outputs()
        embedding_feature = net_out[0].asnumpy()
        norm_embedding_feature = sklearn.preprocessing.normalize(embedding_feature, axis=1)

        return norm_embedding_feature
        pass






