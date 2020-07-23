#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/06/28 16:48
# @Author   : WanDaoYi
# @FileName : calculate_utils.py
# ============================================

import sklearn
import numpy as np
from scipy import interpolate
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold


def fold_split(indices, n_splits=2, shuffle=False):
    """
        K 折分割数据
    :param indices:
    :param n_splits:
    :param shuffle:
    :return:
    """
    if n_splits > 1:
        k_fold = KFold(n_splits=n_splits, shuffle=shuffle)
        return k_fold.split(indices)
        pass
    else:
        return [(indices, indices)]
        pass
    pass


def calculate_accuracy(threshold, dist, actual_is_same, cos_flag=False):
    """
        计算 tpr, fpr, acc
    :param threshold: 阈值
    :param dist: 相似度
    :param actual_is_same: 真实的标签
    :param cos_flag: 是否是余弦相似度比较，True 是; False 为  欧氏距离比较
    :return:
    """
    if cos_flag:
        predict_is_same = np.less(threshold, dist)
        pass
    else:
        predict_is_same = np.less(dist, threshold)
        pass

    # TP 命中
    tp = np.sum(np.logical_and(predict_is_same, actual_is_same))
    # FP 误判
    fp = np.sum(np.logical_and(predict_is_same, np.logical_not(actual_is_same)))
    tn = np.sum(np.logical_and(np.logical_not(predict_is_same), np.logical_not(actual_is_same)))
    fn = np.sum(np.logical_and(np.logical_not(predict_is_same), actual_is_same))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_val_far(threshold, dist, actual_is_same):
    predict_is_same = np.less(dist, threshold)
    # TP 命中
    true_accept = np.sum(np.logical_and(predict_is_same, actual_is_same))
    # FP 误判
    false_accept = np.sum(np.logical_and(predict_is_same, np.logical_not(actual_is_same)))
    # 全部正例: TP + FN，即所有相同的人脸对
    n_same = np.sum(actual_is_same)
    # 全部负例: FP + TN, 即所有不相同的人脸对
    n_diff = np.sum(np.logical_not(actual_is_same))

    # 真正例率: TPR = TP / (TP + FN)
    tpr = float(true_accept) / float(n_same) if n_same != 0 else 0
    # 假正例率: FPR = FP / (FP + TN)
    fpr = float(false_accept) / float(n_diff) if n_diff != 0 else 0
    return tpr, fpr


def calculate_cos(embedding_feature_1, embedding_feature_2,
                  is_same_list, k_folds, thresholds, pca=0):
    """
        计算余弦相似度的 tpr, fpr, acc list
    :param embedding_feature_1: 取所有的第一张图片的向量
    :param embedding_feature_2: 取所有的第二张图片的向量
    :param is_same_list: 两张图像是否相同的 bool list
    :param k_folds: k 折检验
    :param thresholds: 阈值
    :param pca: pca 值
    :return:
    """

    assert (embedding_feature_1.shape[0] == embedding_feature_2.shape[0])
    assert (embedding_feature_1.shape[1] == embedding_feature_2.shape[1])

    is_same_list_len = len(is_same_list)

    # 获得最小长度标签个数和向量个数中
    min_pairs = min(is_same_list_len, embedding_feature_1.shape[0])

    # 获得阈值个数
    thresholds_len = len(thresholds)

    # tpr:正类预测正确（召回率）
    # fpr:正类预测错误
    # acc:准确率

    tpr_list = np.zeros((k_folds, thresholds_len))
    fpr_list = np.zeros((k_folds, thresholds_len))
    accuracy = np.zeros((k_folds,))
    indices = np.arange(min_pairs)

    # 默认没有启动 pca, 即 pca = 0, 求余弦相似度
    # 做点乘积
    dot_value = np.sum(np.multiply(embedding_feature_1, embedding_feature_2), axis=1)
    # 向量绝对值
    embedding_feature_norm_1 = np.linalg.norm(embedding_feature_1, axis=1)
    embedding_feature_norm_2 = np.linalg.norm(embedding_feature_2, axis=1)
    # 余弦相似度度
    cos_value = dot_value / (embedding_feature_norm_1 * embedding_feature_norm_2)

    # k 折分割数据
    data_fold_split = fold_split(indices=indices, n_splits=k_folds, shuffle=False)

    # 分 k 折进行评估
    for fold_index, (train_data, val_data) in enumerate(data_fold_split):

        if pca > 0:
            # 将训练数据进行 pca 处理
            train_feature_1 = embedding_feature_1[train_data]
            train_feature_2 = embedding_feature_2[train_data]
            train_feature = np.concatenate((train_feature_1, train_feature_2), axis=0)

            pca_model = PCA(n_components=pca)
            pca_model.fit(train_feature)

            feature_pca_trans_1 = pca_model.transform(embedding_feature_1)
            feature_pca_trans_2 = pca_model.transform(embedding_feature_2)

            # 归一化处理
            feature_norm_1 = sklearn.preprocessing.normalize(feature_pca_trans_1)
            feature_norm_2 = sklearn.preprocessing.normalize(feature_pca_trans_2)

            # 将 pca 后的特征去余弦相似度
            # 做点乘积
            dot_value_norm = np.sum(np.multiply(feature_norm_1, feature_norm_2), axis=1)
            # 向量绝对值
            embedding_norm_1 = np.linalg.norm(feature_norm_1, axis=1)
            embedding_norm_2 = np.linalg.norm(feature_norm_2, axis=1)
            # 余弦相似度度
            cos_value = dot_value_norm / (embedding_norm_1 * embedding_norm_2)

            pass

        train_acc = np.zeros((thresholds_len,))

        for threshold_idx, threshold in enumerate(thresholds):
            _, _, train_acc[threshold_idx] = calculate_accuracy(threshold,
                                                                cos_value[train_data],
                                                                is_same_list[train_data],
                                                                cos_flag=True)
            pass

        best_threshold_index = np.argmax(train_acc)
        # print("best_threshold: {}".format(thresholds[best_threshold_index]))

        for thresh_idx, thresh in enumerate(thresholds):
            tpr_list[fold_index, thresh_idx], fpr_list[fold_index, thresh_idx], _ = calculate_accuracy(thresh,
                                                                                                       cos_value[
                                                                                                           val_data],
                                                                                                       is_same_list[
                                                                                                           val_data],
                                                                                                       cos_flag=True)
            pass

        # 获取最好的 accuracy
        _, _, accuracy[fold_index] = calculate_accuracy(thresholds[best_threshold_index],
                                                        cos_value[val_data],
                                                        is_same_list[val_data],
                                                        cos_flag=True)
        pass

    tpr = np.mean(tpr_list, axis=0)
    fpr = np.mean(fpr_list, axis=0)
    return tpr, fpr, accuracy
    pass


def calculate_roc(embedding_feature_1, embedding_feature_2,
                  is_same_list, k_folds, thresholds, pca=0):
    """
        计算欧氏距离的 tpr, fpr, acc list
    :param embedding_feature_1: 取所有的第一张图片的向量
    :param embedding_feature_2: 取所有的第二张图片的向量
    :param is_same_list: 两张图像是否相同的 bool list
    :param k_folds: k 折检验
    :param thresholds: 阈值
    :param pca: pca 值
    :return:
    """

    assert (embedding_feature_1.shape[0] == embedding_feature_2.shape[0])
    assert (embedding_feature_1.shape[1] == embedding_feature_2.shape[1])

    is_same_list_len = len(is_same_list)

    # 获得最小长度标签个数和向量个数中
    min_pairs = min(is_same_list_len, embedding_feature_1.shape[0])

    # 获得阈值个数
    thresholds_len = len(thresholds)

    # tpr:正类预测正确（召回率）
    # fpr:正类预测错误
    # acc:准确率

    tpr_list = np.zeros((k_folds, thresholds_len))
    fpr_list = np.zeros((k_folds, thresholds_len))
    accuracy = np.zeros((k_folds,))
    indices = np.arange(min_pairs)

    # 默认没有启动 pca, 即 pca = 0, 求范数距离欧式距离
    # 做减法
    diff = np.subtract(embedding_feature_1, embedding_feature_2)
    # 求平方和
    dist = np.sum(np.square(diff), 1)

    # k 折分割数据
    data_fold_split = fold_split(indices=indices, n_splits=k_folds, shuffle=False)

    # 分 k 折进行评估
    for fold_index, (train_data, val_data) in enumerate(data_fold_split):

        if pca > 0:
            # 将训练数据进行 pca 处理
            train_feature_1 = embedding_feature_1[train_data]
            train_feature_2 = embedding_feature_2[train_data]
            train_feature = np.concatenate((train_feature_1, train_feature_2), axis=0)

            pca_model = PCA(n_components=pca)
            pca_model.fit(train_feature)

            feature_pca_trans_1 = pca_model.transform(embedding_feature_1)
            feature_pca_trans_2 = pca_model.transform(embedding_feature_2)

            feature_norm_1 = sklearn.preprocessing.normalize(feature_pca_trans_1)
            feature_norm_2 = sklearn.preprocessing.normalize(feature_pca_trans_2)

            # 将 pca 后的特征进行 做减法 和 平方和
            diff = np.subtract(feature_norm_1, feature_norm_2)
            dist = np.sum(np.square(diff), 1)
            pass

        train_acc = np.zeros((thresholds_len,))

        for threshold_idx, threshold in enumerate(thresholds):
            _, _, train_acc[threshold_idx] = calculate_accuracy(threshold,
                                                                dist[train_data],
                                                                is_same_list[train_data])
            pass

        best_threshold_index = np.argmax(train_acc)
        # print("best_threshold: {}".format(thresholds[best_threshold_index]))

        for thresh_idx, thresh in enumerate(thresholds):
            tpr_list[fold_index, thresh_idx], fpr_list[fold_index, thresh_idx], _ = calculate_accuracy(thresh,
                                                                                                       dist[val_data],
                                                                                                       is_same_list[
                                                                                                           val_data])
            pass

        # 获取最好的 accuracy
        _, _, accuracy[fold_index] = calculate_accuracy(thresholds[best_threshold_index],
                                                        dist[val_data],
                                                        is_same_list[val_data])
        pass

    tpr = np.mean(tpr_list, axis=0)
    fpr = np.mean(fpr_list, axis=0)
    return tpr, fpr, accuracy
    pass


def calculate_val(embedding_feature_1, embedding_feature_2,
                  is_same_list, k_folds, thresholds, far_target):
    """
        计算欧氏距离的 tpr, fpr, tpr_std值 list
    :param embedding_feature_1: 取所有的第一张图片的向量
    :param embedding_feature_2: 取所有的第二张图片的向量
    :param is_same_list: 两张图像是否相同的 bool list
    :param k_folds: k 折检验
    :param thresholds: 阈值
    :param far_target:
    :return:
    """

    assert (embedding_feature_1.shape[0] == embedding_feature_2.shape[0])
    assert (embedding_feature_1.shape[1] == embedding_feature_2.shape[1])

    is_same_list_len = len(is_same_list)

    # 获得最小长度标签个数和向量个数中
    min_pairs = min(is_same_list_len, embedding_feature_1.shape[0])

    # 获得阈值个数
    thresholds_len = len(thresholds)

    # tpr
    val = np.zeros(k_folds)
    # fpr
    far = np.zeros(k_folds)

    indices = np.arange(min_pairs)

    # 默认没有启动 pca, 即 pca = 0, 求范数距离欧式距离
    # 做减法
    diff = np.subtract(embedding_feature_1, embedding_feature_2)
    # 求平方和
    dist = np.sum(np.square(diff), 1)

    # k 折分割数据
    data_fold_split = fold_split(indices=indices, n_splits=k_folds, shuffle=False)

    # 分 k 折进行评估
    for fold_index, (train_data, val_data) in enumerate(data_fold_split):

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros((thresholds_len,))

        for thresh_idx, thresh in enumerate(thresholds):
            _, far_train[thresh_idx] = calculate_val_far(thresh, dist[train_data], is_same_list[train_data])
            pass

        train_far_max = np.max(far_train)
        if train_far_max > far_target:
            # 插值, kind='slinear' 为线性插值
            far_value = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = far_value(far_target)
            pass
        else:
            threshold = 0.0
            pass

        val[fold_index], far[fold_index] = calculate_val_far(threshold, dist[val_data], is_same_list[val_data])
        pass

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean
    pass


def get_pos_neg_sample(data_list, embedding_feature_1, embedding_feature_2,
                       is_same_list, k_folds, thresholds):
    """

    :param data_list:
    :param embedding_feature_1:
    :param embedding_feature_2:
    :param is_same_list:
    :param k_folds:
    :param thresholds:
    :return:
    """

    assert (embedding_feature_1.shape[0] == embedding_feature_2.shape[0])
    assert (embedding_feature_1.shape[1] == embedding_feature_2.shape[1])

    is_same_list_len = len(is_same_list)

    # 获得最小长度标签个数和向量个数中
    min_pairs = min(is_same_list_len, embedding_feature_1.shape[0])

    # 获得阈值个数
    thresholds_len = len(thresholds)

    # tpr:正类预测正确（召回率）
    # fpr:正类预测错误
    # acc:准确率

    tpr_list = np.zeros((k_folds, thresholds_len))
    fpr_list = np.zeros((k_folds, thresholds_len))
    accuracy = np.zeros((k_folds,))
    indices = np.arange(min_pairs)

    # 默认没有启动 pca, 即 pca = 0, 求范数距离 欧式距离
    # 即: sqrt((x1 - x2) ** 2)
    # 做减法, embedding_feature_1 - embedding_feature_2
    diff = np.subtract(embedding_feature_1, embedding_feature_2)
    # 求平方和, 即每一条数据的 欧氏距离
    dist = np.sum(np.square(diff), 1)

    data = data_list[0]

    pos_output = []
    neg_output = []

    count = 0

    # k 折分割数据
    data_fold_split = fold_split(indices=indices, n_splits=k_folds, shuffle=False)

    # 分 k 折进行评估
    for fold_index, (train_data, val_data) in enumerate(data_fold_split):
        # Find the threshold that gives FAR = far_target
        train_acc = np.zeros((thresholds_len,))

        for threshold_idx, threshold in enumerate(thresholds):
            _, _, train_acc[threshold_idx] = calculate_accuracy(threshold,
                                                                dist[train_data],
                                                                is_same_list[train_data])
            pass

        best_threshold_index = np.argmax(train_acc)
        # print("best_threshold: {}".format(thresholds[best_threshold_index]))

        for thresh_idx, thresh in enumerate(thresholds):
            tpr_list[fold_index, thresh_idx], fpr_list[fold_index, thresh_idx], _ = calculate_accuracy(thresh,
                                                                                                       dist[val_data],
                                                                                                       is_same_list[
                                                                                                           val_data])
            pass

        # 获取最好的 accuracy
        _, _, accuracy[fold_index] = calculate_accuracy(thresholds[best_threshold_index],
                                                        dist[val_data],
                                                        is_same_list[val_data])

        best_threshold = thresholds[best_threshold_index]

        num = 1

        for val_index in val_data:
            index_a = val_index * 2
            index_b = index_a + 1
            is_same = is_same_list[val_index]
            dist_info = dist[val_index]
            print("dist_info: {}".format(dist_info))
            violate = dist_info - best_threshold

            if not is_same:
                violate *= -1.0
                pass

            if violate > 0.0:

                # image to bgr
                image_a = data[index_a].asnumpy().transpose((1, 2, 0))[..., ::-1]
                image_b = data[index_b].asnumpy().transpose((1, 2, 0))[..., ::-1]

                if is_same:
                    pos_output.append((image_a, image_b, dist_info, best_threshold, index_a))
                    pass
                else:
                    neg_output.append((image_a, image_b, dist_info, best_threshold, index_a))
                    pass
                pass

            num += 1
            pass

        count += 1
        pass

    # tpr = np.mean(tpr_list, 0)
    # fpr = np.mean(fpr_list, 0)
    acc = np.mean(accuracy)
    print("acc: {}".format(acc))

    return pos_output, neg_output
    pass
