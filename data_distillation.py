#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/07/18 13:51
# @Author   : WanDaoYi
# @FileName : data_distillation.py
# ============================================

from align import align_lfw
from utils import get_data_utils
from face_recognition import face_classify, face_cluster
from config import cfg


class DataDistillation(object):

    def __init__(self):
        # 人脸检测和矫正的参数
        self.detect_face_model_fold_path = cfg.COMMON.DETECT_FACE_MODEL_PATH
        self.detect_model_name_list = cfg.COMMON.DETECT_MODEL_NAME_LIST

        self.align_input_fold_path = cfg.DATA_SET.INPUT_ALIGN_IMAGE_FILE_PATH
        self.align_output_fold_path = cfg.DATA_SET.OUTPUT_ALIGN_IMAGE_FILE_PATH

        self.face_shape = cfg.COMMON.FACE_SHAPE
        self.image_suffix_list = cfg.COMMON.IMAGE_SUFFIX_LIST
        self.recursive_flag = cfg.DATA_SET.RECURSIVE_FLAG

        # 加载识别模型
        self.recognize_model = get_data_utils.loading_face_recognize_model(cfg.TEST.MODEL_PATH)

        # 人脸分类参数
        self.classify_output_fold_path = cfg.DATA_SET.OUTPUT_FACE_CLASSIFY_FOLD_PATH
        self.info_classify_output_fold_path = cfg.DATA_SET.OUTPUT_INFO_FACE_CLASSIFY_FOLD_PATH
        self.not_recognize_face_fold_path = cfg.DATA_SET.OUTPUT_NOT_RECOGNIZE_FACE_FOLD_PATH
        self.recognize_threshold_value = cfg.DATA_SET.RECOGNIZE_THRESHOLD_VALUE
        self.delete_flag = cfg.DATA_SET.DELETE_FLAG

        # 人脸聚类参数
        self.noise_face_fold_path = cfg.DATA_SET.OUTPUT_CLUSTER_NOISE_FACE_FOLD_PATH
        self.same_face_fold_path = cfg.DATA_SET.OUTPUT_SAME_FACE_FOLD_PATH
        self.info_noise_fold_path = cfg.DATA_SET.OUTPUT_INFO_NOISE_FACE_FOLD_PATH
        self.info_same_face_fold_path = cfg.DATA_SET.OUTPUT_INFO_SAME_FACE_FOLD_PATH
        self.same_threshold_value = cfg.DATA_SET.SAME_THRESHOLD_VALUE

        self.fun_mode = cfg.DATA_SET.FUN_MODE
        pass

    def face_detect(self):
        """
            人脸检测和矫正，lfw 数据类型
        :return:
        """
        align_lfw.do_align_lfw(input_file_path=self.align_input_fold_path,
                               output_file_path=self.align_output_fold_path,
                               model_file_path=self.detect_face_model_fold_path,
                               model_name_list=self.detect_model_name_list,
                               image_shape=self.face_shape,
                               recursive=self.recursive_flag,
                               suffix_info_list=self.image_suffix_list)
        pass

    def face_classifier(self):
        """
            对检测和矫正后的人脸进行分类
        :return:
        """
        face_classify.classifier(input_file_path=self.align_output_fold_path,
                                 output_file_path=self.classify_output_fold_path,
                                 info_file_path=self.info_classify_output_fold_path,
                                 not_recognize_file_path=self.not_recognize_face_fold_path,
                                 face_model=self.recognize_model,
                                 recursive=self.recursive_flag,
                                 image_suffix_list=self.image_suffix_list,
                                 threshold_value=self.recognize_threshold_value,
                                 delete_flag=self.delete_flag)
        pass

    def cluster_face(self):
        face_cluster.cluster(input_class_fold_path=self.classify_output_fold_path,
                             noise_fold_path=self.noise_face_fold_path,
                             same_image_fold_path=self.same_face_fold_path,
                             info_noise_fold_path=self.info_noise_fold_path,
                             info_same_fold_path=self.info_same_face_fold_path,
                             face_model=self.recognize_model,
                             threshold_value=self.same_threshold_value,
                             recursive=self.recursive_flag,
                             image_suffix_list=self.image_suffix_list,
                             delete_flag=self.delete_flag)
        pass


if __name__ == "__main__":
    from datetime import datetime

    # 代码开始时间
    start_time = datetime.now()
    print("开始时间: {}".format(start_time))

    demo = DataDistillation()

    # 对人物图像进行单独检测和矫正人脸
    if demo.fun_mode == 0:
        demo.face_detect()
        pass
    # 单独对检测和矫正后的人脸进行分类(前提是要有检测和矫正后 112 x 112 大小的人脸)
    elif demo.fun_mode == 1:
        demo.face_classifier()
        pass
    # 单独对分类后的人脸进行聚类处理(前提是需要有分类后的 112 x 112 的人脸)
    elif demo.fun_mode == 2:
        demo.cluster_face()
        pass
    # 对人物图像进行人脸检测和矫正后，再对人脸进行分类
    elif demo.fun_mode == 3:
        demo.face_detect()
        demo.face_classifier()
        pass
    # 对检测和矫正后的人脸进行分类，再对人脸进行聚类
    elif demo.fun_mode == 4:
        demo.face_classifier()
        demo.cluster_face()
        pass
    # 对人物图像进行人脸检测和矫正，之后对人脸进行分类，再之后对人脸进行聚类
    elif demo.fun_mode == 5:
        demo.face_detect()
        demo.face_classifier()
        demo.cluster_face()
        pass

    # 代码结束时间
    end_time = datetime.now()
    print("结束时间: {}, 训练模型耗时: {}".format(end_time, end_time - start_time))
