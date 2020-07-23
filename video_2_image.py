#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/07/21 16:06
# @Author   : WanDaoYi
# @FileName : video_2_image.py
# ============================================

import os
import cv2
import shutil


class Video2Image(object):

    def __init__(self):
        pass

    @staticmethod
    def video_2_image(video_path, image_fold_path, image_suffix=".png", image_name_len=6, frame_frequency=25):
        """
            视频流转图像
        :param video_path: 视频路径
        :param image_fold_path: 生成 image 保存的 fold 路径
        :param image_suffix: 图像保存的后缀名
        :param image_name_len: image 名字的长度
        :param frame_frequency: 每多少帧取一张图像
        :return:
        """
        # 视频的名称
        video_name = os.path.split(video_path)[-1]
        name_info = os.path.splitext(video_name)[0]
        save_image_fold_path = os.path.join(image_fold_path, name_info)
        if not os.path.exists(save_image_fold_path):
            os.makedirs(save_image_fold_path)
            pass

        # 读取视频流
        cap = cv2.VideoCapture(video_path)

        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print("视频 width: {}, height: {}, fps: {}, frames_num: {}".format(width, height, fps, frames_num))

        # 保存图像名字的格式
        image_name_format = "{:0" + str(image_name_len) + "d}{}"
        frame_count = 0
        image_count = 0
        while True:

            response_info, image = cap.read()

            # 当有图像的时候 response_info 为 True, 没图像的时候为 False
            if not response_info:
                print("not response_info, not image!")
                break
                pass
            if (frame_count + 1) % frame_frequency == 0:
                image_name = image_name_format.format(image_count, image_suffix)
                image_path = os.path.join(save_image_fold_path, image_name)
                cv2.imwrite(image_path, image)
                print("image_path: {}".format(image_path))
                image_count += 1
                pass

            frame_count += 1
            pass
        cap.release()
        pass

    @staticmethod
    def file_rename_copy(input_fold_path, output_fold_path, image_name_len=7, count_num=0, delete_flag=False):
        """
            将图像重命名并复制到指定文件夹内
        :param input_fold_path: 输入图像的文件夹路径
        :param output_fold_path: 输出图像的文件夹路径
        :param image_name_len: 重命名的长度
        :param count_num: 重命名的起始名字号码
        :param delete_flag: 是否删除原来的图像
        :return:
        """

        image_count = count_num
        # 保存图像名字的格式
        image_name_format = "{:0" + str(image_name_len) + "d}{}"
        image_fold_name_list = os.listdir(input_fold_path)
        for image_fold_name in image_fold_name_list:
            image_fold_path = os.path.join(input_fold_path, image_fold_name)
            image_name_list = os.listdir(image_fold_path)
            for image_name in image_name_list:
                image_suffix = os.path.splitext(image_name)[-1]
                ori_image_path = os.path.join(image_fold_path, image_name)
                new_image_name = image_name_format.format(image_count, image_suffix)
                new_image_path = os.path.join(output_fold_path, new_image_name)

                shutil.copyfile(ori_image_path, new_image_path)
                print("{} ---> {}".format(ori_image_path, new_image_path))

                if delete_flag:
                    os.remove(ori_image_path)
                    pass
                image_count += 1
                pass
            pass
        pass


if __name__ == "__main__":
    from datetime import datetime

    # 代码开始时间
    start_time = datetime.now()
    print("开始时间: {}".format(start_time))

    video = "./data/video/gee.mp4"
    image_fold = "./data/image"
    output_fold = "./data/align_input"
    output_fold_image_name_list = os.listdir(output_fold)
    output_fold_image_name_len = len(output_fold_image_name_list)
    demo = Video2Image()
    demo.video_2_image(video, image_fold)
    # demo.file_rename_copy(image_fold, output_fold, count_num=output_fold_image_name_len)

    # 代码结束时间
    end_time = datetime.now()
    print("结束时间: {}, 训练模型耗时: {}".format(end_time, end_time - start_time))
    pass
