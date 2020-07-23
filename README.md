# [insight_face_pro](https://github.com/wandaoyi/insight_face_pro)
tensorflow, mxnet 版本的 insight_face 人脸识别项目 2020-07-23
-


- [论文地址](https://arxiv.org/abs/1801.07698)
- [论文对应源码地址](https://github.com/deepinsight/insightface)
- [我的 CSDN 博客](https://blog.csdn.net/qq_38299170) 
- 环境依赖(其实版本要求并不严格，你的版本要是能跑起来，那也是OK的)：
```bashrc
pip install easydict
pip install numpy==1.16
conda install tensorflow-gpu==1.12.0
pip install mxnet-cu90
pip install opencv-python
```
-
- [训练和验证数据下载地址](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo)
- [预训练模型下载地址](https://github.com/deepinsight/insightface/wiki/Model-Zoo)
- All face images are aligned by [MTCNN](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html) and cropped to 112x112
- Face Detection, Please check RetinaFace(我的代码中没加入有, 但在作者源码中有) for more details
- 
-
- 将数据放到指定的文件目录下(config.py 文件):
- 其实，做好依赖，拿到数据，就仔细看看 config.py 文件，里面全是配置。配好路径或一些超参，基本上，后面就是一键运行就 OK 了。
- 对 config.py 进行配置设置。

## 数据生成
- 对于模型的训练或验证，我们都需要数据，这些数据，可以去下载开源的，也可以自己制作。
- 这里对于制作人脸数据，我使用的是数据蒸馏的方法。
- data_distillation.py 图原图进行人脸检测，人脸分类，人脸聚类等操作
- prepare.py 对分类或聚类后的人脸数据生成 .bin, .rec, .idx 等数据
- video_2_image.py 是将 视频流 转为 图像流


## 训练模型
- face_train.py 人脸识别训练


## 模型验证
- verification_model.py 人脸识别模型验证
- 
- 对于模型测试，想弄的话，也很简单，利用数据库保存录入人的信息
- 测试的初始化，就将初始化 128 维人脸特征保存到缓存
- 当新的图像进入，先人脸检测，人脸矫正，再提取 128 维人脸特征
- 之后，再将新图像的 128 维特征 与 缓存的 128 维特征进行相似度计算
- 最后，选择大于阈值的人脸结果，再根据缓存获取到该用户的信息


## 本项目的优点
- 就是方便，很多东西，我已经做成傻瓜式一键操作的方式。里面的路径，如果不喜欢用相对路径的，可以在 config.py 里面选择 绝对路径
- 本人和唠叨，里面的代码，基本都做了注解，就怕有人不理解，不懂，我只是希望能给予不同的你，一点点帮助。


