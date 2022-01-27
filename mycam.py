# 对单个图像可视化
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, \
                                         deprocess_image, \
                                         preprocess_image
from torchvision.models import resnet50
import cv2
import numpy as np
import os
import torch
from timm.models import resume_checkpoint
import timm

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def load_model():
    model = timm.create_model('tf_efficientnetv2_s_in21ft1k', num_classes=200)
    # epoch = resume_checkpoint(model, 'output/train/20211221-233218-tf_efficientnetv2_s_in21ft1k-448/model_best.pth.tar')
    img = torch.randn(1, 3, 448, 448)

    x = model.conv_stem(img)
    x = model.bn1(x)
    x = model.act1(x)
    x = model.blocks[0](x)
    x = model.blocks[1](x)
    x = model.blocks[2](x)
    x = model.blocks[3](x)
    x = model.blocks[4](x)
    x = model.blocks[5](x)
    x = model.conv_head(x)
    x = model.bn2(x)
    x = model.act2(x)
    x = model.global_pool(x)
    x = model.classifier(x)

    return model

def onepic(model, image_path='Laysan_Albatross_0003_1033.jpg', use_cuda=False):
    # 2.选择目标层
    target_layers = [model.blocks[-1]]

    # 3. 构建输入图像的Tensor形式
    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]   # 1是读取rgb
    rgb_img = cv2.resize(rgb_img, (448, 448))
    rgb_img = np.float32(rgb_img) / 255.0

    # preprocess_image作用：归一化图像，并转成tensor
    # torch.Size([1, 3, 224, 224])
    input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # Create an input tensor image for your model..
    # Note: input_tensor can be a batch tensor with several images!

    # Construct the CAM object once, and then re-use it on many images:
    # 4.初始化GradCAM，包括模型，目标层以及是否使用cuda
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)

    # If target_category is None, the highest scoring category
    # will be used for every image in the batch.
    # target_category can also be an integer, or a list of different integers
    # for every image in the batch.
    # 5.选定目标类别，如果不设置，则默认为分数最高的那一类
    target_category = None  # 281

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    # 6. 计算cam
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)  # [batch, 224,224]

    # In this example grayscale_cam has only one image in the batch:
    # 7.展示热力图并保存, grayscale_cam是一个batch的结果，只能选择一张进行展示
    grayscale_cam = grayscale_cam[0]
    visualization = show_cam_on_image(rgb_img, grayscale_cam)  # (224, 224, 3)
    cv2.imwrite('{}'.format(savepath(path=image_path)), visualization)

def allpic(path, use_cuda):
    model = load_model()
    folders = os.listdir(path)
    for folderi in folders:
        jpgpath = '{}{}/'.format(path, folderi)
        jpgs = os.listdir(jpgpath)
        for jpgi in jpgs:
            onepic(model, image_path='{}{}'.format(jpgpath, jpgi), use_cuda=use_cuda)

def savepath(path='D:/deeplearning/datasets/official/CaltechBirds-200/val/002.Laysan_Albatross/0001_545.jpg'):
    jpgspath = 'output' + path.split('official')[-1]
    names = path.split('/')[-1]
    folders = jpgspath[:-(len(names))]
    if not os.path.exists(folders):
        os.makedirs(folders)
    print(jpgspath)
    return jpgspath

if __name__ == '__main__':
    path = 'D:/deeplearning/datasets/official/CaltechBirds-200/val/'
    use_cuda = True
    allpic(path, use_cuda)
    print('finish')

# 2021-12-23 guangjinzheng
# https://zhuanlan.zhihu.com/p/371296750
