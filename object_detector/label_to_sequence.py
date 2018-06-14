import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2
import csv


slim = tf.contrib.slim
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
sys.path.append('../')
from nets import ssd_vgg_300, ssd_common, np_methods, ssd_vgg_512
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization

gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)

net_shape = (512, 512)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
# ssd_net = ssd_vgg_300.SSDNet()
ssd_net = ssd_vgg_512.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

# Restore SSD model.
# ckpt_filename = '../checkpoints/ssd_300_vgg.ckpt'
ckpt_filename = './checkpoints/model.ckpt-91516'
print(ckpt_filename)
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)


def process_image(img, select_threshold=0.5, nms_threshold=.45, net_shape=(512, 512)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})

    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
        rpredictions, rlocalisations, ssd_anchors,
        select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)

    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes







cid=0

load_path="/Users/qianzheng/Downloads/构图/"
pathDir=os.listdir(load_path)

with open("test2.csv", "w", encoding='utf-8') as csvfile2:
    writer2 = csv.writer(csvfile2)
    for path_dir in pathDir :
        paths = []
        if path_dir=='高远':
            cid=1
            for ff in  os.listdir(load_path+'高远/'):
                 # print("ff的名字%s"%(ff))
                 if  not ff.startswith('.'):
                     paths.append(str(''+load_path+'高远/'+ff+''))
                     # writer.writerow(['',ff])
        elif path_dir=='深远':
            cid=2
            for ff in os.listdir(load_path + '深远/'):
                if not ff.startswith('.') and len(ff)>1:
                    paths.append(str(''+load_path + '深远/'+ff+''))
                    # writer.writerow(['',ff])

        elif path_dir=='平远':
            cid=3
            for ff in os.listdir(load_path + '平远/'):
                if not ff.startswith('.'):
                    paths.append(str(''+load_path + '平远/'+ff+''))
                    # writer.writerow(['',ff])

        for p in paths:
            print(p)
            codes = []
            # print(p)
            img = mpimg.imread(p)
            rclasses, rscores, rbboxes = process_image(img)
            # print(rbboxes)
            # visualization.plt_bboxes(img, rclasses, rscores, rbboxes)
            height = img.shape[0]
            width = img.shape[1]
            # print(rclasses)
            codes.extend([cid,width,height])
            #看有多少行
            category=[]
            for i in range(rclasses.shape[0]):
                cls_id = int(rclasses[i])
                if cls_id >= 0:
                    score = rscores[i]
                    ymin = int(rbboxes[i, 0] * height)
                    xmin = int(rbboxes[i, 1] * width)
                    ymax = int(rbboxes[i, 2] * height)
                    xmax = int(rbboxes[i, 3] * width)
                    category.extend([cls_id,xmin,ymin,xmax,ymax,score])
            codes.extend(category)
            writer2.writerow(codes)
    csvfile2.close()

