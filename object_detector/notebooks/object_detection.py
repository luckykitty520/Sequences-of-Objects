import os
import math
import random
import numpy as np
import tensorflow as tf
import cv2

from tensorflow.contrib import slim
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
sys.path.append('../')
from nets import ssd_vgg_300, ssd_common, np_methods, ssd_vgg_512
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization


"""
    gpu memory manage
"""
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)


"""
    define graph & restore SSD model 
"""
# input image placeholder
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
ckpt_filename = '../checkpoints/model.ckpt-91516'
# print(ckpt_filename)
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)


"""
    object detection
"""


def process_image(img, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
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

# object detection
image_path = sys.argv[1]
img = mpimg.imread(image_path)
rclasses, rscores, rbboxes = process_image(img)

# print detection results
height = img.shape[0]
width = img.shape[1]
print('number of object', rclasses.shape[0])
print('\n')

for i in range(rclasses.shape[0]):
    cls_id = int(rclasses[i])
    if cls_id >= 0:
        score = rscores[i]
        ymin = int(rbboxes[i, 0] * height)
        xmin = int(rbboxes[i, 1] * width)
        ymax = int(rbboxes[i, 2] * height)
        xmax = int(rbboxes[i, 3] * width)

        print('object {0}'.format(i))
        print('class', rclasses[i])
        print('coordination', xmin,ymin,xmax,ymax)
        print('confidence', score)
        print('\n')


"""
    label in original image
"""
# [label, color]
label_dict = {0: [None, None],        # empty object
              1: [None, None],        # cheng guan
              2: ['Thatched cottage', (100/256, 149/256, 237/256)],
              3: ['Rock', (200/256, 200/256, 0/256)],
              4: ['Bridge', (255/256, 106/256, 106/256)],
              5: ['Mountain slope', (255/256, 105/256, 180/256)],
              6: ['Peak', (255/256, 48/256, 48/256)],
              7: ['Tree', (160/256, 32/256, 240/256)],
              8: ['Cascading peaks', (255/256, 165/256, 0/256)],
              9: ['Inscription', (0/256, 200/256, 0/256)],
              10: ['Stamp', (0/256, 190/256, 200/256)]}

# draw & show
dpi = 100
fig = plt.figure(figsize=(img.shape[1] / dpi, img.shape[0] / dpi), dpi=dpi)    # canvas
plt.imshow(img)     # draw img

for i in range(rclasses.shape[0]):
    cls_id = int(rclasses[i])
    if cls_id >= 0:
        score = rscores[i]
        ymin = int(rbboxes[i, 0] * height)
        xmin = int(rbboxes[i, 1] * width)
        ymax = int(rbboxes[i, 2] * height)
        xmax = int(rbboxes[i, 3] * width)

        # draw rectangle
        rect = plt.Rectangle((xmin, ymin),
                             xmax - xmin,
                             ymax - ymin,
                             fill=False,
                             edgecolor=label_dict[cls_id][1],
                             linewidth=2)
        plt.gca().add_patch(rect)  # draw rect

        # draw label
        class_name = label_dict[cls_id][0]
        plt.gca().text(int(xmin) - 2, int(ymin) - 2,
                       '{:s} | {:.3f}'.format(class_name, score),
                       bbox=dict(facecolor=label_dict[cls_id][1], alpha=0.5),
                       fontsize=10, color='white')

plt.axis('off')     # off axis
plt.show()




