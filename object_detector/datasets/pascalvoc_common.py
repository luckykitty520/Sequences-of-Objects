# Copyright 2015 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides data for the Pascal VOC Dataset (images + annotations).
"""
import os

import tensorflow as tf
from datasets import dataset_utils

slim = tf.contrib.slim

VOC_LABELS0 = {
    'none': (0, 'Background'),
    'aeroplane': (1, 'Vehicle'),
    'bicycle': (2, 'Vehicle'),
    'bird': (3, 'Animal'),
    'boat': (4, 'Vehicle'),
    'bottle': (5, 'Indoor'),
    'bus': (6, 'Vehicle'),
    'car': (7, 'Vehicle'),
    'cat': (8, 'Animal'),
    'chair': (9, 'Indoor'),
    'cow': (10, 'Animal'),
    'diningtable': (11, 'Indoor'),
    'dog': (12, 'Animal'),
    'horse': (13, 'Animal'),
    'motorbike': (14, 'Vehicle'),
    'person': (15, 'Person'),
    'pottedplant': (16, 'Indoor'),
    'sheep': (17, 'Animal'),
    'sofa': (18, 'Indoor'),
    'train': (19, 'Vehicle'),
    'tvmonitor': (20, 'Indoor'),
}

labels = ['chengguan',
    'fangwu',
    'fantou',
    'qiao',
    'shanpo-p',
    'shanpo-t',
    'shanpo-y',
    'shantou-h',
    'shantou-j',
    'shantou-p',
    'shantou-r',
    'shantou-t',
    'shantou-y',
    'shu-cy',
    'shu-dy',
    'shu-qy',
    'shu-ry',
    'shu-s',
    'shu-sy',
    'shu-xz',
    'stzh-c',
    'stzh-cm',
    'stzh-t',
    'stzh-z',
    'tikuan-qc',
    'tikuan-xz',
    'yinzhang-qc',
    'yinzhang-qichang',
    'yinzhang-xz']

VOC_LABELS1 = {
    'chengguan': (1, "ChengGuan"),
    'fangwu': (2, "FangWu"),
    'fantou': (3, "FanTou"),
    'qiao': (4, "Qiao"),
    'shanpo-p': (5, "ShanPo-p"),
    'shanpo-t': (6, "ShanPo-t"),
    'shanpo-y': (7, "ShanPo-y"),
    'shantou-h': (8, "ShanTou-h"),
    'shantou-j': (9, "ShanTou-j"),
    'shantou-p': (10, "ShanTou-p"),
    'shantou-r': (11, "ShanTou-r"),
    'shantou-t': (12, "ShanTou-t"),
    'shantou-y': (13, "ShanTou-y"),
    'shu-cy': (14, "Shu-cy"),
    'shu-dy': (15, "Shu-dy"),
    'shu-qy': (16, "Shu-qy"),
    'shu-ry': (17, "Shu-ry"),
    'shu-s': (18, "Shu-s"),
    'shu-sy': (19, "Shu-sy"),
    'shu-xz': (20, "Shu-xz"),
    'stzh-c': (21, "STZH-c"),
    'stzh-cm': (22, "STZH-cm"),
    'stzh-t': (23, "STZH-t"),
    'stzh-z': (24, "STZH-z"),
    'tikuan-qc': (25, "TiKuan-qc"),
    'tikuan-xz': (26, "TiKuan-xz"),
    'yinzhang-qc': (27, "YinZhang-qc"),
    'yinzhang-qichang': (28, "YinZhang-qichang"),
    'yinzhang-xz': (29, "YinZhang-xz"),
}

VOC_LABELS = {
    'none': (0, 'Background'),
    'chengguan': (1, "ChengGuan"),
    'fangwu': (2, "FangWu"),
    'fantou': (3, "FanTou"),
    'qiao': (4, "Qiao"),
    'shanpo': (5, "ShanPo"),
    'shantou': (6, "ShanTou-h"),
    'shu-cy': (7, "Shu-cy"),
    'shu-dy': (8, "Shu-dy"),
    'shu-qy': (9, "Shu-qy"),
    'shu-ry': (10, "Shu-ry"),
    'shu-s': (11, "Shu-s"),
    'shu-sy': (12, "Shu-sy"),
    'shu-xz': (13, "Shu-xz"),
    'stzh': (14, "STZH-c"),
    'tikuan': (15, "TiKuan-qc"),
    'yinzhang': (16, "YinZhang-qc"),
}

VOC_LABELS = {
    'no'
    'ne': (0, 'Background'),
    'chengguan': (1, "ChengGuan"),
    'fangwu': (2, "FangWu"),
    'fantou': (3, "FanTou"),
    'qiao': (4, "Qiao"),
    'shanpo': (5, "ShanPo"),
    'shantou': (6, "ShanTou-h"),

    'shu': (7, "Shu-cy"),
    'stzh': (8, "STZH-c"),
    'tikuan': (9, "TiKuan-qc"),
    'yinzhang': (10, "YinZhang-qc"),
}


def get_split(split_name, dataset_dir, file_pattern, reader, split_to_sizes, items_to_descriptions, num_classes):
    """Gets a dataset tuple with instructions for reading Pascal VOC dataset.

    Args:
      split_name: A train/test split name.
      dataset_dir: The base directory of the dataset sources.
      file_pattern: The file pattern to use when matching the dataset sources.
        It is assumed that the pattern contains a '%s' string so that the split
        name can be inserted.
      reader: The TensorFlow reader type.

    Returns:
      A `Dataset` namedtuple.

    Raises:
        ValueError: if `split_name` is not a valid train/test split.
    """
    if split_name not in split_to_sizes:
        raise ValueError('split name %s was not recognized.' % split_name)
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

    # Allowing None in the signature so that dataset_factory can use the default.
    if reader is None:
        reader = tf.TFRecordReader
    # Features in Pascal VOC TFRecords.
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/height': tf.FixedLenFeature([1], tf.int64),
        'image/width': tf.FixedLenFeature([1], tf.int64),
        'image/channels': tf.FixedLenFeature([1], tf.int64),
        'image/shape': tf.FixedLenFeature([3], tf.int64),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/difficult': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/truncated': tf.VarLenFeature(dtype=tf.int64),
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'shape': slim.tfexample_decoder.Tensor('image/shape'),
        'object/bbox': slim.tfexample_decoder.BoundingBox(
                ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
        'object/label': slim.tfexample_decoder.Tensor('image/object/bbox/label'),
        'object/difficult': slim.tfexample_decoder.Tensor('image/object/bbox/difficult'),
        'object/truncated': slim.tfexample_decoder.Tensor('image/object/bbox/truncated'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    labels_to_names = None
    if dataset_utils.has_labels(dataset_dir):
        labels_to_names = dataset_utils.read_label_file(dataset_dir)
    # else:
    #     labels_to_names = create_readable_names_for_imagenet_labels()
    #     dataset_utils.write_label_file(labels_to_names, dataset_dir)

    return slim.dataset.Dataset(
            data_sources=file_pattern,
            reader=reader,
            decoder=decoder,
            num_samples=split_to_sizes[split_name],
            items_to_descriptions=items_to_descriptions,
            num_classes=num_classes,
            labels_to_names=labels_to_names)
