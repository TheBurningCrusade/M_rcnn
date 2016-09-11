#-*-coding:utf-8-*-
"""
roidb
basic format [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
extended ['image', 'max_classes', 'max_overlaps', 'bbox_targets']
"""

import cv2
import numpy as np

from bbox_regression import compute_bbox_regression_targets
from rcnn.config import config


def prepare_roidb(imdb, roidb):
    """
    add image path, max_classes, max_overlaps to roidb
    :param imdb: image database, provide path
    :param roidb: roidb
    :return: None
    """
    print 'prepare roidb'
    print len(roidb)
    for i in range(len(roidb)):  # image_index
        # 得到每个图片实际的存储路径
        roidb[i]['image'] = imdb.image_path_from_index(imdb.image_set_index[i])
        if config.TRAIN.ASPECT_GROUPING: # True
            size = cv2.imread(roidb[i]['image']).shape    # 一个三元组,(332, 500, 3), 3表示一个彩色像素有gb表示
            roidb[i]['height'] = size[0]
            roidb[i]['width'] = size[1]
        gt_overlaps = roidb[i]['gt_overlaps'].toarray()
        # 最大的overlap的值, gt_overlaps每一行有21个元素，索引代表类编号
        max_overlaps = gt_overlaps.max(axis=1)
        # 最大值的列号, 即class(类别)号
        max_classes = gt_overlaps.argmax(axis=1)
        roidb[i]['max_overlaps'] = max_overlaps
        # 通过overlap的最大值，为每个box找到一个类别
        roidb[i]['max_classes'] = max_classes

        # background roi => background class
        # overlap等于0的行号
        zero_indexes = np.where(max_overlaps == 0)[0]
        # 判断是不是overlap最大值等于0的行号的列号（即类号)是背景类别0的编号
        assert all(max_classes[zero_indexes] == 0)
        # foreground roi => foreground class
        nonzero_indexes = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_indexes] != 0)


def add_bbox_regression_targets(roidb):
    """
    given roidb, add ['bbox_targets'] and normalize bounding box regression targets
    :param roidb: roidb to be processed. must have gone through imdb.prepare_roidb
    :return: means, std variances of targets
    """
    print 'add bounding box regression targets'
    assert len(roidb) > 0
    assert 'max_classes' in roidb[0]

    num_images = len(roidb)
    num_classes = roidb[0]['gt_overlaps'].shape[1]
    for im_i in range(num_images):
        rois = roidb[im_i]['boxes']
        max_overlaps = roidb[im_i]['max_overlaps']
        max_classes = roidb[im_i]['max_classes']
        roidb[im_i]['bbox_targets'] = compute_bbox_regression_targets(rois, max_overlaps, max_classes)

    if config.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED:
        # use fixed / precomputed means and stds instead of empirical values
        means = np.tile(np.array(config.TRAIN.BBOX_MEANS), (num_classes, 1))
        stds = np.tile(np.array(config.TRAIN.BBOX_STDS), (num_classes, 1))
    else:
        # compute mean, std values
        # 注意这里的规则化，是按照label进行的，每一个label都有一个规则化
        # 这里的规则化这是对上一步生成的'bbox_targets'进行的
        class_counts = np.zeros((num_classes, 1)) + config.EPS    #(21, 1)
        sums = np.zeros((num_classes, 4))     # (21, 4)
        squared_sums = np.zeros((num_classes, 4))    # (21, 4)
        for im_i in range(num_images):
            targets = roidb[im_i]['bbox_targets']
            for cls in range(1, num_classes):
                cls_indexes = np.where(targets[:, 0] == cls)[0]
                if cls_indexes.size > 0:
                    class_counts[cls] += cls_indexes.size
                    sums[cls, :] += targets[cls_indexes, 1:].sum(axis=0)
                    squared_sums[cls, :] += (targets[cls_indexes, 1:] ** 2).sum(axis=0)

        means = sums / class_counts
        # var(x) = E(x^2) - E(x)^2
        stds = np.sqrt(squared_sums / class_counts - means ** 2)

    # normalized targets
    for im_i in range(num_images):
        targets = roidb[im_i]['bbox_targets']
        for cls in range(1, num_classes):
            cls_indexes = np.where(targets[:, 0] == cls)[0]
            roidb[im_i]['bbox_targets'][cls_indexes, 1:] -= means[cls, :]
            roidb[im_i]['bbox_targets'][cls_indexes, 1:] /= stds[cls, :]

    return means.ravel(), stds.ravel()
