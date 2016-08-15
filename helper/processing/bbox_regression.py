#-*-coding:utf-8-*-
"""
This file has functions about generating bounding box regression targets
"""

import numpy as np

from rcnn.config import config
from bbox_transform import bbox_transform


def bbox_overlaps(boxes, query_boxes):
    # 每一个boxes中的box都要计算它和每一个query_boxes在一个图上的重叠面积
    """
    determine overlaps between boxes and query_boxes
    :param boxes: n * 4 bounding boxes
    :param query_boxes: k * 4 bounding boxes
    :return: overlaps: n * k overlaps
    """
    n_ = boxes.shape[0]
    k_ = query_boxes.shape[0]
    overlaps = np.zeros((n_, k_), dtype=np.float)
    for k in range(k_):
        # 每个query_box的面积
        query_box_area = (query_boxes[k, 2] - query_boxes[k, 0] + 1) * (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        for n in range(n_):
            # 计算他们共有的宽度
            iw = min(boxes[n, 2], query_boxes[k, 2]) - max(boxes[n, 0], query_boxes[k, 0]) + 1
            if iw > 0:
                # 计算他们共有的高度
                ih = min(boxes[n, 3], query_boxes[k, 3]) - max(boxes[n, 1], query_boxes[k, 1]) + 1
                if ih > 0:
                    # 每个 box的面积
                    box_area = (boxes[n, 2] - boxes[n, 0] + 1) * (boxes[n, 3] - boxes[n, 1] + 1)
                    # 一个box 和query_box如果重叠的话,他们加起来的面积，注意这里要减去一个他们重合的面积(iw * ih)
                    all_area = float(box_area + query_box_area - iw * ih)
                    # 重叠区域的计算为重叠区域的面积 除以他们加起来的面积
                    overlaps[n, k] = iw * ih / all_area
    return overlaps


def compute_bbox_regression_targets(rois, overlaps, labels):
    """
    given rois, overlaps, gt labels, compute bounding box regression targets
    :param rois: roidb[i]['boxes'] k * 4
    :param overlaps: roidb[i]['max_overlaps'] k * 1
    :param labels: roidb[i]['max_classes'] k * 1
    :return: targets[i][class, dx, dy, dw, dh] k * 5
    """
    # Ensure ROIs are floats
    rois = rois.astype(np.float, copy=False)

    # Indices of ground-truth ROIs
    # 得到overlap=1的行号
    gt_inds = np.where(overlaps == 1)[0]
    if len(gt_inds) == 0:
        print 'something wrong : zero ground truth rois'
    # Indices of examples for which we try to make predictions
    # overlap > 固定值的行号
    ex_inds = np.where(overlaps >= config.TRAIN.BBOX_REGRESSION_THRESH)[0]

    # Get IoU overlap between each ex ROI and gt ROI
    ex_gt_overlaps = bbox_overlaps(rois[ex_inds, :], rois[gt_inds, :])

    # Find which gt ROI each ex ROI has max overlap with:
    # this will be the ex ROI's gt target
    # ex_gt_overlaps 是一个二维数组, 其中列数是gt_inds的长度, 注意列的索引和gt_inds的索引是对应的
    # gt_inds[gt_assignment] 其实就可以取出行号(overlaps)了
    gt_assignment = ex_gt_overlaps.argmax(axis=1)

    gt_rois = rois[gt_inds[gt_assignment], :]
    ex_rois = rois[ex_inds, :]

    targets = np.zeros((rois.shape[0], 5), dtype=np.float32)
    targets[ex_inds, 0] = labels[ex_inds]
    targets[ex_inds, 1:] = bbox_transform(ex_rois, gt_rois)
    return targets


def expand_bbox_regression_targets(bbox_targets_data, num_classes):
    """
    expand from 5 to 4 * num_classes; only the right class has non-zero bbox regression targets
    :param bbox_targets_data: [k * 5]
    :param num_classes: number of classes
    :return: bbox target processed [k * 4 num_classes]
    bbox_inside_weights ! only foreground boxes have bbox regression computation!
    """
    classes = bbox_targets_data[:, 0]
    bbox_targets = np.zeros((classes.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    indexes = np.where(classes > 0)[0]
    for index in indexes:
        cls = classes[index]
        start = int(4 * cls)
        end = start + 4
        bbox_targets[index, start:end] = bbox_targets_data[index, 1:]
        bbox_inside_weights[index, start:end] = config.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights

