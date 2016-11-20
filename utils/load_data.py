#-*-coding:utf-8-*-
from helper.dataset.pascal_voc import PascalVOC
from helper.processing.roidb import prepare_roidb, add_bbox_regression_targets


def load_ss_roidb(image_set, year, root_path, devkit_path, flip=False):
    voc = PascalVOC(image_set, year, root_path, devkit_path)
    gt_roidb = voc.gt_roidb()
    ss_roidb = voc.selective_search_roidb(gt_roidb)
    if flip:
        ss_roidb = voc.append_flipped_images(ss_roidb)
    """这里ss_roidb是一个list，其中每一个元素是一个字典，在执行prepare_roidb之前每个字典包含四个元素
    1:boxes        (存储的是一个图片上的所有物体的位置); 
    2:gt_classes   (所有这些物体所对的类别，注意：新合并进来的box没有赋值类别) 
    3:gt_overlaps  :是一个二维数组，其中行数是物体的个数，列数是一共要判定的类别的个数，这里是21
    4:flipped      标示是否进行了翻转
    在执行完prepare_roidb之后，每个元素的字典中又多了5项
    5:image        (该图片的存储位置)
    6:height       (该图片的高度)
    7:width        (该图片的宽度)
    8:max_overlaps 每个物体的最大覆盖度(覆盖第一次载入的标记物体数据)
    9:max_classes  该物体所覆盖的那个标记物体的类别， 即对应的8中最大覆盖值所覆盖的那个标记物体
    在执行完add_bbox_regression_targets之后，又增加了一列
    10:bbox_target 对于所有标记物体和未标记物体计算他们的位置,一个元素包含5个元素，第一个是物体标记，后面4个是物体的位置标记 
    """
    prepare_roidb(voc, ss_roidb)
    means, stds = add_bbox_regression_targets(ss_roidb)
    return voc, ss_roidb, means, stds


def load_gt_roidb(image_set, year, root_path, devkit_path, flip=False):
    voc = PascalVOC(image_set, year, root_path, devkit_path)
    gt_roidb = voc.gt_roidb()
    if flip:
        gt_roidb = voc.append_flipped_images(gt_roidb)
    prepare_roidb(voc, gt_roidb)
    return voc, gt_roidb


def load_rpn_roidb(image_set, year, root_path, devkit_path, flip=False):
    voc = PascalVOC(image_set, year, root_path, devkit_path)
    gt_roidb = voc.gt_roidb()
    rpn_roidb = voc.rpn_roidb(gt_roidb)
    if flip:
        rpn_roidb = voc.append_flipped_images(rpn_roidb)
    prepare_roidb(voc, rpn_roidb)
    means, stds = add_bbox_regression_targets(rpn_roidb)
    return voc, rpn_roidb, means, stds


def load_test_ss_roidb(image_set, year, root_path, devkit_path):
    voc = PascalVOC(image_set, year, root_path, devkit_path)
    gt_roidb = voc.gt_roidb()
    ss_roidb = voc.selective_search_roidb(gt_roidb)
    prepare_roidb(voc, ss_roidb)
    return voc, ss_roidb


def load_test_rpn_roidb(image_set, year, root_path, devkit_path):
    voc = PascalVOC(image_set, year, root_path, devkit_path)
    gt_roidb = voc.gt_roidb()
    rpn_roidb = voc.rpn_roidb(gt_roidb)
    prepare_roidb(voc, rpn_roidb)
    return voc, rpn_roidb
