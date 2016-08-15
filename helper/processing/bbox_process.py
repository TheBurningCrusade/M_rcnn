#-*-coding:utf-8-*-
import numpy as np


def unique_boxes(boxes, scale=1.0):
    """ return indices of unique boxes """
    v = np.array([1, 1e3, 1e6, 1e9])
    #先对boxes中的每一个元素乘以scale,然后四舍五入(round), 由于boxes是一个二维数组,
    #所以对每个行向量使用dot和v进行求内积，得到一个一维的数组，每一个元素都是一个内积值,
    #对hashes这个数据排序，然后去重，unique返回两个变量，一个下划线_和index，其中下划线_表示的是
    #从小到达去重后排列的值，而index是这些值对应的在原来数组中第一次出现的索引
    hashes = np.round(boxes * scale).dot(v)
    _, index = np.unique(hashes, return_index=True)
    return np.sort(index)


def filter_small_boxes(boxes, min_size):
    # 是图片的宽度，应该是右上减去左上
    w = boxes[:, 2] - boxes[:, 0]
    # 是图片的高度，
    h = boxes[:, 3] - boxes[:, 1]
    keep = np.where((w >= min_size) & (h > min_size))[0]
    return keep
