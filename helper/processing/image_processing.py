#-*-coding:utf-8-*-
import numpy as np
import cv2


def resize(im, target_size, max_size):
    """
    only resize input image to target size and return scale
    :param im: BGR image input by opencv
    :param target_size: one dimensional size (the short side)
    :param max_size: one dimensional max size (the long side)
    :return:
    """
    im_shape = im.shape
    print "cv2 imread shape: %s" % (str(im_shape))
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    """
    图像的几何变换：
    常见的几何变换有缩放，仿射，透视变换，可以通过如下函数完成对图像的上述变换
    dst = cv2.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]]) 
    dst = cv2.warpAffine(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]]) 
    dst = cv2.warpPerspective(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]])
    首先是缩放变换cv2.resize()
    非关键字参数组有2个：src,dsize，分别是源图像与缩放后图像的尺寸
    关键字参数为dst,fx,fy,interpolation
    dst为缩放后的图像，fx,fy为图像x,y方向的缩放比例，
    interplolation为缩放时的插值方式，有三种插值方式：
    cv2.INTER_AREA    # 使用象素关系重采样。当图像缩小时候，该方法可以避免波纹出现。当图像放大时，类似于 CV_INTER_NN 方法　　　　
    cv2.INTER_CUBIC　　# 立方插值
    cv2.INTER_LINEAR  # 双线形插值　
    cv2.INTER_NN      # 最近邻插值
    """
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    return im, im_scale


def transform(im, pixel_means):
    """
    transform into mxnet tensor
    substract pixel size and transform to correct format
    :param im: [height, width, channel] in BGR
    :param pixel_means: [[[R, G, B pixel means]]]
    :return: [batch, channel, height, width]
    """
    im = im.copy()
    im[:, :, (0, 1, 2)] = im[:, :, (2, 1, 0)]
    im = im.astype(float)
    # config.PIXEL_MEANS = np.array([[[123.68, 116.779, 103.939]]])
    im -= pixel_means
    # 这里表示的是给矩阵在最外边增加一维
    im_tensor = im[np.newaxis, :]
    # put channel first
    """
    a = np.array([[[[11, 23, 43],
             [12, 32, 44],
             [88, 74, 97],
             [91, 33, 55]],

             [[66, 44, 33],
             [23, 45, 67],
             [97, 10, 88],
             [13, 99, 76]]]])
    b = a.transpose((0, 3, 1, 2))
    b = array([[[[11, 12, 88, 91],
             [66, 23, 97, 13]],
             
             [[23, 32, 74, 33],
             [44, 45, 10, 99]],

             [[43, 44, 97, 55],
             [33, 67, 88, 76]]]])
    a.shape = (1, 2, 4, 3)
    b.shape = (1, 3, 2, 4)"""
    channel_swap = (0, 3, 1, 2)
    # 对矩阵进行转置，转置后矩阵维度大小是(0, 3, 1, 2), 这里表示的意思是
    # 前面新加的一维不变，图像的大小排列也不便，只是把像素的rgb三元组都进
    # 行分开，即假设图象有（3,2），那么拆分后会有3个（3,2）
    im_tensor = im_tensor.transpose(channel_swap)
    return im_tensor


def transform_inverse(im_tensor, pixel_means):
    """
    transform from mxnet im_tensor to ordinary RGB image
    im_tensor is limited to one image
    :param im_tensor: [batch, channel, height, width]
    :param pixel_means: [[[R, G, B pixel means]]]
    :return: im [height, width, channel(RGB)]
    """
    assert im_tensor.shape[0] == 1
    im_tensor = im_tensor.copy()
    # put channel back
    channel_swap = (0, 2, 3, 1)
    im_tensor = im_tensor.transpose(channel_swap)
    im = im_tensor[0]
    assert im.shape[2] == 3
    im += pixel_means
    im = im.astype(np.uint8)
    return im


def tensor_vstack(tensor_list, pad=0):
    """
    vertically stack tensors
    :param tensor_list: list of tensor to be stacked vertically
    :param pad: label to pad with
    :return: tensor with max shape
    """
    """该函数主要功能是将tensor_list中的矩阵进行对齐，对齐的规则是使用某一维度下
    各个元素在这个维度上的上的最大值，其他达不到这个最大值的元素要按这个最大维度
    进行补齐, 注意在补齐的过程中，矩阵的维度是不变的,即输入和输出的维度是一样的"""
    # print "tensor_list[0].shape"
    # print tensor_list[0].shape
    ndim = len(tensor_list[0].shape)
    if ndim == 1:
        return np.hstack(tensor_list)
    dimensions = [0]
    # 寻找各个维度下，各元素在这个维度的最大值, 但是对第一维不做处理
    for dim in range(1, ndim):
        dimensions.append(max([tensor.shape[dim] for tensor in tensor_list]))
    # 使用最大维度进行补齐,这是都是单方向的补齐，往下或者往右
    for ind, tensor in enumerate(tensor_list):
        pad_shape = [(0, 0)]
        for dim in range(1, ndim):
            # 只在最后进行补0
            pad_shape.append((0, dimensions[dim] - tensor.shape[dim]))
        # pad_shape是一个还有两个元素的tuple表示开始补几个，结尾补几个
        tensor_list[ind] = np.lib.pad(tensor, pad_shape, 'constant', constant_values=pad)
    # 经过vstack，最前面的一维代表的是图片的个数
    all_tensor = np.vstack(tensor_list)
    return all_tensor
