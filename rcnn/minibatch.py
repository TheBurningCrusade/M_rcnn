#-*-coding:utf-8-*-
"""
To construct data iterator from imdb, batch sampling procedure are defined here
RPN:
data =
    {'data': [num_images, c, h, w],
    'im_info': [num_images, 4] (optional)}
label =
prototype: {'gt_boxes': [num_boxes, 5]}
final:  {'label': [batch_size, 1] <- [batch_size, num_anchors, feat_height, feat_width],
         'bbox_target': [batch_size, num_anchors, feat_height, feat_width],
         'bbox_inside_weight': [batch_size, num_anchors, feat_height, feat_width],
         'bbox_outside_weight': [batch_size, num_anchors, feat_height, feat_width]}
Fast R-CNN:
data =
    {'data': [num_images, c, h, w],
    'rois': [num_images, num_rois, 5]}
label =
    {'label': [num_images, num_rois],
    'bbox_target': [num_images, num_rois, 4 * num_classes],
    'bbox_inside_weight': [num_images, num_rois, 4 * num_classes],
    'bbox_outside_weight': [num_images, num_rois, 4 * num_classes]}
"""

import cv2
import numpy as np
import numpy.random as npr

from helper.processing import image_processing
from helper.processing.bbox_regression import expand_bbox_regression_targets
from helper.processing.generate_anchor import generate_anchors
from helper.processing.bbox_regression import bbox_overlaps
from helper.processing.bbox_transform import bbox_transform
from rcnn.config import config


def get_minibatch(roidb, num_classes, mode='test'):
    """
    return minibatch of images in roidb
    :param roidb: a list of dict, whose length controls batch size
    :param num_classes: number of classes is used in bbox regression targets
    :param mode: controls whether blank label are returned
    :return: data, label
    """
    # build im_array: [num_images, c, h, w]
    num_images = len(roidb)
    
    print "len(config.SCALES): %s" % (len(config.SCALES)) #config.SCALES=(600, )
    # print len(config.SCALES)
    # 得到一个从0到high，但不包括high的数据，大小为size， size也可以是一个多维的，比如
    # (4,5)这样就可以得到一个(4X5）的矩阵
    # 在这里的主要目的是为每一个图像获得一个随机缩放的比例
    random_scale_indexes = npr.randint(0, high=len(config.SCALES), size=num_images)
    print "random_scale_indexes"
    print random_scale_indexes
    # 得到关于一批图片的的数据（im_array）和scale比例
    im_array, im_scales = get_image_array(roidb, config.SCALES, random_scale_indexes)

    if mode == 'train':
        cfg_key = 'TRAIN'
    else:
        cfg_key = 'TEST'

    if config[cfg_key].HAS_RPN:
        print "has rpn token"
        assert len(roidb) == 1, 'Single batch only'
        assert len(im_scales) == 1, 'Single batch only'

        print "len(roidb): %s" % (str(len(roidb)))
        print "len(im_scales): %s" % (str(len(im_scales)))
        # im_info 中存储图片的高度，宽度，图片的缩放比例
        im_info = np.array([[im_array.shape[2], im_array.shape[3], im_scales[0]]], dtype=np.float32)

        data = {'data': im_array,
                'im_info': im_info}
        label = {}

        if mode == 'train':
            # gt boxes: (x1, y1, x2, y2, cls)
            gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
            gt_boxes = np.empty((roidb[0]['boxes'].shape[0], 5), dtype=np.float32)
            """这里gt_inds中加入的判断其实应该是不会过滤到任何一个box的，因为如果
            过滤掉了一个box，那么后续的对gt_boxes的赋值就会出错，因为gt_boxes的shape
            和roidb中boxes的shape是一样大小的，而下面的赋值中却使用了gt_inds中的下标
            如果gt_inds中有过滤，那么下面的赋值机会失败
            """
            # print "gt_inds shape: %s" % (str(len(gt_inds)))
            # print "boxes shape: %s" % (str(roidb[0]['boxes'].shape))
            gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
            gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
            label = {'gt_boxes': gt_boxes}
    else:
        if mode == 'train':
            assert config.TRAIN.BATCH_SIZE % config.TRAIN.BATCH_IMAGES == 0, \
                'BATCHIMAGES {} must devide BATCHSIZE {}'.format(config.TRAIN.BATCH_IMAGES, config.TRAIN.BATCH_SIZE)
            rois_per_image = config.TRAIN.BATCH_SIZE / config.TRAIN.BATCH_IMAGES # 128 / 2
            # 64 × 0.25  = 16
            fg_rois_per_image = np.round(config.TRAIN.FG_FRACTION * rois_per_image).astype(int)

            rois_array = list()
            labels_array = list()
            bbox_targets_array = list()
            bbox_inside_array = list()

            for im_i in range(num_images):
                # 这里是从一副图像中筛选制定个数和配比的rois, 其中这幅图像要的得到的roi=rois_per_image=128(配置的)
                # 其中foreground的个数=fg_rois_per_iamge; im_rois还是一个有4列元素的二维array，label是代表每个
                # rois的类别， bbox_targets是一个有4Xn_class(21)列的二维数组
                im_rois, labels, bbox_targets, bbox_inside_weights, overlaps = \
                    sample_rois(roidb[im_i], fg_rois_per_image, rois_per_image, num_classes)
                    # (*, 16, 128, 21)

                # project im_rois
                # do not round roi
                rois = im_rois * im_scales[im_i]
                batch_index = im_i * np.ones((rois.shape[0], 1))
                rois_array_this_image = np.hstack((batch_index, rois))
                rois_array.append(rois_array_this_image)

                # add labels
                labels_array.append(labels)
                bbox_targets_array.append(bbox_targets)
                bbox_inside_array.append(bbox_inside_weights)

            rois_array = np.array(rois_array)
            labels_array = np.array(labels_array)
            bbox_targets_array = np.array(bbox_targets_array)
            bbox_inside_array = np.array(bbox_inside_array)
            bbox_outside_array = np.array(bbox_inside_array > 0).astype(np.float32)
            #print "bbox_inside_array: %s" % (bbox_inside_array)
            #print "bbox_outside_array: %s" % (bbox_outside_array)
            if np.array_equal(bbox_inside_array,bbox_outside_array):
                print "init bbox_inside_array equal bbox_outside_array"
            # 这里的rois是包含5列的一个二维数组，其中第一列用来表示图片，比如如果他是第一个
            # 图片的rois那么就是0，第二个就是1
            """
            这里每张图片的rois_array, bbox_targets_array, bbox_inside_array, bbox_outside_array 
            都是一个二维数组，其中它们的行数代表的就是一张图片的rois的个数，把上述这些数据都分别
            存到各自的list当中，其中list的长度即代表图片的个数。然后用np.array将list转化成array，
            注意这里转化成array之后，数据会增加一维，其中的个数转化后数组的第一维，相当于list的
            长度, labels_array是一个一维数组，转化后成为2维数据，第一维也是代表图片个数
            """
            data = {'data': im_array,
                    'rois': rois_array}
            label = {'label': labels_array,
                     'bbox_target': bbox_targets_array,
                     'bbox_inside_weight': bbox_inside_array,
                     'bbox_outside_weight': bbox_outside_array}
        else:
            rois_array = list()
            for im_i in range(num_images):
                im_rois = roidb[im_i]['boxes']
                rois = im_rois * im_scales[im_i]
                batch_index = im_i * np.ones((rois.shape[0], 1))
                rois_array_this_image = np.hstack((batch_index, rois))
                rois_array.append(rois_array_this_image)
            rois_array = np.vstack(rois_array)

            data = {'data': im_array,
                    'rois': rois_array}
            label = {}

    return data, label


def get_image_array(roidb, scales, scale_indexes):
    # 这里scales是一个数组，而scale_indexes是一个和roidb一样大的数组，它的值是scales的索引下标
    # 主要的功能是将一批图片转换成一个4维矩阵，其中第一维是图片的个数，第二维是代表每个像素是被
    # 几个数值描述的，3,4维为图片的像素宽度和长度.
    # 主要的步骤是进行图片的scale，然后统一按各维度的最大值进行补齐(补零)
    """
    build image array from specific roidb
    :param roidb: images to be processed
    :param scales: scale list
    :param scale_indexes: indexes
    :return: array [b, c, h, w], list of scales
    """
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    for i in range(num_images):
        im = cv2.imread(roidb[i]['image'])
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        target_size = scales[scale_indexes[i]]
        # 对图片进行缩放
        im, im_scale = image_processing.resize(im, target_size, config.MAX_SIZE) #config.MAX_SIZE = 1000
        # config.PIXEL_MEANS = np.array([[[123.68, 116.779, 103.939]]])
        # 对像素值去均值，然后增加一维，用于存储图片的个数
        im_tensor = image_processing.transform(im, config.PIXEL_MEANS)

        processed_ims.append(im_tensor)
        im_scales.append(im_scale)

    array = image_processing.tensor_vstack(processed_ims)
    return array, im_scales


def sample_rois(roidb, fg_rois_per_image, rois_per_image, num_classes):
    """
    generate random sample of ROIs comprising foreground and background examples
    :param roidb: database of selected rois
    :param fg_rois_per_image: foreground roi number
    :param rois_per_image: total roi number
    :param num_classes: number of classes
    :return: (labels, rois, bbox_targets, bbox_inside_weights, overlaps)
    """
    # label = class RoI has max overlap with
    labels = roidb['max_classes']
    overlaps = roidb['max_overlaps']
    rois = roidb['boxes']

    #得到制定个数foreground rois
    # foreground RoI with FG_THRESH overlap
    fg_indexes = np.where(overlaps >= config.TRAIN.FG_THRESH)[0] # 0.5
    # guard against the case when an image has fewer than fg_rois_per_image foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_indexes.size)
    # Sample foreground regions without replacement
    if fg_indexes.size > 0:
        fg_indexes = npr.choice(fg_indexes, size=fg_rois_per_this_image, replace=False)

    #得到制定个数的 background rois
    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_indexes = np.where((overlaps < config.TRAIN.BG_THRESH_HI) & (overlaps >= config.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image, bg_indexes.size)
    # Sample foreground regions without replacement
    if bg_indexes.size > 0:
        bg_indexes = npr.choice(bg_indexes, size=bg_rois_per_this_image, replace=False)

    # indexes selected
    # foreground 和background组合
    keep_indexes = np.append(fg_indexes, bg_indexes)

    # pad more to ensure a fixed minibatch size
    if keep_indexes.shape[0] < rois_per_image:
        gap = rois_per_image - keep_indexes.shape[0]
        gap_indexes = npr.choice(range(len(rois)), size=gap, replace=False)
        keep_indexes = np.append(keep_indexes, gap_indexes)

    # select labels
    labels = labels[keep_indexes]
    # set labels of bg_rois to be 0
    labels[fg_rois_per_this_image:] = 0
    overlaps = overlaps[keep_indexes]
    rois = rois[keep_indexes]

    bbox_targets, bbox_inside_weights = \
        expand_bbox_regression_targets(roidb['bbox_targets'][keep_indexes, :], num_classes)

    return rois, labels, bbox_targets, bbox_inside_weights, overlaps


def assign_anchor(feat_shape, gt_boxes, im_info, feat_stride=16,
                  scales=(8, 16, 32), ratios=(0.5, 1, 2), allowed_border=0):
    """
    assign ground truth boxes to anchor positions
    :param feat_shape: infer output shape
    :param gt_boxes: assign ground truth
    :param im_info: filter out anchors overlapped with edges
    :param feat_stride: anchor position step
    :param scales: used to generate anchors, affects num_anchors (per location)
    :param ratios: aspect ratios of generated anchors
    :param allowed_border: filter out anchors with edge overlap > allowed_border
    :return: dict of label
    'label': of shape (batch_size, 1) <- (batch_size, num_anchors, feat_height, feat_width)
    'bbox_target': of shape (batch_size, num_anchors * 4, feat_height, feat_width)
    'bbox_inside_weight': *todo* mark the assigned anchors
    'bbox_outside_weight': used to normalize the bbox_loss, all weights sums to RPN_POSITIVE_WEIGHT
    """
    def _unmap(data, count, inds, fill=0):
        """" unmap a subset inds of data into original data of size count """
        if len(data.shape) == 1:
            ret = np.empty((count,), dtype=np.float32)
            ret.fill(fill)
            ret[inds] = data
        else:
            ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
            ret.fill(fill)
            ret[inds, :] = data
        return ret

    def _compute_targets(ex_rois, gt_rois):
        """ compute bbox targets for an image """
        assert ex_rois.shape[0] == gt_rois.shape[0]
        assert ex_rois.shape[1] == 4
        assert gt_rois.shape[1] == 5

        return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)

    DEBUG = False
    # print "In assign_anchor im_info: %s" % (str(im_info))
    im_info = im_info[0]
    scales = np.array(scales, dtype=np.float32)
    base_anchors = generate_anchors(base_size=16, ratios=list(ratios), scales=scales)
    num_anchors = base_anchors.shape[0]
    feat_height, feat_width = feat_shape[-2:]

    if DEBUG:
        print 'anchors:'
        print base_anchors
        print 'anchor shapes:'
        print np.hstack((base_anchors[:, 2::4] - base_anchors[:, 0::4],
                         base_anchors[:, 3::4] - base_anchors[:, 1::4]))
        print 'im_info', im_info
        print 'height', feat_height, 'width', feat_width
        print 'gt_boxes shape', gt_boxes.shape
        print 'gt_boxes', gt_boxes

    # 1. generate proposals from bbox deltas and shifted anchors
    # 对卷积后的输出图片的每一个点进行遍历,只是对每一步设置了一个跨度
    # 这个跨度应该也不是随便设置的，应该是最初图片到最终卷积出图片的
    # 一个缩小程度，即缩小了16倍，这样feat_stride才设置成了16
    shift_x = np.arange(0, feat_width) * feat_stride
    shift_y = np.arange(0, feat_height) * feat_stride
    """"
    print "shift_x: %s" % (shift_x)
    print "shift_y: %s" % (shift_y)
    print "shift_x shape: %s" % (str(shift_x.shape))
    print "shift_y shape: %s" % (str(shift_y.shape))"""

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    """"
    print "After shift_x: %s" % (shift_x)
    print "After shift_y: %s" % (shift_y)
    print "After shift_x shape: %s" % (str(shift_x.shape))
    print "After shift_y shape: %s" % (str(shift_y.shape))
    print "shift_x ravel: %s" % (str(shift_x.ravel()))
    print "shift_y ravel: %s" % (str(shift_y.ravel()))"""
    """
    假设一个矩阵有３行５列，那么np.meshgrid就会生成连个３行５列的矩阵
             [[0,1,2,3,4]    　         [[0,0,0,0,0]
    shif_x = [0,1,2,3,4]   和 shift_y = [1,1,1,1,1]  在这里如果使用vstack和transpose()
             [0,1,2,3,4]]               [2,2,2,2,2]] 
    就可以按照行优先的顺序遍历整个原始的３行5列的矩阵，即生成他们的遍历他们的下标.
    np.vstack((shift_x.ravel()),(shift_y.ravel())).tranpos()就会生成一个二维矩阵
    [[0,0],[0,1], [0,2], [0,3], [0,4], [1,0], [1,1], [1,2], .............]
    如果按照行来遍历的话就是一个按行遍历原始矩阵的下标矩阵
    """

    """
    a = np.array([[[1,2,3,4],[-1,-2,-3,-4],[5,6,7,8]]])
    a.shape #(1, 3, 4)
    b = np.array([[[6,7,8,9]],[[10,11,12,13]],[[14,15,16,17]],[[18,19,20,21]]])
    b.shape #(4, 1, 4)
    a+b
    array([[[ 7,  9, 11, 13],[ 5,  5,  5,  5], [11, 13, 15, 17]],
           [[11, 13, 15, 17],[ 9,  9,  9,  9], [15, 17, 19, 21]],
           [[15, 17, 19, 21],[13, 13, 13, 13], [19, 21, 23, 25]],
           [[19, 21, 23, 25],[17, 17, 17, 17], [23, 25, 27, 29]]])
    """

    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
    print "shifts: %s" % (str(shifts))

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    """
    anchors中每一个元素有４个数字组成,比如(-176,-88,191,103)，代表的意思是
    如果个一个坐标点(x1,y1,x2,y2)，其中(x1,y1)代表左上点的坐标，(x2,y2)代表
    右下点的坐标，那么anchor代表这些对应坐标的移动值，(x1-176,y1-88,x2+191,y2+103)
    这里本例中x1=x2且y1=y2就表示是对一个点进行移动，分别移动到左上和右下构成一个矩形
    """
    A = num_anchors
    K = shifts.shape[0]

    # 对每一个shift点都让他和所有anchor进行位移运算
    all_anchors = base_anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    print "base_anchors: %s" % (str(base_anchors.reshape((1,A,4))))
    print "base_anchors shape: %s" % (str(base_anchors.reshape((1,A,4)).shape))
    print "shifts: %s" % (str(shifts.reshape((1, K, 4)).transpose((1, 0, 2))))
    print "shifts shape: %s" % (str(shifts.reshape((1, K, 4)).transpose((1, 0, 2)).shape))
    print "all_anchors: %s" % (str(all_anchors))
    print "all_anchors shape: %s" % (str(all_anchors.shape))
    all_anchors = all_anchors.reshape((K * A, 4))
    total_anchors = int(K * A)

    # only keep anchors inside the image 如英文解释
    inds_inside = np.where((all_anchors[:, 0] >= -allowed_border) &
                           (all_anchors[:, 1] >= -allowed_border) &
                           (all_anchors[:, 2] < im_info[1] + allowed_border) &
                           (all_anchors[:, 3] < im_info[0] + allowed_border))[0]
    if DEBUG:
        print 'total_anchors', total_anchors
        print 'inds_inside', len(inds_inside)

    # keep only inside anchors
    anchors = all_anchors[inds_inside, :]
    if DEBUG:
        print 'anchors shape', anchors.shape

    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)

    if gt_boxes.size > 0:
        # overlap between the anchors and the gt boxes
        # overlaps (ex, gt)
        overlaps = bbox_overlaps(anchors.astype(np.float), gt_boxes.astype(np.float))
        argmax_overlaps = overlaps.argmax(axis=1)
        # 一个anchor和一幅图上的标记的哪一个物体具有最大的覆盖度
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
        print "max_overlaps: %s" % (str(max_overlaps))
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        # 按列进行排序的行号,即第一列中最大的行号，第二列中最大的行号，.....
        # 一幅图上标记的每一个物体和哪一anchor的覆盖面积是最大的
        gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
        print "gt_max_overlaps: %s" % (str(gt_max_overlaps))
        """np.where的一些注意，比如如下例子
        a = np.array([[7, 2, 3],[4, 5, 6]])
        b = np.array([4, 2, 3])
        a == b
        array([[False,  True,  True], [ True, False, False]], dtype=bool)
        np.where(a==b)
        (array([0, 0, 1]), array([1, 2, 0]))
        如上np.where的结果就是a==b中等于True的下标的值，但是这里要注意下标的
        顺序并不是查找b中元素的顺序，而是a==b结果举矩阵中先遍历第一行在遍历
        第二行的顺序得到的，例如第一个[0,1]就是指a==b结果中第一行第二列的下标
        而不是指b中4的下标[1,0]虽然4是b中的第一个元素
        """
        # 通过上述讲解，可知这里对gt_argmax_overlaps进行从新按行排序，之前是按列排序
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

        if not config.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels first so that positive labels can clobber them
            # labels中的判断会自动转化为下标
            labels[max_overlaps < config.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        # fg label: for each gt, anchor with highest overlap
        # 将一副图中标记的n个物体中，和他们覆盖度最大的anchor标记成正样本
        labels[gt_argmax_overlaps] = 1

        # fg label: above threshold IoU
        labels[max_overlaps >= config.TRAIN.RPN_POSITIVE_OVERLAP] = 1

        if config.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels last so that negative labels can clobber positives
            labels[max_overlaps < config.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
    else:
        labels[:] = 0

    # subsample positive labels if we have too many
    num_fg = int(config.TRAIN.RPN_FG_FRACTION * config.TRAIN.RPN_BATCH_SIZE)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        if DEBUG:
            disable_inds = fg_inds[:(len(fg_inds) - num_fg)]
        labels[disable_inds] = -1

    # subsample negative labels if we have too many
    num_bg = config.TRAIN.RPN_BATCH_SIZE - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        if DEBUG:
            disable_inds = bg_inds[:(len(bg_inds) - num_bg)]
        labels[disable_inds] = -1

    bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
    if gt_boxes.size > 0:
        bbox_targets[:] = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])

    bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_inside_weights[labels == 1, :] = np.array(config.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)

    bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    if config.TRAIN.RPN_POSITIVE_WEIGHT < 0:
        # uniform weighting of exampling (given non-uniform sampling)
        num_examples = np.sum(labels >= 0)
        positive_weights = np.ones((1, 4)) * 1.0 / num_examples
        negative_weights = np.ones((1, 4)) * 1.0 / num_examples
    else:
        assert ((config.TRAIN.RPN_POSTIVE_WEIGHT > 0) & (config.TRAIN.RPN_POSTIVE_WEIGHT < 1))
        positive_weights = config.TRAIN.RPN_POSTIVE_WEIGHT / np.sum(labels == 1)
        negative_weights = (1.0 - config.TRAIN.RPN_POSTIVE_WEIGHT) / np.sum(labels == 1)
    bbox_outside_weights[labels == 1, :] = positive_weights
    bbox_outside_weights[labels == 0, :] = negative_weights

    if DEBUG:
        _sums = bbox_targets[labels == 1, :].sum(axis=0)
        _squared_sums = (bbox_targets[labels == 1, :] ** 2).sum(axis=0)
        _counts = config.EPS + np.sum(labels == 1)
        means = _sums / _counts
        stds = np.sqrt(_squared_sums / _counts - means ** 2)
        print 'means', means
        print 'stdevs', stds

    # map up to original set of anchors
    # print "labels: %s" % (str(labels))
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
    bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

    if DEBUG:
        print 'rpn: max max_overlaps', np.max(max_overlaps)
        print 'rpn: num_positives', np.sum(labels == 1)
        print 'rpn: num_negatives', np.sum(labels == 0)
        _fg_sum = np.sum(labels == 1)
        _bg_sum = np.sum(labels == 0)
        _count = 1
        print 'rpn: num_positive avg', _fg_sum / _count
        print 'rpn: num_negative avg', _bg_sum / _count

    labels = labels.reshape((1, feat_height, feat_width, A)).transpose(0, 3, 1, 2)
    labels = labels.reshape((1, A * feat_height * feat_width))
    bbox_targets = bbox_targets.reshape((1, feat_height, feat_width, A * 4)).transpose(0, 3, 1, 2)
    bbox_inside_weights = bbox_inside_weights.reshape((1, feat_height, feat_width, A * 4)).transpose((0, 3, 1, 2))
    bbox_outside_weights = bbox_outside_weights.reshape((1, feat_height, feat_width, A * 4)).transpose((0, 3, 1, 2))

    label = {'label': labels,
             'bbox_target': bbox_targets,
             'bbox_inside_weight': bbox_inside_weights,
             'bbox_outside_weight': bbox_outside_weights}
    return label
