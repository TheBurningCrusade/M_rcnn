#-*-coding:utf-8-*-
"""
Generate base anchors on index 0
"""

import numpy as np


def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2 ** np.arange(3, 6)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """

    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in xrange(ratio_anchors.shape[0])])
    print "generate anchors: %s" % (str(anchors))
    return anchors

# 讲左上，右下的坐标转化成宽，高，中心的坐标的数据
def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

# 将有宽，高，中心点坐标的数据转换成，左上右下的坐标
# 和_whctrs有相反的功能
def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    """[12,16,22] 变成
    [[ 12.] [ 16.] [ 22.]]"""
    #print "add newaxis ws: %s" % (str(ws))
    #print "add newaxis hs: %s" % (str(hs))
    
    #左上点的坐标和右下点的坐标
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    #print "anchors in _mkanchors: %s" % (str(anchors))
    return anchors


def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """
    print "Before _ratio_enum anchor: %s" % (str(anchor))
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h # 宽和高相乘
    # ratios是数组，所以在这里anchors变成了多个
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios)) # 相当于size 除以ratios 在开方
    hs = np.round(ws * ratios) # 相当于sqrt(size/ratios) * ratios
    """
    print "size_ratios: %s" % (str(size_ratios))
    print "ws: %s" % (str(ws))
    print "hs: %s" % (str(hs))"""
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    print "After _ratio_enum anchor: %s" % (str(anchors))
    return anchors


def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """
    print "Before _scale_enum anchor: %s" % (str(anchor))
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    print "After _scale_enum anchor: %s" % (str(anchors))
    return anchors
