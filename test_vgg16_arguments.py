#-*-coding:utf-8-*-
#!/usr/bin/env python


import mxnet as mx

#data = mx.symbol.Variable('data', shape=(2L, 3L, 600L, 904L))

cao= mx.symbol.Variable(name="dd", shape=(2, 4, 3, 4))
aa = mx.symbol.Flatten(data=cao, name="flatten")
print "test Flatten: %s" % (aa.infer_shape(),)

data = mx.symbol.Variable(name="data", shape=(2L, 3L, 600L, 901L))
rois = mx.symbol.Variable(name='rois', shape=(2L, 64L, 5L))
label = mx.symbol.Variable(name='label',shape=(2L, 64L))
bbox_target = mx.symbol.Variable(name='bbox_target',shape=(2L, 64L, 84L))
bbox_inside_weight = mx.symbol.Variable(name='bbox_inside_weight', shape=(2L, 64L, 84L))
bbox_outside_weight = mx.symbol.Variable(name='bbox_outside_weight',shape=(2L, 64L, 84L))
num_classes=21


rois = mx.symbol.Reshape(data=rois, shape=(-1, 5), name='rois_reshape')
label = mx.symbol.Reshape(data=label, shape=(-1, ), name='label_reshape')
bbox_target = mx.symbol.Reshape(data=bbox_target, shape=(-1, 4 * num_classes), name='bbox_target_reshape')
bbox_inside_weight = mx.symbol.Reshape(data=bbox_inside_weight, shape=(-1, 4 * num_classes), name='bbox_inside_weigh    t_reshape')
bbox_outside_weight = mx.symbol.Reshape(data=bbox_outside_weight, shape=(-1, 4 * num_classes), name='bbox_outside_we    ight_reshape')


conv1_1 = mx.symbol.Convolution(data=data, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_1")
print "conv1_1: %s" % (conv1_1.infer_shape(),)
#arg_shapes, out_shapes, aux_shapes =conv1_1.infer_shape()
relu1_1 = mx.symbol.Activation(data=conv1_1, act_type="relu", name="relu1_1")
#relu1_1.list_arguments()
conv1_2 = mx.symbol.Convolution(data=relu1_1, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_2")
conv1_2.infer_shape()
relu1_2 = mx.symbol.Activation(data=conv1_2, act_type="relu", name="relu1_2")
pool1 = mx.symbol.Pooling(data=relu1_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool1")
conv2_1 = mx.symbol.Convolution(data=pool1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_1")
relu2_1 = mx.symbol.Activation(data=conv2_1, act_type="relu", name="relu2_1")
conv2_2 = mx.symbol.Convolution(data=relu2_1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_2")
relu2_2 = mx.symbol.Activation(data=conv2_2, act_type="relu", name="relu2_2")
#print relu2_2.infer_shape()
pool2 = mx.symbol.Pooling(data=relu2_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool2")
#print pool2.infer_shape()
# group 3
conv3_1 = mx.symbol.Convolution(data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_1")
relu3_1 = mx.symbol.Activation(data=conv3_1, act_type="relu", name="relu3_1")
conv3_2 = mx.symbol.Convolution(data=relu3_1, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_2")
relu3_2 = mx.symbol.Activation(data=conv3_2, act_type="relu", name="relu3_2")
conv3_3 = mx.symbol.Convolution(data=relu3_2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_3")
relu3_3 = mx.symbol.Activation(data=conv3_3, act_type="relu", name="relu3_3")
pool3 = mx.symbol.Pooling(data=relu3_3, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool3")
print pool3.infer_shape()
# group 4
conv4_1 = mx.symbol.Convolution(data=pool3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_1")
relu4_1 = mx.symbol.Activation(data=conv4_1, act_type="relu", name="relu4_1")
conv4_2 = mx.symbol.Convolution(data=relu4_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_2")
relu4_2 = mx.symbol.Activation(data=conv4_2, act_type="relu", name="relu4_2")
conv4_3 = mx.symbol.Convolution(data=relu4_2, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_3")
relu4_3 = mx.symbol.Activation(data=conv4_3, act_type="relu", name="relu4_3")
pool4 = mx.symbol.Pooling(data=relu4_3, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool4")
print pool4.infer_shape()
# group 5
conv5_1 = mx.symbol.Convolution(data=pool4, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_1")
relu5_1 = mx.symbol.Activation(data=conv5_1, act_type="relu", name="relu5_1")
conv5_2 = mx.symbol.Convolution(data=relu5_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_2")
relu5_2 = mx.symbol.Activation(data=conv5_2, act_type="relu", name="relu5_2")
conv5_3 = mx.symbol.Convolution(data=relu5_2, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_3")
relu5_3 = mx.symbol.Activation(data=conv5_3, act_type="relu", name="relu5_3")
print relu5_3.infer_shape()

pool5 = mx.symbol.ROIPooling(name='roi_pool5', data=relu5_3, rois=rois, pooled_size=(7, 7), spatial_scale=0.0625)
print "pool5 111111111111111111111"
print pool5.infer_shape()

flatten = mx.symbol.Flatten(data=pool5, name="flatten")
print "flatten 2222222222222"
print flatten.infer_shape()

fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096, name="fc6")
print "flatten 333333333333"
print fc6.infer_shape()

relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6")
print "relu6 4444444444444444"
print relu6.infer_shape()

drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
print "drop6 ××××××××××××××"
print drop6.infer_shape()

fc7 = mx.symbol.FullyConnected(data=drop6, num_hidden=4096, name="fc7")
print "fc7 5555555555555555555"
print fc7.infer_shape()

relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7")
print "relu7"
print relu7.infer_shape()



