#-*-coding:utf-8-*-
import mxnet as mx
import numpy as np
import minibatch
from config import config
from mxnet.executor_manager import _split_input_slice
from helper.processing.image_processing import tensor_vstack


class ROIIter(mx.io.DataIter):
    def __init__(self, roidb, batch_size=2, shuffle=False, mode='train', ctx=None, work_load_list=None):
        """
        This Iter will provide roi data to Fast R-CNN network
        :param roidb: must be preprocessed
        :param batch_size: must divide BATCH_SIZE(128)
        :param shuffle: bool
        :param mode: control returned info
        :param ctx: list of contexts
        :param work_load_list: list of work load
        :return: ROIIter
        """
        super(ROIIter, self).__init__()

        #roidb是个list，每个元素代表一个图片中所需要的全部信息
        self.roidb = roidb
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mode = mode
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = [mx.cpu()]
        self.work_load_list = work_load_list

        self.cur = 0
        self.size = len(roidb) # 1022
        #print "roidb"
        #print len(roidb)
        self.index = np.arange(self.size)
        self.num_classes = self.roidb[0]['gt_overlaps'].shape[1]
        self.reset()

        self.batch = None
        self.data = None
        self.label = None
        self.get_batch() # get_batch的主要目的就是对,ROIIter中的data和label进行赋值
        self.data_name = ['data', 'rois']
        self.label_name = ['label', 'bbox_target', 'bbox_inside_weight', 'bbox_outside_weight']

    @property
    def provide_data(self):
        if self.mode == 'train':
            # 得到data和rois的大小
            return [('data', self.data[0].shape), ('rois', self.data[1].shape)]
        else:
            return [(k, v.shape) for k, v in self.data.items()]

    @property
    def provide_label(self):
        if self.mode == 'train':
            return [('label', self.label[0].shape),
                    ('bbox_target', self.label[1].shape),
                    ('bbox_inside_weight', self.label[2].shape),
                    ('bbox_outside_weight', self.label[3].shape)]
        else:
            return [(k, v.shape) for k, v in self.data.items()]

    def reset(self):
        # 这里只是对index中存储的索引进行了随机化没有改变roidb
        self.cur = 0
        if self.shuffle:
            if config.TRAIN.ASPECT_GROUPING:
                widths = np.array([r['width'] for r in self.roidb])
                heights = np.array([r['height'] for r in self.roidb])
                #print "heigth"
                #print heights.shape
                horz = (widths >= heights)
                vert = np.logical_not(horz)
                horz_inds = np.where(horz)[0] # 所有width>=height的行号,即图片编号
                vert_inds = np.where(vert)[0] # 所有width<heigth的行号
                # 对他们的索引进行随机化处理，并把他们进行列堆叠
                inds = np.hstack((np.random.permutation(horz_inds), np.random.permutation(vert_inds))) 
                #print inds.shape
                inds = np.reshape(inds, (-1, 2)) # 默认有2列的二维数组,inds是2维数组
                #print "reset"
                # 对有2列数据的二维数组的行号进行随机
                row_perm = np.random.permutation(np.arange(inds.shape[0]))
                # 把inds展开成一个一维数组
                inds = np.reshape(inds[row_perm, :], (-1, ))
                #print "sfsd"
                #print inds
                self.index = inds
            else:
                np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def get_batch(self):
        # cur_from 和 cur/cur_to都是以图片为单位的
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        print "cur_form"
        print cur_from
        print "cur_to"
        print cur_to
        # self.index 是一个一维数组，存储的是roidb的行索引号,二cur_from和cur_to存储的是
        # index的索引号
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]
        if self.mode == 'test':
            self.data, self.label = minibatch.get_minibatch(roidb, self.num_classes, self.mode)
        else:
            work_load_list = self.work_load_list
            ctx = self.ctx
            if work_load_list is None:
                print "work_load_list is None"
                work_load_list = [1] * len(ctx)
            assert isinstance(work_load_list, list) and len(work_load_list) == len(ctx), \
                "Invalid settings for work load. "
            slices = _split_input_slice(self.batch_size, work_load_list)

            print "length slices: %s" % (len(slices))

            data_list = []
            label_list = []
            for islice in slices:
                #print "islice: %s" % (islice)

                #print(range(islice.start, islice.stop))   # [0, 1]
                iroidb = [roidb[i] for i in range(islice.start, islice.stop)]
                # 这里将分片之后的iroidb传入get_minibatch里面，iroidb是一个list
                """

                """
                data, label = minibatch.get_minibatch(iroidb, self.num_classes, self.mode)
                data_list.append(data)
                label_list.append(label)

            #print "***********************"
            # data_list是一个包含字典元素的list，其中每个字典有'data' 和 'roi'
            # 两个key，这里是将list整理成为一个字典，整理的规则是将data和roi
            # 进行合并，生成array
            all_data = dict()
            for key in data_list[0].keys():
                all_data[key] = tensor_vstack([batch[key] for batch in data_list])

            all_label = dict()
            for key in label_list[0].keys():
                all_label[key] = tensor_vstack([batch[key] for batch in label_list])

            print "all_data['data'] shape: %s" % (all_data['data'].shape,)
            print "all_data['rois'] shape: %s" % (all_data['rois'].shape,)
            print "all_data['label'] shape: %s" % (all_label['label'].shape,)
            print "all_label['bbox_target'] shape: %s" % (all_label['bbox_target'].shape,)
            print "all_label['bbox_inside_weigth'] shape: %s" % (all_label['bbox_inside_weight'].shape,)
            print "all_label['bbox_outside_weght'] shape: %s" % (all_label['bbox_outside_weight'].shape,)

            self.data = [mx.nd.array(all_data['data']),
                         mx.nd.array(all_data['rois'])]
            self.label = [mx.nd.array(all_label['label']),
                          mx.nd.array(all_label['bbox_target']),
                          mx.nd.array(all_label['bbox_inside_weight']),
                          mx.nd.array(all_label['bbox_outside_weight'])]


class AnchorLoader(mx.io.DataIter):
    def __init__(self, feat_sym, roidb, batch_size=1, shuffle=False, mode='train', ctx=None, work_load_list=None,
                 feat_stride=16, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2), allowed_border=0):
        """
        This Iter will provide roi data to Fast R-CNN network
        :param feat_sym: to infer shape of assign_output
        :param roidb: must be preprocessed
        :param batch_size: must divide BATCH_SIZE(128)
        :param shuffle: bool
        :param mode: control returned info
        :param ctx: list of contexts
        :param work_load_list: list of work load
        :return: AnchorLoader
        """
        super(AnchorLoader, self).__init__()

        self.feat_sym = feat_sym
        # roidb 是一个list，每一个元素代表一张图片，每张图的信息用dict存储
        self.roidb = roidb 
        self.batch_size = batch_size
        print "AnchorLoader's batch size: %d" % (batch_size)
        self.shuffle = shuffle
        self.mode = mode
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = [mx.cpu()]
        self.work_load_list = work_load_list
        self.feat_stride = feat_stride
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.allowed_border = allowed_border

        self.cur = 0
        self.size = len(roidb)
        self.index = np.arange(self.size)
        self.num_classes = self.roidb[0]['gt_overlaps'].shape[1]
        self.reset()

        self.batch = None
        self.data = None
        self.label = None
        """ 这一次将batch的data和label载入进data和label中, get_batch除了
        在初始化的时候被调用一次，其他时候并不会被直接调用，而是由next直接
        掉用，next会修改self.cur的值，然后执行get_batch，但是这里有一个问题
        初始化一个对象时调用完get_batch之后，并没有修改self.cur，所以第一次
        执行next时，和初始化对象调用的get_batch是对同一个batch进行的操作，
        即这里进行了重复的操作
        """
        self.get_batch()
        self.data_name = ['data', 'im_info']
        self.label_name = ['label', 'bbox_target', 'bbox_inside_weight', 'bbox_outside_weight']

    @property
    def provide_data(self):
        if self.mode == 'train':
            return [('data', self.data[0].shape)]
        else:
            return [(k, v.shape) for k, v in self.data.items()]

    @property
    def provide_label(self):
        if self.mode == 'train':
            return [('label', self.label[0].shape),
                    ('bbox_target', self.label[1].shape),
                    ('bbox_inside_weight', self.label[2].shape),
                    ('bbox_outside_weight', self.label[3].shape)]
        else:
            return [(k, v.shape) for k, v in self.data.items()]

    def reset(self):
        self.cur = 0
        if self.shuffle:
            if config.TRAIN.ASPECT_GROUPING:
                widths = np.array([r['width'] for r in self.roidb])
                heights = np.array([r['height'] for r in self.roidb])
                horz = (widths >= heights)    # 宽图
                vert = np.logical_not(horz)
                horz_inds = np.where(horz)[0]
                vert_inds = np.where(vert)[0]
                inds = np.hstack((np.random.permutation(horz_inds), np.random.permutation(vert_inds)))
                inds = np.reshape(inds, (-1, 2))
                row_perm = np.random.permutation(np.arange(inds.shape[0]))
                inds = np.reshape(inds[row_perm, :], (-1, ))
                self.index = inds
            else:
                np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def get_batch(self):
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]
        if self.mode == 'test':
            self.data, self.label = minibatch.get_minibatch(roidb, self.num_classes, self.mode)
        else:
            work_load_list = self.work_load_list
            ctx = self.ctx
            if work_load_list is None:
                work_load_list = [1] * len(ctx)
            assert isinstance(work_load_list, list) and len(work_load_list) == len(ctx), \
                "Invalid settings for work load. "
            # _split_input_slice这个函数返回一个list，其中每个元素的类型为slice
            # slice是一个三元组(start, stop, 未知(None有时取这个值)), 这里应该
            # 是为每个显卡分配一个含有batch_size个数据集, 即如果有3个显卡，那么
            # 将返回一个3个slice组成的list，其中每个slice中包含一个batch的数据索引
            slices = _split_input_slice(self.batch_size, work_load_list)
            print "slices: %s" % (str(slices))
            # print "type slices: %s" % (type(slices))
            print "slices len: %s" % (str(len(slices)))

            data_list = []
            label_list = []
            for islice in slices:
                # 获取一个slice中包含的图片的roidb
                """
                slices: [slice(0, 1, None)]
                slices len: 1
                slicing-----
                slices list: [0]
                slice start: 0
                slice end: 1"""

                """
                print "slicing-----"
                print "slices list: %s" % (str(range(islice.start, islice.stop)))
                print "slice start: %s" % (str(islice.start))
                print "slice end: %s" % (str(islice.stop))"""

                iroidb = [roidb[i] for i in range(islice.start, islice.stop)]
                data, label = minibatch.get_minibatch(iroidb, self.num_classes, self.mode)
                data_list.append(data)
                label_list.append(label)

            # pad data first and then assign anchor (read label)
            data_tensor = tensor_vstack([batch['data'] for batch in data_list])
            #print "data_tensor type: %s" % (type(data_tensor)) # numpy.ndarray
            print "data_tensor shape: %s" % (str(data_tensor.shape))
            data_test = [batch['data'] for batch in data_list]
            print "data_tensor element's shape: %s" % (str(data_test[0].shape))

            # print "data_list len: %s" % (str(len(data_list)))
            # a = np.array([[1,2],[4,5]])
            # for x  in a:
            # ...     print x
            # ... 
            # [1 2]
            # [4 5]
            for data, data_pad in zip(data_list, data_tensor):
                print "data_pad shape %s" % (str(data_pad.shape))
                data['data'] = data_pad[np.newaxis, :]
                print "data[data] shape: %s" % (str(data["data"].shape))

            new_label_list = []
            # 这里的循环的基本单位是一幅图
            for data, label in zip(data_list, label_list):
                # infer label shape
                data_shape = {k: v.shape for k, v in data.items()}
                print "before data shape: %s" % (str(data_shape))
                del data_shape['im_info']
                print "after data shape: %s" % (str(data_shape))
                _, feat_shape, _ = self.feat_sym.infer_shape(**data_shape)
                print "infer output shape: %s" % (str(feat_shape))
                feat_shape = [int(i) for i in feat_shape[0]]
                print "feat_shape: %s" % (str(feat_shape))

                # assign anchor for label
                # print "before im_info: %s" % (str(data["im_info"]))
                label = minibatch.assign_anchor(feat_shape, label['gt_boxes'], data['im_info'],
                                                self.feat_stride, self.anchor_scales,
                                               self.anchor_ratios, self.allowed_border)
                del data['im_info']
                new_label_list.append(label)

            all_data = dict()
            # 对于每次只有一幅图来说，这里的tensor_vstack是不起作用的
            # print "data_list: %s" % (str(data_list))
            for key in ['data']:
                all_data[key] = tensor_vstack([batch[key] for batch in data_list])
            # print "all_data[key]: %s" % (str(all_data[key]))

            all_label = dict()
            # print "len new label list: %s" % (len(new_label_list))
            # print "new label list: %s" % (str(new_label_list[0]['label']))
            all_label['label'] = tensor_vstack([batch['label'] for batch in new_label_list], pad=-1)
            # print "all_label: %s" % (str(all_label["label"]))
            for key in ['bbox_target', 'bbox_inside_weight', 'bbox_outside_weight']:
                all_label[key] = tensor_vstack([batch[key] for batch in new_label_list])

            self.data = [mx.nd.array(all_data['data'])]

            self.label = [mx.nd.array(all_label['label']),
                          mx.nd.array(all_label['bbox_target']),
                          mx.nd.array(all_label['bbox_inside_weight']),
                          mx.nd.array(all_label['bbox_outside_weight'])]
