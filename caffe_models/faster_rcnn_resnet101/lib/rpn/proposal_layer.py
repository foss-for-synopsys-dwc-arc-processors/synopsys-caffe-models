# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import caffe
import numpy as np
import yaml
from fast_rcnn.config import cfg
from generate_anchors import generate_anchors
from fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from fast_rcnn.nms_wrapper import nms

DEBUG = True
DEBUG = False

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    print "IN NMS BASELINE; ndets = ",len(dets)
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    ix = 0
    while order.size > 0:
        i = order[0]
        keep.append(i)
	print ix,"nms: keep ",dets[order[0]]; ix += 1
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

if False:
    print "using py_cpu_nms as nms to avoid getting it from .pyd, which I can't change"
    # Had to put in the code above so I could add print statements for debugging.
    nms = py_cpu_nms

class ProposalLayer(caffe.Layer):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def setup(self, bottom, top):
        # parse the layer parameter string, which must be valid YAML
        # layer_params = yaml.load(self.param_str_)
	# print dir(self)
	# param_str_ is wrong.  In python it's param_str.  C++ uses the extra _.
        layer_params = yaml.load(self.param_str)

        self._feat_stride = layer_params['feat_stride']
        anchor_scales = layer_params.get('scales', (8, 16, 32))
        anchor_ratios = layer_params.get('ratios', ((0.5, 1, 2)))
        self._anchors = generate_anchors(ratios=anchor_ratios, scales=np.array(anchor_scales))
        self._num_anchors = self._anchors.shape[0]

        if DEBUG:
	    print "anchor scales are",anchor_scales
            print 'feat_stride: {}'.format(self._feat_stride)
            print 'anchors:'
            print self._anchors

        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        # top[0].reshape(1, 5)
        top[0].reshape(300, 5)

        # scores blob: holds scores for R regions of interest
        if len(top) > 1:
            top[1].reshape(1, 1, 1, 1)

    def forward(self, bottom, top):
        print "============== forward proposal layer ================="
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)

        assert bottom[0].data.shape[0] == 1, \
            'Only single item batches are supported'

	# On windows this appears to be 0 or 1
	#enum Phase { TRAIN = 0, TEST = 1 };
	# print "phase=",self.phase
        cfg_key = str(self.phase) # either 'TRAIN' or 'TEST'
	cfg_key = ("TRAIN","TEST")[self.phase]
        pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
        nms_thresh    = cfg[cfg_key].RPN_NMS_THRESH
        min_size      = cfg[cfg_key].RPN_MIN_SIZE
	print "post_nms_topN =",post_nms_topN

        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs, which we want
	# Incoming is 1 18 14 14; we take just the last 9 (anchors).  Why?
	# "fg" = foreground?
	if DEBUG:
	    print "FP: incoming scores shape is ",bottom[0].data.shape
	    print "FP: incoming delta  shape is ",bottom[1].data.shape
	    print "FP: num anchors is ",self._num_anchors

        scores = bottom[0].data[:, self._num_anchors:, :, :]
	if DEBUG:
	    print "FP, half scores shape is ",scores.shape

	# This is rpn_bbox_pred
        bbox_deltas = bottom[1].data
	if DEBUG:
	    print "bbox delta shape=",bbox_deltas.shape
	    print "proposal layer: im_info is ",bottom[2].data
        im_info = bottom[2].data[0, :]

        if DEBUG:
            print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
            print 'scale: {}'.format(im_info[2])
	    print "min_size: {}".format(min_size)
	    print "pre_nms_topN", pre_nms_topN
	    print "post_nms_topN" , post_nms_topN
	    print "nms_thresh" ,nms_thresh

        # 1. Generate proposals from bbox deltas and shifted anchors
        height, width = scores.shape[-2:]	# last two elements of the shape

        if DEBUG:
            print 'score map size: {}'.format(scores.shape)

	def debug(msg):
	    if DEBUG:
	        print "debug -- ",msg

	debug(0)
        # Enumerate all shifts
	# If input is 224 x 284 we get 14 x 18.
	# If input is 224 x 224 we get 14 x 14.
	# So, that's the division by feat_stride.
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
	if DEBUG:
	    print "score height, width=",height,width # E.g. 14, 18: pooling from original HxW
	    print "shift_x=",shift_x
	    print "shift_y=",shift_y
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
	# print "wow, meshgrid x!",shift_x
	# print "wow, meshgrid y!",shift_y
	# this sure seems like a waste of time.
	# Construct two 2D arrays (meshgrid) of replicated coordinates
	# and then flatten them.  A whole lot of duplicates!
	# vstack is not particularly useful.  Just call array.
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()
	# shifts = [4][252] then transposed to [252][4]
	# 252 = x * y = width * height.
	if False:
	    print "shifts=",shifts.shape,"\n",shifts

	# I think the idea is that the anchors are bounding boxes [x,y] [x,y]
	# We then replicate them with the feature strides.
	# The secret to understanding this code was to replicate it standalone
	# and make the numbers all smaller so it's easy to see.
        # Enumerate all shifted anchors:
        #
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
	debug(0.5)
        A = self._num_anchors
	# anchors = 9 x 4 = 9 anchors each 4 values: x,y top left x,y bottom right
	# Turn into 1 x 9 x 4
	if DEBUG:
	    print "num anchors = ",A,"shape=",self._anchors.shape,"shift shape=",shifts.shape
        K = shifts.shape[0]
	srt = shifts.reshape((1, K, 4)).transpose((1, 0, 2))
	if DEBUG:
	    print "shifts reshaped and transposed shape = ",srt.shape,"\n",srt
	# The reshape below adds the extra input dimension 1:
	ars = self._anchors.reshape((1, A, 4)) 
	if DEBUG:
	    print "anchors reshaped = ",ars.shape,"\n",ars
	    print "add two arrays:",ars.shape,srt.shape
	# This adds two differently-shaped arrays.
	# E.g. anchors = [1,9,4] and shifts = [196,1,4]
	# The result is [196,9,4].  Broadcasting replicates the 9 and the 196.
	# The replication occurs in C instead of python.
        anchors = ars + srt
        anchors = anchors.reshape((K * A, 4))
	# 2268 x 4 = 224 * 9 x 4
	if False:
	    print "anchors after massive redoing",anchors.shape,"\n",anchors

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        #
        # bbox deltas will be (1, 4 * A, H, W) format
        # transpose to (1, H, W, 4 * A)
        # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
        # in slowest to fastest order
	bbd_tr =  bbox_deltas.transpose((0, 2, 3, 1))
        bbox_deltas = bbd_tr.reshape((-1, 4))
	if DEBUG:
	    print "transposed shape is ",bbd_tr.shape
	    print "tp reshape is ",bbox_deltas.shape

	def show_proposals(where):
	    print "proposals",where
	    ix = 0
	    for Z in proposals:
		print ix, Z
		ix += 1
	def show_scores(where):
	    print "scores",where
	    ix = 0
	    for Z in scores:
		print ix, Z
		ix += 1
	def show_proposal_and_score(where):
	    print "scores & proposals",where
	    for ix in range(len(scores)):
		print ix, scores[ix],proposals[ix]

        # Same story for the scores:
        #
        # scores are (1, A, H, W) format
        # transpose to (1, H, W, A)
        # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
        scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))
	if False:
	    print scores.shape
	    show_scores("transposed & reshaped scores are")

        # Convert anchors into proposals via bbox transformations
        proposals = bbox_transform_inv(anchors, bbox_deltas)
	if DEBUG:
	    print "proposals shape=",proposals.shape,"result:"
	    print proposals

	if False:
	    show_proposals("pre-clipped")

	debug(2)
        # 2. clip predicted boxes to image
	if DEBUG:
	    print "clip with im_info",im_info,im_info[:2]
        proposals = clip_boxes(proposals, im_info[:2])
	if False:
	    show_proposals("clipped")

	debug(3)
        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[2])
        keep = _filter_boxes(proposals, min_size * im_info[2])
        proposals = proposals[keep, :]
        scores = scores[keep]
	if False:
	    show_proposals("post-filter kept")
	    show_scores("post-filter kept")

	debug(4)
        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        order = scores.ravel().argsort()[::-1]
        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]
        proposals = proposals[order, :]
        scores = scores[order]
	if False:
	    show_proposal_and_score("post-sort")

	debug(6)
        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        keep = nms(np.hstack((proposals, scores)), nms_thresh)
        if post_nms_topN > 0:
            keep = keep[:post_nms_topN]
        proposals = proposals[keep, :]
        scores = scores[keep]

	debug("output")
        # Output rois blob
        # Our RPN implementation only supports a single input image, so all
        # batch inds are 0
        batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
        top[0].reshape(*(blob.shape))
        top[0].data[...] = blob
	if DEBUG:
	    print "region proposals are:"
	    print proposals

        # [Optional] output scores blob
        if len(top) > 1:
            top[1].reshape(*(scores.shape))
	    if DEBUG:
		print "outgoing scores shape is",scores.shape
            top[1].data[...] = scores
	debug("return")
	# If I try the below, it says
	# No to_python (by-value) converter found for C++ type: class caffe::Blob<float> * __ptr64
	if False:
	    for i in top:
		print i.data

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep
