#!/usr/bin/env python

import os
os.environ['GLOG_minloglevel'] = '2'

import numpy as np
import caffe

def convertDet1():
  net = caffe.Net('det1.prototxt', 'det1.caffemodel', caffe.TEST)
  netPatched = caffe.Net('det1-patched.prototxt', caffe.TEST)
  
  for layer in ['conv1', 'conv2', 'conv3']:
    netPatched.params[layer][0].data[...] = net.params[layer][0].data
    netPatched.params[layer][1].data[...] = net.params[layer][1].data
  
  netPatched.params['conv4'][0].data[...] = \
    np.concatenate((net.params['conv4-1'][0].data, net.params['conv4-2'][0].data), axis=0)
  netPatched.params['conv4'][1].data[...] = \
    np.concatenate((net.params['conv4-1'][1].data, net.params['conv4-2'][1].data), axis=0)
  
  netPatched.save('det1-patched.caffemodel')

def convertDet2():
  net = caffe.Net('det2.prototxt', 'det2.caffemodel', caffe.TEST)
  netPatched = caffe.Net('det2-patched.prototxt', caffe.TEST)
  
  for layer in ['conv1', 'conv2', 'conv3', 'conv4']:
    netPatched.params[layer][0].data[...] = net.params[layer][0].data
    netPatched.params[layer][1].data[...] = net.params[layer][1].data
  
  netPatched.params['conv5'][0].data[...] = \
    np.concatenate((net.params['conv5-1'][0].data, net.params['conv5-2'][0].data), axis=0)
  netPatched.params['conv5'][1].data[...] = \
    np.concatenate((net.params['conv5-1'][1].data, net.params['conv5-2'][1].data), axis=0)
  
  netPatched.save('det2-patched.caffemodel')


def convertDet3():
  net = caffe.Net('det3.prototxt', 'det3.caffemodel', caffe.TEST)
  netPatched = caffe.Net('det3-patched.prototxt', caffe.TEST)
  
  for layer in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']:
    netPatched.params[layer][0].data[...] = net.params[layer][0].data
    netPatched.params[layer][1].data[...] = net.params[layer][1].data
  
  netPatched.params['conv6'][0].data[...] = \
    np.concatenate((net.params['conv6-1'][0].data,
                    net.params['conv6-2'][0].data,
                    net.params['conv6-3'][0].data), axis=0)
  netPatched.params['conv6'][1].data[...] = \
    np.concatenate((net.params['conv6-1'][1].data,
                    net.params['conv6-2'][1].data,
                    net.params['conv6-3'][1].data), axis=0)
  
  netPatched.save('det3-patched.caffemodel')

convertDet1()
convertDet2()
convertDet3()

