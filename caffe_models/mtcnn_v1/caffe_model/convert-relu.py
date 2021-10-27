#!/usr/bin/env python

import os
os.environ['GLOG_minloglevel'] = '2'

import numpy as np
import caffe

def convertDet1():
  net = caffe.Net('det1.prototxt', 'det1.caffemodel', caffe.TEST)
  netPatched = caffe.Net('det1-relu.prototxt', caffe.TEST)
  
  for layer in ['conv1', 'conv2', 'conv3', 'conv4-1', 'conv4-2']:
    netPatched.params[layer][0].data[...] = net.params[layer][0].data
    netPatched.params[layer][1].data[...] = net.params[layer][1].data
  
  netPatched.save('det1-relu.caffemodel')

def convertDet2():
  net = caffe.Net('det2.prototxt', 'det2.caffemodel', caffe.TEST)
  netPatched = caffe.Net('det2-relu.prototxt', caffe.TEST)
  
  for layer in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5-1', 'conv5-2']:
    netPatched.params[layer][0].data[...] = net.params[layer][0].data
    netPatched.params[layer][1].data[...] = net.params[layer][1].data
  
  netPatched.save('det2-relu.caffemodel')


def convertDet3():
  net = caffe.Net('det3.prototxt', 'det3.caffemodel', caffe.TEST)
  netPatched = caffe.Net('det3-relu.prototxt', caffe.TEST)
  
  for layer in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6-1', 'conv6-2', 'conv6-3']:
    netPatched.params[layer][0].data[...] = net.params[layer][0].data
    netPatched.params[layer][1].data[...] = net.params[layer][1].data
  
  netPatched.save('det3-relu.caffemodel')

convertDet1()
convertDet2()
convertDet3()

