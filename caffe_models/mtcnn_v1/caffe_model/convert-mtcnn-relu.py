#!/usr/bin/env python

import os
#os.environ['GLOG_minloglevel'] = '2'

import numpy as np
import caffe

netPatched = caffe.Net('mtcnn-relu.prototxt', caffe.TEST)

def convertDet1():
  net = caffe.Net('det1.prototxt', 'det1.caffemodel', caffe.TEST)
  
  for layer in ['conv1', 'conv2', 'conv3', 'conv4-1', 'conv4-2']:
    netPatched.params['pnet-'+layer][0].data[...] = net.params[layer][0].data
    netPatched.params['pnet-'+layer][1].data[...] = net.params[layer][1].data

def convertDet2():
  net = caffe.Net('det2.prototxt', 'det2.caffemodel', caffe.TEST)
  
  for layer in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5-1', 'conv5-2']:
    netPatched.params['rnet-'+layer][0].data[...] = net.params[layer][0].data
    netPatched.params['rnet-'+layer][1].data[...] = net.params[layer][1].data


def convertDet3():
  net = caffe.Net('det3.prototxt', 'det3.caffemodel', caffe.TEST)
  
  for layer in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6-1', 'conv6-2', 'conv6-3']:
    netPatched.params['onet-'+layer][0].data[...] = net.params[layer][0].data
    netPatched.params['onet-'+layer][1].data[...] = net.params[layer][1].data

convertDet1()
convertDet2()
convertDet3()

netPatched.save('mtcnn-relu.caffemodel')
