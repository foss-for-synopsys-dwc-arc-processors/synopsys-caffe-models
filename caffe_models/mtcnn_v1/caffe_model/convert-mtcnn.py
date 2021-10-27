#!/usr/bin/env python

import os
os.environ['GLOG_minloglevel'] = '2'

import numpy as np
import caffe
import argparse

parser = argparse.ArgumentParser(description='Conversion options')
parser.add_argument('-relu', action='store_true', help='Use ReLU instead of PReLU')
parser.add_argument('-transp', action='store_true', help='Transpose convolution coefficients')
args = parser.parse_args()

baseName = "mtcnn"
if args.relu:
    baseName += "-relu"

if args.transp:
    baseName += "-transp"

netPatched = caffe.Net("{}.prototxt".format(baseName), caffe.TEST)

def convertDet1():
  net = caffe.Net('det1.prototxt', 'det1.caffemodel', caffe.TEST)
  
  for pnet in ['pnet1', 'pnet2', 'pnet3', 'pnet4', 'pnet5', 'pnet6', 'pnet7', 'pnet8', 'pnet9']: 
    for layer in ['conv1', 'conv2', 'conv3']:
      if args.transp:
        netPatched.params[pnet+'-'+layer][0].data[...] = np.transpose(net.params[layer][0].data, (0,1,3,2))
      else:
        netPatched.params[pnet+'-'+layer][0].data[...] = net.params[layer][0].data
      netPatched.params[pnet+'-'+layer][1].data[...] = net.params[layer][1].data

    netPatched.params[pnet+'-conv4'][0].data[...] = \
      np.concatenate((net.params['conv4-1'][0].data, net.params['conv4-2'][0].data), axis=0)
    netPatched.params[pnet+'-conv4'][1].data[...] = \
      np.concatenate((net.params['conv4-1'][1].data, net.params['conv4-2'][1].data), axis=0)

    if not args.relu:
      for layer in ['PReLU1', 'PReLU2', 'PReLU3']:
        netPatched.params[pnet+'-'+layer][0].data[...] = net.params[layer][0].data

def convertDet2():
  net = caffe.Net('det2.prototxt', 'det2.caffemodel', caffe.TEST)
  
  for layer in ['conv1', 'conv2', 'conv3']:
    if args.transp:
      netPatched.params['rnet-'+layer][0].data[...] = np.transpose(net.params[layer][0].data, (0,1,3,2))
    else:
      netPatched.params['rnet-'+layer][0].data[...] = net.params[layer][0].data

    netPatched.params['rnet-'+layer][1].data[...] = net.params[layer][1].data

  for layer in ['conv4']:
    if args.transp:
      netPatched.params['rnet-'+layer][0].data[...] = np.reshape(np.transpose(np.reshape(net.params[layer][0].data, (128,64,3,3)), (0,1,3,2)), (128,576))
    else:
      netPatched.params['rnet-'+layer][0].data[...] = net.params[layer][0].data

    netPatched.params['rnet-'+layer][1].data[...] = net.params[layer][1].data

  netPatched.params['rnet-conv5'][0].data[...] = \
    np.concatenate((net.params['conv5-1'][0].data, net.params['conv5-2'][0].data), axis=0)
  netPatched.params['rnet-conv5'][1].data[...] = \
    np.concatenate((net.params['conv5-1'][1].data, net.params['conv5-2'][1].data), axis=0)

  if not args.relu:
    for layer in ['prelu1', 'prelu2', 'prelu3', 'prelu4']:
      netPatched.params['rnet-'+layer][0].data[...] = net.params[layer][0].data

def convertDet3():
  net = caffe.Net('det3.prototxt', 'det3.caffemodel', caffe.TEST)
  
  for layer in ['conv1', 'conv2', 'conv3', 'conv4']:
    if args.transp:
      netPatched.params['onet-'+layer][0].data[...] = np.transpose(net.params[layer][0].data, (0,1,3,2))
    else:
      netPatched.params['onet-'+layer][0].data[...] = net.params[layer][0].data
    netPatched.params['onet-'+layer][1].data[...] = net.params[layer][1].data

  for layer in ['conv5']:
    if args.transp:
      netPatched.params['onet-'+layer][0].data[...] = np.reshape(np.transpose(np.reshape(net.params[layer][0].data, (256,128,3,3)), (0,1,3,2)), (256,1152))
    else:
      netPatched.params['onet-'+layer][0].data[...] = net.params[layer][0].data

    netPatched.params['onet-'+layer][1].data[...] = net.params[layer][1].data

  netPatched.params['onet-conv6'][0].data[...] = \
    np.concatenate((net.params['conv6-1'][0].data,
                    net.params['conv6-2'][0].data,
                    net.params['conv6-3'][0].data), axis=0)
  netPatched.params['onet-conv6'][1].data[...] = \
    np.concatenate((net.params['conv6-1'][1].data,
                    net.params['conv6-2'][1].data,
                    net.params['conv6-3'][1].data), axis=0)
  if not args.relu:
    for layer in ['prelu1', 'prelu2', 'prelu3', 'prelu4', 'prelu5']:
      netPatched.params['onet-'+layer][0].data[...] = net.params[layer][0].data

convertDet1()
convertDet2()
convertDet3()

netPatched.save('{}.caffemodel'.format(baseName))
