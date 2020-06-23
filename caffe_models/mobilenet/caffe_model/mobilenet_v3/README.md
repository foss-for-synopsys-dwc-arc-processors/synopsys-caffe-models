The model origin is: https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet  
Jira record: https://jira.internal.synopsys.com/browse/P10019563-39535  

MobileNet v3 model could be converted from frozen pb or tflite to caffe model.  

Conversion command:
1. Model converted from TF: pb_converted/v3-small_224_0.75_float_pb_convert.prototxt
> evconvert tf2ev -g v3-small_224_0.75_float.pb -p input 1 224 224 3  

2. (Old) model converted from tflite: v3-small_224_0.75_float_convert.prototxt
> evconvert tflite2ev -m v3-small_224_0.75_float.tflite  
This model contains Permute layers that should have been optimized away during the conversion.
The model converted from TF does not have these layers.

The *_nohardsiwsh.prototxt models are generated manually by changing all HardSwish layers into ReLU layers

