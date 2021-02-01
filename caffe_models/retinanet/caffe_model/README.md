The model is converted from pytorch model to onnx, then using onnx2ev converter to caffe model. 
Origin model downloaded from: https://github.com/facebookresearch/detectron2 

Please use the optimized prototxt with caffemodel when mapping to EV to get better performance. 

new_detectron2_retinanet_R_101_FPN_3_convert_optimized.prototxt is converted from model: onnx/detectron2_retinanet_R_101_FPN_3_modified.onnx
conversion command: evconvert onnx2ev -m detectron2_retinanet_R_101_FPN_3_modified.onnx -o new_detectron2_retinanet_R_101_FPN_3

model detectron2_retinanet_R_101_FPN_3_modified.onnx was exported by script:
https://github.com/facebookresearch/detectron2/blob/master/tools/deploy/caffe2_converter.py
Then the exported onnx model removed the first Sub and Div nodes to get the same inference result as torch model.
