From https://github.com/chuanqi305/MobileNet-SSD/

Standard graph (300x300):
1. MobileNetSSD\_deploy.prototxt / MobileNetSSD\_deploy.caffemodel

Resized graphs:
1. 640x480: MobileNetSSD\_deploy-640x480.prototxt / MobileNetSSD\_deploy.caffemodel
2. 960x720: MobileNetSSD\_deploy-960x720.prototxt / MobileNetSSD\_deploy.caffemodel
3. 1920x1080: MobileNetSSD\_deploy-1920x1080.prototxt / MobileNetSSD\_deploy.caffemodel

Pruned graphs:
A. random pruned (60% conv, 85% fc):
1. MobileNetSSD\_deploy.prototxt / MobileNetSSD\_deploy\_random\_pruned.caffemodel
2. MobileNetSSD\_deploy-640x480.prototxt / MobileNetSSD\_deploy\_random\_pruned.caffemodel
3. MobileNetSSD\_deploy-960x720.prototxt / MobileNetSSD\_deploy\_random\_pruned.caffemodel
4. MobileNetSSD\_deploy-1920x1080.prototxt / MobileNetSSD\_deploy\_random\_pruned.caffemodel

Structure modified graph:
1. MobileNetSSD\_deploy\_reordered.prototxt  
(The Softmax layer is moved before Concat, and the original DetectionOutput layer is replaced by SSDDecoder and SSDSort layers. 
This prototxt could be used with the MobileNetSSD\_deploy.caffemodel and will generate the same results as MobileNetSSD\_deploy.prototxt.)
