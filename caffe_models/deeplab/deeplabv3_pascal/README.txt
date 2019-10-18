1. The origianl TensorFlow model:
https://gitsnps.internal.synopsys.com/dwc_ev/cnn_models/blob/master/tensorflow/deeplabv3_pascal_train_aug.pb
Download from: http://download.tensorflow.org/models/deeplabv3_pascal_train_aug_2018_01_04.tar.gz

2. evconvert command:
evconvert tfToCaffe -g deeplabv3_pascal_train_aug.pb -p ImageTensor 1 300 300 3 -f sub_7 -l ResizeBilinear_3 --error 2.0e-3 -o deeplabv3_pascal
