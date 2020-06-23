1. download link: http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz
2. The model is converted by evconvert.
3. From the link https://github.com/tensorflow/models/blob/master/research/slim/README.md, input image size is 299.
4. Preprocessing
The model is generated from TensorFlow, so it should use the same preprocessing. 
From https://github.com/tensorflow/models/blob/master/research/slim/README.md
ResNet V2 models use Inception pre-processing and input image size of 299
Therefore, the correct preprocessing is: https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py#L290
There is a simple version from Keras(we only use the mode == 'tf'): https://github.com/keras-team/keras-applications/blob/b34c10628a0ab436542e9160f98de72b49084bbe/keras_applications/imagenet_utils.py#L18
    x /= 127.5
    x -= 1.0
 After them, we should resize and crop it to the expected image size. At last, we should:
• add batch index: x = np.expand_dims(x, axis=0)
• NHWC data format into NCHW data format:
    x = x.transpose(0, 3, 1, 2)
