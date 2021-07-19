# Changelog
## Conversion
This model generated from ../../tensorflow/object_detection/ssd_mobilenet_v2_coco_2018_03_29.pb with evconvert (2019.03)

* Commands:

```shell
evconvert tfToCaffe -g ssd_mobilenet_v2_coco_2018_03_29.pb -p image_tensor 1 300 300 3 -f Preprocessor/sub -l concat concat_1 -o ssd_mobilenet_v2_coco_2018_03_29
```

## Models
### evconvert converted models
* convert_optimized.prototxt
* convert_optimized.caffemodel

### manually optimized models
* convert_optimized_detection_yx.prototxt  
We add an extra input layer `pbox` for priorbox generation, the data can be found in `anchor_yx.pkl` in this folder.  
We also added extra `detection_out` layer to produce the same detection output as the TensorFlow model, 
which means the detection box order is `[ymin, xmin, ymax, xmax]`.  
Here is sample code for how to use this model in caffe (preprocessing and extra handling after post processing):

   ~~~python
    def run_caffe_model_inference(net, img_path, pbox_pkl):
        mean_values = [127.5, 127.5, 127.5]
        scale = 0.00784313771874
        rgb_color_order = True
    
        img = cv2.imread(img_path)
        img = cv2.resize(img, (300, 300), interpolation = cv2.INTER_LINEAR)
        img = img.astype(dtype=np.float32)
        img = np.rollaxis(img, 2) # convert hwc_to_chw
        img[0] = img[0] - mean_values[0]
        img[1] = img[1] - mean_values[1]
        img[2] = img[2] - mean_values[2]
        img = img * scale
        if rgb_color_order:
            img = np.flipud(img) # convert bgr_to_rgb
        img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
        
        with open(pbox_pkl, 'rb') as F:
            pbox = pickle.load(F)
        
        net.blobs['Preprocessor/sub'].data[...] = img
        net.blobs['pbox'].data[...] = pbox
        net.forward()
        return net.blobs['detection_out'].data
  
    def parse_caffe_detections(detections):
        num_detections = len(detections[0][0])
        new_detections = detections.copy()
        # [ymin, xmin, ymax, xmax] -> [xmin, ymin, ymax, xmax]
        for i in range(num_detections):
            new_detections[0][0][i][3] = detections[0][0][i][4]
            new_detections[0][0][i][4] = detections[0][0][i][3]
            new_detections[0][0][i][5] = detections[0][0][i][6]
            new_detections[0][0][i][6] = detections[0][0][i][5]
        return new_detections
  ~~~
    
