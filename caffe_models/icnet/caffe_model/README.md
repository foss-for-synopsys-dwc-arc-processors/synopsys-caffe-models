Model from: https://github.com/hszhao/ICNet

icnet_cityscapes_train_30k.caffemodel: [GoogleDrive](https://drive.google.com/open?id=0BzaU285cX7TCRXpXMnVIbXdfaW8)

(31M, md5: c7038630c4b6c869afaaadd811bdb539; train on trainset for 30k)

icnet_cityscapes_train_30k_bnnomerge.caffemodel: [GoogleDrive](https://drive.google.com/open?id=0BzaU285cX7TCSW9tZlQ2Q0pFeG8)

(31M, md5: 6da59a72a254862106420983a7723624; train on trainset for 30k, original)

icnet_cityscapes_trainval_90k.caffemodel: [GoogleDrive](https://drive.google.com/open?id=0BzaU285cX7TCTFVpZWJINi1Iblk)

(31M, md5: 4f4dd9eecd465dd8de7e4cf88ba5d5d5; train on trainvalset for 90k)

icnet_cityscapes_trainval_90k_bnnomerge.caffemodel: [GoogleDrive](https://drive.google.com/open?id=0BzaU285cX7TCQlpJMkFIYmdjc1U)

(31M, md5: ba3cf6e24beb07068dacc901a9c7f28b; train on trainvalset for 90k, original)

**Notes**:   
1. Model's name that contains phrase 'bnnomerge' is the original trained model, the related one without this phrase is obtained by merging the parameters in batch normlization layers into the closely front convolution layers.  
    a. When testing the mIoU performance, please choose the related prototxt file.   
    b. When testing the inference speed, please choose prototxt without this phrase. That's because the ''Caffe time'' tool runs in training mode while bn layers work in different way as in testing mode (using stored history statistics during testing VS online calculating current batch's statistics during training).  

2. The icnet_cityscapes_merge_subgraph.prototxt is a variant from the icnet_cityscapes.prototxt. It merges several AVE Pooling and Interp with the Eltwise Sum into a single layer called ICNetSubgraph. This prototxt could produce the same final results as the original one while improving the model efficiency, reducing bandwidth consumption and solve the mismatch issue between host_fixed and unmerged_large implementation.

