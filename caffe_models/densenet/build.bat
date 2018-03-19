@set TOOL=evgencnn

@set IMAGES=test_images
@set IMAGES=/local/one_image
@set IMAGES=/local/ILSVRC2012/5images

@set TOOLDIR=N:\git\cnn\tools\cnn_tools\evgencnn\scripts
set show_classify=1
@set __SAVER=1
@set create_convolution=1
@rem --caffe caffe_model/%M%\DenseNet_%M%.prototxt ^
@set M=121
@set __CBSC=
    @rem --caffe caffe_model/%M%\DenseNet_%M%.prototxt 

python %TOOLDIR%/evgencnn ^
    --caffe caffe_model/DenseNet_%M%.prototxt ^
    --weights caffe_model/%M%\DenseNet_%M%.caffemodel ^
    --images %IMAGES% ^
    --name test ^
    --wof bin --vof bin --name test ^
    --fixed_dir fp ^
    --float_dir flt ^
    --outdir . ^
    --signed_blobs ^
    --float_type float ^
    --vdump_top ^
    --ibs 8 ^
    -g host_fixed ^
    --classifier_layer  fc6 ^
    --pixel_mean 103.94,116.78,123.68 ^
    --allow_unsupported_hardware ^
    %*
    
@goto end
    --use_minmax_cache ^
    --unsigned_weights ^
    --no_verify_blobs ^
    -g host_fixed ^
    This is vgg16's pixel mean:
    -g host_fixed ^
    -g host_float ^
    --fpsigned ^
    --lkrelu 12 ^
    --no_verify_blobs ^
    --ibs 8 ^
    --acc_bias
    --nice_format
:end

rd c_ref_fixed c_ref_float
find . -name tnnc!tnnv -rmdir


