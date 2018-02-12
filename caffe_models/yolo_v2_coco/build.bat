@set TOOL=evgencnn
@set IMAGES=test_images

@set TOOLDIR=K:\chuck\evgencnn\scripts
@set TOOLDIR=N:\git\cnn\tools\cnn_tools\evgencnn\scripts
set show_classify=1
@set __SAVER=1
@set M=121
@set create_convolution=1
@set POOL_TRACE=
@rem the need for the darknet env var to issue the
@rem reorg layer is temporary.
@set darknet=1
@set P=caffe_model/chuck.prototxt 
@set W=no_weight_file

@set CM=caffe_model
@set P=%CM%/yolo_voc.prototxt 
@set W=%CM%/yolo_voc.caffemodel 
@del *.pyc
@set g_fixed_float=1
@set P=%CM%/yolo.prototxt
@set W=%CM%/yolo.caffemodel

python %TOOLDIR%/evgencnn ^
    --caffe %P% --weights %W% ^
    --images %IMAGES% ^
    --name test ^
    --wof bin --vof bin --name test ^
    --fixed_dir fp ^
    --float_dir flt ^
    --outdir . ^
    --allow_unsupported_hardware ^
    --signed_blobs ^
    --float_type float ^
    --use_minmax_cache ^
    --color_order RGB ^
    --image_scale 1 ^
    --post_verify_func yolo ^
    --ceng_version=3_0 ^
    --lkrelu 7 ^
    -g host_fixed ^
    %*
    
@goto end
    -g host_float ^
    -g unmerged_large ^
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


