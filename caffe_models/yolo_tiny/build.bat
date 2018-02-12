@set TOOL=tap 
@set TOOL=tom
@set TOOL=evgencnn
@set IMAGES=dogs
@set IMAGES=failures
@set IMAGES=../vgg16/2016.09.13_images
@set IMAGES=22_images
@set IMAGES=500images
@set IMAGES=few_images
@set IMAGES=../vgg16/2016.09.19_1000subset
@set IMAGES=../vgg16/2016.09.19_500subset
@set IMAGES=../vgg16/2016.09.19_20subset
@set IMAGES=test_images
@set IMAGES=one_image
@set IMAGES=/local/pascal/100_subset
@set IMAGES=/local/pascal/1_subset
@set IMAGES=test_image

@set TOOLDIR=N:\git\cnn\tools\cnn_tools\evgencnn\scripts
set show_classify=1
@set __SAVER=1
@set SZ=12

python %TOOLDIR%/evgencnn ^
    --caffe caffe_model/yolo_tiny_deploy.prototxt ^
    --weights caffe_model/yolo_tiny.caffemodel ^
    --images %IMAGES% ^
    --name test ^
    --wof bin --vof bin --name test ^
    --fpsize %SZ% ^ --fixed_dir fp.%SZ% --fpweight_size %SZ% ^
    --float_dir flt ^
    --signed_blobs ^
    -g host_fixed ^
    --ugraph ^
    --outdir . ^
    --lkrelu 8 ^
    --lkrelu_accsize 15 ^
    --use_minmax_cache ^
    --color_order RGB ^
    --image_scale 1 ^
    --post_verify_func yolo ^
    %*
    
@goto end
    --no_verify_blobs ^
    -g host_float ^
    -g host_fixed ^
    -g host_float ^
    --ibs 8 ^
    --acc_bias
    --nice_format
:end

rd c_ref_fixed c_ref_float
find . -name tnnc!tnnv -rmdir


