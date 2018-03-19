@set TOOL=evgencnn
@set IMAGES=test_images

@set TOOLDIR=N:\git\cnn\tools\cnn_tools\evgencnn\scripts
set show_classify=1
@set __SAVER=1
@set WF=no_weight_file
@set WF=caffe_model/deploy.caffemodel 
@set __CBSC=

python %TOOLDIR%/evgencnn ^
    --caffe caffe_model/deploy.prototxt ^
    --images %IMAGES% ^
    --weights %WF% ^
    --name test ^
    --wof bin --vof bin --name test ^
    --fpsize 12 --fixed_dir fp.12 ^
    --float_dir flt ^
    --outdir . ^
    --signed_blobs ^
    --allow_unsupported_hardware ^
    --ignore_unsupported_layers ^
    --discard_after_softmax ^
    --float_type float ^
    --pixel_mean 104,117,123 ^
    -g host_fixed ^
    %*
    
@goto end
    -g host_float ^
    --use_minmax_cache ^
    --no_verify_blobs ^
    -g host_fixed ^
    --ibs 8 ^
    --pixel_mean 103.939,116.779,123.68 ^
    --no_verify_blobs ^
    --acc_bias
    --nice_format
:end

rd c_ref_fixed c_ref_float


