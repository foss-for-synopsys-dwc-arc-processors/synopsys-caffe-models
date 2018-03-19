@set TOOL=tap 
@set TOOL=tom
@set TOOL=evgencnn
@set IMAGES=test_images
@set IMAGES=/local/one_image

@set TOOLDIR=N:\git\cnn\tools\cnn_tools\evgencnn\scripts
set show_classify=1
@set __SAVER=1
@set WF=caffe_model/VGG_VOC0712_SSD_300x300_iter_60000.caffemodel
@set DETECT_TRACE=
@set DETECT_TRACE=1

@set MINMAX_PER_MAP=1
@set MINMAX_PER_MAP=

@call :doit 12
@goto exit
@call :doit 8
@call :doit 11
@call :doit 10
@call :doit 9

:doit
@set SZ=%1
@set WSZ=%1
python %TOOLDIR%/evgencnn ^
    --caffe caffe_model/small.prototxt ^
    --caffe caffe_model/deploy.prototxt ^
    --weights %WF% ^
    --images %IMAGES% ^
    --name test ^
    --wof bin --vof bin --name test ^
    --fpsize %SZ%  --fixed_dir fp.%SZ% --fpweight_size %WSZ% ^
    --fpweight_opnd_size 12 --fp_opnd_size 12 ^
    --float_dir flt ^
    --outdir . ^
    --signed_blobs ^
    --allow_unsupported_hardware ^
    --ignore_unsupported_layers ^
    --float_type float ^
    --discard_after_softmax ^
    --implement_softmax_layer ^
    -g host_float ^
    -g host_fixed ^
    %9
@cd fp.%SZ%
@call \s
@cd ..
@goto:EOF
    
    --use_minmax_cache ^
    --no_verify_blobs ^
    --ibs 8 ^
    --pixel_mean 103.939,116.779,123.68 ^
    --no_verify_blobs ^
    --acc_bias
    --nice_format
:exit

rd c_ref_fixed c_ref_float


