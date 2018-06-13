@set TOOL=evgencnn
@set IMAGES=/local/one_image
@set TOOLDIR=N:\git\cnn\tools\cnn_tools\evgencnn\scripts
set show_classify=1
@set __SAVER=1
@set g_fixed_float=1
@set W=no_weight_file
@set W=dummy_file
@set W=caffe_model\resnet101_faster_rcnn_bn_scale_merged_end2end_iter_70000.caffemodel
@set ROI_TRACE=1
@set RPN_POST_NMS_TOP_N=10

@call :doit 12
@goto exit
@call :doit 8
@call :doit 7
@call :doit 9
@call :doit 10
@call :doit 11

:doit
@set SZ=%1
@set WSZ=%1
python %TOOLDIR%/evgencnn ^
    --caffe caffe_model/test.pt ^
    --weights %W% ^
    --images %IMAGES% ^
    --name test ^
    --wof bin --vof bin --name test ^
    --fpsize %SZ%  --fixed_dir fp.%SZ% --fpweight_size %WSZ% ^
    --float_dir flt ^
    --outdir . ^
    --allow_unsupported_hardware ^
    --ignore_unsupported_layers ^
    --float_type float ^
    --signed_blobs ^
    -g host_float ^
    -g host_fixed ^
    --ibs 8 ^
    --subgraph G1 proposal cls_score bbox_pred ^
    --batch_iterate G1 iobj %RPN_POST_NMS_TOP_N% ^
    --pixel_mean 102.9801,115.9465,122.7717 ^
    %9
@goto :EOF
    
@goto end
    -g host_fixed ^
    -g host_float ^
    --no_verify_blobs ^
:end

:exit
@find . -rmdir -s
