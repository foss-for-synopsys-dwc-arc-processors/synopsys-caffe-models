@set TOOL=evgencnn
@set IMAGES=test_images

@set TOOLDIR=N:\git\cnn\tools\cnn_tools\evgencnn\scripts
set show_classify=1
@set __SAVER=1

@set CM=caffe_model
@set g_fixed_float=1
@set P=%CM%/yolov3-tiny.prototxt
@set W=%CM%/yolov3-tiny.caffemodel
@set P=%CM%/yolov3.prototxt
@set W=%CM%/yolov3.caffemodel

@set MINMAX_PER_MAP=
@set MINMAX_PER_MAP=1
@set WEIGHTS_PER_MAP=
@set WEIGHTS_PER_MAP=1

@call :doit 8
@goto end
@call :doit 12
@call :doit 9
@call :doit 10
@call :doit 11

:doit
@set SZ=%1
@set WSZ=12
python %TOOLDIR%/evgencnn ^
    --caffe %P% --weights %W% ^
    --images %IMAGES% ^
    --name test ^
    --wof bin --vof bin --name test ^
    --fixed_dir fp.%SZ%  ^
    --fpsize %SZ% --fp_opnd_size %SZ% ^
    --fpweight_size %WSZ% --fpweight_opnd_size %WSZ%   ^
    --float_dir flt ^
    --outdir . ^
    --allow_unsupported_hardware ^
    --signed_blobs ^
    --float_type double ^
    --float_type float ^
    --color_order RGB ^
    --image_scale 1 ^
    --letterbox 127.5 ^
    --ceng_version=3_0 ^
    --support_caffe_source true ^
    --caffe_layer_type_pixel_only  UpsampleDarknet ^
    --ibs 8 ^
    --post_verify_func v2:yolo ^
    -g host_fixed ^
    %9
@rem @cd fp.%SZ%
@rem @call \s.bat
@rem @cd ..
@goto:EOF
    
@goto end
    -g host_float ^
    --use_minmax_cache ^
    --no_verify_blobs ^
    --post_verify_func yolo ^
    --lkrelu 7 ^
    -g host_float ^
    -g unmerged_large ^
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

find . -rmdir -s


