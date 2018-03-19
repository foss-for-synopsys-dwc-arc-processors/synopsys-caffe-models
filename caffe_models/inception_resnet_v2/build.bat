@set TOOL=evgencnn
@set IMAGES=/local/one_image
@set IMAGES=/local/vgg16/2016.09.19_5subset
@set IMAGES=/local/vgg16/2016.09.19_500subset


@set TOOLDIR=%DRIVE%\git\cnn\tools\cnn_tools\evgencnn\scripts
set show_classify=1
@set __SAVER=1
@set signed_last_ip_layer=
@set g_fixed_float=1

@set MINMAX_PER_MAP=1
@set MINMAX_PER_MAP=
@set print_csv=1
@set print_csv=

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
    --images %IMAGES% ^
    --caffe caffe_model/deploy_inception-resnet-v2.prototxt ^
    --weights caffe_model/deploy_inception-resnet-v2.caffemodel ^
    --vof bin --wof bin ^
    --name test ^
    --fpsize %SZ%  --fixed_dir fp.%SZ% --fpweight_size %WSZ% ^
    --fpweight_opnd_size %WSZ% --fp_opnd_size %SZ% ^
    --fpweight_opnd_size 12 --fp_opnd_size 12 ^
    --ibs 1 ^
    --float_dir flt ^
    --outdir . ^
    --ceng_version 3_0 ^
    --color_order RGB ^
    --image_scale 1 ^
    -g host_float ^
    -g host_fixed ^
    --use_minmax_cache ^
    %9
@goto:EOF
    
    --no_verify_blobs ^
    --blob_size 8 datA ^
    --signed_blobs ^
    --blob_size 12 datA pool3 conv3 ^
    --weight_size 10 re:conv.* ^
    --mean_subtraction false ^
    --mean_subtraction false ^
:exit
@mwfind . -rmdir -s
