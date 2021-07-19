@set TOOL=evgencnn
@set IMAGES=/local/one_image
@set TOOLDIR=%DRIVE%\git\cnn\tools\cnn_tools\evgencnn\scripts
set show_classify=1
@set __SAVER=1
@set g_fixed_float=1
@set __CBSC=
@set show_prototxt=

@set which=3
@set P=det%which%.prototxt
@set W=det%which%.caffemodel

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
    --caffe caffe_model/%P% ^
    --weights caffe_model/%W% ^
    --vof bin --wof both ^
    --name test ^
    --fpsize %SZ%  --fixed_dir fp.%SZ% --fpweight_size %WSZ% ^
    --fpweight_opnd_size %WSZ% --fp_opnd_size %SZ% ^
    --fpweight_opnd_size 12 --fp_opnd_size 12 ^
    --ibs 1 ^
    --float_dir flt ^
    --outdir . ^
    --allow_unsupported_hardware ^
    --ceng_version 3_0 ^
    -g host_float ^
    -g host_fixed ^
    %9
@goto:EOF
    
    --no_verify_blobs ^
:exit
@find . -rmdir -s
