@set TOOL=evgencnn
@set IMAGES=test_images
@set IMAGES=/local/one_image
@set IMAGES=/local/ILSVRC2012/100images

@set TOOLDIR=N:\git\cnn\tools\cnn_tools\evgencnn\scripts
set show_classify=1
@set __SAVER=1
@rem @set __IDEMPOTENT=1
@set WF=no_weight_file
@set WF=dummy_weights
@set WF=caffe_model/mobilenet.caffemodel
@set CF=caffe_model/deploy.prototxt
@set MINMAX_PER_MAP=1
@set MINMAX_PER_MAP=

@call :doit 12
@goto end
@call :doit 8
@call :doit 9
@call :doit 10
@call :doit 11

:doit
@set SZ=%1
@set WSZ=%1

:doit
@set SZ=%1
@set WSZ=%1
python %TOOLDIR%/evgencnn ^
    --caffe %CF% --weights %WF% ^
    --images %IMAGES% ^
    --name test ^
    --wof bin --vof bin --name test ^
    --fpsize %SZ% --fixed_dir fp.%SZ% ^
    --float_dir flt ^
    --outdir . ^
    --allow_unsupported_hardware ^
    --float_type float ^
    --pixel_mean 103.94,116.78,123.68 ^
    --image_scale 4.335 ^
    --classifier_layer fc7 ^
    --ibs 8 ^
    --blob_size 12 data ^
    --use_minmax_cache ^
    --calibrate_file 100.bin ^
    --noZblob_scale fc7 ^
    --no_verify_blobs ^
    -g host_fixed ^
    %9
@cd fp.%SZ%
@call \s.bat
@cd ..
@goto:EOF
    
@goto end
    --signed_blobs ^
    -g host_float ^
    -g host_fixed ^
    --pixel_mean 103.939,116.779,123.68 ^
    --no_verify_blobs ^
    --acc_bias
    --nice_format
:end

@find . -rmdir -s
