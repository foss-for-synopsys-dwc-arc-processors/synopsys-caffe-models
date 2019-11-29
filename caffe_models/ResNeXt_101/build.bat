@set TOOL=evgencnn
@set TOOLDIR=%DRIVE%\git\cnn\tools\cnn_tools\evgencnn\scripts
@set EV_CNNSDK_HOME=k:/

@set IMAGES=/local/vgg16/2016.09.19_5subset
@set IMAGES=/local/vgg16/2016.09.19_20subset
@set IMAGES=/local/vgg16/2016.09.19_100subset
@set IMAGES=/local/vgg16/2016.09.19_500subset
@set IMAGES=/local/one_image

set show_classify=1
@set __SAVER=1
@set g_fixed_float=1
@set __CBSC=
@set IBS=8
@set RN=101

@set MINMAX_PER_MAP=1
@set MINMAX_PER_MAP=

@set __CONCAT_CONV=1
@set __CONCAT_CONV=
@set __CONCAT_CONV=2

@set WF=dummy_weights
@set WF=no_weight_file
@set mean=../resnet_50/caffe_model/ResNet_mean.binaryproto

@call :doit 12
@goto end
@call :doit 8
@call :doit 9
@call :doit 10
@call :doit 11

:doit
@set SZ=%1
@set WSZ=%1
python %TOOLDIR%/evgencnn ^
    --caffe caffe_model/ResNeXt-%RN%-deploy.prototxt ^
    --weights %WF% ^
    --images %IMAGES% ^
    --vof bin --wof bin ^
    --name test ^
    --fpsize %SZ% --fixed_dir fp.%SZ% --fpweight_size %WSZ% ^
    --fpweight_opnd_size %WSZ%  --fp_opnd_size %SZ% ^
    --fp_opnd_size 12 --fpweight_opnd_size 12 ^
    --ibs %IBS% ^
    --float_dir flt ^
    --outdir . ^
    --shift_round even ^
    --float_type float ^
    --signed_blobs ^
    --signed_blobs ^
    -g host_fixed ^
    %9
@rem @cd fp.%SZ%
@rem @call \s.bat
@rem @cd ..
@goto:EOF
    
    --blob_size 12 data ^
    --no_verify_blobs ^
    --image_mean %mean% ^
    --image_mean caffe_model/ResNet_mean.binaryproto ^
    --use_minmax_cache ^
    --calibrate_file 500.bin ^
    -g unmerged_large ^
    --ceng_version 3_0 ^
    --eltwise_folding ^
    -g host_float ^
    --blob_size 12 data res2b_branch2b res2b_branch2a ^
	  res2a_branch1 res2a_branch2a res2a_branch2b res2a_branch2c re:.*res4.* ^
    --weight_size 12 re:res2b*
    --weight_size 12 res2b_branch2b res2b_branch2a ^
	  res2a_branch1 res2a_branch2a res2a_branch2b res2a_branch2c ^
    --signed_blobs ^
    --calibrate_file 100histo.bin ^
    --histosat ^
    --eltwise_folding ^
    --calibrate_file 500.bin ^
    --unsigned_weights ^
    --fpsigned --fixed_dir s12 ^
    -g host_fixed ^
    --ugraph ^
    --allow_unsupported_hardware ^
--nice_format
    --fixed_dir fp12 ^
    --image_mean ../fddb/mean_file.bin ^
--wof bin ^
-g host_fixed ^
--acc_bias
:end

@find . -rmdir -s
