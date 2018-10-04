echo on
if "%~1"=="" goto :help
if "%~1"=="help" goto :help
if "%~1"=="-h" goto :help
if [%~1]== [/?] goto :help

goto :main

:help
echo.
echo Usage:
echo   -h, help, /? - print this help
echo   The script requires a list of model folders as them named in GitHub repo
echo   At least one folder has to be added
echo.
echo   Example:
echo   %~nx0 facedetect_v1 
echo.
echo  NOTE: The script does not check if there is any mistakes in folders' name!!!
echo.
exit /B

:main
echo "Main"

:: Prepare sparse-checkout list
echo caffe_models/image* > sparse-checkout

:loop
if not "%1"=="" (
    echo Param=%1
	echo caffe_models/%1 >> sparse-checkout
    shift
    goto :loop
)

md cnn_models
cd cnn_models
:: it makes .git folder in ./cnn_models
git init  
git remote add -t master origin https://github.com/foss-for-synopsys-dwc-arc-processors/synopsys-caffe-models.git
git config core.sparseCheckout true
move ..\sparse-checkout .git\info
git fetch --depth 1
:: git checkout master # checkout power_scripts of $GIT_BRANCH branch

