
set DRIVE=N:
set GITC=%DRIVE%/git/caffe
@rem J:\git\caffe\wincaffe\Build\x64\Release\pycaffe\caffe
@rem has the __init__.py to turn caffe into a package.
@rem set PYTHONPATH=%GITC%/caffe/src/caffe;%GITC%/wincaffe/build/x64/release/pycaffe/caffe
@rem set PYTHONPATH=%GITC%/wincaffe/build/x64/release/pycaffe/caffe
@rem Must point to the parent of the desired package
@set CAFFE=wincaffe
@set CAFFE=wakim.caffe
set PYTHONPATH=%GITC%/wakim.caffe;%DRIVE%/git/jinja2-2.8;%DRIVE%/git/markupsafe-0.23
set PYTHONPATH=K:\local\v2yolo;%GITC%/wincaffe/build/x64/release/pycaffe;%DRIVE%/git/jinja2-2.8;%DRIVE%/git/markupsafe-0.23;.
set EV_CNNSDK_HOME=k:/
