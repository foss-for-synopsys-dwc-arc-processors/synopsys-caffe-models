
CNN Caffe Models
================

This directory contains a number of caffe models used for CNN SDK examples.
Consult the Caffe Model Zoo for details: 

    https://github.com/BVLC/caffe/wiki/Model-Zoo

and the following for Caffe Zoo license terms and conditions:

    http://caffe.berkeleyvision.org/model_zoo.html#bvlc-model-license

## Usage Instructions
**IMPORTANT NOTE:  This repository uses git-lfs for large file storage.  You can't use zip and tar files listed in the "Assets" section above (added by default by github).  You must clone the repository using the instructions below** 

1. Install git-lfs

2.  Ensure git-lfs and git versions you use are compatible (equal or greater than below)
```
$ git lfs version
git-lfs/2.0.2 # or newer

$ git –version
git version 2.9.3 # or newer
```
3.  Add [SSH key](https://help.github.com/articles/adding-a-new-ssh-key-to-your-github-account/) to your GitHub account (if you haven't already)

4. clone the full repo:
```
$ git clone https://github.com/foss-for-synopsys-dwc-arc-processors/synopsys-caffe-models.git
```

5. clone a part of the repo:
```
If you don't need all models and want to save disc space you can use special scripts:

git_sparse_download.sh  - for Linux
git_sparse_download.bat - for Windows

They set-up git repo for working in space-checkout mode, with minimum git history

1. Choose a folder where "cnn_models" folder will be created. 
2. Select on "Save link as .." to save the script in that folder
2. Choose a list of models which you want to work with ( in caffe_models folder), for instance: googlenet mobilenet
3. Run the script with params - names of selected models:
Example:
git_sparse_download.sh googlenet mobilenet

It creates "cnn_models" folder, init git repo in there, dowload common files and models you select.

```

## Alternative way to download (Beta)

In `caffe_models_zipped` sub-folder we hold all NN models in zip format.  
You can use the special Python utility that can download and unpack selected models.  
Please read a description in README.md that sub-folder.  
