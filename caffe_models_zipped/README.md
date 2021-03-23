
Zipped NN models
================

The folder consists of the list of ZIP archivated models. They can be
directly downloaded, unpacked on local PC.

Donwload and unpack utility
===========================
Instead of manual downloading/unpacking you can use special utility  
`donwnoad_unpack_cnn_models.py`

## Usage Instructions
1. Download `donwnoad_unpack_cnn_models.py` and `model_list.txt`
   Click `model_list.txt`, press `Raw` button, press Right-Click and choose "Save As .." in contect menu  
   
2. Edit `model_list.txt`

   Downloading and unpacking of all models can take formidable time expecially in case of slow Internet. And it requiers about 50Gb of free disc space.  
   Do reduce download time and disc space customers can reduce number of models. 
   Make copy of `model_list.txt` to `model_list_shorten.txt` and remove models you don't want to use now.
   It's possible download other models lately  
> **Note:**  
> `imagenet_mean` and `images` are image data-sets that are used by many models.  
> `Don't remove them from `model_list_shorten.txt`  

3. Download and unpack models

   `python donwnoad_unpack_cnn_models.py --model_list
   model_list_shorten.txt`  
   It will download and unpack models in path that is defined as
   %EV_CNNMODELS_HOME%\caffe_models, where %EV_CNNMODELS_HOME% -
   enviroment variable
   You can select your path instead  
   `python donwnoad_unpack_cnn_models.py --model_list
   model_list_shorten.txt --model_path <your_path>`  
   You need at least 50Gb of disc space to download and unpack all models

4. Download new models

   You can add new models in `model_list_shorten.txt` and re-start  
   `python donwnoad_unpack_cnn_models.py --model_list
   model_list_shorten.txt`  
   It will download/unpack only missed parts

5. Update models

   You can update existed models. You can re-download them with extra option --force:  
   `python donwnoad_unpack_cnn_models.py --model_list
   model_list_shorten.txt --model_path <your_path>` --force  

