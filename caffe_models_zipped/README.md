
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
2. Edit `model_list.txt`

   Downloading and unpacking of all models can take formidable time
   expecially in case of slow Internet. Do reduce time customers reduce
   number of models. Make copy of `model_list.txt` to
   `model_list_shorten.txt` and remove models you don't want to use now.
   You can download other models lately
> **Note:**  
> `imagenet_mean` and `images` are image data-sets that are used many
> `model_list_shorten.txt` models. Don't remove them from
> `model_list_shorten.txt`

3. Run the script

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

