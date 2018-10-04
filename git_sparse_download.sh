#!/bin/bash -e

me="git_sparse_gownload.sh"

if [[ "$1" = "help" || "$1" = "-h" || "$#" -eq 0 ]] ; then
echo ""
echo "Usage:"
echo "    ./${me} -h, help - print this help"
echo ""
echo "    The script uses git to download common part + required models"
echo "    It requires a list of model folders as them named in GitHub repo"
echo ""
echo "    Example"
echo "    ./${me} facedetect_v1"
echo ""
exit -1
fi

echo "caffe_models/image*" > sparse-checkout

for i in "$@"
do
    echo "caffe_models/$i" >> sparse-checkout
done

mkdir -p cnn_models
cd cnn_models
# it makes .git folder in ./cnn_models
git init  
git remote add -t master origin https://github.com/foss-for-synopsys-dwc-arc-processors/synopsys-caffe-models.git
git config core.sparseCheckout true
mv -f ../sparse-checkout .git/info
git fetch --depth 1
# checkout part of repo
git checkout master

