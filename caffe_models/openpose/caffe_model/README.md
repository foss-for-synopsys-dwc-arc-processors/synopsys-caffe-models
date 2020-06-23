# Model Description
* **Model Repo:** https://github.com/CMU-Perceptual-Computing-Lab/openpose
* Prototxt and caffemodel
  + **pose_deploy_linevec.prototxt**: Small edits in dims of https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/models/pose/coco/pose_deploy_linevec.prototxt
  + **pose_deploy_linevec.caffemodel**: random generated weights
  + **pose_iter_440000.caffemodel**: coco weights file sent by M.Tomono, maybe download by this script https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/models/getModels.sh
  + **pose_deploy.prototxt**: Small edits in dims of https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/models/pose/body_25/pose_deploy.prototxt
  + **pose_iter_584000.caffemodel**: body_25 weights file downloaded with https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/models/getModels.sh

* Pruned Graphs
1. pose\_deploy\_linevec.prototxt / pose\_iter\_440000\_random\_pruned.caffemodel
- random pruned (conv: 60%, fc: 85%)
