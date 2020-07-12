docker run --gpus all -it --rm \
 -v `pwd`:/workspace \
 -v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
 -v /data/dataset/coco/train2014:/workspace/data/coco/images/train2014 \
 -v /data/dataset/coco/val2014:/workspace/data/coco/images/val2014 \
 pytorch/pytorch:1.4-cuda10.1-cudnn7-devel bash
