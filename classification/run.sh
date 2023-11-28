NUM_GPUS=8
./dist_train.sh $NUM_GPUS -c\
    /workspace/akane/NAT/classification/configs/vit_small_patch16_augreg_21k_attnsum.yml \
    /workspace/datasets/ImageNet