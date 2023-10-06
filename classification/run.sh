NUM_GPUS=2
./dist_train.sh $NUM_GPUS -c\
    /workspace/akane/NAT/classification/configs/wintome_dinat_s_tiny_imagenette.yml \
    /workspace/datasets/akane/imagenette2
