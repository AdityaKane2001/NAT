NUM_GPUS=1
SAMPLES_PER_GPU=1
WORKERS_PER_GPU=1
./dist_train.sh configs/localglobal_dinat_s_norpb/mask_rcnn_localglobal_dinat_s_norpb_tiny_3x_coco.py \
   $NUM_GPUS --cfg-options data.samples_per_gpu=$SAMPLES_PER_GPU data.workers_per_gpu=$WORKERS_PER_GPU