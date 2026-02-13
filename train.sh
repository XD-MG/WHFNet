# Xian Dataset
CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_train.sh configs/whfnet/whfnet_swin_convnextv2-tiny_2xb4-15k_xian-opt-sar-128x128.py 2

# Korea Dataset
CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_train.sh configs/whfnet/whfnet_swin_convnextv2-tiny_2xb4-40k_korea-opt-sar-256x256.py 2

# WHU Dataset
CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_train.sh configs/whfnet/whfnet_swin_convnextv2-tiny_2xb4-80k_whu-opt-sar-512x512.py 2
