# Example run script for sender side
# -p set port
# -R list of used net devices (if more than selected rails only the first few are used)
# -A Address of peer node

CUDA_VISIBLE_DEVICES=2,0,4,6 ./build/mr_basic_sender -P 13337 -R mlx5_0:1,mlx5_2:1,mlx5_8:1,mlx5_4:1 -A hawk-ai10