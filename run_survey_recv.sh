# Example run script for survey benchmark receiver side with messages up to 16 GB
# -p set port
# -R list of used net devices (if more than selected rails only the first few are used)
# -T type of Test (MR, SINGLE, SPLIT, TEST_SPLIT)
# -m memory type (CUDA, HOST)
# -n Number of rails (2, 4)
# -r Split raio in percent
# -k Number of pipeline stages (< MAX_PIPELINE_STAGES)
# -s Message Size in bytes

source ./example/ucx-1_15_0.sh

BINDING="--physcpubind=16-55 --membind=1,3"

msg_size=1073741824;

while getopts T:n:k:r: flag
do
    case "${flag}" in
        T) test=${OPTARG};;
        n) nof_rails=${OPTARG};;
        k) pipe_stages=${OPTARG};;
        r) ratio=${OPTARG};;
    esac
done

if [ $test = "SPLIT" ]
then
  CUDA_VISIBLE_DEVICES=2,0,4,6 numactl ${BINDING} \
  ./build/mr_bench_receiver -P 13337 -R mlx5_0:1,mlx5_2:1,mlx5_8:1,mlx5_4:1 \
  -m CUDA -s $msg_size  -T SPLIT -n $nof_rails -r $ratio -k $pipe_stages -p 1
fi

if [ $test = "MR" ]
then
  CUDA_VISIBLE_DEVICES=2,0,4,6 numactl ${BINDING} \
  ./build/mr_bench_receiver -P 13337 -R mlx5_0:1,mlx5_2:1,mlx5_8:1,mlx5_4:1 \
  -m CUDA -s $msg_size  -T MR -n $nof_rails -r $ratio -k $pipe_stages -p 1
fi

if [ $test = "SINGLE" ]
then
  CUDA_VISIBLE_DEVICES=2,0,4,6 numactl ${BINDING} \
  ./build/mr_bench_receiver -P 13337 -R mlx5_0:1,mlx5_2:1,mlx5_8:1,mlx5_4:1 \
  -m CUDA -s $msg_size  -T SINGLE -n $nof_rails -r $ratio -k $pipe_stages -p 1
fi

