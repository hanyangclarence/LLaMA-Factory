
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export FORCE_TORCHRUN=1
export NPROC_PER_NODE=8
export MASTER_ADDR=z-chuang-ebd-master-0

llamafactory-cli train examples/train_full/qwen2_5_vl_full_sft_vela.yaml