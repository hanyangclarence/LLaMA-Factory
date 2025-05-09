# Run training

Format training data:
```
python src/llamafactory/data/reformat_data.py
```
You can also change the load paths to generate validationd data 



On aimos, run
```
bash launch_job_auto.sh
```
which will automatically and continuously submit job and resume training

On vela, run
```
bash launch_job_vela.sh
```


# Run eval
```
CUDA_VISIBLE_DEVICES=7 FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/qwen2_5_vl_full_sft_vela_eval.yaml report_to=tensorboard max_samples=xxx resume_from_checkpoint=xxx
```