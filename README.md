**Training**

```
python train.py --output_dir wbase --num_train_epochs 100 --warmup_steps 100 --lr_scheduler_type cosine --learning_rate 1e-4 --model_name base --per_device_train_batch_size 16 --do_train --save_total_limit 2 --evaluation_strategy 'epoch' --logging_strategy 'steps' --logging_step 20 --weight_decay 1e-2 --dataloader_num_workers 32 --save_strategy 'epoch' --load_best_model_at_end True --metric_for_best_model 'eval_IoU' --fp16 --resume old_checkpoint.ckpt 
```

### Pretrained data
```
find . -maxdepth 1 -name '*.wav' -exec bash -c "ffmpeg -i {} -ac 1 -ar 16000 /home/thanh/shared_disk/thanh/data/{}"  \;
find . -maxdepth 1 -name '*.wav' -exec bash -c "ffmpeg -i {} -f segment -segment_time 30 -c copy test_chunk/{}%03d.wav" \;
```


- https://docs.google.com/spreadsheets/d/1z48UKleAbqFd0leO8_fAkOHMvXWIpFpt30QNncX4cAQ/edit?usp=sharing


```
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 python train_ctc.py \
    --dataset_name="custom" \
	--custom_train_dataset_file="/home/thanh/zalocl2022/data/wav2vec2_train.csv" \
	--custom_val_dataset_file="/home/thanh/zalocl2022/data/wav2vec2_val.csv" \
	--model_name_or_path="facebook/wav2vec2-large-xlsr-53" \
	--dataset_config_name="tr" \
	--output_dir="./wav2vec2-common_voice-tr-demo" \
	--overwrite_output_dir \
	--num_train_epochs="15" \
	--per_device_train_batch_size="16" \
	--gradient_accumulation_steps="2" \
	--learning_rate="3e-4" \
	--warmup_steps="500" \
	--evaluation_strategy="steps" \
	--save_steps="400" \
	--eval_steps="100" \
	--layerdrop="0.0" \
	--save_total_limit="3" \
	--gradient_checkpointing \
	--chars_to_ignore , ? . ! - \; \: \" “ % ‘ ” � \
	--fp16 \
	--group_by_length \
	--do_train --do_eval \
    --preprocessing_num_workers 16
```