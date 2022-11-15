**Training**

```
python train.py --output_dir wbase --num_train_epochs 100 --warmup_steps 100 --lr_scheduler_type cosine --learning_rate 1e-4  --optim adamw_torch --adam_eps 1e-6 --model_name base --per_device_train_batch_size 16 --do_train --save_total_limit 2 --evaluation_strategy 'epoch' --logging_strategy 'steps' --logging_step 20 --weight_decay 1e-2 --dataloader_num_workers 8 --save_strategy 'epoch' --load_best_model_at_end True --metric_for_best_model 'eval_IoU' --fp16 --use_external False --gradient_accumulation_steps 1
```

### Pretrained data
```
find . -maxdepth 1 -name '*.wav' -exec bash -c "ffmpeg -i {} -ac 1 -ar 16000 /home/thanh/shared_disk/thanh/data/{}"  \;
find . -maxdepth 1 -name '*.wav' -exec bash -c "ffmpeg -i {} -f segment -segment_time 30 -c copy test_chunk/{}%03d.wav" \;
```


- https://docs.google.com/spreadsheets/d/1z48UKleAbqFd0leO8_fAkOHMvXWIpFpt30QNncX4cAQ/edit?usp=sharing


### Installation
```
pip install torch==1.12.1+cu113 torchaudio===0.12.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html

pip install jiwer pyopenjtalk==0.3.0 pytorch-lightning==1.7.7 evaluate==0.2.2 pandas termcolor h5py matplotlib ipympl scikit-learn ffmpeg albumentations sentencepiece
```

### data problem
1. s > e
2. s < 0, e < 0
3. multiple words in word["d"]
4. word mapping: ko -> không

### augment
1. sub audios
2. audio = audio.flip()
