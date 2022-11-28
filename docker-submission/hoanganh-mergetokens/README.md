**Training**

```
python train.py --output_dir wbase --num_train_epochs 100 --warmup_steps 100 --lr_scheduler_type cosine --learning_rate 5e-5 --model_name base --per_device_train_batch_size 16 --do_train --save_total_limit 2 --evaluation_strategy 'epoch' --logging_strategy 'steps' --logging_step 20 --weight_decay 1e-2 --resume old_checkpoint.ckpt --dataloader_num_workers 32 --save_strategy 'epoch' --load_best_model_at_end True --metric_for_best_model 'eval_IoU'
```