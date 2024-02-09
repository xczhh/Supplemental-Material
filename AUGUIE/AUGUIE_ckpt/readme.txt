1. We leverage UIE-base model from https://github.com/PaddlePaddle/PaddleNLP, then use 
few-shot fine-tune method and adapt it to domain-specific domain.   
2. Training data and test data are listed in finetune_train.txt and finetune_test for examples.
3. The combination of in-domain UIE and rule-based algorithm can sample more available and diverse data for training. 

python finetune.py  \
    --device gpu:1 \
    --logging_steps 5 \
    --save_steps 25 \
    --eval_steps 25 \
    --seed 42 \
    --model_name_or_path uie-base \
    --output_dir ./checkpoint/ \
    --train_path finetune_train.txt \
    --dev_path finetune_test.txt  \
    --max_seq_len 512  \
    --per_device_train_batch_size  8 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 10 \
    --learning_rate 1e-5 \
    --do_train \
    --do_eval \
    --do_export \
    --export_model_dir ./checkpoint/ \
    --overwrite_output_dir \
    --disable_tqdm True \
    --metric_for_best_model eval_f1 \
    --load_best_model_at_end  True \
    --save_total_limit 1

