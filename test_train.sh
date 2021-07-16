rm -rf models tensorboard $HOME/.cache/huggingface/datasets

python run_clm.py \
    --train_file ./fb_data/sanitized/fb_train.json \
    --validation_file ./fb_data/sanitized/fb_test.json \
    --block_size 128 \
    --output_dir ./models/test \
    --overwrite_cache true \
    --do_train \
    --do_eval \
    --max_steps 10 \
    --report_to none \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4