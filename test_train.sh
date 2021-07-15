rm -rf models tensorboard $HOME/.cache/huggingface/datasets

python run_clm.py \
    --train_file ./fb_data/small_train.json \
    --validation_file ./fb_data/small_test.json \
    --block_size 50 \
    --output_dir ./models \
    --overwrite_cache true \
    --do_train \
    --do_eval