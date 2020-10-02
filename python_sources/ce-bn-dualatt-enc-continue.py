# %% [code]
import os

os.system("python /kaggle/input/captioncodepy3/train.py   \
    --checkpoint_path /kaggle/working/model_weights  \
    --start_from ../input/ce-bn-dualatt-enc/model_weights  \
    --label_smoothing 0.2\
    --max_epochs 16  \
    --save_checkpoint_every 1255 \
    --losses_log_every 941  \
    --batch_size 30  \
    --save_history_ckpt 0 \
    --learning_rate 5e-4 \
    --num_layers 6 \
    --input_encoding_size 512 \
    --rnn_size 2048")