#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('apt-get install -y libsndfile1')


# In[ ]:


import numpy as np
import pandas as pd
import cffi
import torchvision
import joblib
import librosa
import jieba

from sklearn.model_selection import train_test_split

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


get_ipython().system('git clone https://github.com/OpenNMT/OpenNMT-py')
get_ipython().system('mkdir /kaggle/working/data')
get_ipython().system('mkdir /kaggle/working/output')


# # Train & test split

# In[ ]:


train_count = 1000000
val_count = 5000
test_count = 5000


# In[ ]:


with open('/kaggle/input/zh-ru-parallel-corpus/un_ru.txt', mode='r', encoding='utf-8') as ru_in, open('/kaggle/working/data/sent_ru_train.txt', mode='w', encoding='utf-8') as ru_out_train,      open('/kaggle/working/data/sent_ru_val.txt', mode='w', encoding='utf-8') as ru_out_val, open('/kaggle/working/data/sent_ru_test.txt', mode='w', encoding='utf-8') as ru_out_test:
    
    for i in range(train_count):
        if i < train_count-1:
            ru_out_train.write(ru_in.readline().strip() + '\n')
        else:
            ru_out_train.write(ru_in.readline().strip())
        
    for i in range(val_count):
        if i < val_count-1:
            ru_out_val.write(ru_in.readline().strip() + '\n')
        else:
            ru_out_val.write(ru_in.readline().strip())
        
    for i in range(test_count):
        if i < test_count-1:
            ru_out_test.write(ru_in.readline().strip() + '\n')
        else:
            ru_out_test.write(ru_in.readline().strip())    


# In[ ]:


with open('/kaggle/input/zh-ru-parallel-corpus/un_zh.txt', mode='r', encoding='utf-8') as zh_in, open('/kaggle/working/data/sent_zh_train.txt', mode='w', encoding='utf-8') as zh_out_train,      open('/kaggle/working/data/sent_zh_val.txt', mode='w', encoding='utf-8') as zh_out_val, open('/kaggle/working/data/sent_zh_test.txt', mode='w', encoding='utf-8') as zh_out_test:
    
    for i in range(train_count):
        if i < train_count-1:
            zh_out_train.write(zh_in.readline().strip() + '\n')
        else:
            zh_out_train.write(zh_in.readline().strip())
        
    for i in range(val_count):
        if i < val_count-1:
            zh_out_val.write(zh_in.readline().strip() + '\n')
        else:
            zh_out_val.write(zh_in.readline().strip())
        
    for i in range(test_count):
        if i < test_count-1:
            zh_out_test.write(zh_in.readline().strip() + '\n')
        else:
            zh_out_test.write(zh_in.readline().strip())   


# # Make dictionary

# In[ ]:


get_ipython().system('python /kaggle/working/OpenNMT-py/tools/learn_bpe.py -i /kaggle/working/data/sent_zh_train.txt -o /kaggle/working/data/src.code -s 36000')


# In[ ]:


get_ipython().system('python /kaggle/working/OpenNMT-py/tools/learn_bpe.py -i /kaggle/working/data/sent_ru_train.txt -o /kaggle/working/data/trg.code -s 30000')


# # BPE

# In[ ]:


get_ipython().system('python /kaggle/working/OpenNMT-py/tools/apply_bpe.py -c /kaggle/working/data/src.code -i /kaggle/working/data/sent_zh_train.txt -o /kaggle/working/data/src-train-bpe.txt')

get_ipython().system('python /kaggle/working/OpenNMT-py/tools/apply_bpe.py -c /kaggle/working/data/src.code -i /kaggle/working/data/sent_zh_val.txt -o /kaggle/working/data/src-val-bpe.txt')

get_ipython().system('python /kaggle/working/OpenNMT-py/tools/apply_bpe.py -c /kaggle/working/data/src.code -i /kaggle/working/data/sent_zh_test.txt -o /kaggle/working/data/src-test-bpe.txt')

get_ipython().system('python /kaggle/working/OpenNMT-py/tools/apply_bpe.py -c /kaggle/working/data/trg.code -i /kaggle/working/data/sent_ru_train.txt -o /kaggle/working/data/tgt-train-bpe.txt')

get_ipython().system('python /kaggle/working/OpenNMT-py/tools/apply_bpe.py -c /kaggle/working/data/trg.code -i /kaggle/working/data/sent_ru_val.txt -o /kaggle/working/data/tgt-val-bpe.txt')


# # Preprocessing

# In[ ]:


get_ipython().system('python /kaggle/working/OpenNMT-py/preprocess.py -train_src /kaggle/working/data/src-train-bpe.txt -train_tgt /kaggle/working/data/tgt-train-bpe.txt     -valid_src /kaggle/working/data/src-val-bpe.txt -valid_tgt /kaggle/working/data/tgt-val-bpe.txt -save_data /kaggle/working/data/ondatr     -src_vocab_size 36000 -tgt_vocab_size 30000')


# # Train

# In[ ]:


#!python /kaggle/working/OpenNMT-py/train.py -data /kaggle/working/data/ondatr -save_model /kaggle/working/data/ondatr-trans -world_size 1 -gpu_rank 0 --train_steps 60000 -save_checkpoint_steps 20000 --keep_checkpoint 1


# In[ ]:


# !python /kaggle/working/OpenNMT-py/train.py -data /kaggle/working/data/ondatr -save_model /kaggle/working/data/ondatr-trans -layers 6 -rnn_size 512 \
#     -word_vec_size 512 -transformer_ff 2048 -heads 8 -encoder_type transformer -decoder_type transformer -position_encoding \
#     -train_steps 10000 -max_generator_batches 2 -dropout 0.1 -batch_size 4096 -batch_type tokens -normalization tokens \
#     -accum_count 2 -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 -max_grad_norm 0 \
#     -param_init 0 -param_init_glorot -label_smoothing 0.1 -valid_steps 1000 -save_checkpoint_steps 10000 -world_size 1 -gpu_rank 0 --keep_checkpoint 1


# # Translate

# In[ ]:


# !python /kaggle/working/OpenNMT-py/translate.py -model /kaggle/working/data/ondatr-trans_step_80000.pt -src /kaggle/input/opennmtpretrained/src-test-bpe.txt \
#     -output /kaggle/working/output/pred80.bpe -replace_unk -verbose


# In[ ]:


# !python /kaggle/working/OpenNMT-py/translate.py -model /kaggle/working/data/ondatr-trans_step_90000.pt -src /kaggle/input/opennmtpretrained/src-test-bpe.txt \
#     -output /kaggle/working/output/pred90.bpe -replace_unk -verbose


# In[ ]:


# !python /kaggle/working/OpenNMT-py/translate.py -model /kaggle/working/data/ondatr-trans_step_100000.pt -src /kaggle/input/opennmtpretrained/src-test-bpe.txt \
#     -output /kaggle/working/output/pred100.bpe -replace_unk -verbose


# In[ ]:


# !python /kaggle/working/OpenNMT-py/translate.py -model /kaggle/working/data/ondatr-trans_step_110000.pt -src /kaggle/input/opennmtpretrained/src-test-bpe.txt \
#     -output /kaggle/working/output/pred110.bpe -replace_unk -verbose


# In[ ]:


# !python /kaggle/working/OpenNMT-py/translate.py -model /kaggle/working/data/ondatr-trans_step_120000.pt -src /kaggle/input/opennmtpretrained/src-test-bpe.txt \
#     -output /kaggle/working/output/pred120.bpe -replace_unk -verbose


# In[ ]:


# !python /kaggle/working/OpenNMT-py/translate.py -model /kaggle/working/data/ondatr-trans_step_130000.pt -src /kaggle/input/opennmtpretrained/src-test-bpe.txt \
#     -output /kaggle/working/output/pred130.bpe -replace_unk -verbose


# In[ ]:


# !python /kaggle/working/OpenNMT-py/translate.py -model /kaggle/working/data/ondatr-trans_step_140000.pt -src /kaggle/input/opennmtpretrained/src-test-bpe.txt \
#     -output /kaggle/working/output/pred140.bpe -replace_unk -verbose


# In[ ]:


# !python /kaggle/working/OpenNMT-py/translate.py -model /kaggle/working/data/ondatr-trans_step_150000.pt -src /kaggle/input/opennmtpretrained/src-test-bpe.txt \
#     -output /kaggle/working/output/pred150.bpe -replace_unk -verbose


# # Detokenization

# In[ ]:


# !sed -i "s/@@ //g" /kaggle/working/output/pred80.bpe
# !sed -i "s/@@ //g" /kaggle/working/output/pred90.bpe
# !sed -i "s/@@ //g" /kaggle/working/output/pred100.bpe
# !sed -i "s/@@ //g" /kaggle/working/output/pred110.bpe
# !sed -i "s/@@ //g" /kaggle/working/output/pred120.bpe
# !sed -i "s/@@ //g" /kaggle/working/output/pred130.bpe
# !sed -i "s/@@ //g" /kaggle/working/output/pred140.bpe
# !sed -i "s/@@ //g" /kaggle/working/output/pred150.bpe


# # BLEU

# In[ ]:


# !perl /kaggle/working/OpenNMT-py/tools/multi-bleu.perl /kaggle/working/data/sent_ru_test.txt < /kaggle/working/output/pred80.bpe


# In[ ]:


# !perl /kaggle/working/OpenNMT-py/tools/multi-bleu.perl /kaggle/working/data/sent_ru_test.txt < /kaggle/working/output/pred90.bpe


# In[ ]:


# !perl /kaggle/working/OpenNMT-py/tools/multi-bleu.perl /kaggle/working/data/sent_ru_test.txt < /kaggle/working/output/pred100.bpe


# In[ ]:


# !perl /kaggle/working/OpenNMT-py/tools/multi-bleu.perl /kaggle/working/data/sent_ru_test.txt < /kaggle/working/output/pred110.bpe


# In[ ]:


# !perl /kaggle/working/OpenNMT-py/tools/multi-bleu.perl /kaggle/working/data/sent_ru_test.txt < /kaggle/working/output/pred120.bpe


# In[ ]:


# !perl /kaggle/working/OpenNMT-py/tools/multi-bleu.perl /kaggle/working/data/sent_ru_test.txt < /kaggle/working/output/pred130.bpe


# In[ ]:


# !perl /kaggle/working/OpenNMT-py/tools/multi-bleu.perl /kaggle/working/data/sent_ru_test.txt < /kaggle/working/output/pred140.bpe


# In[ ]:


# !perl /kaggle/working/OpenNMT-py/tools/multi-bleu.perl /kaggle/working/data/sent_ru_test.txt < /kaggle/working/output/pred150.bpe

