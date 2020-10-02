#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('git clone https://github.com/zhiqwang/fairseq.git')


# In[ ]:


cd fairseq


# In[ ]:


get_ipython().system('git checkout text-recognition')


# In[ ]:


mkdir data-bin


# In[ ]:


cd data-bin/


# In[ ]:


get_ipython().system('ln -s ../../../input/number_sequence/number_sequence number_sequence')


# In[ ]:


cd ..


# In[ ]:


rm -r checkpoints/


# In[ ]:


get_ipython().system("python -m train data-bin/number_sequence     --task image_captioning     --arch decoder_attention     --decoder-layers 2     --batch-size 16     --max-epoch 50     --criterion cross_entropy     --num-workers 4     --optimizer adam     --adam-eps 1e-04     --lr 0.001     --min-lr 1e-09     --adam-betas '(0.9, 0.98)'     --clip-norm 0.0     --weight-decay 0.0     --no-token-crf     --save-interval 5 #     --no-token-rnn \\")
#     --no-token-positional-embeddings \


# In[ ]:


rm fairseq/models/image_captioning.py


# In[ ]:


get_ipython().run_cell_magic('writefile', 'fairseq/models/image_captioning.py', '')


# In[ ]:


get_ipython().system('git status')


# In[ ]:


get_ipython().system('python -m interactive_image_captioning data-bin/number_sequence     --arch decoder_crnn     --path checkpoints/checkpoint_best.pt:checkpoints/checkpoint30.pt:checkpoints/checkpoint35.pt:checkpoints/checkpoint40.pt:checkpoints/checkpoint45.pt:checkpoints/checkpoint50.pt     --task image_captioning     --num-workers 4     --criterion ctc_loss     --batch-size 16     --num-workers 4     --gen-subset valid')

