#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('git clone https://github.com/NVIDIA/apex')
get_ipython().run_line_magic('cd', 'apex')
get_ipython().system('pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./')


# In[ ]:


get_ipython().system('pip install pycocotools')


# In[ ]:


get_ipython().run_line_magic('cd', '')


# In[ ]:


get_ipython().run_line_magic('cd', '/.')


# In[ ]:


get_ipython().system('git clone https://github.com/swetha2410/Visual-Question-Answering.git')


# In[ ]:


get_ipython().run_line_magic('cd', '')


# In[ ]:


get_ipython().run_line_magic('cd', '/')


# In[ ]:


get_ipython().system('ls')


# In[ ]:


get_ipython().run_line_magic('cd', 'kaggle')


# In[ ]:


get_ipython().run_line_magic('cd', 'working')


# In[ ]:


get_ipython().run_line_magic('mkdir', 'processed')


# In[ ]:


get_ipython().run_line_magic('mkdir', 'results_log')


# In[ ]:


get_ipython().run_line_magic('cd', '')


# In[ ]:


get_ipython().run_line_magic('cd', '/')


# In[ ]:


get_ipython().run_line_magic('cd', './Visual-Question-Answering')


# In[ ]:


get_ipython().system('python3 prepare_data.py --balanced_real_images -s train \\-a ../kaggle/input/vqa-data/v2_mscoco_train2014_annotations.json \\-q ../kaggle/input/vqa-data/v2_OpenEnded_mscoco_train2014_questions.json \\-o ../kaggle/working/processed/vqa_train2014.txt \\-v ../kaggle/working/processed/vocab_count_5_K_1000.pickle -c 5 -K 1000')


# In[ ]:


get_ipython().system('python3 prepare_data.py --balanced_real_images -s val \\-a ../kaggle/input/vqa-data/v2_mscoco_val2014_annotations.json \\-q ../kaggle/input/vqa-data/v2_OpenEnded_mscoco_val2014_questions.json \\-o ../kaggle/working/processed/vqa_val2014.txt ')


# In[ ]:


get_ipython().system('python main.py --mode train --expt_name K_1000_Attn --expt_dir /../kaggle/working/results_log --train_img ../kaggle/input/coco2014/train2014/train2014 --train_file /../kaggle/working/processed/vqa_train2014.txt --val_img ../kaggle/input/coco2014/val2014/val2014 --val_file ../kaggle/working/processed/vqa_val2014.txt--vocab_file ../kaggle/working/processed/vocab_count_5_K_1000.pickle --save_interval 1000 --log_interval 100 --gpu_id 0 --num_epochs 2 --batch_size 64 -K 1000 -lr 1e-4 --opt_lvl 1 --num_workers 6 --run_name O1_wrk_6_bs_160 --model attention ')


# In[ ]:


get_ipython().run_line_magic('cd', '..')


# In[ ]:


get_ipython().system('ls')


# In[ ]:


get_ipython().run_line_magic('rm', '-R Visual-Question-Answering')


# In[ ]:


import torch
torch.device('cuda:0')


# In[ ]:



CUDA_VISIBLE_DEVICES = torch.device('cuda')


# In[ ]:


torch.cuda.set_device(0)


# In[ ]:





# In[ ]:




