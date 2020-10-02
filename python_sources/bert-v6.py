#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().system('ls /kaggle/input/jigsaw-bert-batch10-length120-epochs1/bert-bat10-len120-ep1/bert-bat10-len120-ep1/Bert')


# In[ ]:


get_ipython().system('ls /kaggle/input/bertbase-cased/cased_l-12_h-768_a-12/cased_L-12_H-768_A-12')


# In[ ]:


get_ipython().system('mkdir /kaggle/working/bert_output')


# In[ ]:


get_ipython().system('python /kaggle/input/jigsaw-bert-batch10-length120-epochs1/bert-bat10-len120-ep1/bert-bat10-len120-ep1/Bert/bert-master/run_classifier.py --task_name=cola --do_predict=true --data_dir=/kaggle/input/jigsaw-bert-batch10-length120-epochs1/bert-bat10-len120-ep1/bert-bat10-len120-ep1/Bert/bert_data --vocab_file=/kaggle/input/bertbase-cased/cased_l-12_h-768_a-12/cased_L-12_H-768_A-12/vocab.txt --bert_config_file=/kaggle/input/bertbase-cased/cased_l-12_h-768_a-12/cased_L-12_H-768_A-12/bert_config.json --init_checkpoint=/kaggle/input/jigsaw-bert-batch10-length120-epochs1/bert-bat10-len120-ep1/bert-bat10-len120-ep1/Bert/bert_output/model.ckpt-162438 --max_seq_length=120 --save_checkpoints_steps=99999999 --output_dir=/kaggle/working/bert_output --do_lower_case=False')


# In[ ]:


get_ipython().system('ls /kaggle/working/')


# In[ ]:


sample_submission = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv')
predictions = pd.read_csv('//kaggle/working/bert_output/test_results.tsv', header=None, sep='\t')

submission = pd.concat([sample_submission.iloc[:,0], predictions.iloc[:,1]], axis=1)
submission.columns = ['id','prediction']
submission.to_csv('submission.csv', index=False, header=True)


# In[ ]:


submission.head()


# In[ ]:




