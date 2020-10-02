#!/usr/bin/env python
# coding: utf-8

# # Install and Import Libraries

# In[ ]:


#!pip install /kaggle/input/pytorch-fairseq/fairseq-0.9.0/ > /dev/null
#! pip install /kaggle/input/pytorchtransformers/transformers-2.5.1 > /dev/null


# In[ ]:


## Basic Library
import os, time, sys, gc
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import json

## Pytorch
import torch
import pytorch_transformers


for dirname, _, filenames in os.walk('/kaggle/input/xlnet-pretrained-models-pytorch'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Global Parameters

# In[ ]:


INPUTDIR = '/kaggle/input/tweet-sentiment-extraction/'


# # Define Functions

# # Import Data

# In[ ]:


get_ipython().run_cell_magic('time', '', "train_df = pd.read_csv(f'{INPUTDIR}/train.csv')\ntest_df = pd.read_csv(f'{INPUTDIR}/test.csv')\nprint('train shape is {}, and test shape is {}'.format(train_df.shape, test_df.shape))")


# In[ ]:


train = np.array(train_df)
test = np.array(test_df)

get_ipython().system('mkdir -p data')


# In[ ]:


"""
Prepare training data in QA-compatible format
"""

# Adpated from https://www.kaggle.com/cheongwoongkang/roberta-baseline-starter-simple-postprocessing
def find_all(input_str, search_str):
    l1 = []
    length = len(input_str)
    index = 0
    while index < length:
        i = input_str.find(search_str, index)
        if i == -1:
            return l1
        l1.append(i)
        index = i + 1
    return l1

def do_qa_train(train):

    output = {}
    output['version'] = 'v1.0'
    output['data'] = []
    
    for line in tqdm(train):
        context = line[1]
        paragraphs = []
        qas = []
        question = line[-1]
        qid = line[0]
        answers = []
        answer = line[2]
        if type(answer) != str or type(context) != str or type(question) != str:
            print(context, type(context))
            print(answer, type(answer))
            print(question, type(question))
            continue
        answer_starts = find_all(context, answer)
        for answer_start in answer_starts:
            answers.append({'answer_start': answer_start, 'text': answer.lower()})
            break
        qas.append({'question': question, 'id': qid, 'is_impossible': False, 'answers': answers})

        paragraphs.append({'context': context.lower(), 'qas': qas})
        output['data'].append({'title': 'None', 'paragraphs': paragraphs})
        
    return output

qa_train = do_qa_train(train)

with open('data/train.json', 'w') as outfile:
    json.dump(qa_train, outfile)


# In[ ]:


"""
Prepare testing data in QA-compatible format
"""

output = {}
output['version'] = 'v1.0'
output['data'] = []

def do_qa_test(test):
    paragraphs = []
    for line in tqdm(test):
        paragraphs = []
        context = line[1]
        qas = []
        question = line[-1]
        qid = line[0]
        if type(context) != str or type(question) != str:
            print(context, type(context))
            print(answer, type(answer))
            print(question, type(question))
            continue
        answers = []
        answers.append({'answer_start': 1000000, 'text': '__None__'})
        qas.append({'question': question, 'id': qid, 'is_impossible': False, 'answers': answers})

        paragraphs.append({'context': context.lower(), 'qas': qas})
        output['data'].append({'title': 'None', 'paragraphs': paragraphs})
    return output

qa_test = do_qa_test(test)

with open('data/test.json', 'w') as outfile:
    json.dump(qa_test, outfile)


# In[ ]:


get_ipython().system('mkdir -p results_roberta_large')


# In[ ]:


import sentencepiece


# In[ ]:


get_ipython().system('python -m torch.distributed.launch --nproc_per_node=1 /kaggle/input/bert-squad/BERT-SQuAD-master/training/run_squad.py --model_type xlnet --model_name_or_path /kaggle/input/xlnet-pretrained-models-pytorch/xlnet-large-cased-pytorch_model.bin --tokenizer_name /kaggle/input/xlnet-pretrained-models-pytorch/xlnet-large-cased-spiece.model --config_name /kaggle/input/xlnet-pretrained-models-pytorch/xlnet-large-cased-config.json --do_train --do_eval --do_lower_case --train_file ./data/train.json --predict_file ./data/test.json --learning_rate 3e-5 --num_train_epochs 2 --max_seq_length 192 --doc_stride 64 --output_dir ./results_roberta_large/ --per_gpu_eval_batch_size=10   --per_gpu_train_batch_size=10   --save_steps=100000')


# # Submission

# In[ ]:


# Copy predictions to submission file.
predictions = json.load(open('results_roberta_large/predictions_.json', 'r'))
submission = pd.read_csv(open('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv', 'r'))
for i in range(len(submission)):
    id_ = submission['textID'][i]
    if test_df['sentiment'][i] == 'neutral': # neutral postprocessing
        submission.loc[i, 'selected_text'] = test_df['text'][i]
    else:
        submission.loc[i, 'selected_text'] = predictions[id_]


# In[ ]:


submission.head()


# In[ ]:


# Save the submission file.
submission.to_csv('submission.csv', index=False)

