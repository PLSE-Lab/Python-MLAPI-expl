#!/usr/bin/env python
# coding: utf-8

# ## Albert XL Inference. Score 0.703
# 
# I made this notebook by basically copying and pasting a lot of different versions of other notebooks from @cheongwoongkang https://www.kaggle.com/cheongwoongkang/roberta-baseline-starter-simple-postprocessing, so please upvote that and refer to it for the training part. I however made the notebook only do inference for the Alberta XL model.
# 
# # Parameters used for training:
# LR: 3e-5
# Batch size: 20
# Doc stride: 64 
# Max Seq: 192
# Epochs: 3 
# 
# 
# The model was trained locally due to time limitations on kaggle, albert xlarge is not able to be trained on those parameters on kaggle kernels. However if you want to train locally on your computer you can checkout the notebook mentioned above and change the lines in the run_squad.py agruments that say roberta and change the model type to albert and the model path to one of those albert models https://huggingface.co/transformers/pretrained_models.html. In my case I used `albert-xlarge-v2`

# In[ ]:


import numpy as np 
import pandas as pd 
import json


# In[ ]:


pd_train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
pd_test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')


# In[ ]:


train = np.array(pd_train)
test = np.array(pd_test)


# In[ ]:


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


# In[ ]:


get_ipython().system('mkdir data')


# In[ ]:


# Convert training data

output = {}
output['version'] = 'v1.0'
output['data'] = []

for line in train:
    paragraphs = []
    
    context = line[1]
    
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
        answers.append({'answer_start': answer_start, 'text': answer})
    qas.append({'question': question, 'id': qid, 'is_impossible': False, 'answers': answers})
    
    paragraphs.append({'context': context, 'qas': qas})
    output['data'].append({'title': 'None', 'paragraphs': paragraphs})

with open('data/train.json', 'w') as outfile:
    json.dump(output, outfile)


# In[ ]:


# Convert test data

output = {}
output['version'] = 'v1.0'
output['data'] = []

for line in test:
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
    
    paragraphs.append({'context': context, 'qas': qas})
    output['data'].append({'title': 'None', 'paragraphs': paragraphs})

with open('data/test.json', 'w') as outfile:
    json.dump(output, outfile)


# In[ ]:


get_ipython().system('cd /kaggle/input/pytorchtransformers/transformers-2.5.1; pip install .')


# In[ ]:


get_ipython().system('python /kaggle/input/pytorchtransformers/transformers-2.5.1/examples/run_squad.py --model_type albert --model_name_or_path "../input/albertqa/goodalbert" --do_lower_case --do_eval --data_dir ./data --predict_file test.json --output_dir results_roberta_large --per_gpu_eval_batch_size=20')


# In[ ]:


predictions = json.load(open('results_roberta_large/predictions_.json', 'r'))
submission = pd.read_csv(open('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv', 'r'))
for i in range(len(submission)):
    id_ = submission['textID'][i]
    if pd_test['sentiment'][i] == 'neutral': # neutral postprocessing
        submission.loc[i, 'selected_text'] = pd_test['text'][i]
    else:
        submission.loc[i, 'selected_text'] = predictions[id_]


# In[ ]:


submission.to_csv('submission.csv', index=False)

