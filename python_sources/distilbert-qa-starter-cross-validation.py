#!/usr/bin/env python
# coding: utf-8

# ## Problem Formulation
# I formulate this task as an extractive question answering problem, such as SQuAD.  
# Given a question and context, the model is trained to find the answer spans in the context.
# 
# Therefore, I use sentiment as question, text as context, selected_text as answer.
# - Question: sentiment
# - Context: text
# - Answer: selected_text

# ## Hyperparameters & Options 

# In[ ]:


# Hyperparameters
batch_size = 16 # batch size
lr = 5e-5 # learning rate
epochs = 2 # number of epochs
max_seq_len = 128 # max sequence length
doc_stride = 64 # document stride

# Options
cross_validation = True # whether to use cross-validation
K = 2 # number of CV splits
post_processing = True # whether to use post-processing


# ## Import Packages

# In[ ]:


import numpy as np
import pandas as pd
import json
import os


# ## Data Preprocessing
# ### Load Data

# In[ ]:


pd_train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
pd_test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')


# In[ ]:


np_train = np.array(pd_train)
np_test = np.array(pd_test)


# ### K-fold Split
# Split the data into K folds for cross validation. Use the fixed random seed for reproducibility.

# In[ ]:


# Given a data size, return the train/valid indicies for K splits.
def split_data(num_examples, K):
    np.random.seed(0)
    idx = np.arange(num_examples)
    np.random.shuffle(idx)
    
    boundary = num_examples // K
    splits = [{} for _ in range(K)]
    for i in range(K):
        splits[i]['valid_idx'] = idx[i*boundary:(i+1)*boundary]
        splits[i]['train_idx'] = np.concatenate((idx[:i*boundary], idx[(i+1)*boundary:]))

        valid = np_train[splits[i]['valid_idx']]
        d = {'neutral':0, 'positive':0, 'negative':0}
        for line in valid:
            d[line[-1]] += 1
        print(d)
        
    return splits


# In[ ]:


splits = split_data(len(np_train), K)


# ### Convert Data to SQuAD-style
# In this part, I convert the data into SQuAD-style.  
# Since I think most of the errors in the dataset are irreducible, I do not use additional preprocessing methods to handle them.

# In[ ]:


# Convert data to SQuAD-style
def convert_data(data, directory, filename):
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
    
    output = {}
    output['version'] = 'v1.0'
    output['data'] = []
    
    for line in data:
        paragraphs = []
        context = line[1]
        qas = []
        question = line[-1]
        qid = line[0]
        answers = []
        answer = line[2]
        if type(context) != str:
            print(context, type(context))
            continue
        answer_starts = find_all(context, answer)
        for answer_start in answer_starts:
            answers.append({'answer_start': answer_start, 'text': answer})
        qas.append({'question': '[MASK]', 'id': qid, 'is_impossible': False, 'answers': answers})

        paragraphs.append({'context': context, 'qas': qas})
        output['data'].append({'title': 'None', 'paragraphs': paragraphs})

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(os.path.join(directory, filename), 'w') as outfile:
        json.dump(output, outfile)


# In[ ]:


# convert k-fold train data
for i, split in enumerate(splits):
    data = np_train[split['train_idx']]
    directory = 'split_' + str(i+1)
    filename = 'train.json'
    convert_data(data, directory, filename)


# In[ ]:


# convert original train/test data
data = np_train
directory = 'original'
filename = 'train.json'
convert_data(data, directory, filename)

data = np_test
filename = 'test.json'
convert_data(data, directory, filename)


# ## Finetuning
# Install the pytorch-transformers package (v2.5.1) of [huggingface](https://github.com/huggingface/transformers).

# In[ ]:


get_ipython().system('cd /kaggle/input/pytorchtransformers/transformers-2.5.1; pip install .')


# ### Cross-Validation
# Finetune QA models for cross-validation.

# In[ ]:


def run_script(train_file, predict_file, batch_size=16, lr=5e-5, epochs=2, max_seq_len=128, doc_stride=64):
    get_ipython().system('python /kaggle/input/pytorchtransformers/transformers-2.5.1/examples/run_squad.py     --model_type distilbert     --model_name_or_path distilbert-base-uncased     --cache_dir /kaggle/input/cached-distilbert-base-uncased/cache     --do_lower_case     --do_train     --do_eval     --train_file=$train_file     --predict_file=$predict_file     --overwrite_cache     --learning_rate=$lr     --num_train_epochs=$epochs     --max_seq_length=$max_seq_len     --doc_stride=$doc_stride     --output_dir ./results     --overwrite_output_dir     --per_gpu_eval_batch_size=$batch_size     --per_gpu_train_batch_size=$batch_size     --save_steps=100000')


# In[ ]:


get_ipython().system('mkdir results')


# In[ ]:


if cross_validation:
    for i in range(1, K+1):
        train_file = "split_" + str(i) + "/train.json"
        predict_file = "original/train.json"
        run_script(train_file, predict_file, batch_size, lr, epochs, max_seq_len, doc_stride)
        get_ipython().system('mv "results/predictions_.json" "results/predictions_"$i".json"')


# ### Evaluation
# Calculate train/valid scores.

# In[ ]:


def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


# In[ ]:


def evaluate(splits, np_train, post_processing=False):
    K = len(splits)
    predictions = [json.load(open('results/predictions_' + str(i+1) + '.json', 'r')) for i in range(K)]

    train_score = [{'neutral':[], 'positive':[], 'negative':[], 'total':[]} for _ in range(K+1)]
    valid_score = [{'neutral':[], 'positive':[], 'negative':[], 'total':[]} for _ in range(K+1)]

    for train_idx, line in enumerate(np_train):
        text_id = line[0]
        text = line[1]
        answer = line[2]
        sentiment = line[-1]

        if type(text) != str:
            continue

        for i, prediction in enumerate(predictions):
            if text_id not in prediction:
                print('key error:', text_id)
                continue
            else:
                if post_processing and (sentiment == 'neutral' or len(text.split()) <= 0): # post-processing
                    score = jaccard(answer, text)
                else:
                    score = jaccard(answer, prediction[text_id])

                if train_idx in splits[i]['valid_idx']:
                    valid_score[i][sentiment].append(score)
                    valid_score[i]['total'].append(score)
                    valid_score[K][sentiment].append(score)
                    valid_score[K]['total'].append(score)

                else:
                    train_score[i][sentiment].append(score)
                    train_score[i]['total'].append(score)
                    train_score[K][sentiment].append(score)
                    train_score[K]['total'].append(score)

    for i, score_dict in enumerate([train_score, valid_score]):
        if i == 0:
            print('train score \n')
        else:
            print('valid score \n')
        for j in range(K+1):
            for sentiment in ['neutral', 'positive', 'negative', 'total']:
                score = np.array(score_dict[j][sentiment])
                if j < K:
                    print('split', j+1)
                else:
                    print('all data')
                print(sentiment + ' - ' + str(len(score)) + ' examples, average score: ' + str(score.mean()))
            print()


# In[ ]:


if cross_validation:
    evaluate(splits, np_train, post_processing)


# ### Test
# Finetune a model for the test.

# In[ ]:


train_file = "original/train.json"
predict_file = "original/test.json"
run_script(train_file, predict_file, batch_size, lr, epochs, max_seq_len, doc_stride)
get_ipython().system('mv results/predictions_.json results/test_predictions.json')


# ## Submission

# In[ ]:


# Copy predictions to submission file.
predictions = json.load(open('results/test_predictions.json', 'r'))
submission = pd.read_csv(open('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv', 'r'))
for i in range(len(submission)):
    id_ = submission['textID'][i]
    if post_processing and (pd_test['sentiment'][i] == 'neutral' or len(pd_test['text'][i].split()) <= 0): # post-processing
        submission.loc[i, 'selected_text'] = pd_test['text'][i]
    else:
        submission.loc[i, 'selected_text'] = predictions[id_]


# In[ ]:


submission.head()


# In[ ]:


# Save the submission file.
submission.to_csv('submission.csv', index=False)

