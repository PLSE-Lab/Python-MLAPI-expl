#!/usr/bin/env python
# coding: utf-8

# ## DATA

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
from transformers import BertTokenizer, BertModel
import re
import pandas as pd


# In[ ]:


sample_submission = pd.read_csv("../input/google-quest-challenge/sample_submission.csv")
test = pd.read_csv("../input/google-quest-challenge/test.csv")
train = pd.read_csv("../input/google-quest-challenge/train.csv")


# In[ ]:


train.columns.values


# In[ ]:


output_columns = train.columns.values[11:]
input_columns = train.columns.values[[1, 2, 5]]


# In[ ]:


question_output_columns = [col for col in output_columns                            if 'question' in col]
answer_output_colmns = [col for col in output_columns 
                        if col not in question_output_columns]


# In[ ]:


len(question_output_columns)


# In[ ]:


tokenizer = BertTokenizer.from_pretrained('../input/huggingfacetransformermodels/model_classes/bert/bert-base-uncased-tokenizer')


# In[ ]:


max_length_map = {'question_title': 32,
                   'question_body': 512,
                   'answer': 512
                   }


# In[ ]:


def txt_re(txt):
    txt = txt.strip()
    txt = re.sub('https?.*$', '', txt)
    txt = re.sub('https?.*\s', '', txt)
    txt = re.sub('\n+', ' ', txt)
    txt = re.sub('\r+', ' ', txt)
    txt = re.sub('\t+', ' ', txt)
    txt = re.sub('&gt;', '>', txt)
    txt = re.sub('&lt;', '<', txt)
    txt = re.sub('&amp;', '&', txt)
    txt = re.sub('&quot;', '\"', txt)
    return txt


# In[ ]:


def get_input(txt, pair_txt, tokenizer, max_length):
    txt = txt_re(txt)
    txt = tokenizer.encode_plus(txt, pair_txt, add_special_tokens=True,max_length=max_length,                           pad_to_max_length='right')
    input_ids = txt['input_ids']
    segment_masks = txt['token_type_ids']
    input_masks = txt['attention_mask']
    
    return input_ids, segment_masks, input_masks


# In[ ]:


def computer_input_array(df):
    # t_input_ids, t_segment_masks, t_input_masks = [], [], []
    q_input_ids, q_segment_masks, q_input_masks = [], [], []
    a_input_ids, a_segment_masks, a_input_masks = [], [], []
    for _, instance in df[input_columns].iterrows():
        title, question, answer = instance.question_title, instance.question_body, instance.answer
        """
        input_ids, segment_masks, input_masks = get_input(title, tokenizer, max_length_map['question_title'])
        t_input_ids.append(input_ids)
        t_segment_masks.append(segment_masks)
        t_input_masks.append(input_masks)
        """
    
        input_ids, segment_masks, input_masks = get_input(title, question, tokenizer, max_length_map['question_body'])
        q_input_ids.append(input_ids)
        q_segment_masks.append(segment_masks)
        q_input_masks.append(input_masks)
    
        input_ids, segment_masks, input_masks = get_input(answer, None, tokenizer, max_length_map['answer'])
        a_input_ids.append(input_ids)
        a_segment_masks.append(segment_masks)
        a_input_masks.append(input_masks)
    # title = [[input_id, segment_mask, input_mask] for input_id, segment_mask, input_mask in \
              # zip(t_input_ids, t_segment_masks, t_input_masks)]
    question = [[input_id, segment_mask, input_mask] for input_id, segment_mask, input_mask in               zip(q_input_ids, q_segment_masks, q_input_masks)]
    answer = [[input_id, segment_mask, input_mask] for input_id, segment_mask, input_mask in               zip(a_input_ids, a_segment_masks, a_input_masks)]
    
    return question, answer


# In[ ]:


# title_train,question_train, answer_train = computer_input_array(train)
question_train, answer_train = computer_input_array(train)
# title_test, question_test, answer_test = computer_input_array(test)
question_test, answer_test = computer_input_array(test)


# In[ ]:


answer_test


# In[ ]:


labels = train[output_columns].values.tolist()


# In[ ]:


# train_dict = {'title': title_train, 'question': question_train, 'answer': answer_train, 'label': labels}
train_dict = {'question': question_train, 'answer': answer_train, 'label': labels}
# test_dict = {'title': title_test, 'question': question_test, 'answer': answer_test}
test_dict = {'question': question_test, 'answer': answer_test}


# In[ ]:


ls


# In[ ]:


import os
os.mkdir('./data')


# In[ ]:


os.mkdir('./model')


# In[ ]:


import torch
torch.save(train_dict, './data/train_data.t7')
torch.save(test_dict, './data/test_data.t7')


# ## MODEL

# In[ ]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from scipy.stats import spearmanr


# In[ ]:


class Model_v1(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('../input/huggingfacetransformermodels/model_classes/bert/bert-base-uncased-pytorch-model')
        self.dropout = nn.Dropout(0.2)
        self.pool = nn.AvgPool2d((512, 1))

        self.output = nn.Linear(768 * 2, 30)
        
    def forward(self, q_inputs, q_input_masks, q_segment_masks,                 a_inputs, a_input_masks, a_segment_masks):

        q_outputs = self.bert(q_inputs, attention_mask=q_input_masks, token_type_ids=q_segment_masks)
        q_x = q_outputs[0]
        q_x = self.dropout(q_x)
        q_x = q_x.unsqueeze(1)
        q_x = self.pool(q_x)
        q_x = q_x.squeeze(1).squeeze(1)
        
        a_outputs = self.bert(a_inputs, attention_mask=a_input_masks, token_type_ids=a_segment_masks)
        a_x = a_outputs[0]
        a_x = self.dropout(a_x)
        a_x = a_x.unsqueeze(1)
        a_x = self.pool(a_x)
        a_x = a_x.squeeze(1).squeeze(1)
        
        t_q_a = torch.cat((q_x, a_x), -1)
        output = self.output(t_q_a)
        
        x = torch.sigmoid(output)
        return x


# In[ ]:


class Model_v2(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('../input/huggingfacetransformermodels/model_classes/bert/bert-base-uncased-pytorch-model')
        self.dropout = nn.Dropout(0.2)
        self.pool = nn.AvgPool2d((512, 1))

        self.output_q = nn.Linear(768, 21)
        self.output_a = nn.Linear(768, 9)    
        
    def forward(self, q_inputs, q_input_masks, q_segment_masks,                 a_inputs, a_input_masks, a_segment_masks):
        """
        t_outputs = self.bert(t_inputs, attention_mask=t_input_masks, token_type_ids=t_segment_masks)
        t_x = t_outputs[0]
        t_x = self.dropout(t_x)
        t_x = t_x.unsqueeze(1)
        t_x = self.pool_t(t_x)
        t_x = t_x.squeeze(1).squeeze(1)
        """
        q_outputs = self.bert(q_inputs, attention_mask=q_input_masks, token_type_ids=q_segment_masks)
        q_x = q_outputs[0]
        q_x = self.dropout(q_x)
        q_x = q_x.unsqueeze(1)
        q_x = self.pool(q_x)
        q_x = q_x.squeeze(1).squeeze(1)
        
        a_outputs = self.bert(a_inputs, attention_mask=a_input_masks, token_type_ids=a_segment_masks)
        a_x = a_outputs[0]
        a_x = self.dropout(a_x)
        a_x = a_x.unsqueeze(1)
        a_x = self.pool(a_x)
        a_x = a_x.squeeze(1).squeeze(1)
        
        output_q = self.output_q(q_x)
        output_a = self.output_a(a_x)
        
        output = torch.cat((output_q, output_a), -1)
        
        x = torch.sigmoid(output)
        return x


# ## RUN

# In[ ]:


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupKFold
from transformers import AdamW, get_linear_schedule_with_warmup


# In[ ]:


data = torch.load('./data/train_data.t7')
test_data = torch.load('./data/test_data.t7')


# In[ ]:


test_set = TensorDataset(torch.LongTensor(np.array(test_data['question'])),
                        torch.LongTensor(np.array(test_data['answer'])))


# In[ ]:


test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False)


# In[ ]:


criterion = nn.BCELoss()


# In[ ]:


def compute_spearmanr_ignore_nan(trues, preds):
    rhos = []
    for tcol, pcol in zip(np.transpose(trues), np.transpose(preds)):
        rhos.append(spearmanr(tcol, pcol).correlation)
    return np.nanmean(rhos)


# In[ ]:


"""
gkf = GroupKFold(n_splits=5).split(X=train.question_body, groups=train.question_body)
final_predicts = []
for fold, (train_idx, valid_idx) in enumerate(gkf):   
    if fold in [1]:
        model = Model()
        model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
        train_set = TensorDataset(torch.LongTensor(np.array(data['question'])[train_idx]), \
                                  torch.LongTensor(np.array(data['answer'])[train_idx]), \
                                  torch.FloatTensor(np.array(data['label'])[train_idx])) 
        dev_set = TensorDataset(torch.LongTensor(np.array(data['question'])[valid_idx]), \
                                  torch.LongTensor(np.array(data['answer'])[valid_idx]), \
                                  torch.FloatTensor(np.array(data['label'])[valid_idx]))
        train_loader = DataLoader(
            train_set,
            batch_size=6,
            shuffle=True, drop_last=True)
        dev_loader = DataLoader(
            dev_set,
            batch_size=min(len(dev_set), 1),
            shuffle=False)
        for epoch_idx in range(3):
            for batch_idx, (input_question, input_answer, labels) in enumerate(train_loader):
                model.train()
                optimizer.zero_grad()
                
                input_question, input_answer = input_question.cuda(), input_answer.cuda()
                labels = labels.cuda()
                
                input_question, input_answer = Variable(input_question, requires_grad=False), Variable(input_answer, requires_grad=False)
                scores = model(input_question[:,0],
                               input_question[:,2], 
                               input_question[:,1],
                               input_answer[:,0], 
                               input_answer[:,2], 
                               input_answer[:,1])
                
                labels = Variable(labels, requires_grad=False)
                labels = labels.transpose(0, 1)
                scores = scores.transpose(0, 1)
                losses = [criterion(score, label) for score, label in zip(scores, labels)]
                loss = sum(losses)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                # scheduler.step()
            print("train epoch: {} loss: {}".format(epoch_idx, loss.item()/30))
            torch.save(model.state_dict(), './model/model_{}_{}.t7'.format(fold, epoch_idx))
        
        torch.cuda.empty_cache()
        model.eval()
        pre_list = []
        tru_list = []
        with torch.no_grad():
            for input_question, input_answer, labels in dev_loader:
                input_question, input_answer = input_question.cuda(), input_answer.cuda()
                
                input_question, input_answer = Variable(input_question, requires_grad=False), Variable(input_answer, requires_grad=False)
                
                scores = model(input_question[:,0],
                               input_question[:,2], 
                               input_question[:,1],
                               input_answer[:,0], 
                               input_answer[:,2], 
                               input_answer[:,1])
                pre_list.append(scores)
                tru_list.append(labels)
        dev_predicts = [pre.squeeze(0).cpu().numpy().tolist() for pre in pre_list]
        truthes = [t.squeeze(0).numpy().tolist() for t in tru_list]
        dev_rho = compute_spearmanr_ignore_nan(dev_predicts, truthes)
        print("dev score: ", dev_rho)
"""


# ## INFERENCE

# In[ ]:


models = []
for fold in range(5):
    model_path = f'../input/google-qa-labeling-pretrained-v3/model_{fold}_2.t7'
    if os.path.exists(model_path):
        print(f'model available for prediction at {model_path}')
        model = Model_v1()
        model.load_state_dict(torch.load(model_path))
        models.append(model)
for fold in range(5):
    model_path = f'../input/google-qa-labeling-pretrained-v4/model_{fold}_2.t7'
    if os.path.exists(model_path):
        print(f'model available for prediction at {model_path}')
        model = Model_v2()
        model.load_state_dict(torch.load(model_path))
        models.append(model)


# In[ ]:


len(models)


# In[ ]:


final_predicts = []
for model in models:
    model = model.cuda()
    model.eval()
    test_predicts = []
    with torch.no_grad():
        for input_question, input_answer in test_loader:
            print(input_answer.shape)
            input_question, input_answer = input_question.cuda(), input_answer.cuda()
                
            input_question, input_answer = Variable(input_question, requires_grad=False), Variable(input_answer, requires_grad=False)
                
            scores = model(input_question[:,0],
                           input_question[:,2], 
                           input_question[:,1],
                           input_answer[:,0], 
                           input_answer[:,2], 
                           input_answer[:,1])
            test_predicts.append(scores.reshape(scores.shape[-1]))
    final_predicts.append(test_predicts)


# In[ ]:


pres = np.average(final_predicts, axis=0)
test_output = [[p.item() for p in pre] for pre in pres]


# In[ ]:


output_cols = ['question_asker_intent_understanding',
      'question_body_critical', 'question_conversational',
      'question_expect_short_answer', 'question_fact_seeking',
      'question_has_commonly_accepted_answer',
      'question_interestingness_others', 'question_interestingness_self',
      'question_multi_intent', 'question_not_really_a_question',
      'question_opinion_seeking', 'question_type_choice',
      'question_type_compare', 'question_type_consequence',
      'question_type_definition', 'question_type_entity',
      'question_type_instructions', 'question_type_procedure',
      'question_type_reason_explanation', 'question_type_spelling',
      'question_well_written', 'answer_helpful',
      'answer_level_of_information', 'answer_plausible',
      'answer_relevance', 'answer_satisfaction',
      'answer_type_instructions', 'answer_type_procedure',
      'answer_type_reason_explanation', 'answer_well_written']


# In[ ]:


output_values = np.transpose(test_output).tolist()
output_dict = {k: v for k, v in zip(output_cols, output_values)}
output_dict['qa_id'] = sample_submission['qa_id'].values.tolist()
output = pd.DataFrame.from_dict(output_dict)


# In[ ]:


order = ['qa_id', 'question_asker_intent_understanding',
       'question_body_critical', 'question_conversational',
       'question_expect_short_answer', 'question_fact_seeking',
       'question_has_commonly_accepted_answer',
       'question_interestingness_others', 'question_interestingness_self',
       'question_multi_intent', 'question_not_really_a_question',
       'question_opinion_seeking', 'question_type_choice',
       'question_type_compare', 'question_type_consequence',
       'question_type_definition', 'question_type_entity',
       'question_type_instructions', 'question_type_procedure',
       'question_type_reason_explanation', 'question_type_spelling',
       'question_well_written', 'answer_helpful',
       'answer_level_of_information', 'answer_plausible',
       'answer_relevance', 'answer_satisfaction',
       'answer_type_instructions', 'answer_type_procedure',
       'answer_type_reason_explanation', 'answer_well_written']


# In[ ]:


output = output[order]


# In[ ]:


output.head()


# In[ ]:


output.to_csv('submission.csv', index=False)


# In[ ]:




