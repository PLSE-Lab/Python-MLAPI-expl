#!/usr/bin/env python
# coding: utf-8

# ##### This notebook is an implementation of code from the coursera guided project: 
# 
# Sentiment Analysis with Deep Learning using BERT
# 
# The intructor was:Ari Anastassiou
# 

# In[ ]:


## First run of BERT with pytorch,


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


get_ipython().run_line_magic('pwd', '')


# In[ ]:


import torch
import pandas as pd
from tqdm.notebook import tqdm


# In[ ]:


df = pd.read_csv('../input/nlp-getting-started/train.csv')


# In[ ]:


df_test = pd.read_csv('../input/nlp-getting-started/test.csv')
df_test['target'] = 2
df_test.head()


# In[ ]:


df.target.value_counts()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(df.id.values, 
                                                  df.target.values, 
                                                  test_size=0.15, 
                                                  random_state=17, 
                                                  stratify=df.target.values)


# In[ ]:


df['data_type'] = ['not_set']*df.shape[0]


# In[ ]:


df.head()


# In[ ]:


df.loc[df['id'].isin(X_train),'data_type']= 'train'
df.loc[df['id'].isin(X_val),'data_type']= 'val'


# **Loading Tokenizer and Encoding our Data**

# In[ ]:


from transformers import BertTokenizer
from torch.utils.data import TensorDataset


# In[ ]:


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', 
                                          do_lower_case=True)


# In[ ]:


encoded_data_train = tokenizer.batch_encode_plus(
    df[df.data_type=='train'].text.values, 
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=256, 
    return_tensors='pt'
)

encoded_data_val = tokenizer.batch_encode_plus(
    df[df.data_type=='val'].text.values, 
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=256, 
    return_tensors='pt'
)

encoded_data_test = tokenizer.batch_encode_plus(
    df_test.text.values, 
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=256, 
    return_tensors='pt'
)


input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(df[df.data_type=='train'].target.values)

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(df[df.data_type=='val'].target.values)

input_ids_test = encoded_data_test['input_ids']
attention_masks_test = encoded_data_test['attention_mask']
labels_test = torch.tensor(df_test.target.values)


# In[ ]:


dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)
dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test)


# In[ ]:


print(len(dataset_train))
print(len(dataset_val))
print(len(dataset_test))


# In[ ]:


from transformers import BertForSequenceClassification


# In[ ]:


len(df['target'].unique())


# In[ ]:


model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=len(df['target'].unique()),
                                                      output_attentions=False,
                                                      output_hidden_states=False)


# ## Creating Data Loaders

# In[ ]:


from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


# In[ ]:


#batch_size = 32
batch_size = 4

dataloader_train = DataLoader(dataset_train, 
                              sampler=RandomSampler(dataset_train), 
                              batch_size=batch_size)

dataloader_validation = DataLoader(dataset_val, 
                                   sampler=SequentialSampler(dataset_val), 
                                   batch_size=32)

dataloader_test = DataLoader(dataset_test, 
                                   sampler=SequentialSampler(dataset_test), 
                                   batch_size=32)


# In[ ]:


from transformers import AdamW, get_linear_schedule_with_warmup


# In[ ]:


optimizer = AdamW(model.parameters(),
                  lr=1e-5, 
                  eps=1e-8)


# In[ ]:


epochs = 3

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train)*epochs)


# ## Defining our Performance Metrics

# In[ ]:


import numpy as np
from sklearn.metrics import f1_score


# In[ ]:


def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')


# In[ ]:


def accuracy_per_class(preds, labels):
    
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')


# In[ ]:


import random

seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print(device)


# In[ ]:


def evaluate(dataloader_val):

    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in dataloader_val:
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
    
    return loss_val_avg, predictions, true_vals
            


# In[ ]:


for epoch in tqdm(range(1, epochs+1)):
    
    model.train()
    
    loss_train_total = 0

    progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
    for batch in progress_bar:

        model.zero_grad()
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }       

        outputs = model(**inputs)
        
        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
         
        
    torch.save(model.state_dict(), f'finetuned_BERT_epoch.model')
        
    tqdm.write(f'\nEpoch {epoch}')
    
    loss_train_avg = loss_train_total/len(dataloader_train)            
    tqdm.write(f'Training loss: {loss_train_avg}')
    
    val_loss, predictions, true_vals = evaluate(dataloader_validation)
    val_f1 = f1_score_func(predictions, true_vals)
    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'F1 Score (Weighted): {val_f1}')


# In[ ]:


'''
model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False)

model.to(device)

'''


# In[ ]:


'''
model.load_state_dict(torch.load('Models/<<INSERT MODEL NAME HERE>>.model', map_location=torch.device('cpu')))
'''


# In[ ]:


_, predictions, true_vals = evaluate(dataloader_validation)


# In[ ]:


accuracy_per_class(predictions, true_vals)


# In[ ]:


_, test_predictions, true_vals = evaluate(dataloader_test)


# In[ ]:


test_predictions.shape


# In[ ]:


test_predictions[0:3]


# In[ ]:


test_preds_flat = np.argmax(test_predictions, axis=1).flatten()
test_preds_flat.shape


# In[ ]:


test_preds_flat.sum()


# In[ ]:


df_test['predict'] = test_preds_flat


# In[ ]:


df_submit = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
df_submit.head()


# In[ ]:


df_test['target'] = df_test['predict']


# In[ ]:


df_test = df_test[['id','target']]
df_test.head()


# In[ ]:


df_test.to_csv("submission.csv", index=False)


# In[ ]:


get_ipython().run_line_magic('ls', '')


# In[ ]:


get_ipython().run_line_magic('rm', '-f finetuned*')


# In[ ]:


len(df_test)


# In[ ]:




