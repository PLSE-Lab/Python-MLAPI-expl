#!/usr/bin/env python
# coding: utf-8

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


get_ipython().system('unzip /kaggle/input/quora-question-pairs/sample_submission.csv.zip')
get_ipython().system('unzip /kaggle/input/quora-question-pairs/test.csv.zip')
get_ipython().system('unzip /kaggle/input/quora-question-pairs/train.csv.zip')


# In[ ]:


import pandas as pd
from sklearn.metrics import log_loss
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, random_split
from transformers import BertForSequenceClassification
from time import time
from sklearn.metrics import log_loss
from tqdm import tqdm


# In[ ]:


class Data():
    def __init__(self, path : str,
                 dev : bool = True,
                 dev_samples : int = 10000):
        self.dev = dev
        if dev:
            self.train_df = pd.read_csv(path+'train.csv', nrows=dev_samples)
            self.train_df.drop(columns=['id'], inplace=True)
        else:
            self.train_df = pd.read_csv(path+'train.csv')
            self.test_df = pd.read_csv(path+'test.csv')
            self.submission = pd.read_csv(path+'sample_submission.csv')
            self.test_id = self.test_df.test_id
        
        self.train_df.drop(columns=['qid1', 'qid2'], inplace=True)
        
    def get_train_data(self):
        return self.train_df    
    
    def get_test_data(self):
        if self.dev:
            return None
        else:
            return self.test_df
    
    def get_working_sample(self, n : int):
        return self.train_df.sample(n)
    
    def __repr__(self):
        if self.dev:
            return 'Data train_samples:{} < in development mode >'.format(self.train_df.shape[0])
        else:
            return 'Data train_samples:{} test_samples:{}'.format(self.train_df.shape[0], self.test_df.shape[0])
        



def calc_metrics(logits, labels):
    preds = np.argmax(logits, axis=1).flatten()
    labels = labels.flatten()
    accuracy = np.sum(preds == labels) / len(labels)
    l_loss = log_loss(labels, preds)
    return {'accuracy': accuracy,
            'log_loss': l_loss}


# In[ ]:


data = Data('./', dev_samples=100000)
bert_model_string = 'bert-base-uncased'

tokenizer = BertTokenizer.from_pretrained(bert_model_string)


# In[ ]:


tr_ques1 = data.train_df.question1.values
tr_ques2 = data.train_df.question2.values
labels = torch.tensor(data.train_df.is_duplicate)


input_ids = []
token_type_ids = []
attention_masks = []

for qp in zip(tr_ques1, tr_ques2):
    tok_out = tokenizer.encode_plus(qp[0],
                            text_pair=qp[1],
                            add_special_tokens=False,
                            max_length=128,
                            return_tensors='pt',
                            return_attention_mask=True,
                            pad_to_max_length=True)
    
    input_ids.append(tok_out['input_ids'].squeeze(dim=0))
    token_type_ids.append(tok_out['token_type_ids'].squeeze(dim=0))
    attention_masks.append(tok_out['attention_mask'].squeeze(dim=0))
    
input_ids = torch.stack(input_ids)
token_type_ids = torch.stack(token_type_ids)
attention_masks = torch.stack(attention_masks)
   
print('input_ids shape: \t', input_ids.shape)
print('token_type_ids shape: \t', token_type_ids.shape)
print('attention_masks shape: \t', attention_masks.shape)
print('labels shape: \t\t', labels.shape)


# In[ ]:


VAL_RATIO = 0.1

VAL_SPLIT = int(data.train_df.shape[0]*VAL_RATIO)
TR_SPLIT = data.train_df.shape[0] - VAL_SPLIT
BATCH_SIZE = 48
EPOCHS = 2

print('Samples for trainind: ',TR_SPLIT)
print('Samples for validation: ', VAL_SPLIT)


# In[ ]:


dataset = TensorDataset(input_ids, token_type_ids, attention_masks, labels)
tr_dataset, val_dataset = random_split(dataset, [TR_SPLIT, VAL_SPLIT])

tr_dataloader = DataLoader(tr_dataset, shuffle=True, batch_size = BATCH_SIZE)
val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size = BATCH_SIZE)


# In[ ]:


model = BertForSequenceClassification.from_pretrained(bert_model_string,
                                                      num_labels = 2,
                                                      output_attentions = False,
                                                      output_hidden_states = False)


optimizer = AdamW(model.parameters(), lr = 2e-5, eps = 1e-8)

total_steps = len(tr_dataloader) * EPOCHS

lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)




if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using device: ', torch.cuda.get_device_name(0))
    model.cuda()
else:
    device = torch.device('cpu')
    print('Using device CPU.' )


# In[ ]:



for i in range(0, EPOCHS):
    
    print('========== Epoch {} of {}'.format(i+1, EPOCHS))
    print('Training. . .')
    
    epoch_elapsed = time()
    epoch_loss = 0
    model.train()
    
    for step, batch in enumerate(tr_dataloader):
        model.train()
        step += 1
        b_elapsed = time()
        
        b_input_ids = batch[0].to(device)
        b_token_type_ids = batch[1].to(device)
        b_attention_masks = batch[2].to(device)
        b_labels = batch[3].to(device)
        
        model.zero_grad()
        
        output = model(b_input_ids,
                        attention_mask = b_attention_masks,
                        token_type_ids = b_token_type_ids,
                        labels = b_labels)
        
        b_loss = output[0]
        
        epoch_loss += b_loss.item()
        
        b_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        lr_scheduler.step()
        
        b_elapsed = time() - b_elapsed
        
        if step%50 == 0 or step == 1:
            print('Batch {} of {}. Elapsed: {}. Batch Loss:{}'.format(step, len(tr_dataloader), b_elapsed, b_loss), end=' ')
            val_elapsed = time()
            model.eval()

            validation_logits = []
            validation_labels = []

            for batch in val_dataloader:
                b_input_ids = batch[0].to(device)
                b_token_type_ids = batch[1].to(device)
                b_attention_masks = batch[2].to(device)
                b_labels = batch[3]

                with torch.no_grad():
                    outputs = model(b_input_ids,
                                   attention_mask = b_attention_masks,
                                   token_type_ids = b_token_type_ids)
                logits = outputs[0]

                b_logits = logits.detach().cpu().numpy()
                validation_logits.append(b_logits)
                validation_labels.extend(b_labels)


            validation_logits = np.vstack(validation_logits)
            validation_labels = np.vstack(validation_labels)

            metrics = calc_metrics(validation_logits, validation_labels)
            print('\t >>> Validation Accuracy: {} Validation Log Loss: {}'.format(metrics['accuracy'], metrics['log_loss']))
            
    avg_b_loss = epoch_loss / len(tr_dataloader)
    
    epoch_elapsed = time() - epoch_elapsed
    
    print('Average training Loss : ', avg_b_loss)
    print('Time took for epoch : ', epoch_elapsed)
    
    
    
    
    
    print('==========')
    print('Running Validation. . .')
    
    val_elapsed = time()
    
    model.eval()
    
    validation_logits = []
    validation_labels = []
    
    for batch in tqdm(val_dataloader):
        b_input_ids = batch[0].to(device)
        b_token_type_ids = batch[1].to(device)
        b_attention_masks = batch[2].to(device)
        b_labels = batch[3]
        
        with torch.no_grad():
            outputs = model(b_input_ids,
                           attention_mask = b_attention_masks,
                           token_type_ids = b_token_type_ids)
        logits = outputs[0]
        
        b_logits = logits.detach().cpu().numpy()
        validation_logits.append(b_logits)
        validation_labels.extend(b_labels)

        
    validation_logits = np.vstack(validation_logits)
    validation_labels = np.vstack(validation_labels)
    
    metrics = calc_metrics(validation_logits, validation_labels)
    print('Validation Accuracy: ', metrics['accuracy'])
    print('Validation Log Loss: ', metrics['log_loss'])
    
    
print("Training Complete")


# In[ ]:




