#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis with Deep Learning using BERT

# ## Introduction

# ### What is BERT
# 
# BERT is a large-scale transformer-based Language Model that can be finetuned for a variety of tasks.
# 
# For more information, the original paper can be found [here](https://arxiv.org/abs/1810.04805). 
# 
# [HuggingFace documentation](https://huggingface.co/transformers/model_doc/bert.html)
# 
# [Bert documentation](https://characters.fandom.com/wiki/Bert_(Sesame_Street) ;)

# ## Exploratory Data Analysis and Preprocessing

# We will use the SMILE Twitter dataset.
# 
# _Wang, Bo; Tsakalidis, Adam; Liakata, Maria; Zubiaga, Arkaitz; Procter, Rob; Jensen, Eric (2016): SMILE Twitter Emotion dataset. figshare. Dataset. https://doi.org/10.6084/m9.figshare.3187909.v2_

# In[ ]:


import torch
import pandas as pd
from tqdm.notebook import tqdm


# In[ ]:


df = pd.read_csv('/kaggle/input/smile-annotations-final.csv', names=['id', 'text', 'category'])
df.set_index('id', inplace=True)


# In[ ]:


df.head()


# In[ ]:


df.category.value_counts()


# In[ ]:


df = df[~df.category.str.contains('\|')]


# In[ ]:


df = df[df.category != 'nocode']


# In[ ]:


df.category.value_counts()


# In[ ]:


possible_labels = df.category.unique()


# In[ ]:


label_dict = {}
for index, possible_label in enumerate(possible_labels):
    label_dict[possible_label] = index


# In[ ]:


df['label'] = df.category.replace(label_dict)


# In[ ]:


df.head()


# ## Training/Validation Split

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(df.index.values, 
                                                  df.label.values, 
                                                  test_size=0.15, 
                                                  random_state=17, 
                                                  stratify=df.label.values)


# In[ ]:


df['data_type'] = ['not_set']*df.shape[0]


# In[ ]:


df.loc[X_train, 'data_type'] = 'train'
df.loc[X_val, 'data_type'] = 'val'


# In[ ]:


df.groupby(['category', 'label', 'data_type']).count()


# ## Loading Tokenizer and Encoding our Data

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


input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(df[df.data_type=='train'].label.values)

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(df[df.data_type=='val'].label.values)


# In[ ]:


dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)


# In[ ]:


len(dataset_train)


# In[ ]:


len(dataset_val)


# ## Setting up BERT Pretrained Model

# In[ ]:


from transformers import BertForSequenceClassification


# In[ ]:


model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False)


# ## Creating Data Loaders

# In[ ]:


from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


# In[ ]:


batch_size = 32

dataloader_train = DataLoader(dataset_train, 
                              sampler=RandomSampler(dataset_train), 
                              batch_size=batch_size)

dataloader_validation = DataLoader(dataset_val, 
                                   sampler=SequentialSampler(dataset_val), 
                                   batch_size=batch_size)


# ## Setting Up Optimiser and Scheduler

# In[ ]:


from transformers import AdamW, get_linear_schedule_with_warmup


# In[ ]:


optimizer = AdamW(model.parameters(),
                  lr=1e-5, 
                  eps=1e-8)


# In[ ]:


epochs = 10

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train)*epochs)


# ## Defining our Performance Metrics

# Accuracy metric approach originally used in accuracy function in [this tutorial](https://mccormickml.com/2019/07/22/BERT-fine-tuning/#41-bertforsequenceclassification).

# In[ ]:


import numpy as np


# In[ ]:


from sklearn.metrics import f1_score


# In[ ]:


def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')


# In[ ]:


def accuracy_per_class(preds, labels):
    label_dict_inverse = {v: k for k, v in label_dict.items()}
    
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')


# ## Creating our Training Loop

# Approach adapted from an older version of HuggingFace's `run_glue.py` script. Accessible [here](https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128).

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
    
    for batch in tqdm(dataloader_val):
        
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
         
        
    torch.save(model.state_dict(), f'finetuned_BERT_epoch_{epoch}.model')
        
    tqdm.write(f'\nEpoch {epoch}')
    
    loss_train_avg = loss_train_total/len(dataloader_train)            
    tqdm.write(f'Training loss: {loss_train_avg}')
    
    val_loss, predictions, true_vals = evaluate(dataloader_validation)
    val_f1 = f1_score_func(predictions, true_vals)
    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'F1 Score (Weighted): {val_f1}')


# In[ ]:


model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False)

model.to(device)


# In[ ]:


#we have save the model for each epoch of out training loop. Here we can load them and see which one gives the best results for the different classes
model.load_state_dict(torch.load('/kaggle/working/finetuned_BERT_epoch_10.model', map_location=device))


# In[ ]:


_, predictions, true_vals = evaluate(dataloader_validation)


# In[ ]:


accuracy_per_class(predictions, true_vals)


# In[ ]:





# In[ ]:




