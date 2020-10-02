#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import re
import os
import torch
import numpy as np


# Install HuggingFace implementation of bert (https://huggingface.co/).

# In[ ]:


get_ipython().system('pip install transformers')


# In[ ]:


from transformers import BertModel, BertTokenizer


# In[ ]:


path_to_dataset = '/kaggle/input/nlp-getting-started/'


# In[ ]:


test_df = pd.read_csv(os.path.join(path_to_dataset, 'test.csv'))


# Defining our simple model (logistic regression over the bert base model).

# In[ ]:


class Model(torch.nn.Module):
    
    def __init__(self, ):
        
        super(Model, self).__init__()
        self.base_model = BertModel.from_pretrained('bert-base-uncased') #pretrained bert model
        self.fc1 = torch.nn.Linear(768, 1) #use logistic regression
        
    def forward(self, ids, masks):
        
        x = self.base_model(ids, attention_mask=masks)[1]
        x = self.fc1(x)
        return x
        


# In[ ]:


path_to_model = '/kaggle/input/nlpgetstartedbertbasemoel/'


# Load the pretrained model

# In[ ]:


model = torch.load(os.path.join(path_to_model, 'model.pth'))


# In[ ]:


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# In[ ]:


model = model.to(device)


# In[ ]:


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# In[ ]:


def bert_encode(text, max_len=512):
    
    text = tokenizer.tokenize(text)
    text = text[:max_len-2]
    input_sequence = ["[CLS]"] + text + ["[SEP]"]
    tokens = tokenizer.convert_tokens_to_ids(input_sequence)
    tokens += [0] * (max_len - len(input_sequence))
    pad_masks = [1] * len(input_sequence) + [0] * (max_len - len(input_sequence))

    return tokens, pad_masks


# In[ ]:


class TestDataset(torch.utils.data.Dataset):
    
    def __init__(self, test_tokens, test_pad_masks):
        
        super(TestDataset, self).__init__()
        self.test_tokens = test_tokens
        self.test_pad_masks = test_pad_masks
        
    def __getitem__(self, index):
        
        tokens = self.test_tokens[index]
        masks = self.test_pad_masks[index]
        
        return (tokens, masks)
    
    def __len__(self,):
        
        return len(self.test_tokens)


# In[ ]:


test_tokens = []
test_pad_masks = []
for text in test_df.text:
    tokens, masks = bert_encode(text)
    test_tokens.append(tokens)
    test_pad_masks.append(masks)
    
test_tokens = np.array(test_tokens)
test_pad_masks = np.array(test_pad_masks)


# In[ ]:


test_dataset = TestDataset(
    test_tokens=test_tokens,
    test_pad_masks=test_pad_masks
)


# In[ ]:


test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=3, shuffle=False)


# In[ ]:


model.eval()
y_preds = []
for (tokens, masks) in test_dataloader:

    y_pred = model(
                torch.tensor(tokens, dtype=torch.long).to(device),
                torch.tensor(masks, dtype=torch.long).to(device),
            )
    y_preds += y_pred.detach().cpu().numpy().squeeze().tolist()


# In[ ]:


submission_df = pd.read_csv(os.path.join(path_to_dataset, 'sample_submission.csv'))


# Target is 1 if the output is greater than 0.

# In[ ]:


submission_df['target'] = (np.array(y_preds) >= 0).astype('int')


# In[ ]:


submission_df.target.value_counts()


# In[ ]:


submission_df.head()


# Writing to submission.csv file

# In[ ]:


submission_df.to_csv('submission.csv', index=False)


# In[ ]:




