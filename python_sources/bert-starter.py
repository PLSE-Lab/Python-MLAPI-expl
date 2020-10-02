#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import re
import string
import os
import torch
import numpy as np
import tqdm


# In[ ]:


get_ipython().system('pip install transformers')


# In[ ]:


from transformers import BertModel, BertTokenizer


# In[ ]:


path_to_dataset = '/kaggle/input/nlp-getting-started/'


# In[ ]:


test = pd.read_csv(os.path.join(path_to_dataset, 'test.csv'))
train = pd.read_csv(os.path.join(path_to_dataset, 'train.csv'))


# In[ ]:


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# In[ ]:


class Model(torch.nn.Module):
    
    def __init__(self, ):
        
        super(Model, self).__init__()
        self.base_model = BertModel.from_pretrained('bert-base-uncased') # use pre-trained BERT model by HuggingFace
        self.fc1 = torch.nn.Linear(768, 1) # simple logistic regression above the bert model
        
    def forward(self, ids, masks):
        
        x = self.base_model(ids, attention_mask=masks)[1]
        x = self.fc1(x)
        return x
        


# In[ ]:


model = Model()


# In[ ]:


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# In[ ]:


model = model.to(device)


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


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9] \n', '', text)
    return text


# Use first 6000 for training, rest for validation

# In[ ]:


train_text = train.text[:6000]
val_text = train.text[6000:]


# In[ ]:


train_text = train_text.apply(clean_text)
val_text = val_text.apply(clean_text)


# In[ ]:


train_tokens = []
train_pad_masks = []
for text in train_text:
    tokens, masks = bert_encode(text)
    train_tokens.append(tokens)
    train_pad_masks.append(masks)
    
train_tokens = np.array(train_tokens)
train_pad_masks = np.array(train_pad_masks)


# In[ ]:


val_tokens = []
val_pad_masks = []
for text in val_text:
    tokens, masks = bert_encode(text)
    val_tokens.append(tokens)
    val_pad_masks.append(masks)
    
val_tokens = np.array(val_tokens)
val_pad_masks = np.array(val_pad_masks)


# In[ ]:



class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, train_tokens, train_pad_masks, targets):
        
        super(Dataset, self).__init__()
        self.train_tokens = train_tokens
        self.train_pad_masks = train_pad_masks
        self.targets = targets
        
    def __getitem__(self, index):
        
        tokens = self.train_tokens[index]
        masks = self.train_pad_masks[index]
        target = self.targets[index]
        
        return (tokens, masks), target
    
    def __len__(self,):
        
        return len(self.train_tokens)


# In[ ]:


train_dataset = Dataset(
                    train_tokens=train_tokens,
                    train_pad_masks=train_pad_masks,
                    targets=train.target[:6000]
)


# In[ ]:


batch_size = 6
EPOCHS = 2


# In[ ]:


train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


# In[ ]:


criterion = torch.nn.BCEWithLogitsLoss()


# Use Adam Optimizer with learning rate of 0.00001

# In[ ]:


opt = torch.optim.Adam(model.parameters(), lr=0.00001)


# Train for 2 epochs.

# In[ ]:


model.train()
y_preds = []

for epoch in range(EPOCHS):
        for i, ((tokens, masks), target) in enumerate(train_dataloader):

            y_pred = model(
                        tokens.long().to(device), 
                        masks.long().to(device)
                    )
            loss = criterion(y_pred, target[:, None].float().to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            print('\rEpoch: %d/%d, %f%% loss: %0.2f'% (epoch+1, EPOCHS, i/len(train_dataloader)*100, loss.item()), end='')
        print()


# Test the model on the validation dataset

# In[ ]:


val_dataset = Dataset(
                    train_tokens=val_tokens,
                    train_pad_masks=val_pad_masks,
                    targets=train.target[6000:].reset_index(drop=True)
)


# In[ ]:


val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=3, shuffle=False)


# Define accuracy metric

# In[ ]:


def accuracy(y_actual, y_pred):
    y_ = y_pred > 0
    return np.sum(y_actual == y_).astype('int') / y_actual.shape[0]


# In[ ]:


model.eval()
avg_acc = 0
for i, ((tokens, masks), target) in enumerate(val_dataloader):

    y_pred = model(
                tokens.long().to(device), 
                masks.long().to(device), 
            )
    loss = criterion(y_pred,  target[:, None].float().to(device))
    acc = accuracy(target.cpu().numpy(), y_pred.detach().cpu().numpy().squeeze())
    avg_acc += acc
    print('\r%0.2f%% loss: %0.2f, accuracy %0.2f'% (i/len(val_dataloader)*100, loss.item(), acc), end='')
print('\nAverage accuracy: ', avg_acc / len(val_dataloader))


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
for text in test.text:
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
                tokens.long().to(device), 
                masks.long().to(device), 
            )
    y_preds += y_pred.detach().cpu().numpy().squeeze().tolist()


# In[ ]:


submission_df = pd.read_csv(os.path.join(path_to_dataset, 'sample_submission.csv'))


# In[ ]:


submission_df['target'] = (np.array(y_preds) > 0).astype('int')


# In[ ]:


submission_df.target.value_counts()


# In[ ]:


submission_df


# In[ ]:


submission_df.to_csv('submission.csv', index=False)


# In[ ]:




