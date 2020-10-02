#!/usr/bin/env python
# coding: utf-8

# # Bert Model with Pytorch 
# 
# This notebook is basic on Abhishek Thakur YouTube video "Training Sentiment Model Using BERT and Serving it with Flask API" you can find the video here : https://www.youtube.com/watch?v=hinZO--TEk4 

# # Import libraries and Config 

# In[ ]:


import transformers 
import torch.nn as nn 
import torch 
from tqdm import tqdm
import torch 
import torch.nn as nn 
import pandas as pd
import torch.nn as nn
import numpy as np
from sklearn import model_selection
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup


# In[ ]:


MAX_Len = 512 
TRAIN_BATCH_SIZE =8 
VALID_BATCH_SIZE = 4
BERT_PATH = '../input/bert-base-uncased'
TOKENZIER = transformers.BertTokenizer.from_pretrained(BERT_PATH ,do_lower_case = True )


# # Creat Model 

# In[ ]:


class BertBaseUncased(nn.Module) :
    def __init__(self) : 
        super(BertBaseUncased,self).__init__() 
        self.bert = transformers.BertModel.from_pretrained(BERT_PATH) 
        self.bert_drop = nn.Dropout(0.4) 
        self.out = nn.Linear(768,1) 
    def forward(self,ids,mask,token_type_ids) : 
        out1,out2 = self.bert( 
            ids , 
            attention_mask = mask , 
            token_type_ids = token_type_ids 
        )
        bo = self.bert_drop(out2) 
        output = self.out(bo) 
        return output 


# # Data set 

# In[ ]:


class BERTDataset : 
    def __init__(self,df) : 
        
        self.text = df['text'].values
        self.target = df['target'].values 
        self.tokenizer = TOKENZIER 
        self.max_len = MAX_Len 
    def __len__(self) : 
        return len(self.text) 
    def __getitem__(self, item):
        text = str(self.text[item])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
                text,
                None,
                add_special_tokens=True,
                max_length=self.max_len
            )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        padding_length = self.max_len - len(ids)
        ids = ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
                'targets': torch.tensor(self.target[item], dtype=torch.float)
            }


# # Engine 

# In[ ]:


def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))


# In[ ]:


def train_fn(data_loader, model, optimizer, scheduler):
    model.train()

    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        targets = d["targets"]

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        optimizer.zero_grad()
        outputs = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )

        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
       


# In[ ]:


def eval_fn(data_loader, model):
   model.eval()
   fin_targets = []
   fin_outputs = []
   with torch.no_grad():
       for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
           ids = d["ids"]
           token_type_ids = d["token_type_ids"]
           mask = d["mask"]
           targets = d["targets"]

           ids = ids.to(device, dtype=torch.long)
           token_type_ids = token_type_ids.to(device, dtype=torch.long)
           mask = mask.to(device, dtype=torch.long)
           targets = targets.to(device, dtype=torch.float)

           outputs = model(
               ids=ids,
               mask=mask,
               token_type_ids=token_type_ids
           )
           fin_targets.extend(targets.cpu().detach().numpy().tolist())
           fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
   return fin_outputs, fin_targets


# # Training the model 

# In[ ]:


DEVICE =torch.device("cuda")
device = torch.device("cuda")
def run(model,EPOCHS):
    dfx = pd.read_csv('../input/nlp-getting-started/train.csv').fillna("none")
    df_train, df_valid = model_selection.train_test_split(
        dfx,
        test_size=0.1,
        random_state=42,
        stratify=dfx.target.values
    )



    train_dataset = BERTDataset(
        df_train
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        num_workers=4
    )

    valid_dataset = BERTDataset(
        df_valid

    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=VALID_BATCH_SIZE,
        num_workers=1
    )

    device = torch.device("cuda")
    
    
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]

    num_train_steps = int(len(train_data_loader)) * EPOCHS
    optimizer = AdamW(optimizer_parameters, lr=1e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )


    model = nn.DataParallel(model)

    best_accuracy = 0
    for epoch in range(EPOCHS):
        train_fn(train_data_loader, model, optimizer, scheduler)
        outputs, targets = eval_fn(valid_data_loader, model)
        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(targets, outputs)
        print(f"Accuracy Score = {accuracy}")
        scheduler.step()


# In[ ]:


model = BertBaseUncased()
model.to(device)
getattr(tqdm, '_instances', {}).clear()
run(model,1)


# # Make Prediction 

# In[ ]:


def sentence_prediction(sentence):
    tokenizer = TOKENZIER
    max_len = MAX_Len
    text = str(sentence)
    text = " ".join(text.split())

    inputs = tokenizer.encode_plus(
        text,
        None,
        add_special_tokens=True,
        max_length=max_len
    )

    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]

    padding_length = max_len - len(ids)
    ids = ids + ([0] * padding_length)
    mask = mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)

    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0)

    ids = ids.to(DEVICE, dtype=torch.long)
    token_type_ids = token_type_ids.to(DEVICE, dtype=torch.long)
    mask = mask.to(DEVICE, dtype=torch.long)

    outputs = model(
        ids=ids,
        mask=mask,
        token_type_ids=token_type_ids
    )

    outputs = torch.sigmoid(outputs).cpu().detach().numpy()
    return outputs[0][0]


# In[ ]:


test = pd.read_csv('../input/nlp-getting-started/test.csv')
test['target'] = test['text'].apply(sentence_prediction)


# In[ ]:


test


# In[ ]:


sub = test[['id','target']]


# In[ ]:


sub['target'] = sub['target'].round().astype('int')


# In[ ]:


sub['target'] 


# # Using keywords for prediction improvement

# In[ ]:


train = pd.read_csv('../input/nlp-getting-started/train.csv')
train = train.fillna('None')
ag = train.groupby('keyword').agg({'text':np.size, 'target':np.mean}).rename(columns={'text':'Count', 'target':'Disaster Probability'})
ag.sort_values('Disaster Probability', ascending=False).head(10)


# In[ ]:


keyword_list = list(ag[(ag['Count']>2) & (ag['Disaster Probability']>=0.9)].index)
keyword_list


# In[ ]:


ids = test['id'][test.keyword.isin(keyword_list)].values
sub['target'][sub['id'].isin(ids)] = 1
sub.head()


# In[ ]:


sub.to_csv('submission.csv',index=False)


# I hope you find this kernel useful and enjoyable.
# 
# 
# and here you will find my notebook about working with text data in pandas : https://www.kaggle.com/nhoues1997/working-with-text-data-in-pandas
# 
# 
# Your comments and feedback are most welcome. 
# 
