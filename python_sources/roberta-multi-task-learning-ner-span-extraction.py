#!/usr/bin/env python
# coding: utf-8

# **Description**
# * Added NER [0-X,1-Positive,2-Negative,3-Neutral],[0- not part of selected sentence],[1,2,3- part of pos,neg,neu,sentences respectively] to train both tasks simultaneously 
# * Extracted Sentence using Span Extraction (I failed using NER predictions during Inference) 
# * Got a lower score Compared to reference Kernel, I'm not sure if doing two tasks at once is helpful [Pls share your insights on this]
# * Reference Code: [Tweet Sentiment RoBERTa PyTorch](https://www.kaggle.com/shoheiazuma/tweet-sentiment-roberta-pytorch)

# # Libraries

# In[ ]:


import numpy as np
import pandas as pd
import os
import warnings
import random
import torch 
from torch import nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
import tokenizers
from transformers import RobertaModel, RobertaConfig
from transformers import get_linear_schedule_with_warmup
from tqdm.autonotebook import tqdm
warnings.filterwarnings('ignore')


# ### Checking CUDA Avail

# In[ ]:


cuda_yes = torch.cuda.is_available()
# cuda_yes = False
print('Cuda is available?', cuda_yes)
device = torch.device("cuda:0" if cuda_yes else "cpu")
print('Device:', device)


# # Seed

# In[ ]:


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

seed = 42
seed_everything(seed)


# # Data Loader

# In[ ]:


class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, df, max_len=96):
        self.df = df
        self.max_len = max_len
        self.labeled = 'selected_text' in df
        self.tokenizer = tokenizers.ByteLevelBPETokenizer(
            vocab_file='../input/roberta-base/vocab.json', 
            merges_file='../input/roberta-base/merges.txt', 
            lowercase=True,
            add_prefix_space=True)
#         self._label_types = ['X', 'positive', 'negative','neutral']
#         self._label_map = {label: i for i,
#                            label in enumerate(self._label_types)}

#         self.num_labels=num_labels

    def __getitem__(self, index):
        data = {}
        row = self.df.iloc[index]
        
        ids, masks, tweet, offsets = self.get_input_data(row)
        data['ids'] = ids
        data['masks'] = masks
        data['tweet'] = tweet
        data['offsets'] = offsets
        data['sentiment'] = row.sentiment
        if self.labeled:
            start_idx, end_idx,labels = self.get_target_idx(row, tweet, offsets)
            data['start_idx'] = start_idx
            data['end_idx'] = end_idx
            data['labels']=labels
            data['selected_text']=row.selected_text
        
        return data

    def __len__(self):
        return len(self.df)
    
    def get_input_data(self, row):
        tweet = " " + " ".join(row.text.lower().split())
        encoding = self.tokenizer.encode(tweet)
        sentiment_id = self.tokenizer.encode(row.sentiment).ids
        ids = [0] + sentiment_id + [2, 2] + encoding.ids + [2]
        offsets = [(0, 0)] * 4 + encoding.offsets + [(0, 0)]
                
        pad_len = self.max_len - len(ids)
        if pad_len > 0:
            ids += [1] * pad_len
            offsets += [(0, 0)] * pad_len
        
        ids = torch.tensor(ids)
        masks = torch.where(ids != 1, torch.tensor(1), torch.tensor(0))
        offsets = torch.tensor(offsets)
        
        return ids, masks, tweet, offsets
        
    def get_target_idx(self, row, tweet, offsets):
        # [0,1,2,3]
        selected_text = " " +  " ".join(row.selected_text.lower().split())
        if row.sentiment=="positive":
            sent=1
        elif row.sentiment=="negative":
            sent=2
        else:
            sent=3
            
        len_st = len(selected_text) - 1
        idx0 = None
        idx1 = None

        for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):
            if " " + tweet[ind: ind+len_st] == selected_text:
                idx0 = ind
                idx1 = ind + len_st - 1
                break

        char_targets = [0] * len(tweet)
        if idx0 is not None and idx1 is not None:
            for ct in range(idx0, idx1 + 1):
                char_targets[ct] = 1

        target_idx = []
        labels = []
        for j, (offset1, offset2) in enumerate(offsets):
            if sum(char_targets[offset1: offset2]) > 0:
                target_idx.append(j)
                labels.append(sent)
            elif j==1: # Sentiment in input
                labels.append(sent)
            else:
                labels.append(0)
        start_idx = target_idx[0]
        end_idx = target_idx[-1]
        labels = torch.tensor(labels)
        return start_idx, end_idx, labels
        
def get_train_val_loaders(df, train_idx, val_idx, batch_size=8):
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    train_loader = torch.utils.data.DataLoader(
        TweetDataset(train_df), 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        TweetDataset(val_df), 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2)

    dataloaders_dict = {"train": train_loader, "val": val_loader}

    return dataloaders_dict

def get_test_loader(df, batch_size=32):
    loader = torch.utils.data.DataLoader(
        TweetDataset(df), 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2)    
    return loader


# # Model

# In[ ]:


class TweetModel(nn.Module):
    def __init__(self):
        super(TweetModel, self).__init__()
        
        config = RobertaConfig.from_pretrained(
            '../input/roberta-base/config.json', output_hidden_states=True,num_labels=NUM_LABELS)    
        self.roberta = RobertaModel.from_pretrained(
            '../input/roberta-base/pytorch_model.bin', config=config)
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(config.hidden_size, 2)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask, labels=None):
        sequence_output, _, hs = self.roberta(input_ids, attention_mask)
         
        x = torch.stack([hs[-1], hs[-2], hs[-3],hs[-4]])
        x = torch.mean(x, 0)
        x_drop = self.dropout(x)
        x = self.fc(x_drop)
        start_logits, end_logits = x.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        logits = self.classifier(x_drop)
        
        outputs= (logits,start_logits, end_logits)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
                
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        
        return outputs


# # Loss Function

# In[ ]:


def loss_fn(start_logits, end_logits, start_positions, end_positions):
    ce_loss = nn.CrossEntropyLoss()
    start_loss = ce_loss(start_logits, start_positions)
    end_loss = ce_loss(end_logits, end_positions)    
    total_loss = start_loss + end_loss
    return total_loss


# # Evaluation Function

# In[ ]:


def get_selected_text(text, start_idx, end_idx, offsets):
    selected_text = ""
    for ix in range(start_idx, end_idx + 1):
        selected_text += text[offsets[ix][0]: offsets[ix][1]]
        if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
            selected_text += " "
    return selected_text

def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def compute_jaccard_score(text,selected_text,start_idx, end_idx, start_logits, end_logits, offsets):
    start_pred = np.argmax(start_logits)
    end_pred = np.argmax(end_logits)
    if start_pred > end_pred:
        pred = text
    else:
        pred = get_selected_text(text, start_pred, end_pred, offsets)
        
    true = get_selected_text(text, start_idx, end_idx, offsets)
    jaccard_approx = jaccard(true, pred)
    jaccard_true = jaccard(selected_text,pred)
    return jaccard_approx,jaccard_true


# # Training Function

# In[ ]:


def train_model(model, dataloaders_dict, criterion, optimizer, num_epochs, filename):
    model.to(device)

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            epoch_loss = 0.0
            epoch_approx_jaccard = 0.0
            epoch_true_jaccard = 0.0
            epoch_start_end_loss=0.0
            epoch_ner_loss=0.0
            tk0 = tqdm(dataloaders_dict[phase], total=len(dataloaders_dict[phase]))
            for data in tk0:
                ids = data['ids'].to(device)
                masks = data['masks'].to(device)
                tweet = data['tweet']
                selected_text=data['selected_text']
                sentiment = data['sentiment']
                offsets = data['offsets'].numpy()
                start_idx = data['start_idx'].to(device)
                end_idx = data['end_idx'].to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    ner_loss,logits,start_logits, end_logits = model(ids, masks,labels)

                    start_end_loss = criterion(start_logits, end_logits, start_idx, end_idx)
                    loss=ner_loss+start_end_loss
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                    epoch_loss += loss.item() * len(ids)
                    epoch_start_end_loss += start_end_loss.item() * len(ids)
                    epoch_ner_loss += ner_loss.item() * len(ids)
                    start_idx = start_idx.cpu().detach().numpy()
                    end_idx = end_idx.cpu().detach().numpy()
                    start_logits = torch.softmax(start_logits, dim=1).cpu().detach().numpy()
                    end_logits = torch.softmax(end_logits, dim=1).cpu().detach().numpy()
                    
                    for i in range(len(ids)):                        
                        jaccard_approx_score,jaccard_true_score = compute_jaccard_score(
                            tweet[i],
                            selected_text[i],
                            start_idx[i],
                            end_idx[i],
                            start_logits[i], 
                            end_logits[i], 
                            offsets[i])
                        
                        epoch_approx_jaccard += jaccard_approx_score
                        epoch_true_jaccard += jaccard_true_score
                        
                    
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_ner_loss = epoch_ner_loss / len(dataloaders_dict[phase].dataset)
            epoch_start_end_loss = epoch_start_end_loss / len(dataloaders_dict[phase].dataset)
            epoch_approx_jaccard = epoch_approx_jaccard / len(dataloaders_dict[phase].dataset)
            epoch_true_jaccard = epoch_true_jaccard / len(dataloaders_dict[phase].dataset)
            
            print('Epoch {}/{} | {:^5} | Loss: {:.4f} | NER_Loss: {:.4f} | Start_End_Loss: {:.4f} | Approx_Jaccard: {:.4f} | True_Jaccard: {:.4f}'.format(
                epoch + 1, num_epochs, phase, epoch_loss,epoch_ner_loss,epoch_start_end_loss,epoch_approx_jaccard,epoch_true_jaccard))
    
    torch.save(model.state_dict(), filename)


# # Training [Skipped]

# In[ ]:


num_epochs = 3
batch_size = 32
gradient_accumulation_steps = 1
warmup_proportion=0.1
NUM_LABELS=4
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)


# In[ ]:


# %%time

# train_df = pd.read_csv('../input/tweet-sentiment-extraction/train.csv')
# train_df['text'] = train_df['text'].astype(str)
# train_df['selected_text'] = train_df['selected_text'].astype(str)

# for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df.sentiment), start=1): 
#     print(f'Fold: {fold}')

#     model = TweetModel()
#     optimizer = optim.AdamW(model.parameters(), lr=3e-5, betas=(0.9, 0.999))
#     total_train_steps = int(len(train_df) / batch_size / gradient_accumulation_steps * num_epochs)

#     criterion = loss_fn    
#     dataloaders_dict = get_train_val_loaders(train_df, train_idx, val_idx, batch_size)
#     scheduler = get_linear_schedule_with_warmup(optimizer,warmup_proportion*total_train_steps,total_train_steps)

#     train_model(
#         model, 
#         dataloaders_dict,
#         criterion, 
#         optimizer, 
#         num_epochs,
#         f'roberta_fold{fold}.pth')


# # Inference

# In[ ]:


get_ipython().run_cell_magic('time', '', '\ntest_df = pd.read_csv(\'../input/tweet-sentiment-extraction/test.csv\')\ntest_df[\'text\'] = test_df[\'text\'].astype(str)\ntest_loader = get_test_loader(test_df)\npredictions = []\nmodels = []\nmodel_dir = "../input/tweet-sentiment-roberta-pytorch/"\nfor fold in range(skf.n_splits):\n    model = TweetModel()\n    model.to(device)\n    model.load_state_dict(torch.load(model_dir+f\'roberta_fold{fold+1}.pth\',map_location=torch.device(device)))\n    model.eval()\n    models.append(model)\n\nfor data in tqdm(test_loader):\n    ids = data[\'ids\'].to(device)\n    masks = data[\'masks\'].to(device)\n    tweet = data[\'tweet\']\n    sentiment = data[\'sentiment\']\n    offsets = data[\'offsets\'].numpy()\n\n    start_logits = []\n    end_logits = []\n    for model in models:\n        with torch.no_grad():\n            output = model(ids, masks)\n            start_logits.append(torch.softmax(output[1], dim=1).cpu().detach().numpy())\n            end_logits.append(torch.softmax(output[2], dim=1).cpu().detach().numpy())\n\n    start_logits = np.mean(start_logits, axis=0)\n    end_logits = np.mean(end_logits, axis=0)\n    for i in range(len(ids)):    \n        start_pred = np.argmax(start_logits[i])\n        end_pred = np.argmax(end_logits[i])\n        \n        if (start_pred > end_pred):\n            pred = tweet[i]\n        else:\n            pred = get_selected_text(tweet[i], start_pred, end_pred, offsets[i])\n        predictions.append(pred)')


# # Submission

# In[ ]:


sub_df = pd.read_csv('../input/tweet-sentiment-extraction/sample_submission.csv')
sub_df['selected_text'] = predictions

sub_df['selected_text'] = sub_df['selected_text'].apply(lambda x: x.replace('!!!!', '!') if len(x.split())==1 else x)
sub_df['selected_text'] = sub_df['selected_text'].apply(lambda x: x.replace('..', '.') if len(x.split())==1 else x)
sub_df['selected_text'] = sub_df['selected_text'].apply(lambda x: x.replace('...', '.') if len(x.split())==1 else x)
sub_df.to_csv('submission.csv', index=False)
sub_df.head()

