#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
get_ipython().system('pip install pytorch-pretrained-bert')
get_ipython().system('pip install seqeval')

# Any results you write to the current directory are saved as output.


# In[ ]:


dframe = pd.read_csv("../input/ner.csv", encoding = "ISO-8859-1", error_bad_lines=False)


# In[ ]:


dframe.head()


# In[ ]:


dataset=dframe.drop(['Unnamed: 0', 'lemma', 'next-lemma', 'next-next-lemma', 'next-next-pos',
       'next-next-shape', 'next-next-word', 'next-pos', 'next-shape',
       'next-word', 'prev-iob', 'prev-lemma', 'prev-pos',
       'prev-prev-iob', 'prev-prev-lemma', 'prev-prev-pos', 'prev-prev-shape',
       'prev-prev-word', 'prev-shape', 'prev-word','shape'],axis=1)


# In[ ]:


dataset.head()


# In[ ]:


class SentenceGetter(object):
    
    def __init__(self, dataset):
        self.n_sent = 1
        self.dataset = dataset
        self.empty = False
        agg_func = lambda s: [(w,p, t) for w,p, t in zip(s["word"].values.tolist(),
                                                       s['pos'].values.tolist(),
                                                        s["tag"].values.tolist())]
        self.grouped = self.dataset.groupby("sentence_idx").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


# In[ ]:


getter = SentenceGetter(dataset)


# In[ ]:


sentences = [" ".join([s[0] for s in sent]) for sent in getter.sentences]
sentences[0]


# In[ ]:


labels = [[s[2] for s in sent] for sent in getter.sentences]
print(np.array(labels))


# In[ ]:


tags_vals = list(set(dataset["tag"].values))
tag2idx = {t: i for i, t in enumerate(tags_vals)}


# In[ ]:


tags_vals


# In[ ]:


import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig, BertModel
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam
from tqdm import tqdm, trange


# In[ ]:


MAX_LEN = 200 # maximum length of the sentence
bs = 32


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()


# In[ ]:


torch.cuda.get_device_name(0) 


# In[ ]:


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


# In[ ]:


tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
print(tokenized_texts[0])


# In[ ]:


input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")


# In[ ]:


tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                     maxlen=MAX_LEN, value=tag2idx["O"], padding="post",
                     dtype="long", truncating="post")


# In[ ]:


attention_masks = [[float(i>0) for i in ii] for ii in input_ids]


# In[ ]:


tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags, 
                                                            random_state=2018, test_size=0.1)
tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                             random_state=2018, test_size=0.1)


# In[ ]:


tr_inputs = torch.tensor(tr_inputs)
val_inputs = torch.tensor(val_inputs)
tr_tags = torch.tensor(tr_tags)
val_tags = torch.tensor(val_tags)
tr_masks = torch.tensor(tr_masks)
val_masks = torch.tensor(val_masks)


# In[ ]:


train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)


# In[ ]:


model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(tag2idx))


# In[ ]:


# model


# In[ ]:


model.cuda()


# In[ ]:


# no_decay = ['bias', 'gamma', 'beta']
# [n for n, p in list(model.named_parameters()) if any(nd in n for nd in no_decay)]

# print('ac' in 'abc')


# In[ ]:


FULL_FINETUNING = True
if FULL_FINETUNING:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
else:
    param_optimizer = list(model.classifier.named_parameters()) 
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)


# In[ ]:


from seqeval.metrics import f1_score

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# In[ ]:


# import sys
# def q():
#     sys.exit()


# In[ ]:


epochs = 10
max_grad_norm = 1.0

for _ in trange(epochs, desc="Epoch"):
    # TRAIN loop
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(train_dataloader):
        # add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
#         print(b_input_ids, b_input_mask, b_labels)
        # forward pass
        loss = model(b_input_ids, token_type_ids=None,
                     attention_mask=b_input_mask, labels=b_labels)
#         print(loss)
#         q()
#         print(b_input_ids.shape, " ---", b_input_mask.shape, "---", b_labels)
#         print(loss)  
#         q()
        # backward pass
        loss.backward()
        # track train loss
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        # update parameters
        optimizer.step()
        model.zero_grad()
    # print train loss per epoch
    print("Train loss: {}".format(tr_loss/nb_tr_steps))
    # VALIDATION on validation set
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions , true_labels = [], []
    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        
        with torch.no_grad():
            tmp_eval_loss = model(b_input_ids, token_type_ids=None,
                                  attention_mask=b_input_mask, labels=b_labels)
            logits = model(b_input_ids, token_type_ids=None,
                           attention_mask=b_input_mask)
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.append(label_ids)
        
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        
        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy
        
        nb_eval_examples += b_input_ids.size(0)
        nb_eval_steps += 1
    eval_loss = eval_loss/nb_eval_steps
    print("Validation loss: {}".format(eval_loss))
    print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
    pred_tags = [tags_vals[p_i] for p in predictions for p_i in p]
    valid_tags = [tags_vals[l_ii] for l in true_labels for l_i in l for l_ii in l_i]
    print("F1-Score: {}".format(f1_score(pred_tags, valid_tags)))


# In[ ]:


model.eval()
predictions = []
true_labels = []
eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0
for batch in valid_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch

    with torch.no_grad():
        tmp_eval_loss = model(b_input_ids, token_type_ids=None,
                              attention_mask=b_input_mask, labels=b_labels)
        logits = model(b_input_ids, token_type_ids=None,
                       attention_mask=b_input_mask)
        
    logits = logits.detach().cpu().numpy()
    predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
    label_ids = b_labels.to('cpu').numpy()
    true_labels.append(label_ids)
    tmp_eval_accuracy = flat_accuracy(logits, label_ids)

    eval_loss += tmp_eval_loss.mean().item()
    eval_accuracy += tmp_eval_accuracy

    nb_eval_examples += b_input_ids.size(0)
    nb_eval_steps += 1

pred_tags = [[tags_vals[p_i] for p_i in p] for p in predictions]
valid_tags = [[tags_vals[l_ii] for l_ii in l_i] for l in true_labels for l_i in l ]
print("Validation loss: {}".format(eval_loss/nb_eval_steps))
print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
print("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags)))


# In[ ]:


# import torch.nn.functional as F


# In[ ]:


# #prediction function
# text = "This is my first NER model. Working from Gurgaon."
# tokenized_text = tokenizer.tokenize(text)
# input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(tokenized_text)],
#                           maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

# attention_masks = [[float(i>0) for i in ii] for ii in input_ids]

# #convert to tensor
# input_ids_tensor = torch.tensor(input_ids).to(device)
# attention_masks_tensor = torch.tensor(attention_masks).to(device)

# # print(input_ids_tensor, attention_masks_tensor)

# prediction = model(input_ids_tensor, token_type_ids=None,
#                      attention_mask=attention_masks_tensor)

# logits = F.softmax(prediction,dim=2)

# logits_label = torch.argmax(logits,dim=2)

# logits_label = logits_label.detach().cpu().numpy().tolist()[0]
# prediction.shape


# In[ ]:


# len(tokenized_text), len(logits_label[0:len(tokenized_text)])
# for text, label in zip(tokenized_text,logits_label[0:len(tokenized_text)]):
#     print(text, tags_vals[label])


# In[ ]:


# logits_confidence = [values[label].item() for values,label in zip(logits[0],logits_label)]
# logits_confidence

# logits = []
# pos = 0
# for index,mask in enumerate(valid_ids[0]):
#     if index == 0:
#         continue
#     if mask == 1:
#         logits.append((logits_label[index-pos],logits_confidence[index-pos]))
#     else:
#         pos += 1
# logits.pop()

# labels = [(self.label_map[label],confidence) for label,confidence in logits]
# words = word_tokenize(text)
# assert len(labels) == len(words)
# output = [{"word":word,"tag":label,"confidence":confidence} for word,(label,confidence) in zip(words,labels)]

# print(output)


# In[ ]:


import pickle
pickle.dump(model, open('BERT_NER', 'wb'))
pickle.dump(model.state_dict(), open('BERT_NER_state_dict', 'wb'))

torch.save(model, 'bert_model')
torch.save(model.state_dict(), 'bert_model_state_dict')

