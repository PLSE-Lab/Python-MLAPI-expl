#!/usr/bin/env python
# coding: utf-8

# Using pytorch_pretrained_bert for token level classification - Followed the tutorial at  https://www.depends-on-the-definition.com/named-entity-recognition-with-bert/

# 

# 

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

# Any results you write to the current directory are saved as output.
from tqdm import tqdm, trange


# In[ ]:


input_data = pd.read_csv("../input/ner_dataset.csv", encoding="latin1")
#input_data


# In[ ]:


input_data = input_data.fillna(method="ffill")
input_data.tail(10)


# In[ ]:


words_list = list(set(input_data["Word"].values))
words_list[:10]


# In[ ]:


number_words = len(words_list); number_words # number of unique words in the corpus


# In[ ]:


class RetrieveSentance(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        function = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(function)
        self.sentences = [s for s in self.grouped]
    
    def retrieve(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


# In[ ]:


Sentences = RetrieveSentance(input_data)
Sentences


# In[ ]:


Sentences_list = [" ".join([s[0] for s in sent]) for sent in Sentences.sentences]
Sentences_list[0]


# In[ ]:


len(Sentences_list) #number of sentences 


# In[ ]:


labels = [[s[2] for s in sent] for sent in Sentences.sentences]
print(labels[0])


# In[ ]:


#labels [0] # list of lists of dimension (sentences,labels)


# In[ ]:


tags2vals = list(set(input_data["Tag"].values))
tag2idx = {t: i for i, t in enumerate(tags2vals)}


# In[ ]:


tags2vals # 17 kinds of tags 


# In[ ]:


tag2idx # indexing the tag 


# In[ ]:


import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam


# In[ ]:


max_seq_len = 75 # tokens
batch_s = 32 # batch size


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()


# In[ ]:


torch.cuda.get_device_name(0) 


# In[ ]:


tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=True)


# In[ ]:


tokenized_texts = [tokenizer.tokenize(sent) for sent in Sentences_list]
print(tokenized_texts[0])


# In[ ]:


len(tokenized_texts)


# In[ ]:


print(tokenized_texts[1])


# In[ ]:


X = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=max_seq_len, dtype="long", truncating="post", padding="post")


# In[ ]:


Y = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                     maxlen=max_seq_len, value=tag2idx["O"], padding="post",
                     dtype="long", truncating="post")


# In[ ]:


X.shape # (sentences, maximum sequence length)


# In[ ]:


Y.shape


# In[ ]:


X


# In[ ]:


Y


# In[ ]:


attention_masks = [[float(i>0) for i in ii] for ii in X]


# In[ ]:


len(attention_masks) # list of lists of shape (sentences, labels )


# In[ ]:


#attention_masks[0]


# In[ ]:


X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, 
                                                            random_state=20, test_size=0.1)
Mask_train, Mask_valid, _, _ = train_test_split(attention_masks, X,
                                             random_state=20, test_size=0.1)


# In[ ]:


X_train = torch.tensor(X_train)
X_valid = torch.tensor(X_valid)
Y_train = torch.tensor(Y_train)
Y_valid = torch.tensor(Y_valid)
Mask_train = torch.tensor(Mask_train)
Mask_valid = torch.tensor(Mask_valid)


# In[ ]:


data_train = TensorDataset(X_train, Mask_train, Y_train)
data_train_sampler = RandomSampler(data_train)
DL_train = DataLoader(data_train, sampler=data_train_sampler, batch_size=batch_s)

data_valid = TensorDataset(X_valid, Mask_valid, Y_valid)
data_valid_sampler = SequentialSampler(data_valid)
DL_valid = DataLoader(data_valid, sampler=data_valid_sampler, batch_size=batch_s)


# In[ ]:


model = BertForTokenClassification.from_pretrained("bert-base-cased", num_labels=len(tag2idx))


# In[ ]:


model.cuda();


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


#from seqeval.metrics import f1_score
from sklearn.metrics import f1_score
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# In[ ]:


epochs = 5
max_grad_norm = 1.0

for _ in trange(epochs, desc="Epoch"):
    # TRAIN loop
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(DL_train):
        # add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        # forward pass
        loss = model(b_input_ids, token_type_ids=None,
                     attention_mask=b_input_mask, labels=b_labels)
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
    for batch in DL_valid:
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
    pred_tags = [tags2vals[p_i] for p in predictions for p_i in p]
    valid_tags = [tags2vals[l_ii] for l in true_labels for l_i in l for l_ii in l_i]
    #print("F1-Score: {}".format(f1_score(pred_tags, valid_tags)))


# In[ ]:


model.eval()
predictions = []
true_labels = []
eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0
for batch in DL_valid:
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

pred_tags = [[tags2vals[p_i] for p_i in p] for p in predictions]
valid_tags = [[tags2vals[l_ii] for l_ii in l_i] for l in true_labels for l_i in l ]
print("Validation loss: {}".format(eval_loss/nb_eval_steps))
print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
#print("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags)))


# In[ ]:


valid_data = [tokenizer.convert_ids_to_tokens(id.numpy()) for id in X_valid]


# In[ ]:


submission = pd.DataFrame.from_dict({
    'text': valid_data,
    'valid_tags': valid_tags,
    'prediction': pred_tags
})
submission.to_csv('predicated_data.csv', index=False)


# In[ ]:


model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

# If we save using the predefined names, we can load using `from_pretrained`
#output_model_file = os.path.join(OUTPUT_DIR, WEIGHTS_NAME)
#output_config_file = os.path.join(OUTPUT_DIR, CONFIG_NAME)

torch.save(model_to_save.state_dict(), "pytorch_model.bin")
model_to_save.config.to_json_file("config.json")
#tokenizer.save_vocabulary("/vocab1.txt")


# In[ ]:



