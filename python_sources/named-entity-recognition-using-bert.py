#!/usr/bin/env python
# coding: utf-8

# f1 socre: 0.816981
# 
# Accuracy score: 0.967118

# In[ ]:


get_ipython().system(' pip install seqeval')


# In[ ]:


get_ipython().system(' pip install pytorch-transformers')


# The process of doing NER with BERT contains 4 steps:
# 1. Load data
# 2. Set data into training embeddings
# 3. Train model
# 4. Evaluate model performance

# In[ ]:


import pandas as pd
import math
import numpy as np
from seqeval.metrics import f1_score
from seqeval.metrics import classification_report,accuracy_score,f1_score
import torch.nn.functional as F
import torch
import os
from tqdm import tqdm,trange
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_transformers import BertTokenizer, BertConfig
from pytorch_transformers import BertForTokenClassification, AdamW


# In[ ]:


get_ipython().system(' ls ../input')


# In[ ]:


df_data = pd.read_csv("../input/ner_dataset.csv",sep=",",encoding="latin1").fillna(method='ffill')
df_data.head()


# In[ ]:


df_data.POS.unique()


# In[ ]:


df_data.Tag.value_counts()


# In[ ]:


class SentenceGetter(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


# In[ ]:


getter = SentenceGetter(df_data)
sentences = [[s[0] for s in sent] for sent in getter.sentences]
print(sentences[0])


# In[ ]:


poses = [[s[1] for s in sent] for sent in getter.sentences]
labels = [[s[2] for s in sent] for sent in getter.sentences]

print (sentences[0])
print (poses[0])
print(labels[0])


# In[ ]:


tags_vals = list(set(df_data["Tag"].values))


# In[ ]:


# Add X  label for word piece support
# Add [CLS] and [SEP] as BERT need
tags_vals.append('X')
tags_vals.append('[CLS]')
tags_vals.append('[SEP]')
tags_vals = set(tags_vals)
tags_vals


# In[ ]:


# Set a dict for mapping id to tag name
#tag2idx = {t: i for i, t in enumerate(tags_vals)}

# Recommend to set it by manual define, good for reusing
tag2idx={'B-art': 14,
 'B-eve': 16,
 'B-geo': 0,
 'B-gpe': 13,
 'B-nat': 12,
 'B-org': 10,
 'B-per': 4,
 'B-tim': 2,
 'I-art': 5,
 'I-eve': 7,
 'I-geo': 15,
 'I-gpe': 8,
 'I-nat': 11,
 'I-org': 3,
 'I-per': 6,
 'I-tim': 1,
 'X':17,
 'O': 9,
 '[CLS]':18,
 '[SEP]':19}


# In[ ]:


tag2idx


# In[ ]:


# Mapping index to name
tag2name={tag2idx[key] : key for key in tag2idx.keys()}


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()


# In[ ]:


n_gpu


# In[ ]:


# Len of the sentence must be the same as the training model
# See model's 'max_position_embeddings' = 512
max_len  = 45
# load tokenizer, with manual file address or pretrained address
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# In[ ]:


tokenized_texts = []
word_piece_labels = []
i_inc = 0
for word_list,label in (zip(sentences,labels)):
    temp_lable = []
    temp_token = []
    
    # Add [CLS] at the front 
    temp_lable.append('[CLS]')
    temp_token.append('[CLS]')
    
    for word,lab in zip(word_list,label):
        token_list = tokenizer.tokenize(word)
        for m,token in enumerate(token_list):
            temp_token.append(token)
            if m==0:
                temp_lable.append(lab)
            else:
                temp_lable.append('X')  
                
    # Add [SEP] at the end
    temp_lable.append('[SEP]')
    temp_token.append('[SEP]')
    
    tokenized_texts.append(temp_token)
    word_piece_labels.append(temp_lable)
    
    if 5 > i_inc:
        print("No.%d,len:%d"%(i_inc,len(temp_token)))
        print("texts:%s"%(" ".join(temp_token)))
        print("No.%d,len:%d"%(i_inc,len(temp_lable)))
        print("lables:%s"%(" ".join(temp_lable)))
    i_inc +=1


# In[ ]:


# Make text token into id
input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=max_len, dtype="long", truncating="post", padding="post")
print(input_ids[0])


# In[ ]:


# Make label into id, pad with "O" meaning others
tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in word_piece_labels],
                     maxlen=max_len, value=tag2idx["O"], padding="post",
                     dtype="long", truncating="post")
print(tags[0])


# In[ ]:


# For fine tune of predict, with token mask is 1,pad token is 0
attention_masks = [[int(i>0) for i in ii] for ii in input_ids]
attention_masks[0];


# In[ ]:


# Since only one sentence, all the segment set to 0
segment_ids = [[0] * len(input_id) for input_id in input_ids]
segment_ids[0];


# In[ ]:



tr_inputs, val_inputs, tr_tags, val_tags,tr_masks, val_masks,tr_segs, val_segs = train_test_split(input_ids, tags,attention_masks,segment_ids, 
                                                            random_state=4, test_size=0.3)


# In[ ]:


tr_inputs = torch.tensor(tr_inputs)
val_inputs = torch.tensor(val_inputs)
tr_tags = torch.tensor(tr_tags)
val_tags = torch.tensor(val_tags)
tr_masks = torch.tensor(tr_masks)
val_masks = torch.tensor(val_masks)
tr_segs = torch.tensor(tr_segs)
val_segs = torch.tensor(val_segs)


# In[ ]:


batch_num = 32


# In[ ]:


# Only set token embedding, attention embedding, no segment embedding
train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
train_sampler = RandomSampler(train_data)
# Drop last can make batch training better for the last one
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_num,drop_last=True)

valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=batch_num)


# **TRAIN THE MODEL**

# In[ ]:


model = BertForTokenClassification.from_pretrained("bert-base-uncased",num_labels=len(tag2idx))


# In[ ]:


model;


# In[ ]:


model.cuda();


# In[ ]:


# Add multi GPU support
if n_gpu >1:
    model = torch.nn.DataParallel(model)


# In[ ]:


# Set epoch and grad max num
epochs = 5
max_grad_norm = 1.0


# In[ ]:


# Cacluate train optimiazaion num
num_train_optimization_steps = int( math.ceil(len(tr_inputs) / batch_num) / 1) * epochs


# In[ ]:


# True: fine tuning all the layers 
# False: only fine tuning the classifier layers
FULL_FINETUNING = True


# In[ ]:


if FULL_FINETUNING:
    # Fine tune model all layer parameters
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
else:
    # Only fine tune classifier parameters
    param_optimizer = list(model.classifier.named_parameters()) 
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5)


# In[ ]:


# TRAIN loop
model.train();


# In[ ]:


print("***** Running training *****")
print("  Num examples = %d"%(len(tr_inputs)))
print("  Batch size = %d"%(batch_num))
print("  Num steps = %d"%(num_train_optimization_steps))
for _ in trange(epochs,desc="Epoch"):
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(train_dataloader):
        # add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        
        # forward pass
        outputs = model(b_input_ids, token_type_ids=None,
        attention_mask=b_input_mask, labels=b_labels)
        loss, scores = outputs[:2]
        if n_gpu>1:
            # When multi gpu, average it
            loss = loss.mean()
        
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
        optimizer.zero_grad()
        
    # print train loss per epoch
    print("Train loss: {}".format(tr_loss/nb_tr_steps))


# In[ ]:


bert_out_address = 'models/bert_out_model/en09'
# Make dir if not exits
if not os.path.exists(bert_out_address):
        os.makedirs(bert_out_address)


# In[ ]:


model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self


# In[ ]:


# If we save using the predefined names, we can load using `from_pretrained`
output_model_file = os.path.join(bert_out_address, "pytorch_model.bin")
output_config_file = os.path.join(bert_out_address, "config.json")

# Save model into file
torch.save(model_to_save.state_dict(), output_model_file)
model_to_save.config.to_json_file(output_config_file)
tokenizer.save_vocabulary(bert_out_address)


# In[ ]:


model = BertForTokenClassification.from_pretrained(bert_out_address,num_labels=len(tag2idx))

model.cuda(); # Set model to GPU

if n_gpu >1:
    model = torch.nn.DataParallel(model)


# In[ ]:


model.eval();


# In[ ]:


eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0
y_true = []
y_pred = []

print("***** Running evaluation *****")
print("  Num examples ={}".format(len(val_inputs)))
print("  Batch size = {}".format(batch_num))
for step, batch in enumerate(valid_dataloader):
    batch = tuple(t.to(device) for t in batch)
    input_ids, input_mask, label_ids = batch
    
#     if step > 2:
#         break
    
    with torch.no_grad():
        outputs = model(input_ids, token_type_ids=None,
        attention_mask=input_mask,)
        # For eval mode, the first result of outputs is logits
        logits = outputs[0] 
    
    # Get NER predict result
    logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
    logits = logits.detach().cpu().numpy()
    
    
    # Get NER true result
    label_ids = label_ids.to('cpu').numpy()
    
    
    # Only predict the real word, mark=0, will not calculate
    input_mask = input_mask.to('cpu').numpy()
    
    # Compare the valuable predict result
    for i,mask in enumerate(input_mask):
        # Real one
        temp_1 = []
        # Predict one
        temp_2 = []
        
        for j, m in enumerate(mask):
            # Mark=0, meaning its a pad word, dont compare
            if m:
                if tag2name[label_ids[i][j]] != "X" and tag2name[label_ids[i][j]] != "[CLS]" and tag2name[label_ids[i][j]] != "[SEP]" : # Exclude the X label
                    temp_1.append(tag2name[label_ids[i][j]])
                    temp_2.append(tag2name[logits[i][j]])
            else:
                break
        
            
        y_true.append(temp_1)
        y_pred.append(temp_2)

        

print("f1 socre: %f"%(f1_score(y_true, y_pred)))
print("Accuracy score: %f"%(accuracy_score(y_true, y_pred)))

# Get acc , recall, F1 result report
report = classification_report(y_true, y_pred,digits=4)

# Save the report into file
output_eval_file = os.path.join(bert_out_address, "eval_results.txt")
with open(output_eval_file, "w") as writer:
    print("***** Eval results *****")
    print("\n%s"%(report))
    print("f1 socre: %f"%(f1_score(y_true, y_pred)))
    print("Accuracy score: %f"%(accuracy_score(y_true, y_pred)))
    
    writer.write("f1 socre:\n")
    writer.write(str(f1_score(y_true, y_pred)))
    writer.write("\n\nAccuracy score:\n")
    writer.write(str(accuracy_score(y_true, y_pred)))
    writer.write("\n\n")  
    writer.write(report)


# **PREDICT**

# In[ ]:


import nltk
sentences = "I bought a new iphone and its Iphone X from Apple"
word_tokens = nltk.word_tokenize(sentences)
pos_tags = nltk.pos_tag(word_tokens)
tokenized_texts = []
word_piece_labels = []
i_inc = 0
temp_token = []
# Add [CLS] at the front 
temp_token.append('[CLS]')
for word,lab in pos_tags:
    token_list = tokenizer.tokenize(word)
    for m,token in enumerate(token_list):
        temp_token.append(token)
# Add [SEP] at the end
temp_token.append('[SEP]')
tokenized_texts.append(temp_token)
print("texts:%s"%(" ".join(temp_token)))
input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=max_len, dtype="long", truncating="post", padding="post")
print(input_ids[0])
b_input_mask = ""


# In[ ]:





# In[ ]:





# In[ ]:


# source: https://github.com/billpku/NLP_In_Action/blob/master/NER_with_BERT.ipynb


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




