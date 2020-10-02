#!/usr/bin/env python
# coding: utf-8

# This is adpated from [this](https://www.kaggle.com/ozdata/bert-huggingface-pytorch), and a few changes are added to try different options. 
# 
# It turns out higher number of epoch (5) or larger learning rates(X5 or X10) doesn't help(decrease the performance of both crossvalidation and final). One epoch with a little bit train is enough. bert-large-uncased not help as well

# In[ ]:


import pandas as pd
import numpy as np

import torch
import tensorflow as tf
import transformers
from transformers import *
from sklearn import metrics
from sklearn.model_selection import KFold

import time
import datetime
import random

print('Transformers version: ', transformers.__version__)
print('Tensorflow version: ', tf.__version__)


# # Importing Data

# In[ ]:


randomseed=42
bertmodel='bert-large-uncased'#bert-base-uncased
data_dir = '/kaggle/input/nlp-getting-started/'
train_df = pd.read_csv(data_dir+'train.csv')
test_df = pd.read_csv(data_dir+'test.csv')
train_df = train_df.sample(n=len(train_df), random_state=randomseed)# random permutation of the sample
sample_submission = pd.read_csv(data_dir+'sample_submission.csv')
# print(train_df['target'].value_counts())
# train_df.head(2)


# In[ ]:


x_train = train_df['text']
y_train = train_df['target']
x_test = test_df['text']


# # Tokenization and Input Formatting
# ### Sequence to IDs

# In[ ]:


tokenizer = transformers.BertTokenizer.from_pretrained(bertmodel, do_lower_case=True)


# In[ ]:


def encode_tweet(df):
    input_ids = []

    for x in df:
        encoded_x = tokenizer.encode(x,
                                    add_special_tokens = True)
        input_ids.append(encoded_x)
    return input_ids


# In[ ]:


test_input = encode_tweet(x_test)
# print('Original: ', x_test[0])
# print('Encoded: ', test_input[0])


# In[ ]:


train_input = encode_tweet(x_train)


# some self test on bert and it's found some 'word' are encoded by more than one long value

# In[ ]:


# a=encode_tweet(x_train)
# a=encode_tweet(x_train[0])
# a=encode_tweet(['Forest fire near La Ronge Sask. Canada'])
# a=tokenizer.encode(x_train[0],add_special_tokens = True)
# a[0]
# len(a)
# a
# x_train[1]
# x_train[0]
# x_train[0:3]
# a[1]
# len(a[0])
# len(x_train)
# len(x_train)
# x_train.shape
# help(x_train)
# type(x_train)
# x_train
# df=x_train[1:2]
# df
# input_ids = []

# for x in df:
#     encoded_x = tokenizer.encode(x,add_special_tokens = True)
#     input_ids.append(encoded_x)
#     print(encoded_x)
#     print(len(encoded_x))
#     print(x)
#     print(len(x))
# encoded_x = tokenizer.encode('un-imaginable',add_special_tokens = True)
# print(encoded_x)
# print(len(encoded_x))


# ### Padding

# In[ ]:


# print('Max sentence length in Train Ids: ', max([len(sen) for sen in train_input]))
# print('Max sentence length in Test Ids: ', max([len(sen) for sen in test_input]))
# train_input[1]


# In[ ]:


from keras.preprocessing.sequence import pad_sequences
MAX_LEN = 84

def pad_tweets(df):
    df = pad_sequences(df, maxlen=MAX_LEN, dtype="long", 
                       value=0, truncating="post", padding="post")
    return df


# In[ ]:


train_input = pad_tweets(train_input)
test_input = pad_tweets(test_input)


# ### Attention Masks

# In[ ]:


def get_att_mask(df):
    attention_masks = []

    for tweet in df:
        att_mask = [int(token_id > 0) for token_id in tweet]
        attention_masks.append(att_mask)
    return attention_masks


# In[ ]:


train_att = get_att_mask(train_input)
test_att = get_att_mask(test_input)


# In[ ]:


# test_att[1]


# In[ ]:


from sklearn.model_selection import train_test_split

tr_input, val_input, tr_label, val_label = train_test_split(train_input, y_train, 
                                                            random_state=2020, test_size=0.15)
# Do the same for the masks.
tr_mask, val_mask, _, _ = train_test_split(train_att, y_train,
                                             random_state=2020, test_size=0.15)


# Converting to Torch Tensors

# In[ ]:


# For Training and Validation data and masks
tr_input = torch.tensor(tr_input)
val_input = torch.tensor(val_input)

#convert to np.array, otherwise throws a mysterious 'KeyError: 4' error
tr_label = torch.tensor(np.array(tr_label))
val_label = torch.tensor(np.array(val_label))

tr_mask = torch.tensor(tr_mask)
val_mask = torch.tensor(val_mask)

# For Test data and mask
te_input = torch.tensor(test_input)
te_mask = torch.tensor(test_att)


# Using helper classes in order to use batches for training. It creates an iterator, which should save on memory during training. The same must be repeated on the test set once we have prediction labels.

# In[ ]:


from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

BATCH_SIZE = 32 

# For training
train_data = TensorDataset(tr_input, tr_mask, tr_label)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler = train_sampler, batch_size = BATCH_SIZE)

# For validation
val_data = TensorDataset(val_input, val_mask, val_label)
val_sampler = RandomSampler(val_data)
val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size = BATCH_SIZE)


# # Training the Model
# Using the pre-trained model, documentation can be found here
# https://huggingface.co/transformers/v2.2.0/main_classes/model.html#transformers.PreTrainedModel.from_pretrained

# In[ ]:


from transformers import BertForSequenceClassification, AdamW, BertConfig

model = BertForSequenceClassification.from_pretrained(bertmodel,
                                                      num_labels = 2,
                                                      output_attentions = False,
                                                      output_hidden_states = False)


# In[ ]:


model.cuda()


# Displaying some of the model's parameters:

# In[ ]:


# Get all of the model's parameters as a list of tuples.
params = list(model.named_parameters())

print('The BERT model has {:} different named parameters.\n'.format(len(params)))

print('==== Embedding Layer ====\n')

for p in params[0:5]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== First Transformer ====\n')

for p in params[5:21]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== Output Layer ====\n')

for p in params[-4:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))


# ### AdamW optimizer
# 
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L109

# In[ ]:


from transformers import get_linear_schedule_with_warmup

# AdamW is a class from the Huggingface library
optimizer = AdamW(model.parameters(),
                  lr = 1e-5, # default is 5e-5
                  eps = 1e-8 # default is 1e-8
                )
epochs = 1 # 1 epoch gave the best result

# the number of batches times the number of epochs
total_steps = len(train_dataloader) * epochs

# the learning rate scheduler
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)


# ### Training Loop

# In[ ]:


# Accuracy helper function
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# In[ ]:


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

# In[ ]:


if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


# In[ ]:


seed = 50

random.seed(seed)
np.random.seed(seed)

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

loss_arr = []

for i in range(0, epochs):
    
    # ========= Training ==========
    
    print('====== Epoch {:} of {:}'.format(i+1, epochs))
    print('Training...')
    
    t0 = time.time()
    
    total_loss = 0
    # initialize training mode
    model.train()
    
    for step, batch in enumerate(train_dataloader):
        if step % 30 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('Batch {:>5,} of {:>5,}. Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            
            # Unpacking the training batch from dataloader and copying each tensor to the GPU
            
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        # pytorch doesn't clear previously calculated gradients
        # before performing backward pass, so clearing here:
        model.zero_grad()
        
        outputs = model(b_input_ids,
                       token_type_ids = None, 
                       attention_mask = b_input_mask,
                       labels = b_labels)
        loss = outputs[0]
        
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        #update the learning rate
        scheduler.step()
    
    avg_train_loss = total_loss / len(train_dataloader)
    
    loss_arr.append(avg_train_loss)
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(format_time(time.time() - t0)))
    
    # ========= Validation ==========
    
    print("")
    print("Running Validation...")
    t0 = time.time()
    # evaluation mode
    model.eval()
    
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    
    for batch in val_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        with torch.no_grad():
            
            outputs = model(b_input_ids, 
                           token_type_ids = None, 
                           attention_mask = b_input_mask)
            
        logits = outputs[0]
        # move logits to cpu
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        # get accuracy
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        
        eval_accuracy += tmp_eval_accuracy
        
        nb_eval_steps += 1
    
    print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))
    
print("")
print("Training complete!")


# In[ ]:


import matplotlib.pyplot as plt


import seaborn as sns

# Use plot styling from seaborn.
sns.set(style='darkgrid')

# Increase the plot size and font size.
sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (12,6)

# Plot the learning curve.
plt.plot(loss_arr, 'b-o')

# Label the plot.
plt.title("Training loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.show()


# # Testing

# In[ ]:


pred_labels = np.array(sample_submission['target'])


# In[ ]:


# b_labels


# In[ ]:


te_labels = torch.tensor(pred_labels)


# In[ ]:


prediction_data = TensorDataset(te_input, te_mask, te_labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler = prediction_sampler, batch_size = BATCH_SIZE)


# In[ ]:


print('Predicting labels for {:,} test sentences...'.format(len(te_input)))

# Put model in evaluation mode
model.eval()

# Tracking variables 
predictions , true_labels = [], []

# Predict 
for batch in prediction_dataloader:
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    b_labels = batch[2].to(device)
    
    # Telling the model not to compute or store gradients, saving memory and 
    # speeding up prediction
    with torch.no_grad():
      # Forward pass, calculate logit predictions
        outputs = model(b_input_ids, token_type_ids=None,
                        attention_mask=b_input_mask)

    logits = outputs[0]

    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
  
    # Store predictions and true labels
    predictions.append(logits)
    true_labels.append(label_ids)

print('    DONE.')


# In[ ]:


flat_predictions = [item for sublist in predictions for item in sublist]
# flat_predictions
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()


# In[ ]:


# predictions


# In[ ]:


flat_predictions


# In[ ]:


sample_submission['target'] = flat_predictions
sample_submission.to_csv('submission.csv', index = False)


# In[ ]:


# sample_submission

