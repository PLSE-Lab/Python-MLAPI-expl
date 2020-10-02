#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


get_ipython().system('nvidia-smi')


# In[ ]:


#!pip install -qq transformers


# In[ ]:


import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch

import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 12, 8

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device


# In[ ]:


df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
df.head()


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


sns.countplot(df.target)
plt.xlabel('Target')


# In[ ]:


PRE_TRAINED_MODEL_NAME = 'bert-base-cased'


# In[ ]:


tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)


# In[ ]:


sample_txt=df.iloc[0,3]


# In[ ]:


tokens = tokenizer.tokenize(sample_txt)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print(f' Sentence: {sample_txt}')
print(f'   Tokens: {tokens}')
print(f'Token IDs: {token_ids}')


# In[ ]:


tokenizer.sep_token, tokenizer.sep_token_id


# In[ ]:


tokenizer.cls_token, tokenizer.cls_token_id


# In[ ]:


tokenizer.pad_token, tokenizer.pad_token_id


# In[ ]:




encoding = tokenizer.encode_plus(
  sample_txt,
  max_length=32,
  add_special_tokens=True, # Add '[CLS]' and '[SEP]'
  return_token_type_ids=False,
  pad_to_max_length=True,
  return_attention_mask=True,
  return_tensors='pt',  # Return PyTorch tensors
)

encoding.keys()


# In[ ]:


token_lens = []

for txt in df.text:
  tokens = tokenizer.encode(txt, max_length=512)
  token_lens.append(len(tokens))


# In[ ]:


sns.distplot(token_lens)
plt.xlim([0, 256]);
plt.xlabel('Token count');


# In[ ]:


MAX_LEN = 120


# In[ ]:


class Dataset(Dataset):

  def __init__(self, reviews, targets, tokenizer, max_len):
    self.reviews = reviews
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len
  
  def __len__(self):
    return len(self.reviews)
  
  def __getitem__(self, item):
    review = str(self.reviews[item])
    target = self.targets[item]

    encoding = self.tokenizer.encode_plus(
      review,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
    )

    return {
      'review_text': review,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }


# In[ ]:


df_train, df_val = train_test_split(df, test_size=0.1, random_state=RANDOM_SEED)


# In[ ]:


df_train.shape, df_val.shape


# In[ ]:


def create_data_loader(df, tokenizer, max_len, batch_size):
  ds = Dataset(
    reviews=df.text.to_numpy(),
    targets=df.target.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
  )

  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=4
  )


# In[ ]:


BATCH_SIZE = 16

train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)


# In[ ]:


data = next(iter(train_data_loader))
data.keys()


# In[ ]:


print(data['input_ids'].shape)
print(data['attention_mask'].shape)
print(data['targets'].shape)


# In[ ]:


bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)


# In[ ]:


last_hidden_state, pooled_output = bert_model(
  input_ids=encoding['input_ids'], 
  attention_mask=encoding['attention_mask']
)


# In[ ]:


last_hidden_state.shape


# In[ ]:


bert_model.config.hidden_size


# In[ ]:


pooled_output.shape


# In[ ]:


class SentimentClassifier(nn.Module):

  def __init__(self, n_classes):
    super(SentimentClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
  
  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    output = self.drop(pooled_output)
    return self.out(output)


# In[ ]:


class_names=['real','fake']


# In[ ]:


model = SentimentClassifier(len(class_names))
model = model.to(device)


# In[ ]:


input_ids = data['input_ids'].to(device)
attention_mask = data['attention_mask'].to(device)

print(input_ids.shape) # batch size x seq length
print(attention_mask.shape) # batch size x seq length


# In[ ]:


F.softmax(model(input_ids, attention_mask), dim=1)


# In[ ]:


EPOCHS = 10

optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)

loss_fn = nn.CrossEntropyLoss().to(device)


# In[ ]:




def train_epoch(
  model, 
  data_loader, 
  loss_fn, 
  optimizer, 
  device, 
  scheduler, 
  n_examples
):
  model = model.train()

  losses = []
  correct_predictions = 0
  
  for d in data_loader:
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    targets = d["targets"].to(device)

    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )

    _, preds = torch.max(outputs, dim=1)
    loss = loss_fn(outputs, targets)

    correct_predictions += torch.sum(preds == targets)
    losses.append(loss.item())

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

  return correct_predictions.double() / n_examples, np.mean(losses)


# In[ ]:


def eval_model(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()

  losses = []
  correct_predictions = 0

  with torch.no_grad():
    for d in data_loader:
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)

      loss = loss_fn(outputs, targets)

      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())

  return correct_predictions.double() / n_examples, np.mean(losses)


# In[ ]:


get_ipython().run_cell_magic('time', '', "\nhistory = defaultdict(list)\nbest_accuracy = 0\n\nfor epoch in range(EPOCHS):\n\n  print(f'Epoch {epoch + 1}/{EPOCHS}')\n  print('-' * 10)\n\n  train_acc, train_loss = train_epoch(\n    model,\n    train_data_loader,    \n    loss_fn, \n    optimizer, \n    device, \n    scheduler, \n    len(df_train)\n  )\n\n  print(f'Train loss {train_loss} accuracy {train_acc}')\n\n  val_acc, val_loss = eval_model(\n    model,\n    val_data_loader,\n    loss_fn, \n    device, \n    len(df_val)\n  )\n\n  print(f'Val   loss {val_loss} accuracy {val_acc}')\n  print()\n\n  history['train_acc'].append(train_acc)\n  history['train_loss'].append(train_loss)\n  history['val_acc'].append(val_acc)\n  history['val_loss'].append(val_loss)\n\n  if val_acc > best_accuracy:\n    torch.save(model.state_dict(), 'best_model_state.bin')\n    best_accuracy = val_acc")


# In[ ]:


plt.plot(history['train_acc'], label='train accuracy')
plt.plot(history['val_acc'], label='validation accuracy')

plt.title('Training history')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.ylim([0, 1]);


# In[ ]:


test=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

sub=pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')


# In[ ]:


test.head()


# In[ ]:


class test_Dataset(Dataset):

  def __init__(self, reviews, tokenizer, max_len):
    self.reviews = reviews
    #self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len
  
  def __len__(self):
    return len(self.reviews)
  
  def __getitem__(self, item):
    review = str(self.reviews[item])
    #target = self.targets[item]

    encoding = self.tokenizer.encode_plus(
      review,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
    )

    return {
      'review_text': review,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten()
    }


# In[ ]:


def test_create_data_loader(df, tokenizer, max_len, batch_size):
  ds = test_Dataset(
    reviews=df.text.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
  )

  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=4
  )


# In[ ]:


test_data_loader = test_create_data_loader(test, tokenizer, MAX_LEN, BATCH_SIZE)


# In[ ]:


test_data = next(iter(test_data_loader))
test_data.keys()

print(test_data['input_ids'].shape)
print(test_data['attention_mask'].shape)


# In[ ]:


def get_predictions(model, data_loader):
  model = model.eval()
  
  review_texts = []
  predictions = []
  prediction_probs = []
  real_values = []

  with torch.no_grad():
    for d in data_loader:

      texts = d["review_text"]
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      #targets = d["targets"].to(device)

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)

      probs = F.softmax(outputs, dim=1)

      review_texts.extend(texts)
      predictions.extend(preds)
      prediction_probs.extend(probs)
      #real_values.extend(targets)

  predictions = torch.stack(predictions).cpu()
  prediction_probs = torch.stack(prediction_probs).cpu()
  #real_values = torch.stack(real_values).cpu()
  return review_texts, predictions, prediction_probs


# In[ ]:


y_review_texts, y_pred, y_pred_probs = get_predictions(model,test_data_loader)


# In[ ]:


predict=y_pred.numpy()


# In[ ]:


sub.head()


# In[ ]:


sub.target = predict


# In[ ]:


sub.head()


# In[ ]:


sub.to_csv('submit_bert.csv',index=False)


# In[ ]:




