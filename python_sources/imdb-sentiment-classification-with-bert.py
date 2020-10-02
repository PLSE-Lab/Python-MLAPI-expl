#!/usr/bin/env python
# coding: utf-8

# Here is a statring kernel that aims to do sentiment prediction on IMDB dataset using the famous BERT model in a simple way that allow everyone of you to understand how to apply it.
# I hope you'll enjoy this
# This notebook is inspired from this tuto:
# https://www.curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/

# # Setup and imports
# 
# Let's start with installing the transformers library and some required imports for the rest of the notebook

# In[ ]:


get_ipython().system('pip install transformers')


# In[ ]:


import torch
from transformers import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# # Data exploration & visualization

# We use the IMDB reviews dataset available from kaggle datasets through this link:
# https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

# In[ ]:


df = pd.read_csv("../input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv")
df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x=='positive' else 0)
df.head()


# In[ ]:


#dataset dimensions
df.shape


# In[ ]:


#Labeled reviews barplot
sns.countplot(df.sentiment)
plt.xlabel('sentiments')


# As it's shown by the figure above the dataset is perfectly balanced with equal number of reviews text for the two classes.
# 0 means that the review is negative and 1 for positive

# # BERT (Bidirectional Encoder Representations from Transformers)

# The Transformers library provides a wide variety of Transformer models (including BERT). It works with TensorFlow and PyTorch! It also includes prebuild tokenizers to do the heavy work required for bert model input.

# In[ ]:


#Selecting the bert-base-cased
PRE_TRAINED_MODEL_NAME = '../input/bert-base-cased/'


# In[ ]:


# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)


# In[ ]:


#Let's see how Bertencoder is encoding a sentence
sample_txt = 'When was I last outside? I am stuck at home for 2 weeks.'
tokens = tokenizer.tokenize(sample_txt)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print(f' Sentence: {sample_txt}')
print(f'   Tokens: {tokens}')
print(f'Token IDs: {token_ids}')


# Some special tokens id:

# In[ ]:


tokenizer.sep_token, tokenizer.sep_token_id


# In[ ]:


tokenizer.cls_token, tokenizer.cls_token_id


# In[ ]:


tokenizer.pad_token, tokenizer.pad_token_id


# In[ ]:


tokenizer.unk_token, tokenizer.unk_token_id


# In[ ]:


#To do all preprocessing you specify some parameters in the encod_plus() method of the tokenizer
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


#The tokens ids list
print(len(encoding['input_ids'][0]))
encoding['input_ids'][0]


# In[ ]:


#The attentions masked tokens
print(len(encoding['attention_mask'][0]))
encoding['attention_mask']


# In[ ]:


#let's see how the sentence is tokenized with bert tokenizer
tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])


# In order to get the suitable sequence length we need to take a look at all reviews length and then select the most appropriate one

# In[ ]:


token_lens = []

for txt in df.review:
  tokens = tokenizer.encode(txt, max_length=512)
  token_lens.append(len(tokens))
sns.distplot(token_lens)
plt.xlim([0, 500]);
plt.xlabel('Token count')


# In[ ]:


MAX_LEN = 200      #for not consuming much resources
RANDOM_SEED = 42
device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )


# In[ ]:


#doing the split of the dataset into training, validation and testing sets
df_train, df_test = train_test_split(df, test_size=0.1, random_state=RANDOM_SEED)
df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)


# In[ ]:


df_train.shape, df_val.shape, df_test.shape


# # Creating Dataset and Dataloader

# In[ ]:


class IMDBDataset(Dataset):

  def __init__(self, reviews, sentiments, tokenizer, max_len):
    self.reviews = reviews
    self.sentiments = sentiments
    self.tokenizer = tokenizer
    self.max_len = max_len
  
  def __len__(self):
    return len(self.reviews)
  
  def __getitem__(self, item):
    review = str(self.reviews[item])
    sentiment = self.sentiments[item]

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
      'review': review,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'sentiments': torch.tensor(sentiment, dtype=torch.long)
    }


# In[ ]:


def create_data_loader(df, tokenizer, max_len, batch_size):
  ds = IMDBDataset(
    reviews=df.review.to_numpy(),
    sentiments=df.sentiment.to_numpy(),
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
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)


# In[ ]:


data = next(iter(train_data_loader))
data.keys()


# In[ ]:


print(data['input_ids'].shape)
print(data['attention_mask'].shape)
print(data['sentiments'].shape)


# As you see here we get a training data tensor of size (batch_size * max_length) 

# # Reviews Classification with BERT and Hugging Face

# In[ ]:


import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# In[ ]:


class IMDBClassifier(nn.Module):

  def __init__(self, n_classes):
    super(IMDBClassifier, self).__init__()
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


model = IMDBClassifier(len(df['sentiment'].unique()))
model = model.to(device)


# In[ ]:


input_ids = data['input_ids'].to(device)
attention_mask = data['attention_mask'].to(device)

print(input_ids.shape) # batch size x seq length
print(attention_mask.shape) # batch size x seq length


# In[ ]:


F.softmax(model(input_ids, attention_mask), dim=1)


# # Training (fine-tuning of BERT for classification task)
# 
# How to come up with all hyperparameters? The BERT authors in this [paper](https://arxiv.org/pdf/1810.04805.pdf) have some recommendations for fine-tuning:
# 
# * Batch size: 16, 32
# * Learning rate (Adam): 5e-5, 3e-5, 2e-5
# * Number of epochs: 2, 3, 4

# In[ ]:


EPOCHS = 4

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
    sentiments = d["sentiments"].to(device)

    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )

    _, preds = torch.max(outputs, dim=1)
    loss = loss_fn(outputs, sentiments)

    correct_predictions += torch.sum(preds == sentiments)
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
      sentiments = d["sentiments"].to(device)

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)

      loss = loss_fn(outputs, sentiments)

      correct_predictions += torch.sum(preds == sentiments)
      losses.append(loss.item())

  return correct_predictions.double() / n_examples, np.mean(losses)


# In[ ]:


get_ipython().run_cell_magic('time', '', "\ntrain_a = []\ntrain_l = []\nval_a = []\nval_l = []\nbest_accuracy = 0\n\nfor epoch in range(EPOCHS):\n\n  print(f'Epoch {epoch + 1}/{EPOCHS}')\n  print('-' * 10)\n\n  train_acc, train_loss = train_epoch(\n    model,\n    train_data_loader,    \n    loss_fn, \n    optimizer, \n    device, \n    scheduler, \n    len(df_train)\n  )\n\n  print(f'Train loss {train_loss} accuracy {train_acc}')\n\n  val_acc, val_loss = eval_model(\n    model,\n    val_data_loader,\n    loss_fn, \n    device, \n    len(df_val)\n  )\n\n  print(f'Val   loss {val_loss} accuracy {val_acc}')\n  print()\n\n  train_a.append(train_acc)\n  train_l.append(train_loss)\n  val_a.append(val_acc)\n  val_l.append(val_loss)\n\n  if val_acc > best_accuracy:\n    torch.save(model.state_dict(), 'best_model_state.bin')\n    best_accuracy = val_acc")


# In[ ]:


plt.plot(train_a, label='train accuracy')
plt.plot(val_a, label='validation accuracy')

plt.title('Training history')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.ylim([0, 1]);


# In[ ]:


test_acc, _ = eval_model(
  model,
  test_data_loader,
  loss_fn,
  device,
  len(df_test)
)

test_acc.item()


# # Predictions

# In[ ]:


def get_predictions(model, data_loader):
  model = model.eval()
  
  review = []
  predictions = []
  prediction_probs = []
  real_values = []

  with torch.no_grad():
    for d in data_loader:

      reviews = d["review"]
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      sentiments = d["sentiments"].to(device)

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)

      probs = F.softmax(outputs, dim=1)

      review.extend(reviews)
      predictions.extend(preds)
      prediction_probs.extend(probs)
      real_values.extend(sentiments)

  predictions = torch.stack(predictions).cpu()
  prediction_probs = torch.stack(prediction_probs).cpu()
  real_values = torch.stack(real_values).cpu()
  return review, predictions, prediction_probs, real_values


# In[ ]:


y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(
  model,
  test_data_loader
)


# # Performance report 

# In[ ]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
class_names = ['negative', 'positive']


# In[ ]:


print(classification_report(y_test, y_pred, target_names=class_names))


# In[ ]:


def show_confusion_matrix(confusion_matrix):
  hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
  hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
  hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
  plt.ylabel('True sentiment')
  plt.xlabel('Predicted sentiment')

cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
show_confusion_matrix(df_cm)


# In[ ]:


idx = 2

review_text = y_review_texts[idx]
true_sentiment = y_test[idx]
pred_df = pd.DataFrame({
  'class_names': class_names,
  'values': y_pred_probs[idx]
})


# In[ ]:


print(review_text)
print()
print(f'True sentiment: {class_names[true_sentiment]}')


# In[ ]:


sns.barplot(x='values', y='class_names', data=pred_df, orient='h')
plt.ylabel('sentiment')
plt.xlabel('probability')
plt.xlim([0, 1])


# # Predict on raw text

# In[ ]:


review_text = "The film is too terrible I hate the characters actions and there's a lot of language mistakes"


# In[ ]:


encoded_review = tokenizer.encode_plus(
  review_text,
  max_length=MAX_LEN,
  add_special_tokens=True,
  return_token_type_ids=False,
  pad_to_max_length=True,
  return_attention_mask=True,
  return_tensors='pt',
)


# In[ ]:


input_ids = encoded_review['input_ids'].to(device)
attention_mask = encoded_review['attention_mask'].to(device)

output = model(input_ids, attention_mask)
_, prediction = torch.max(output, dim=1)

print(f'Review text: {review_text}')
print(f'Sentiment  : {class_names[prediction]}')


# If you like it help with a simple up clic
