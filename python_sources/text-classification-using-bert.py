#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap

from tqdm import tqdm


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")

sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8

RANDOM_SEED = 23
np.random.seed(RANDOM_SEED)

torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[ ]:


get_ipython().system('pip freeze > requirements.txt')


# In[ ]:


data = pd.read_json('../input/news-category-dataset/News_Category_Dataset_v2.json', lines=True)
data.head()

# data = data.sample(n=1000)


# In[ ]:


data.category.unique()


# In[ ]:


text = pd.DataFrame({
    'text':data.headline + data.short_description,
    'label':data.category
})


# In[ ]:


text.head()


# In[ ]:


encoder = LabelEncoder()
text.label = encoder.fit_transform(text.label)

text.head()


# In[ ]:


train, test = train_test_split(text, test_size=0.1, random_state=23)
train, val = train_test_split(train, test_size=0.3,random_state=23)


# In[ ]:


# train.reset_index(drop=True, inplace=True)
# val.reset_index(drop=True, inplace=True)
# test.reset_index(drop=True, inplace=True)


# In[ ]:


# test.drop('label', 1, inplace=True)


# ## Tokenisation or data-preprocessing

# In[ ]:


PRE_TRAINED_MODEL_NAME = 'bert-base-cased'


# In[ ]:


tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)


# ### Sample Test
# 
# The idea is to preserve the purity of the sentence and not to convert it into all lower cases and vise-versa.

# In[ ]:


sample_txt = 'So give me reason to prove me wrong to wash this memory clean. Let the flood the cross the distance in your eyes.'

tokens = tokenizer.tokenize(sample_txt)

token_ids = tokenizer.convert_tokens_to_ids(tokens)

print(f' Sentence: {sample_txt}')
print(f'   Tokens: {tokens}')
print(f'Token IDs: {token_ids}')


# Bert uses special tokens to navigate through the training process. The tokens include:
# 1. Unknown
# 2. Seperate
# 3. Padding
# 4. Class
# 5. Mask

# In[ ]:


tokenizer.special_tokens_map


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


tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])


# In[ ]:


class Tokenisation(Dataset):
    
    
    def __init__(self, data, targets, tokenizer, max_len):
        self.data = data
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_vid = self.tokenizer.vocab["[PAD]"]
        
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        data = str(self.data[item])
        data = " ".join(data.split())
        target = self.targets
        
        encoding = self.tokenizer.encode_plus(
          data,
          add_special_tokens=True,
          max_length=self.max_len,
          return_token_type_ids=False,
          pad_to_max_length=True,
          return_attention_mask=True,
          return_tensors='pt',
        )
        
        ids = encoding['input_ids']
        masks = encoding['attention_mask']
        token_type_ids = encoding['input_ids']
        
        true_seq_length = len(encoding['input_ids'][0])
        pad_size = self.max_len - true_seq_length
        pad_ids = torch.Tensor([self.pad_vid] * pad_size).long()
        ids = torch.cat((encoding['input_ids'][0], pad_ids))
        
        
#         padding_len = self.max_len - len(ids)
#         ids = ids + ([0] * padding_len)
#         masks = ids + ([0] * padding_len)
#         token_type_ids = token_type_ids + ([0] * padding_len)
        
        return {
          'text': data,
          'input_ids': ids.flatten(),
          'attention_mask': encoding['attention_mask'].flatten(),
          'targets': torch.tensor(target[item], dtype=torch.long)
        }


# In[ ]:


def dataLoader(df, tokenizer, max_len, batch_size):
    ds = Tokenisation(
    data=df['text'].to_numpy(),
    targets=df['label'].to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
    )
    return DataLoader(ds,batch_size=batch_size, num_workers=4)


# In[ ]:


BATCH_SIZE = 16
MAX_LEN = 128

train_data_loader = dataLoader(train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = dataLoader(val, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = dataLoader(test, tokenizer, MAX_LEN, BATCH_SIZE)


# In[ ]:


data = next(iter(train_data_loader))
data.keys()


# In[ ]:


print(data['input_ids'].shape)
print(data['attention_mask'].shape)
print(data['targets'].shape)


# ## Building the model

# In[ ]:


class TextClassifier(nn.Module):
    
    def __init__(self, n_classes):
        super(TextClassifier, self).__init__()
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
    
    def unfreeze(self,start_layer,end_layer):
        def children(m):
            return m if isinstance(m, (list, tuple)) else list(m.children())
        def set_trainable_attr(m, b):
            m.trainable = b
            for p in m.parameters():
                p.requires_grad = b
        def apply_leaf(m, f):
            c = children(m)
            if isinstance(m, nn.Module):
                f(m)
            if len(c) > 0:
                for l in c:
                    apply_leaf(l, f)
        def set_trainable(l, b):
            apply_leaf(l, lambda m: set_trainable_attr(m, b))

        # You can unfreeze the last layer of bert by calling set_trainable(model.bert.encoder.layer[23], True)
        set_trainable(self.bert, False)
        for i in range(start_layer, end_layer+1):
            set_trainable(self.bert.encoder.layer[i], True)


# In[ ]:


device


# In[ ]:


len(text.label.unique())


# In[ ]:


model = TextClassifier(len(text.label.unique()))
model = model.to(device)


# In[ ]:


input_ids = data['input_ids'].to(device)
attention_mask = data['attention_mask'].to(device)

print(input_ids.shape) # batch size x seq length
print(attention_mask.shape) # batch size x seq length


# In[ ]:


F.softmax(model(input_ids, attention_mask), dim=1)


# In[ ]:


EPOCHS = 20
MAX_LENGTH = 128
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4

LearningRate = 5e-5

BETAS = (0.9, 0.999)
BERT_WEIGHT_DECAY = 0.01
EPS = 1e-8

# Define identifiers & group model parameters accordingly 
bert_identifiers = ['embedding', 'encoder', 'pooler']
no_weight_decay_identifiers = ['bias', 'LayerNorm.weight']
grouped_model_parameters = [
        {'params': [param for name, param in model.named_parameters()
                    if any(identifier in name for identifier in bert_identifiers) and
                    not any(identifier_ in name for identifier_ in no_weight_decay_identifiers)],
        'lr': LearningRate,
        'betas': BETAS,
        'weight_decay': BERT_WEIGHT_DECAY,
        'eps': EPS},
        {'params': [param for name, param in model.named_parameters()
                    if any(identifier in name for identifier in bert_identifiers) and
                    any(identifier_ in name for identifier_ in no_weight_decay_identifiers)],
        'lr': LearningRate,
        'betas': BETAS,
        'weight_decay': 0.0,
        'eps': EPS},
        {'params': [param for name, param in model.named_parameters()
                    if not any(identifier in name for identifier in bert_identifiers)],
        'lr': LearningRate,
        'betas': BETAS,
        'weight_decay': 0.0,
        'eps': EPS}
]

# Define optimizer
optimizers = AdamW(grouped_model_parameters)

optimizer = AdamW(model.parameters(), lr=5e-5, correct_bias=False)

total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)

loss_fn = nn.CrossEntropyLoss().to(device)


# In[ ]:


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    
    
    model = model.train()
    losses = []
    correct_predictions = 0
    for d in tqdm(data_loader):
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
        for d in tqdm(data_loader):
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


get_ipython().run_cell_magic('time', '', "\nhistory = defaultdict(list)\nbest_accuracy = 0\n\nfor epoch in range(EPOCHS):\n    print(f'Epoch {epoch + 1}/{EPOCHS}')\n    print('-' * 10)\n    train_acc, train_loss = train_epoch(\n                                        model,\n                                        train_data_loader,\n                                        loss_fn,\n                                        optimizer,\n                                        device,\n                                        scheduler,\n                                        len(train)\n    )\n    print(f'Train loss {train_loss} accuracy {train_acc}')\n    val_acc, val_loss = eval_model(\n                                    model,\n                                    val_data_loader,\n                                    loss_fn,\n                                    device,\n                                    len(val)\n  )\n    print(f'Val   loss {val_loss} accuracy {val_acc}')\n    print()\n    history['train_acc'].append(train_acc)\n    history['train_loss'].append(train_loss)\n    history['val_acc'].append(val_acc)\n    history['val_loss'].append(val_loss)\n    \n    if val_acc > best_accuracy:\n        torch.save(model.state_dict(), 'best_model_state.bin')\n        best_accuracy = val_acc")


# In[ ]:


plt.plot(history['train_acc'], label='train accuracy')
plt.plot(history['val_acc'], label='validation accuracy')
plt.title('Training history')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()


# In[ ]:


test


# In[ ]:


test_acc, _ = eval_model(
  model,
  test_data_loader,
  loss_fn,
  device,
  len(df_test)
)


test_acc.item()


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
        targets = d["targets"].to(device)
        outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
        )
        _, preds = torch.max(outputs, dim=1)
        review_texts.extend(texts)
        predictions.extend(preds)
        prediction_probs.extend(outputs)
        real_values.extend(targets)
    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return review_texts, predictions, prediction_probs, real_values


# In[ ]:


y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(
  model,
  test_data_loader
)


# In[ ]:


print(classification_report(y_test, y_pred, target_names=class_names))


# In[ ]:


def show_confusion_matrix(confusion_matrix):
    hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
    plt.ylabel('True sentiment')
    plt.xlabel('Predicted sentiment');


# In[ ]:


cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
show_confusion_matrix(df_cm)

