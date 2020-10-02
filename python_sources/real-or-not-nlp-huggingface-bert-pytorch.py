#!/usr/bin/env python
# coding: utf-8

# # <html> <h1> <font color="#555F61"> Real or Not? NLP + HuggingFace + BERT + PyTorch  </font> </h1>
# <hr/>
#     
# This notebook provides a simple, easy-to-understand implementation of BERT for the [Real or Not? NLP with Tweets](https://www.kaggle.com/c/nlp-getting-started)
#  competition. We'll use [Hugging Face's transformers](https://github.com/huggingface/transformers) package and provide a detailed explanation of the special pre-processing required for BERT along with a step-by-step implementation and then train a model for this competition.
#  
# References: <br/>
# 1) Hugging Face: https://github.com/huggingface/transformers & https://huggingface.co/transformers/index.html <br/>
# 2) A Visual Guide to Using BERT by Jay Alammar : http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/ <br/>
# 3) Pytorch BERT Inference: https://www.kaggle.com/abhishek/pytorch-bert-inference
# 
# First, lets import all the required libraries and take a look at the data.

# In[ ]:


import numpy as np 
import pandas as pd 
import random
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup

import warnings
warnings.filterwarnings('ignore')

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda")


# In[ ]:


train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
print("Shape of train data : ",train.shape)
print("Shape of test data : ",test.shape)


# In[ ]:


###############################################################################################
#
# Since this notebook is primarily about BERT:
# 1. We'll skip the basic EDA as there are a lot of other notebooks in this competition that cover that.
# 2. Remove the mislabelled tweets.
# 3. Add the keyword column to the text column to retain that information (We've left out the location column as it has a lot of missing values and is relatively dirty)
#
###############################################################################################

# Remove the mislabelled tweets
incorrect_labels_df = train.groupby(['text']).nunique().sort_values(by='target', ascending=False)
incorrect_labels_df = incorrect_labels_df[incorrect_labels_df['target'] > 1]
incorrect_texts = incorrect_labels_df.index.tolist()
train = train[~train.text.isin(incorrect_texts)]

# Add the keyword column to the text column
train['keyword'].fillna('', inplace=True)
train['final_text'] = train['keyword'] + ' ' + train['text'] 
test['keyword'].fillna('', inplace=True)
test['final_text'] = test['keyword'] + ' ' + test['text'] 


# ## <html> <h2> <font color="#555F61"> Pre-processing for BERT   </font> </h2>
# <hr/>

# In[ ]:


# Get text values and labels
text_values = train.final_text.values
labels = train.target.values

# Load the pretrained Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


# First, we'll tokenize the first tweet and take a look at the tokens and token IDs.

# In[ ]:


print('Original Text : ', text_values[1])
print('Tokenized Text: ', tokenizer.tokenize(text_values[1]))
print('Token IDs     : ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text_values[1])))


# <html> 
#     <ul> That seems fairly straightforward but BERT has a few special tokens and these need to be added for sentence classification. Here's an overview of these special tokens:     
#      <li>   [CLS] - This token is added at the beginning, that is, before all the other tokens in the sentence.</li>
#     <li>  [SEP] -  This token is added at the end of the sentence.</li>
#     </ul>
#     
#      Also, at a later step, we'll the tweets to make them a uniform length and the [PAD] token is used for that. (Truncating the tweets is also an option)</li>
# 
# </html>

# In[ ]:


text = '[CLS]'
print('Original Text : ', text)
print('Tokenized Text: ', tokenizer.tokenize(text))
print('Token IDs     : ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text)))
print('\n')

text = '[SEP]'
print('Original Text : ', text)
print('Tokenized Text: ', tokenizer.tokenize(text))
print('Token IDs     : ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text)))
print('\n')

text = '[PAD]'
print('Original Text : ', text)
print('Tokenized Text: ', tokenizer.tokenize(text))
print('Token IDs     : ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text)))


# Now, let's tokenize the first tweet again and add the special tokens as well.

# In[ ]:


print('Original Text                          : ', text_values[1])
print('Tokenized Text                         : ', tokenizer.tokenize(text_values[1]))
print('Token IDs                              : ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text_values[1])))
print('Adding Special Tokens                  :', tokenizer.build_inputs_with_special_tokens(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text_values[1]))))
print('Adding Special Tokens Using Encode Func:', tokenizer.encode(text_values[1]))


# We'll also have to pad or truncate the token IDs for all tweets to make them the same length. This can easily be done using the max_length and  pad_to_max_length parameters of the tokenizer.encode() function. 
# 
# Next, let's take a look at the lengths of the tweets and the number of words in a tweet to decide on appropriate initial values for these parameters. (We can experiment and fine tune them later as these parameters can affect the outputs of the model)

# In[ ]:


# Add a length column which contains the length of the tweet
train['length'] = train['final_text'].apply(len)
test['length'] = test['final_text'].apply(len)

# Plot
sns.set_style('whitegrid', {'axes.grid' : False})
fig, ax = plt.subplots(1, 2)
fig.set_size_inches(18, 6)

sns.distplot(train['length'], color='#20A387',ax=ax[0])
sns.distplot(test['length'], color='#440154',ax=ax[1])

fig.suptitle("Length of Tweets", fontsize=14)
ax[0].set_title('Train')
ax[1].set_title('Test')

plt.show()


# In[ ]:


# Add column for number of words
train['num_of_words'] = train['final_text'].apply(lambda x:len(str(x).split())) 
test['num_of_words'] = test['final_text'].apply(lambda x:len(str(x).split())) 

# Plot
sns.set_style('whitegrid', {'axes.grid' : False})
fig, ax = plt.subplots(1, 2)
fig.set_size_inches(18, 6)

sns.distplot(train['num_of_words'], color='#20A387',ax=ax[0])
sns.distplot(test['num_of_words'], color='#440154',ax=ax[1])

fig.suptitle("Number of Words in Tweets", fontsize=14)
ax[0].set_title('Train')
ax[1].set_title('Test')

plt.show()


# In[ ]:


# Function to get token ids for a list of texts 
def encode_fn(text_list):
    all_input_ids = []    
    for text in text_values:
        input_ids = tokenizer.encode(
                        text,                      
                        add_special_tokens = True, 
                        max_length = 160,           
                        pad_to_max_length = True,
                        return_tensors = 'pt'  
                   )
        all_input_ids.append(input_ids)    
    all_input_ids = torch.cat(all_input_ids, dim=0)
    return all_input_ids


# Next, we'll use the padded token IDs to create the training and validation dataloaders.

# In[ ]:


epochs = 4
batch_size = 32

# Split data into train and validation 
all_input_ids = encode_fn(text_values)
labels = torch.tensor(labels)

dataset = TensorDataset(all_input_ids, labels)
train_size = int(0.95 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create train and validation dataloaders
train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)


# In[ ]:


# Load the pretrained BERT model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", 
    num_labels = 2, 
    output_attentions = False, 
    output_hidden_states = False, 
)

model.cuda()

# Create optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr = 2e-5)
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0,num_training_steps = total_steps)


# ## <html> <h2> <font color="#555F61"> Training and Validation   </font> </h2>
# <hr/>

# In[ ]:


##### Training #####
for epoch in range(epochs):    
    
    model.train()   
    total_loss, total_val_loss = 0, 0   
    for step, batch in enumerate(train_dataloader):     
        model.zero_grad()        
        loss, logits = model(batch[0].to(device), 
                             token_type_ids = None, 
                             attention_mask = (batch[0] > 0).to(device), 
                             labels = batch[1].to(device))
        
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

##### Validation #####
    model.eval()
    for i, batch in enumerate(val_dataloader):
        with torch.no_grad():  
            loss, logits = model(batch[0].to(device), 
                                   token_type_ids = None, 
                                   attention_mask = (batch[0] > 0).to(device),
                                   labels = batch[1].to(device))
        total_val_loss += loss.item()
                    
    avg_train_loss = total_loss / len(train_dataloader)      
    avg_val_loss = total_val_loss / len(val_dataloader)     
    print('Train Loss     : ', avg_train_loss)
    print('Validation Loss: ', avg_val_loss)
    print('\n')


# ## <html> <h2> <font color="#555F61"> Predictions  </font> </h2>
# <hr/>

# In[ ]:


# Create the test data loader
text_values = test.final_text.values
all_input_ids = encode_fn(text_values)
pred_data = TensorDataset(all_input_ids)
pred_dataloader = DataLoader(pred_data, batch_size = batch_size, shuffle = False)


# In[ ]:


##### Predictions #####
model.eval()
preds = []
for i, (batch,) in enumerate(pred_dataloader):    
    with torch.no_grad():
        outputs = model(batch.to(device), 
                        token_type_ids = None, 
                        attention_mask = (batch > 0).to(device))

    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    preds.append(logits)
    
final_preds = np.concatenate(preds, axis=0)
final_preds = np.argmax(final_preds, axis=1)


# In[ ]:


# Create submission file
submission = pd.DataFrame()
submission['id'] = test['id']
submission['target'] = final_preds
submission.to_csv('submission.csv', index=False)

