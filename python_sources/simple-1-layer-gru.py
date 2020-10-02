#!/usr/bin/env python
# coding: utf-8

# # Challenge Kaggle - Quora Insincere Questions Classification - Simple Version

# This notebook presents a simple solution for the Kaggle Challenge: [Quora Insincere Questions Classification](https://www.kaggle.com/c/quora-insincere-questions-classification).
# 
# This problem can be understood as a sentiment analysis problem, one of the most common downstream tasks in NLP.
# 
# The objective is to build a simple RNN model step-by-step in order to understand how such problem can be tackled.
# 
# I am going to use the PyTorch library as deep-learning framework.
# 
# In this solution, I am not going to use one of pre-trained embedding made available by the challenge. In order to present a more didactic solution, I am going to hand-craft the tokenization and we are going to add a embedding layer for training in the model. Please note that this approach will increase the training time, as the model will need to learn the paramaters of the embeddings, which quantity can be huge given the size of the vocabulary.

# ## Import of useful libraries

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


# 
# ## Loading the data

# ### Loading data and visualizing data

# In[ ]:


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print("Train shape : ",train_df.shape)
print("Test shape : ",test_df.shape)


# Let's visualizing the structure of the dataframes and the Quora's questions

# In[ ]:


train_df.head(5)


# In[ ]:


for question in train_df.question_text[:5]:
    print(question)


# Some statitistics of the datasets:

# In[ ]:


labels = np.array(train_df.target)

print('Number of training samples:', len(train_df))
print('Percentage of insincere questions: {:.2%}'.format(labels.sum()/len(train_df)))
print('Number of test samples:', len(test_df))


# It can be noted that the data set is very unballanced (only 6.19% of the training samples are labeled 1), therefore accuracy is not a good metric to evalutate performance. The metric chosen by the competition, F1-score, is a good metric for evluation of unbalaced datasets.

# Visualizing a few examples of insincere questions:

# In[ ]:


ins_idx = np.where(np.array(train_df.target)==1)[0]
for question in train_df.question_text[ins_idx[:5]]:
    print(question)


# ## Data preprocessing

# ### Eliminate punctuation and lower the sentences
# 
# For this simpler notebook, I am only going to deal with the words in the sentences, without taking into account the punctuation. It is however important to notice that ponctuation might play a big role in this task, as insincere questions are likely to have some particular patterns of punctuation.

# In[ ]:


from string import punctuation

def lower_eliminate_punctuation(sentence):
    '''
    Function that takes an string as input, lower it and get rid of its punctuation
    '''
    filtered_sentence = ''.join([c for c in sentence if c not in punctuation])
    return filtered_sentence.lower()

# Getting rid of punctuation in both datasets
train_df.question_text = train_df.question_text.apply(lower_eliminate_punctuation)
test_df.question_text = test_df.question_text.apply(lower_eliminate_punctuation)


# ### Dealing with outliers
# 
# If there are empty sentences in the datasets, we should not take them into account

# In[ ]:


# Training set
empty_idx_train = []
for i, question in enumerate(train_df.question_text):
    if question == '':
        print('Empty question at index', i, 'with label ', train_df.target[i])
        empty_idx_train.append(i)


# In[ ]:


# Test set
empty_idx_test = []
for i, question in enumerate(test_df.question_text):
    if question == '':
        print('Empty question at index', i, 'with label ', test_df.target[i])
        empty_idx_test.append(i)


# In[ ]:


# Eliminating sample with empty question in the training set
train_df = train_df.drop(empty_idx_train, axis=0)
labels = np.delete(labels, empty_idx_train)


# ### Creating vocabulary
# 
# In other to tokenize the sentences, we need to create a vocabulary. I will use a dictionnaire to map word and integer. 
# 
# Note: the first number of the vocabulary will be a 1, as we will use 0 for the padding of the sentences.

# In[ ]:


# Getting all words in both samples
all_text_list = list(train_df.question_text) + list(test_df.question_text)
all_text = ' '.join(all_text_list)
words = set(all_text.split())

print('Number of unique words:', len(words))


# In[ ]:


# Dictionary that maps words to integers
word_to_int = {word: i for i, word in enumerate(words, 1)}


# ### Tokenizing and padding

# Tokeninzing questions in both datasets:

# In[ ]:


def tokenize(sentence):
    '''
    Function that tokenize a sentence using the word_to_int dictionnaire and 
    return a list of tokens
    '''
    tokens = []
    for word in sentence.split():
        tokens.append(word_to_int[word])
    return tokens


train_tokens = train_df.question_text.apply(tokenize)
test_tokens = test_df.question_text.apply(tokenize)


# Checking the size of the longest question in both datasets

# In[ ]:


print('Size of longest question:')
print('Training set:', max(train_tokens.apply(len)))
print('Test set:', max(test_tokens.apply(len)))


# As the questions are not too long, we can set the sequence length of the samples to the highest value and we will not need to deal with truncating. 
# 
# I will then pad the questions at the left using the token 0.

# In[ ]:


seq_length = max(train_tokens.apply(len)) # == 132


# In[ ]:


def pad(questions, seq_length):
    '''
    This function pad the questions fed as series of tokens with 0 at left
    and returns a numpy array
    '''
    
    features = np.zeros((len(questions), seq_length), dtype=int)
    for i, sentence in enumerate(questions):
        features[i, -len(sentence):] = sentence
    
    return features


# In[ ]:


train = pad(train_tokens, seq_length)
test = pad(test_tokens, seq_length)


# ### Splitting training data and creating dataloaders

# In order to avoid overfitting, we need to use some data as validation during the training phase.
# 
# I am going to use a 90/10 ratio for the split of training and validation sets, in order to get a training dataset the closest as possible to the original training dataset

# In[ ]:


x_train, x_val, label_train, label_val = train_test_split(train, labels, test_size=0.1, random_state=0) 


# Now, let's create Dataloaders for the datasets that will help us with batch iteration.

# In[ ]:


# Create Tensor datasets
train_data = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(label_train))
valid_data = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(label_val))
test_data = TensorDataset(torch.from_numpy(test))

# Create Dataloaders
batch_size = 56

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)


# ## Building the model

# Now that we have processesed the data, I am going to build a RNN model using 1-Layer GRUs. Such RNN structure is pretty simple and it usually give good results. One can choose to use LSTM instead, I chose GRU because it will have to train less parameters while keeping good performance.

# ### Training on GPU or CPU

# In[ ]:


# Checking if GPU is available
train_on_gpu=torch.cuda.is_available()

if train_on_gpu:
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')


# ### The model
# 
# The structure of the model is simple:
# 1. Embedding layer. As said above, we are not going to use pre-trained embeddings.
# 2. 1-Layer GRU
# 3. Dropout Layer to avoid overfitting
# 4. Fully connected layer followed by the application of a sigmoid
# 5. Use the output of the last position of the setence as prediction probability

# In[ ]:


class RNN_model(nn.Module):
    """
    The RNN model that will be used for our classification task
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        """
        Initialize the model by setting up the layers
        """
        super(RNN_model, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim       
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # GRU layer
        self.gru = nn.GRU(embedding_dim, hidden_dim, n_layers, batch_first=True, dropout = drop_prob)
        
        # Dropout layer
        self.dropout = nn.Dropout(p=drop_prob)
        
        # Fully-connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
        
        # Sigmoid layer
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        
        batch_size = x.size(0)
        
        # Deal with cases were the current batch_size is different from general batch_size
        # It occurrs at the end of iteration with the Dataloaders
        if hidden.size(1) != batch_size:
            hidden = hidden[:, :batch_size, :].contiguous()
        
        # Apply embedding
        x = self.embedding(x)
        
        # GRU Layer
        out, hidden = self.gru(x, hidden)
        
        # Stack up GRU outputs --> preparation for the fully-connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        
        # Dropout and fully-connected layers
        out = self.dropout(out)
        sig_out = self.sigmoid(self.fc(out))
        
        # Unstack outputs to come back to correct dimensions per sample (batch_size, seq_length)
        sig_out = sig_out.contiguous().view(batch_size, -1)
        
        # return last sigmoid output and hidden state
        return sig_out[:, -1], hidden
    
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create a new tensor with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero
        
        weight = next(self.parameters()).data
        
        if train_on_gpu:
            hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda()
            
        else:
            hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()
        
        return hidden
        


# ### Defining hyperparameters and initiating the model

# In[ ]:


vocab_size = len(word_to_int) + 1 # including token 0
output_size = 1 # binary classification task 
embedding_dim = 256
hidden_dim = 256
n_layers = 1

# Initiating the model
model = RNN_model(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)


# ## Training
# 
# We are going to use binary cross-entropy loss (BCELoss()) as loss function and the ADAM optimizer.
# 
# We are algo going to print the F1-score on valuation set each 1000 steps to follow the progress of the training. Please note that, for simplicity, we are considering a threshold of 0.5 for prediction of a positive label. This is also a hyperparameter that can be learned by cross-validation.
# 
# We are also going to clip the gradient whenever its norm is higher than 5 in order to avoid the explosion gradient effect that can happen often in RNNs.

# In[ ]:


# Training parameters

epochs = 4

print_every = 1000
clip = 5 # gradient clipping - to avoid gradient explosion

lr=0.001

# Defining loss and optimization functions

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# In[ ]:


def train_model(model, train_loader, valid_loader, batch_size, epochs, optimizer, criterion, print_every, clip):
    
    # move model to GPU, if available
    if(train_on_gpu):
        model.cuda()
    
    counter = 0
    
    # Model in training mode
    model.train()
    breaker = False
    for e in range(epochs):

        # Batch loop
        for inputs, labels in train_loader:
            counter += 1

            # move data to GPU, if available
            if(train_on_gpu):
                inputs, labels = inputs.cuda(), labels.cuda()

            # Initialize hidden state
            h = model.init_hidden(batch_size)

            # Setting accumulated gradients to zero before backward step
            model.zero_grad()

            # Output from the model
            output, _ = model(inputs, h)

            # Calculate the loss and perform backprop
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()

            # Clipping the gradient to avoid explosion
            nn.utils.clip_grad_norm_(model.parameters(), clip)

            # Backpropagation step
            optimizer.step()

            # Validation stats
            if counter % print_every == 0:

                with torch.no_grad():

                    # Get validation loss and F1-score on validation set

                    val_losses = []
                    all_val_labels = []
                    all_val_preds = []

                    # Model in evaluation mode
                    model.eval()
                    for inputs, labels in valid_loader:

                        all_val_labels += list(labels)

                        # Sending data to GPU
                        if(train_on_gpu):
                            inputs, labels = inputs.cuda(), labels.cuda()

                        # Initiating hidden state for the validation set
                        val_h = model.init_hidden(batch_size)

                        output, _ = model(inputs, val_h)

                        # Computing validation loss
                        val_loss = criterion(output.squeeze(), labels.float())

                        val_losses.append(val_loss.item())

                        # Computing validation F1-score

                        preds = torch.round(output.squeeze())  # 1 if output probability >= 0.5
                        preds = np.squeeze(preds.numpy()) if not train_on_gpu else np.squeeze(preds.cpu().numpy())
                        all_val_preds += list(preds)

                current_loss = np.mean(val_losses)
                
                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(loss.item()),
                      "Val Loss: {:.6f}...".format(current_loss),
                      "F1-score: {:.3%}".format(f1_score(all_val_labels, all_val_preds)))
                
                # Saving the best model and stopping if there is no improvement after 10 evaluations
                
                if counter == print_every: # first evaluation
                    best_loss = current_loss
                    counter_eval = 0  
                    
                if current_loss < best_loss:
                    best_loss = current_loss
                    torch.save(model.state_dict(), 'checkpoint.pth')
                    counter_eval = 0 
                    
                counter_eval += 1
                if counter_eval == 10:
                    breaker = True
                    break

                # Put model back to training mode
                model.train()
        
        # breaking outer loop on epochs
        if breaker:
            break
    
    # Loading best model
    state_dict = torch.load('checkpoint.pth')
    model.load_state_dict(state_dict)


# In[ ]:


train_model(model, train_loader, valid_loader, batch_size, epochs, optimizer, criterion, print_every, clip)


# ## Predictions on test set

# In[ ]:


# Model in evaluation mode
model.eval()

with torch.no_grad():
    all_test_preds = []

    for inputs in test_loader:
        inputs = inputs[0]
        
        # Sending data to GPU
        if(train_on_gpu):
            inputs = inputs.cuda()
            
        test_h = model.init_hidden(batch_size)
        output, _ = model(inputs, test_h)
        
        preds = torch.round(output.squeeze())  # 1 if output probability >= 0.5
        preds = np.squeeze(preds.numpy()) if not train_on_gpu else np.squeeze(preds.cpu().numpy())
        all_test_preds += list(preds.astype(int))


# In[ ]:


sub = pd.DataFrame({
    'qid': test_df.qid,
    'prediction': all_test_preds
})

# Make sure the columns are in the correct order
sub = sub[['qid', 'prediction']]


# In[ ]:


sub.to_csv('submission.csv', index=False, sep=',')

