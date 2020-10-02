#!/usr/bin/env python
# coding: utf-8

# ### Inspired by:
# * https://www.kaggle.com/sudalairajkumar/a-look-at-different-embeddings
# * https://www.kaggle.com/shujian/single-rnn-with-4-folds-v1-9
# * http://mlexplained.com/2018/01/13/weight-normalization-and-layer-normalization-explained-normalization-in-deep-learning-part-2/
# * https://arxiv.org/abs/1607.06450
# * https://github.com/keras-team/keras/issues/3878
# * https://www.kaggle.com/lystdo/lstm-with-word2vec-embeddings
# * https://www.kaggle.com/jhoward/improved-lstm-baseline-glove-dropout
# * https://www.kaggle.com/aquatic/entity-embedding-neural-net
# * https://www.kaggle.com/hireme/fun-api-keras-f1-metric-cyclical-learning-rate
# * https://ai.google/research/pubs/pub46697
# * https://blog.openai.com/quantifying-generalization-in-reinforcement-learning/
# * https://www.kaggle.com/rasvob/let-s-try-clr-v3
# * https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/3%20-%20Faster%20Sentiment%20Analysis.ipynb
# * https://www.kaggle.com/ziliwang/pytorch-text-cnn
# * https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/bidirectional_recurrent_neural_network/main.py
# * https://github.com/clairett/pytorch-sentiment-classification/blob/master/bilstm.py
# * https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
# 
# 
# trying torch...
# 
# much harder then keras, but feels more rewarding when done

# In[ ]:


import numpy as np # linear algebra
import sys
np.set_printoptions(threshold=sys.maxsize)
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
print(os.listdir("../input/embeddings"))
print(os.listdir("../input/embeddings/GoogleNews-vectors-negative300"))

# Any results you write to the current directory are saved as output.

import gensim
from gensim.utils import simple_preprocess
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,f1_score,precision_recall_fscore_support,recall_score,precision_score
from keras import backend as K
from sklearn.utils import class_weight
import matplotlib.pyplot as plt

import tensorflow as tf

SEED = 2019

np.random.seed(SEED)

#https://www.kaggle.com/shujian/single-rnn-with-4-folds-v1-9
def threshold_search(y_true, y_proba):
    best_threshold = 0
    best_score = 0
    for threshold in [i * 0.01 for i in range(100)]:
        score = f1_score(y_true=y_true, y_pred=y_proba > threshold)
#         print('\rthreshold = %f | score = %f'%(threshold,score),end='')
        if score > best_score:
            best_threshold = threshold
            best_score = score
#     print('best threshold is % f with score %f'%(best_threshold,best_score))
    search_result = {'threshold': best_threshold, 'f1': best_score}
    return search_result


# In[ ]:


import torchtext
import random
from nltk import word_tokenize

text = torchtext.data.Field(lower=True, batch_first=True, tokenize=word_tokenize, fix_length=100)
qid = torchtext.data.Field()
target = torchtext.data.Field(sequential=False, use_vocab=False, is_target=True)
train_dataset = torchtext.data.TabularDataset(path='../input/train.csv', format='csv',
                                      fields={'question_text': ('text',text),
                                              'target': ('target',target)})

train, val,test = train_dataset.split(split_ratio=[0.8,0.1,0.1],stratified=True,strata_field='target',random_state=random.getstate())

submission_x = torchtext.data.TabularDataset(path='../input/test.csv', format='csv',
                                     fields={'qid': ('qid', qid),
                                             'question_text': ('text', text)})

text.build_vocab(train_dataset, submission_x, min_freq=3)
qid.build_vocab(submission_x)
print('train dataset len:',len(train_dataset))
print('train len:',len(train))
print('val len:',len(val))
print('test len:',len(test))


# In[ ]:


glove = torchtext.vocab.Vectors('../input/embeddings/glove.840B.300d/glove.840B.300d.txt')
text.vocab.set_vectors(glove.stoi, glove.vectors, dim=300)


# In[ ]:


#src: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss dosen't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            if self.verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss


# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchtext.data
import warnings
from sklearn.metrics import accuracy_score
from torch.autograd import Variable

torch.cuda.init()
torch.cuda.empty_cache()
print('CUDA MEM:',torch.cuda.memory_allocated())

print('cuda:', torch.cuda.is_available())
print('cude index:',torch.cuda.current_device())


# lr = 1e-3
# batch_size = int(len(train_dataset)/100)
# batch_size = int(lr*len(train))
batch_size = 512
print('batch_size:',batch_size)
print('---')

train_loader = torchtext.data.BucketIterator(dataset=train,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               sort=False)
val_loader = torchtext.data.BucketIterator(dataset=val,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               sort=False)
test_loader = torchtext.data.BucketIterator(dataset=test,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               sort=False)

class SentimentLSTM(nn.Module):
    
    def __init__(self,vocab_vectors,padding_idx,batch_size):
        super(SentimentLSTM,self).__init__()
        print('Vocab vectors size:',vocab_vectors.shape)
        self.batch_size = batch_size
        self.hidden_dim = 128
        self.n_layers = 2 #bidirectional has 2 layers - forward and backward seq
        
        self.embedding = nn.Embedding.from_pretrained(vocab_vectors)
        self.embedding.weight.requires_grad = False
        self.embedding.padding_idx = padding_idx
        
        self.lstm = nn.LSTM(input_size=vocab_vectors.shape[1], hidden_size=self.hidden_dim, bidirectional=True,batch_first=True)        
        self.linear1 = nn.Linear(self.n_layers*self.hidden_dim,self.hidden_dim)        
        self.linear2 = nn.Linear(self.hidden_dim,1)
        self.dropout = nn.Dropout(0.2)

        
    def forward(self,x):
        #init h0,c0
        hidden = (torch.zeros(self.n_layers, x.shape[0], self.hidden_dim).cuda(),
                torch.zeros(self.n_layers, x.shape[0], self.hidden_dim).cuda())
        e = self.embedding(x)
        _, hidden = self.lstm(e, hidden)
        out = torch.cat((hidden[0][-2,:,:], hidden[0][-1,:,:]), dim=1).cuda()
        out = self.linear1(F.relu(out))
        return self.linear2( self.dropout(out))
    
class SentimentBase(nn.Module):
    
    def __init__(self):
        super(SentimentBase,self).__init__()
        
        self.embedding = nn.Embedding(75966,300)        
        self.linear1 = nn.Linear(300*100,128)
        self.linear2 = nn.Linear(128,1)
    
    def forward(self,x):
        emb = self.embedding(x)
        pooled = emb.reshape((emb.shape[0],emb.shape[1]*emb.shape[2]))
        out = self.linear1(F.relu(pooled))
        out = self.linear2(out)
        return out

    
class SentimentCNN(nn.Module):
    
    def __init__(self,vocab_vectors,padding_idx,batch_size):
        super(SentimentCNN,self).__init__()
        print('Vocab vectors size:',vocab_vectors.shape)
        self.batch_size = batch_size
        self.hidden_dim = 128
        
        self.embedding = nn.Embedding.from_pretrained(vocab_vectors)
        self.embedding.weight.requires_grad = False
        self.embedding.padding_idx = padding_idx
        
        self.cnns =  nn.ModuleList([nn.Conv1d(in_channels=vocab_vectors.shape[1], out_channels=self.hidden_dim, kernel_size=k) for k in [3,4,5]])
        
        self.linear1 = nn.Linear(3*self.hidden_dim,self.hidden_dim)        
        self.linear2 = nn.Linear(self.hidden_dim,1)
        self.dropout = nn.Dropout(0.2)

    @staticmethod
    def conv_and_max_pool(x, conv):
        """Convolution and global max pooling layer"""
        return F.relu(conv(x).permute(0, 2, 1).max(1)[0])
        
    # https://github.com/gaussic/text-classification/blob/master/cnn_pytorch.py
    def forward(self,x):
        e = self.embedding(x)
         # Conv1d takes in (batch, channels, seq_len), but raw embedded is (batch, seq_len, channels)
        e = e.permute(0,2,1)
        cnn_outs = []
        for conv in self.cnns:
            f =self.conv_and_max_pool(e,conv)
            cnn_outs.append(f)
        out = torch.cat(cnn_outs, dim=1).cuda()
        out = self.linear1(F.relu(out))
        return self.linear2( self.dropout(out))


class SentimentGRU(nn.Module):
    
    def __init__(self,vocab_vectors,padding_idx,batch_size):
        super(SentimentGRU,self).__init__()
        print('Vocab vectors size:',vocab_vectors.shape)
        self.batch_size = batch_size
        self.hidden_dim = 128
        self.n_layers = 2 #bidirectional has 2 layers - forward and backward seq
        
        self.embedding = nn.Embedding.from_pretrained(vocab_vectors)
        self.embedding.weight.requires_grad = False
        self.embedding.padding_idx = padding_idx
        
        self.gru = nn.GRU(input_size=vocab_vectors.shape[1], hidden_size=self.hidden_dim, bidirectional=True,batch_first=True)        
        self.linear1 = nn.Linear(self.n_layers*self.hidden_dim,self.hidden_dim)        
        self.linear2 = nn.Linear(self.hidden_dim,1)
        self.dropout = nn.Dropout(0.2)

        
    def forward(self,x):
        #init h0,c0
        hidden = torch.zeros(self.n_layers, x.shape[0], self.hidden_dim).cuda()
        e = self.embedding(x)
        _, hidden = self.gru(e, hidden)
        out = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1).cuda()
        out = self.linear1(F.relu(out))
        return self.linear2( self.dropout(out))
    

class Ensemble(nn.Module):
    
    def __init__(self,vocab_vectors,padding_idx,batch_size):
        super(Ensemble,self).__init__()
        self.lstm = SentimentLSTM(text.vocab.vectors, padding_idx=text.vocab.stoi[text.pad_token], batch_size=batch_size).cuda()
        self.gru = SentimentGRU(text.vocab.vectors, padding_idx=text.vocab.stoi[text.pad_token], batch_size=batch_size).cuda()
        self.cnn = SentimentCNN(text.vocab.vectors, padding_idx=text.vocab.stoi[text.pad_token], batch_size=batch_size).cuda()
        self.base = SentimentBase().cuda()
        self.soft = nn.Softmax(dim=1)
#         self.out_layer = nn.Linear(4,1)
          
    def forward(self,x):        
        o1 = self.lstm(x)
        o2 = self.gru(x)
        o3 = self.cnn(x)
        o4 = self.base(x)
        out = torch.cat([o1,o2,o3,o4],1)
        s_out = self.soft(out)
        return torch.sum(torch.mul(out,s_out),dim=1).reshape(x.shape[0],1)
#         return self.out_layer(s_out)
        
model = Ensemble(text.vocab.vectors, padding_idx=text.vocab.stoi[text.pad_token], batch_size=batch_size).cuda()
print(model)
print('-'*80)

early_stopping = EarlyStopping(patience=2,verbose=True)
loss_function = nn.BCEWithLogitsLoss().cuda()        
optimizer = optim.Adam(model.parameters(),lr=1e-3)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

    
losses = []
val_losses=[]
epoch_acc=[]
epoch_val_acc=[]
lrs = []

for epoch in range(100):
#     print('-----%d-----'%epoch)
    epoch_losses=[]
    epoch_val_losses = []
    preds = []
    val_preds=[]
    targets = []
    acc = []
    model.train()
    for batch,train_batch in enumerate(list(iter(train_loader)),1):
        optimizer.zero_grad()
        
        y_pred = model(train_batch.text.cuda()).squeeze(1)
        y_numpy_pred =torch.sigmoid(y_pred).cpu().detach().numpy()
        preds += y_numpy_pred.tolist()
        
        y_true = train_batch.target.float().cuda()
        y_numpy_true = train_batch.target.cpu().detach().numpy()
        targets += y_numpy_true.tolist()
        loss = loss_function(y_pred,y_true)
        epoch_losses.append(loss.item())

        loss.backward()
        optimizer.step()
        lrs.append(get_lr(optimizer))
        acc.append(accuracy_score(y_numpy_true,np.round(y_numpy_pred)))
        if batch % 100 == 0:
            print('\rtraining (batch,loss,acc) | ',batch,' ===>',loss.item(),' acc ',np.mean(acc),end='')
    
    losses.append(np.mean(epoch_losses))
    targets =  np.array(targets)
    preds = np.array(preds)
    search_result = threshold_search(targets, preds)
    train_f1 = search_result['f1']
    epoch_acc.append(np.mean(acc))
    
    targets = []
    val_acc=[]
    model.eval()
    with torch.no_grad():
        for batch,val_batch in enumerate(list(val_loader),1):
            y_pred = model(val_batch.text.cuda()).squeeze(1)
            y_numpy_pred = torch.sigmoid(y_pred).cpu().detach().numpy()
            val_preds += y_numpy_pred.tolist()        
            y_true = val_batch.target.float().cuda()
            y_numpy_true = val_batch.target.cpu().detach().numpy()
            targets += y_numpy_true.tolist()
            val_loss = loss_function(y_pred,y_true)
            epoch_val_losses.append(val_loss.item())
            val_acc.append(accuracy_score(y_numpy_true,np.round(y_numpy_pred)))
            if batch % 100 == 0:
                print('\rvalidation (batch,acc) | ',batch,' ===>', np.mean(val_acc),end='')
    
    val_losses.append(np.mean(epoch_val_losses))
    epoch_val_acc.append(np.mean(val_acc))
    
    targets =  np.array(targets)
    val_preds =  np.array(val_preds)
    search_result = threshold_search(targets, val_preds)
    val_f1 = search_result['f1']
    
    print('\nEPOCH: ',epoch,'\n has acc of ',epoch_acc[-1],' ,has loss of ',losses[-1], ' ,f1 of ',train_f1,'\nval acc of ',epoch_val_acc[-1],' ,val loss of ',val_losses[-1],' ,val f1 of ',val_f1)
    print('-'*80)
            
    if early_stopping.early_stop:        
        print("Early stopping at ",epoch," epoch")
        break
    else:
        early_stopping(1.-val_f1, model)

    
print('Training finished....')


# In[ ]:


print(os.listdir())

model = Ensemble(text.vocab.vectors, padding_idx=text.vocab.stoi[text.pad_token], batch_size=batch_size).cuda()
model.load_state_dict(torch.load('checkpoint.pt'))


# In[ ]:


_,ax = plt.subplots(2,1,figsize=(20,10))
ax[0].plot(losses,label='loss')
ax[0].plot(val_losses,label='val_loss')

ax[1].plot(epoch_acc,label='acc')
ax[1].plot(epoch_val_acc,label='val_acc')

plt.legend()
plt.show()

pred = []
targets = []
with torch.no_grad():
    for test_batch in list(test_loader):
        model.eval()
        x = test_batch.text.cuda()
        pred += torch.sigmoid(model(x).squeeze(1)).cpu().data.numpy().tolist()
        targets += test_batch.target.cpu().data.numpy().tolist()

pred = np.array(pred)
targets =  np.array(targets)
search_result = threshold_search(targets, pred)
pred = (pred > search_result['threshold']).astype(int)
print('test acc:',accuracy_score(pred,targets))
print('test f1:',search_result['f1'])

print('RESULTS ON TEST SET:\n',classification_report(targets,pred))


# In[ ]:


print('Threshold:',search_result['threshold'])

submission_list = list(torchtext.data.BucketIterator(dataset=submission_x,
                                    batch_size=batch_size,
                                    sort=False,
                                    train=False))
pred = []
with torch.no_grad():
    for submission_batch in submission_list:
        model.eval()
        x = submission_batch.text.cuda()
        pred += torch.sigmoid(model(x).squeeze(1)).cpu().data.numpy().tolist()

pred = np.array(pred)

df_subm = pd.DataFrame()
df_subm['qid'] = [qid.vocab.itos[j] for i in submission_list for j in i.qid.view(-1).numpy()]
# df_subm['prediction'] = test_meta > search_result['threshold']
df_subm['prediction'] = (pred > search_result['threshold']).astype(int)
print(df_subm.head())
df_subm.to_csv('submission.csv', index=False)

