#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
import time
import datetime
import gc
import random
import re
import operator

from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.metrics import f1_score,precision_score,recall_score

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader,TensorDataset,Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim.optimizer import Optimizer

from keras.preprocessing.text import Tokenizer,text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

def seed_everything(SEED=42):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED']=str(SEED)
    # torch.backends.cudnn.benchmark = False

def init_func(worker_id):
    np.random.seed(SEED+worker_id)

    
tqdm.pandas()
SEED=42
seed_everything(SEED=SEED)


# ## EMBEDDINGS

# In[ ]:


get_ipython().run_cell_magic('time', '', 'def load_embed(file):\n    def get_coefs(word,*arr): \n        return word, np.asarray(arr, dtype=\'float32\')\n    \n    if file == \'../input/quora-insincere-questions-classification/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec\':\n        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file) if len(o)>100)\n    else:\n        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding=\'latin\'))\n        \n    return embeddings_index\n\nglove = \'../input/quora-insincere-questions-classification/embeddings/glove.840B.300d/glove.840B.300d.txt\'\n\nprint("Extracting GloVe embedding")\nembeddings_dict_glove= load_embed(glove)\nprint("Number of embeddings loaded:",len(embeddings_dict_glove))')


# ## DATA

# In[ ]:


path="../input/"
train=pd.read_csv(path+"hackereathmlinterntest/756269323c1011e9/dataset/hm_train.csv")
test=pd.read_csv(path+"hackereathmlinterntest/756269323c1011e9/dataset/hm_test.csv")
sample=pd.read_csv(path+"hackereathmlinterntest/756269323c1011e9/dataset/sample_submission.csv")

print(train.shape,test.shape,sample.shape)
train.head()


# In[ ]:


test.head()


# ## DROPPING THE DUPLICATES

# Here we will check the text.

# In[ ]:


print("For ID",train['hmid'].nunique()==train.shape[0])
print("For text",train['cleaned_hm'].nunique()==train.shape[0])


# So there are some rows in the dataset which have same text and different ID. So the rows are duplicated only with ID changed. So we need to remove those duplicated rows in the train dataset. Let us see how is the case in test dataset.

# In[ ]:


print("For ID",train['hmid'].nunique()==train.shape[0])
print("For text",train['cleaned_hm'].nunique()==train.shape[0])


# So even in the test data we have duplicated rows , but we have to just the predict the category, so it's fine here. Now I will remove duplicated rows in the train dataset. As ID is not at all used even in the future, so I am dropping the ID from the train dataset.

# In[ ]:


train_id=train['hmid']
train.drop(columns=['hmid'],inplace=True)
print(train.shape)
train.head()


# In[ ]:


# dropping the duplicates 
train.drop_duplicates(inplace=True)
print(train.shape)
train.head()


# So we can see easily that from 60321 to the examples dropped to 58682. That's a decrease of 1639 examples.

# ## TARGET VARIABLE
# 
# Distribution of target variable.

# In[ ]:


sns.countplot(train['predicted_category'])
plt.xticks(rotation='90')
plt.show()


# In[ ]:


target_info=train['predicted_category'].value_counts().reset_index()
target_info['percentage']=(target_info['predicted_category']/train.shape[0]*100).astype(str)+" %"
target_info


# In[ ]:





# The dataset looks highly imbalanced. There are 7 categories. 

# ## NUMBER OF SENTENCES

# In[ ]:


plt.figure(figsize=(15,5))
plt.subplot(131)
sns.distplot(train['num_sentence'],kde=False)
plt.title("Train Distribution")
plt.yscale("log")
plt.subplot(132)
sns.boxplot(train['predicted_category'],y=train['num_sentence'])
plt.xticks(rotation="90")
plt.subplot(133)
sns.distplot(test['num_sentence'],kde=False)
plt.title("Test Distribution")
plt.yscale("log")
plt.show()


# So there are statments with number of sentences nearly 60 (which is very high). And you can see from the other graph that the sentences which have 60 sentences are mostly belong to "affection". in test data the num of sentences are even more with approx 70 sentences.

# ## REFLECTION PEROID
# 
# It represents the time of happiness. This variable only takes two values 24h and 3m. Let us see the distribution in train and test and also the relation with target.

# In[ ]:


plt.figure(figsize=(15,5))
plt.subplot(131)
sns.countplot(train['reflection_period'])
plt.title("Train Distribution")
plt.subplot(132)
sns.countplot(train['reflection_period'],hue=train['predicted_category'])
plt.subplot(133)
sns.countplot(test['reflection_period'])
plt.title("Test Distribution")
plt.tight_layout()
plt.show()


# Number of 3m are more in test data than in train data.

# ## WORD LENGTH
# 
#    In this we will examin the word length of the sentences.

# In[ ]:


train['num_words']=train['cleaned_hm'].apply(lambda x:len(x.split()))
test['num_words']=test['cleaned_hm'].apply(lambda x:len(x.split()))
print(train.shape,test.shape)
train.head()


# In[ ]:


print("The average word length in train is",train['num_words'].mean(),
                      "and in test is",test['num_words'].mean())

plt.figure(figsize=(12,5))
plt.subplot(121)
sns.distplot(train['num_words'],kde=False)
plt.yscale("log")
plt.title("Train Distribution")
plt.subplot(122)
sns.distplot(test['num_words'],kde=False)
plt.yscale("log")
plt.title("Test Distribution")
plt.show()


# The word length looks same. But there are some sentences for which the length is 1200 which is very high for RNN,LSTM. But in train and test , the average word length is 19 and 20. As there are outliers , it's better we see the median length also. 

# In[ ]:


print("The median word length in train is",train['num_words'].median(),
                      "and in test is",test['num_words'].median())


# So median word length is 14 for train and 13 for test.

# ## TEXT

# In[ ]:


def build_vocab(sentences,verbose=True):
    vocab={}
    for sentence in tqdm(sentences,disable=(not verbose)):
        for word in sentence:
            try:
                vocab[word]+=1
            except KeyError:
                vocab[word]=1
                
    print("Number of words found in vocab are",len(vocab.keys()))
    return dict(sorted(vocab.items(), key=operator.itemgetter(1))[::-1])

def sen(x):
    return x.split()

def check_coverage(vocab,embeddings_dict):
    # words that dont have embeddings
    oov={}
    # stores words that have embeddings
    a=[]
    i=0
    k=0
    for word in tqdm(vocab.keys()):
        if embeddings_dict.get(word) is not None:                    # implies that word has embedding
            a.append(word)
            k=k+vocab[word]
        else:
            oov[word]=vocab[word]
            i=i+vocab[word]
    
    print("Total embeddings found in vocab are",len(a)/len(vocab)*100,"%")
    print("Total embeddings found in text are",k/(k+i)*100,"%")
    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]
    return dict(sorted_x)


# The code for the analysis written below is removed.
# 
# As you can see the words that dont have embedding are in compact form an not in loose form. For example if haven't was written as have not then there is no problem finding a embedding. As haven't and have not doesn't change the meaning of the sentence we can replace them easily. The mappings are defined below.

# In[ ]:


puncts=[',','.','!','$','(',')','%','[',']','?',':',";","#",'/','"',"'","-","|",'*']

contraction_mapping={"haven't":"have not","hadn't":"had not","wasn't":"was not","he's":"he is",
                     "couldn't":"could not","she's":"she is","i'm":"i am","we've":"we have",
                     "wouldn't":"would not","That's":"That is","we're":"we are","isn't":"is not",
                     "hasn't":"has not","they're":"they are","She's":"She is","He's":"He is","weren't":"were not",
                    "there's":"there is","i've":"i have","you've":"you have","We've":"We have","we'd":"we would",
                    "We're":"We are","who's":"who is","they'll":"they will","what's":"what is","she'd":"she would",
                    "They're":"They are","aren't":"are not","shouldn't":"should not","There's":"There is",
                     "we'll":"we will","I`m":"I am","You're":"You are","i'd":"i would","he'll":"he will",
                    "they'd":"they would","Didn't":"Did not","CAN'T":"CANNOT","THAT'S":"That is","you;ll":"you will",
                    "You'll":"You will","Can't":"cannot","would've":"would have","you\'re":"you are","i'll":"i will",
                    "DIDN'T":"did not","film/theater":"film or theater","that\'s":"that is","Let's":"Lets","We'd":"We would",
                    "They've":"They have","she'll":"she will","Haven't":"Have not","it'll":"it will","you'd":"you would",
                    "I'VE":"I have","I`ve":"I have","She'll":"She will","It'll":"It will","Hadn't":"Had not","I\'ve":"I have",
                     "You\'re":"You are","b'day":"birthday","DON'T":"Do not","it'd":"it would","You've":"You have",
                     "I'LL":"I will","don\'t":"do not","what\'s":"what is","won't":"will not",
                    "he'd":"he would","I'M":"I AM"}

misspelled_words={"fiancA":"fiance","couldnat":"could not","giftz":"gifts",
                "othersa":"others","nerous":"nervous","wasnat":"was not",
                "aIam":"I am","arace":"a race","10class":"10 class",
                 "aHow":"How","aWait":"Wait","aMumma":"Mumma","aWhy":"Why",
                "B+":"B +","INAGURATION":"INAUGURATION","wonat":"will not",
                "3+":"3 +","Thereas":"There is","Letas":"Let's",
                "Valentineas":"Valentines","genervous":"generous",
                 "brotheras":"brother's","Dhubai":"Dubai",
                 "shiridi":"shirdi","PAPPER":"PAPER","booka|":"book",
                 "aHey":"Hey","seek&hide":"seek and hide"}

def replace_misspelled(x):
    for word in misspelled_words.keys():
        x=x.replace(word,misspelled_words[word])
        
    return x

def replace_contraction_mapping(x):
    for contract in contraction_mapping.keys():
        x=x.replace(contract,contraction_mapping[contract])
    
    return x
    
def replace_puncts(x):
    for p in puncts:
        x=x.replace(p,f' {p} ')
        
    return x

cleaned_sen=train['cleaned_hm'].progress_apply(replace_misspelled)
cleaned_sen=cleaned_sen.progress_apply(replace_contraction_mapping)
cleaned_sen=cleaned_sen.progress_apply(replace_puncts)
sentences=cleaned_sen.progress_apply(sen)
vocab=build_vocab(sentences)
oov=check_coverage(vocab,embeddings_dict_glove)

# for index,sen in enumerate(cleaned_sen):
#     if "Valentineas" in sen.split():
#         print(sen,index)
#         print("\n")


# In[ ]:


# cleaning the train data and test data

# replace the misspelled word
train_clean_hm=train['cleaned_hm'].progress_apply(replace_misspelled)
test_clean_hm=test['cleaned_hm'].progress_apply(replace_misspelled)

# replace contraction mapping
train_clean_hm=train_clean_hm.progress_apply(replace_contraction_mapping)
test_clean_hm=test_clean_hm.progress_apply(replace_contraction_mapping)

# replace punctuations
train_clean_hm=train_clean_hm.progress_apply(replace_puncts)
test_clean_hm=test_clean_hm.progress_apply(replace_puncts)


# ## TOKENIZING AND PADDING

# In[ ]:


max_words=20000
max_len=70
embed_dim=300


tokenizer=Tokenizer(num_words=max_words,filters=None,lower=False)
tokenizer.fit_on_texts(list(train_clean_hm.apply(sen).values))

print("The length of vocabulary is",len(tokenizer.word_index))

X=tokenizer.texts_to_sequences(list(train_clean_hm.apply(sen).values))
X_test=tokenizer.texts_to_sequences(list(test_clean_hm.apply(sen).values))

# padding and truncating
X=np.array(pad_sequences(X,maxlen=max_len,padding='pre',truncating='pre'))
X_test=np.array(pad_sequences(X_test,maxlen=max_len,padding='pre',truncating='pre'))


# target
target_encoder={"affection":0,"achievement":1,
                "bonding":2,"enjoy_the_moment":3,
               "leisure":4,"nature":5,
               "exercise":6}

target_decoder=dict(zip(target_encoder.values(),target_encoder.keys()))

y=pd.get_dummies(train['predicted_category'].map(target_encoder)).values

print("Training Shape",X.shape,y.shape)
print("Test Shape",X_test.shape)


# ## EMBEDDINGS MATRIX

# In[ ]:


def give_embed_glove(word_index):
    nb_words=min(max_words,len(word_index)+1)
    embeddings_matrix_glove = np.zeros((nb_words, embed_dim))
    for word,index in word_index.items():
        if index>=max_words:
            continue
        # implies that word has embedding
        if embeddings_dict_glove.get(word) is not None:
            embeddings_matrix_glove[index]=embeddings_dict_glove.get(word)
            
    
    return embeddings_matrix_glove


embeddings_matrix_glove=give_embed_glove(tokenizer.word_index)
print("The Shape of the glove matrix is",embeddings_matrix_glove.shape)


# ## METRICS

# In[ ]:


def cross_entropy(y_true,y_pred,eps=1e-8):
    """
    y_true : (m,classes) contaning true values.
    y_pred : (m,classes) contaning predictions.

    Returns : Cross entropy loss between y_true and y_pred.
    """
    predictions=np.clip(y_pred,eps,1-eps)
    m=predictions.shape[0]
    loss=-np.sum(y_true*np.log(predictions))/m
    return loss

def f1(y_true,y_pred,threshold=0.5,eps=1e-8):
    """    
    y_true : (m,classes) contaning true values.
    y_pred : (m,classes) contaning predictions.
    
    Returns : Average Weighted F1 Score.
    """
    predictions=np.argmax(y_pred,axis=1)
    targets=np.argmax(y_true,axis=1)
    return f1_score(targets,predictions,average="weighted")


# ## LOSS

# In[ ]:


class CrossEntropyLoss(nn.Module):
    """
    y_true = (N,C)
    y_pred = (N,C)
    Cross Entropy Loss
    """
    def __init__(self,eps=1e-8):
        super(CrossEntropyLoss,self).__init__()
        self.eps=eps
    
    def forward(self,y_true,y_pred):
        y_pred=torch.clamp(y_pred,self.eps,1-self.eps)
        m=y_pred.shape[0]
        loss=-torch.sum(y_true*torch.log(y_pred))/m
        
        return loss


# ## CYCLIC LEARNING RATE

# In[ ]:


# code inspired from: https://github.com/anandsaha/pytorch.cyclic.learning.rate/blob/master/cls.py
class CyclicLR(object):
    def __init__(self, optimizer, base_lr=1e-3, max_lr=6e-3,
                 step_size=2000, mode='triangular', gamma=1.,
                 scale_fn=None, scale_mode='cycle', last_batch_iteration=-1):

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(base_lr, list) or isinstance(base_lr, tuple):
            if len(base_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} base_lr, got {}".format(
                    len(optimizer.param_groups), len(base_lr)))
            self.base_lrs = list(base_lr)
        else:
            self.base_lrs = [base_lr] * len(optimizer.param_groups)

        if isinstance(max_lr, list) or isinstance(max_lr, tuple):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} max_lr, got {}".format(
                    len(optimizer.param_groups), len(max_lr)))
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self.step_size = step_size

        if mode not in ['triangular', 'triangular2', 'exp_range']                 and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.batch_step(last_batch_iteration + 1)
        self.last_batch_iteration = last_batch_iteration

    def batch_step(self, batch_iteration=None):
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma**(x)

    def get_lr(self):
        step_size = float(self.step_size)
        cycle = np.floor(1 + self.last_batch_iteration / (2 * step_size))
        x = np.abs(self.last_batch_iteration / step_size - 2 * cycle + 1)

        lrs = []
        param_lrs = zip(self.optimizer.param_groups, self.base_lrs, self.max_lrs)
        for param_group, base_lr, max_lr in param_lrs:
            base_height = (max_lr - base_lr) * np.maximum(0, (1 - x))
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_batch_iteration)
            lrs.append(lr)
        return lrs


# ## MODEL

# In[ ]:


class Attention(nn.Module):
    def __init__(self,hidden_dim,max_len):
        super(Attention,self).__init__()
        
        self.hidden_dim=hidden_dim
        self.max_len=max_len
        
        self.tanh=nn.Tanh()
        self.linear=nn.Linear(in_features=self.hidden_dim,out_features=1,bias=False)
        self.softmax=nn.Softmax(dim=1)        
        
    def forward(self,h):
        
        m=self.tanh(h)
        
        alpha=self.linear(m)
        
        alpha=torch.squeeze(alpha)           # shape of alpha will be batch_size*max_len
        
#         print("Alpha shape:",alpha.shape)
        
        # softmax(note that softmax is along dimension 1)
        alpha=self.softmax(alpha)
        
        # unsequezzing alpha to get shape as batch_size*max_len*1
        alpha=torch.unsqueeze(alpha,-1)
        
        # we have to define r
        r=h*alpha
        
        # now we have to take sum and shape of r is batch_size*hidden_size
        r=torch.sum(r,dim=1)
        
        return r


# In[ ]:


# model params
hidden_units=64

class LstmGru(nn.Module):
    def __init__(self,embeddings_matrix):
        super(LstmGru,self).__init__()

        self.embedding=nn.Embedding.from_pretrained(torch.Tensor(embeddings_matrix),freeze=True)
        
        self.lstm=nn.LSTM(input_size=embed_dim,hidden_size=hidden_units,
                              bidirectional=True,batch_first=True)
        
        # gru output goes to lstm
        self.gru=nn.GRU(input_size=2*hidden_units,hidden_size=hidden_units,
                           bidirectional=True,batch_first=True)
        
        # lstm attention
        self.lstm_attention=Attention(2*hidden_units,max_len)
        
        # gru attention
        self.gru_attention=Attention(2*hidden_units,max_len)
        
        
        self.linear1=nn.Linear(in_features=8*hidden_units,out_features=32)
        self.batch1=nn.BatchNorm1d(32)
        self.relu1=nn.ReLU()
        self.drop1=nn.Dropout(0.25)
        
        self.linear2=nn.Linear(in_features=32,out_features=7)
        self.softmax=nn.Softmax(dim=1)
        
        
    def forward(self,X):
        batch_size=X.shape[0]
        
        embeds=self.embedding(X.long())
        
        h_lstm,_=self.lstm(embeds)
        h_gru,_=self.gru(h_lstm)
        
        # max pooling over time
        h_lstm_max,_=torch.max(h_lstm,1)
        h_gru_max,_=torch.max(h_gru,1)
        
        # mean average pooling over time
        h_lstm_mean=torch.mean(h_lstm,1)
        h_gru_mean=torch.mean(h_gru,1)
        
        # attention
        h_lstm_attend=self.lstm_attention(h_lstm)
        h_gru_attend=self.gru_attention(h_gru)

        h=torch.cat((h_lstm_attend,h_gru_attend,h_lstm_max,h_gru_max),dim=1)
        
        output=self.relu1(self.batch1(self.linear1(h)))
        output=self.drop1(output)
        
        output=(self.linear2(output))
        output=self.softmax(output)
        
        return output
    
def initialize_model(embeddings_matrix):
    model=LstmGru(embeddings_matrix)
    
    # setting all the dtypes to float
    model.float()
    
    # pushing the code to gpu
    model.cuda()
    
    # params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("Total trainiable Param's are",trainable_params)
    
    return model


# In[ ]:


initialize_model(embeddings_matrix_glove)


# In[ ]:





# ## SOME USEFUL FUNCTIONS

# In[ ]:


def fit_data(model,optimizer,loss_fn,scheduler=None,
             train_iterator=None,val_iterator=None,
            m_train=None,m_val=None,num_classes=None,epochs=None,fname=None):
    """
        model : pytroch model
        optimizer : any optimizer from torch.optim
        m_train : number of training examples
        m_val : number of validation examples
        epochs : number of epochs.
        fname : file name to save the model(it always saves the best)
        
        returns
        best train preds and best val preds(selected based on validation score)
        and evaluations(which cotains losses,accuracy and f1 score of every epoch)
    """
    
    train_loss=[]
    val_loss=[]
    train_f1=[]
    val_f1=[]
    evals={}
    
    best_train_preds=np.zeros((m_train,num_classes))
    best_val_preds=np.zeros((m_val,num_classes))
    best_val_f1=0
    
    for ep_num in range(epochs):
        print("Epoch","{0}/{1}:".format(ep_num+1,epochs))
        
        # measuring the current time
        start=datetime.datetime.now()
        
        # iterating through batch
        train_targets=np.zeros((m_train,num_classes))
        train_preds=np.zeros((m_train,num_classes))
    
        # start train_index
        train_index=0
        
        # setting up the model in train
        model.train()
    
        for batch,(X_train,y_train) in enumerate(train_iterator):
            optimizer.zero_grad()
            
            X_train=Variable(X_train.cuda())
            y_train=Variable(y_train.type('torch.FloatTensor').cuda())
            
            y_pred=model.forward(X_train)
            
            if scheduler:
                scheduler.batch_step()
                
            loss=loss_fn(y_train,y_pred)
            
            # appending the preds
            train_preds[train_index:train_index+X_train.shape[0]]=y_pred.cpu().detach().numpy().reshape((-1,num_classes))
            
            # appending the targets
            train_targets[train_index:train_index+X_train.shape[0]]=y_train.cpu().detach().numpy().reshape((-1,num_classes))
            
            # backprop
            loss.backward()
            
            # update the weights
            optimizer.step()
            
            train_index=train_index+X_train.shape[0]
            
            logger=str(train_index)+"/"+str(m_train)

            print(logger,end='\r')
            
        # setting up in evaluation mode
        model.eval()
        
        val_targets=np.zeros((m_val,num_classes))
        val_preds=np.zeros((m_val,num_classes))
        val_index=0
        
        for batch,(X_val,y_val) in enumerate(val_iterator):
            
            X_val=Variable(X_val.cuda())
            y_val=Variable(y_val.type('torch.FloatTensor').cuda())
            
            y_pred=model.forward(X_val)
            
            # appending the preds
            val_preds[val_index:val_index+X_val.shape[0]]=y_pred.cpu().detach().numpy().reshape((-1,num_classes))
            
            # appending the targets
            val_targets[val_index:val_index+X_val.shape[0]]=y_val.cpu().detach().numpy().reshape((-1,num_classes))
            
            val_index=val_index+X_val.shape[0]
            
        # finding the losses and f1 score 
        trainloss=cross_entropy(train_targets,train_preds)
        valloss=cross_entropy(val_targets,val_preds)
        
        trainf1=f1(train_targets,train_preds)
        valf1=f1(val_targets,val_preds)
        
        train_loss.append(trainloss),val_loss.append(valloss)
        train_f1.append(trainf1),val_f1.append(valf1)
        
        # end measuring time 
        end=datetime.datetime.now()
        
        print("Seconds = ",round((end-start).total_seconds()),end=" ")
        
        print("train loss = ",round(trainloss,5),end=" ")
        print("train f1 = ",round(trainf1,5),end=" ")

        print("val loss = ",round(valloss,5),end=" ")
        print("val f1 = ",round(valf1,5))
        
        if valf1>best_val_f1:
            print("Validation F1 score increased from",round(best_val_f1,5),"to",round(valf1,5),                                      "Saving the model at",fname)
            
            torch.save(model.state_dict(),fname)
            
            best_val_f1=valf1
            best_train_preds=train_preds
            best_val_preds=val_preds
        print("\n")
    
    # outside of epoch loop
    evals['train_loss']=train_loss
    evals['val_loss']=val_loss
    evals['train_f1']=train_f1
    evals['val_f1']=val_f1
    evals['best_val_f1']=best_val_f1
    
    return best_train_preds,best_val_preds,evals


# In[ ]:


def predict_on_test(model,test_iterator,m_test):
    # model at evaluation mode
    model.eval()
    
    test_preds=np.zeros((m_test,num_classes))
    test_index=0
    
    start=datetime.datetime.now()
    
    for batch,X_test in enumerate(test_iterator):
        X_test=Variable(X_test[0].cuda())
        
        y_pred=model.forward(X_test)
        # appending the preds
        test_preds[test_index:test_index+X_test.shape[0]]=y_pred.cpu().detach().numpy().reshape((-1,num_classes))
        
        test_index=test_index+X_test.shape[0]
        
        logger=str(test_index)+"/"+str(m_test)
        
        if batch<len(test_iterator)-1:
            print(logger,end='\r')
        else:
            print(logger,end=" ")
            
    end=datetime.datetime.now()
    print("Predictions done on test data in",round((end-start).total_seconds()),"seconds")
    
    return test_preds


# In[ ]:


print("Training Shape",X.shape,y.shape)
print("Testing Shape",X_test.shape)


# In[ ]:


def make_dataset(X_train,y_train,X_val,y_val,batch_size):
    X_train,y_train=torch.Tensor(X_train),torch.Tensor(y_train)
    X_val,y_val=torch.Tensor(X_val),torch.Tensor(y_val)

    # train dataset and val dataset contains pair of X and y for each example 
    train_dataset=TensorDataset(X_train,y_train)
    val_dataset=TensorDataset(X_val,y_val)
    test_dataset=TensorDataset(X_test)

    # now I will pass this to data loader
    # shuffle set to true imples for every epoch data is shuffled
    train_iterator=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    val_iterator=DataLoader(val_dataset,batch_size=batch_size,shuffle=False)
    
    return train_iterator,val_iterator


# In[ ]:





# ## TRAINING

# In[ ]:


n_folds=5
num_classes=7
kfold=StratifiedKFold(n_splits=n_folds,shuffle=True,random_state=SEED)

scores=np.zeros((n_folds,))

# oof preds 
oof_preds=np.zeros((X.shape[0],num_classes))
# test preds
test_preds=np.zeros((X_test.shape[0],num_classes))


# batch size and epochs per fold
batch_size=64
ep=[8]*n_folds

# making the test iterator  and for predictions the batch size can be more , so that we can predict fast
X_test=torch.Tensor(X_test)
test_dataset=TensorDataset(X_test)
test_iterator=DataLoader(test_dataset,batch_size=256,shuffle=False)
m_test=X_test.shape[0]

# one point to note is that kfold.split works for labels of form [0,1,1,2]
for fold,(train_index,val_index) in enumerate(kfold.split(X,np.argmax(y,axis=1))):
    X_train,X_val=X[train_index],X[val_index]
    y_train,y_val=y[train_index],y[val_index]
    
    m_train,m_val=X_train.shape[0],X_val.shape[0]
    
    print("================================= FOLD",fold+1,"=============================================")

    print("Training Shape:",X_train.shape,y_train.shape)
    print("Validation Shape:",X_val.shape,y_val.shape)
    
    train_iterator,val_iterator=make_dataset(X_train,y_train,X_val,y_val,batch_size=batch_size)
    
    gc.enable()
    del X_train,y_train,X_val,y_val
    gc.collect()
    
    model=initialize_model(embeddings_matrix_glove)
    
    base_lr=1e-3
    max_lr=1e-2
    is_scheduler=True
    if is_scheduler:
        step_size=1468                   # 2 times the iteration in an epoch
        optimizer=torch.optim.Adam(model.parameters(),lr=max_lr)
        scheduler=CyclicLR(optimizer,base_lr=base_lr,max_lr=max_lr,
                              step_size=step_size,mode='triangular')
    else:
        optimizer=torch.optim.Adam(model.parameters(),lr=base_lr)
        scheduler=None
        
    
    loss_fn=CrossEntropyLoss()
    epochs=ep[fold]
    fname="LstmGru"+str(fold+1)+".pt"
    best_train_preds,best_val_preds,evals=fit_data(model,optimizer,loss_fn,scheduler,train_iterator,val_iterator,
                                              m_train,m_val,num_classes,epochs,fname)
    print("Loading the model")
    model=LstmGru(embeddings_matrix_glove)
    model.float()
    model.cuda()
    model.load_state_dict(torch.load(fname))
    preds=predict_on_test(model,test_iterator,m_test)
    
    gc.enable()
    del model
    gc.collect()
    
    # saving the oof preds
    oof_preds[val_index]=best_val_preds
    
    # test predictions
    test_preds=test_preds+preds/n_folds
    
    # storing the scores
    scores[fold]=evals['best_val_f1']


# In[ ]:


print("The F1 Score on the total data is",f1(y,oof_preds))
print("\n")
print("The fold scores are",scores,"and the mean is",np.mean(scores),"and std is",np.std(scores))


# ## MAKING THE SUBMISSION FILE

# In[ ]:


sub=pd.DataFrame()
sub['hmid']=test['hmid']
sub['predicted_category']=pd.Series(np.argmax(test_preds,axis=1)).map(target_decoder)
sub.head()


# In[ ]:


plt.figure(figsize=(12,5))
sns.countplot(sub['predicted_category'])
plt.show()


# In[ ]:


sub.to_csv("first_sub.csv",index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




