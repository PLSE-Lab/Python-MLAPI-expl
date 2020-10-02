#!/usr/bin/env python
# coding: utf-8

# ### This is my first competition entry to Kaggle.  The Deep learning model has been developed using Pytorch
# The exploration and data transformation has been heavily borrowed from Noobhound's 'A simple nn solution with Keras (~0.48611 PL)' and ThyKhueLy 'Mercari Interactive EDA + Topic Modelling' Kernels.  All the Pytorch model is mine.  
# #### I am not having success with the deep learning model - very poor validation performance.  I have tried different types of convolution types on the item_description and name embeddings.  Any pointers to a better Deep learning model from the forum would be helpful.  Not sure If I am missing something basic in my models.  
# 
# 

# In[ ]:


import os
os.environ['OMP_NUM_THREADS'] = '4'


# In[ ]:


# used for developing deep learning models
import torch
from torch.autograd import Variable
from torch import optim
#from torch.optim import lr_scheduler
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as F


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import time          #to get the system time


# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import math


# In[ ]:


train = pd.read_csv('../input/train.tsv', sep='\t')
test = pd.read_csv('../input/test.tsv', sep='\t')


# In[ ]:


train.describe()


# In[ ]:


test.describe()


# In[ ]:


train.head()


# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# ## Item Category

# In[ ]:


print("There are %d unique values in the category column." % train['category_name'].nunique())
# TOP 5 RAW CATEGORIES
print(train['category_name'].value_counts()[:5])
# missing categories
print("There are %d items that do not have a label." % train['category_name'].isnull().sum())


# In[ ]:


# reference: BuryBuryZymon at https://www.kaggle.com/maheshdadhich/i-will-sell-everything-for-free-0-55
def split_cat(text):
    try: return text.split("/")
    except: return ("None", "None", "None")


# In[ ]:


train['general_cat'], train['subcat_1'], train['subcat_2'] = zip(*train['category_name'].apply(lambda x: split_cat(x)))
train.head()
test['general_cat'], test['subcat_1'], test['subcat_2'] = zip(*test['category_name'].apply(lambda x: split_cat(x)))
test.head()


# In[ ]:


print("There are %d unique first sub-categories." % train['general_cat'].nunique())
print("There are %d unique first sub-categories." % train['subcat_1'].nunique())
print("There are %d unique second sub-categories." % train['subcat_2'].nunique())


# In[ ]:


### Plotting some histograms of categorical Variables
plt.figure(figsize=(10,10))
plt.subplot(3,3,1)
count_classes_general_cat = pd.value_counts(train.general_cat, sort = True)
count_classes_general_cat.plot(kind = 'bar')
plt.title("General Category histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")
# subcategory 1
plt.subplot(3,3,3)
count_classes_subcat_1 = pd.value_counts(train.subcat_1, sort = True)[:15]
count_classes_subcat_1.plot(kind = 'bar')
plt.title("Sub Category 1 histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")
# subcategory 2
plt.subplot(3,3,9)
count_classes_subcat_2 = pd.value_counts(train.subcat_2, sort = True)[:15]
count_classes_subcat_2.plot(kind = 'bar')
plt.title("Sub Category 2 histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")


# In[ ]:


print("There are %d unique brand names in the training dataset." % train['brand_name'].nunique())


# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(11, 7), sharex=True)
sns.distplot(np.log(train['price'].values+1))


# In[ ]:


def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    return (sum(to_sum) * (1.0/len(y))) ** 0.5
#Source: https://www.kaggle.com/marknagelberg/rmsle-function


# In[ ]:


#HANDLE MISSING VALUES
print("Handling missing values...")
def handle_missing(dataset):
    #dataset.category_name.fillna(value="na", inplace=True)
    dataset.brand_name.fillna(value="None", inplace=True)
    dataset.item_description.fillna(value="None", inplace=True)
    dataset.category_name.fillna(value="None", inplace=True)
    return (dataset)

train = handle_missing(train)
test = handle_missing(test)
print(train.shape)
print(test.shape)


# In[ ]:


train.isnull().sum()
# Not to worry about the nulls in Category name as the nulls have been taken care of earlier when splitting the
# category name into general, sub cat1 and sub cat2


# In[ ]:


#PROCESS CATEGORICAL DATA
#print("Handling categorical variables...")
def encode_text(column):
    le = LabelEncoder()
    le.fit(np.hstack([train[column], test[column]]))
    train[column+'_index'] = le.transform(train[column])
    test[column+'_index'] = le.transform(test[column])


# In[ ]:


encode_text('brand_name')
encode_text('general_cat')
encode_text("subcat_1")
encode_text('subcat_2')
encode_text('category_name')


# In[ ]:


test.head()


# In[ ]:


class Category:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# In[ ]:


# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
import unicodedata
import re
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    #s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def normalizeLine(sentence):
    return [normalizeString(s) for s in sentence.split('\t')]


# In[ ]:


def prepareData(lang1,data):
    input_cat = Category(lang1)
    for sentence in data:
        normalize_line = [normalizeString(s) for s in sentence.split('\t')]
        input_cat.addSentence(normalize_line[0])
        
    print("Counted words:")
    print(input_cat.name, input_cat.n_words)
    return input_cat


# In[ ]:


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def variableFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    #indexes.append(EOS_token)
    return indexes


# In[ ]:


def token_fit(column):
    raw_text = np.hstack([(train[column]).str.lower(), (test[column]).str.lower()])
    cat1 = prepareData(column,raw_text)
    print ("adding train data")
    train[column + '_seq'] = [variableFromSentence(cat1,normalizeLine(sentence.lower())[0])                                                       for sentence in train[column]]
    print ("adding test data")
    test[column + '_seq'] = [variableFromSentence(cat1,normalizeLine(sentence.lower())[0])                                                       for sentence in test[column]]


# In[ ]:


token_fit('name')


# In[ ]:


token_fit('item_description')


# In[ ]:


train.head()


# In[ ]:


#SEQUENCES VARIABLES ANALYSIS
max_name_seq = np.max([np.max(train.name_seq.apply(lambda x: len(x))), np.max(test.name_seq.apply(lambda x: len(x)))])
max_item_description_seq = np.max([np.max(train.item_description_seq.apply(lambda x: len(x)))
                                   , np.max(test.item_description_seq.apply(lambda x: len(x)))])
print("max name seq "+str(max_name_seq))
print("max item desc seq "+str(max_item_description_seq))


# In[ ]:


#EMBEDDINGS MAX VALUE
#Base on the histograms, we select the next lengths
MAX_NAME_SEQ = 10
MAX_ITEM_DESC_SEQ = 75
MAX_TEXT = np.max([np.max(train.name_seq.max()) 
                   , np.max(test.name_seq.max())
                  , np.max(train.item_description_seq.max())
                  , np.max(test.item_description_seq.max())])+2
MAX_GEN_CATEGORY = np.max([train.general_cat_index.max(), test.general_cat_index.max()])+1
MAX_SUB_CAT1_CATEGORY = np.max([train.subcat_1_index.max(), test.subcat_1_index.max()])+1
MAX_SUB_CAT2_CATEGORY = np.max([train.subcat_2_index.max(), test.subcat_2_index.max()])+1
MAX_BRAND = np.max([train.brand_name_index.max(), test.brand_name_index.max()])+1
MAX_CONDITION = np.max([train.item_condition_id.max(), test.item_condition_id.max()])+1
MAX_CATEGORY_NAME = np.max([train.category_name_index.max(), test.category_name_index.max()])+1


# In[ ]:


MAX_BRAND


# In[ ]:


#SCALE target variable
train["target"] = np.log(train.price+1)
target_scaler = MinMaxScaler(feature_range=(-1, 1))
train["target"] = target_scaler.fit_transform(train.target.values.reshape(-1,1))
pd.DataFrame(train.target).hist()


# In[ ]:


print (target_scaler.scale_,target_scaler.min_,target_scaler.data_min_,target_scaler.data_max_)


# In[ ]:


train.head()


# In[ ]:


#EXTRACT DEVELOPTMENT TEST
dtrain, dvalid = train_test_split(train, random_state=123, train_size=0.99)
print(dtrain.shape)
print(dvalid.shape)


# In[ ]:


def pad(tensor, length):
    if length > tensor.size(0):
        return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])
    else:
        return torch.split(tensor, length, dim=0)[0]


# In[ ]:


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        name, item_desc,brand_name,cat_name,general_category,subcat1_category,subcat2_category,         item_condition,shipping,target = sample['name'], sample['item_desc'], sample['brand_name'],         sample['cat_name'], sample['general_category'], sample['subcat1_category'], sample['subcat2_category'],         sample['item_condition'], sample['shipping'],sample['target']
        #item_desc, brand_name = sample['item_desc'], sample['brand_name']       
        return {'name': pad(torch.from_numpy(np.asarray(name)).long().view(-1),MAX_NAME_SEQ),
                'item_desc': pad(torch.from_numpy(np.asarray(item_desc)).long().view(-1),MAX_ITEM_DESC_SEQ),
               'brand_name':torch.from_numpy(np.asarray(brand_name)),
               'cat_name':torch.from_numpy(np.asarray(cat_name)),
               'general_category':torch.from_numpy(np.asarray(general_category)),
               'subcat1_category':torch.from_numpy(np.asarray(subcat1_category)),
               'subcat2_category':torch.from_numpy(np.asarray(subcat2_category)),
               'item_condition':torch.from_numpy(np.asarray(item_condition)),
               'shipping':torch.torch.from_numpy(np.asarray(shipping)),
               'target':torch.from_numpy(np.asarray(target))}


# In[ ]:


# Define the Dataset to use in a DataLoader
class MercariDataset(Dataset):
    """Mercari Challenge dataset."""

    def __init__(self, data_pd, transform=None):
        """
        Args:
            data_pd: Data frame with the used columns.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.mercari_frame = data_pd
        self.transform = transform

    def __len__(self):
        return len(self.mercari_frame)

    def __getitem__(self, idx):
        name = [self.mercari_frame.name_seq.iloc[idx]]
        item_desc = [self.mercari_frame.item_description_seq.iloc[idx]]
        brand_name = [self.mercari_frame.brand_name_index.iloc[idx]]
        cat_name = [self.mercari_frame.category_name_index.iloc[idx]]
        general_category = [self.mercari_frame.general_cat_index.iloc[idx]]
        subcat1_category = [self.mercari_frame.subcat_1_index.iloc[idx]]
        subcat2_category = [self.mercari_frame.subcat_2_index.iloc[idx]]
        item_condition = [self.mercari_frame.item_condition_id.iloc[idx]]
        shipping = [self.mercari_frame.shipping.iloc[idx]]
        target = [self.mercari_frame.target.iloc[idx]]
        sample = {'name': name,
                'item_desc': item_desc,
               'brand_name': brand_name,
               'cat_name': cat_name,   
               'general_category': general_category,
               'subcat1_category': subcat1_category,
               'subcat2_category': subcat2_category,
               'item_condition': item_condition,
               'shipping': shipping,
               'target': target}

        if self.transform:
            sample = self.transform(sample)

        return sample


# In[ ]:


mercari_datasets = {'train': MercariDataset(dtrain,transform=transforms.Compose([ToTensor()])), 
                    'val': MercariDataset(dvalid,transform=transforms.Compose([ToTensor()]))
                   }
dataset_sizes = {x: len(mercari_datasets[x]) for x in ['train', 'val']}


# In[ ]:


mercari_dataloaders = {x: torch.utils.data.DataLoader(mercari_datasets[x], batch_size=50, shuffle=True) 
                                                           for x in ['train', 'val']}


# In[ ]:


mercari_dataloaders


# In[ ]:


# Some Useful Time functions
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


# In[ ]:


# Definition of the Pytorch Model
class RegressionNeural(nn.Module):
    def __init__(self, max_sizes):
        super(RegressionNeural, self).__init__()
        # declaring all the embedding for the various items
        self.name_embedding = nn.Embedding(np.asscalar(max_sizes['max_text']), 50)
        self.item_embedding = nn.Embedding(np.asscalar(max_sizes['max_text']), 50)
        self.brand_embedding = nn.Embedding(np.asscalar(max_sizes['max_brand']), 10)
        self.gencat_embedding = nn.Embedding(np.asscalar(max_sizes['max_gen_category']), 10)
        self.subcat1_embedding = nn.Embedding(np.asscalar(max_sizes['max_subcat1_category']), 10)
        self.subcat2_embedding = nn.Embedding(np.asscalar(max_sizes['max_subcat2_category']), 10)
        self.condition_embedding = nn.Embedding(np.asscalar(max_sizes['max_condition']), 5)
        # I am adding an embedding just based on Category name without separating it into the 3 pieces
        self.catname_embedding = nn.Embedding(np.asscalar(max_sizes['max_cat_name']), 10)
        
        ## I am trying to throw all types of convolutional model on the name and item embedding and haven't
        ## had any luck.  
        #self.conv1_name = nn.Conv1d(max_sizes['max_name_seq'], 1, 3, stride=1)
        #self.conv1_item_desc = nn.Conv1d(max_sizes['max_item_desc_seq'], 1, 5, stride=5) 
        
        self.conv1_name = nn.Conv1d(50, 1, 2, stride=1)
        # I am not using these other convolutions as they didn't seem to improve my result
        self.conv2_name = nn.Conv1d(16, 8, 2, stride=1)
        self.conv3_name = nn.Conv1d(8, 4, 2, stride=1)
        
        self.conv1_item_desc = nn.Conv1d(50, 1, 5, stride=5) 
        # I am not using these other convolutions as they didn't see to improve my result
        self.conv2_item_desc = nn.Conv1d(64, 16, 5, stride=1)
        self.conv3_item_desc = nn.Conv1d(16, 4, 5, stride=1)
        
        #self.conv1 = nn.Conv1d(64, 32, 3, stride=1)
        #self.conv2 = nn.Conv1d(32, 16, 3, stride=1)
        self.dropout = nn.Dropout(p=0.2)
        
        self.input_fc1_count = 50 #1214 #206 #16+10+10+10+10+5+1
        self.fc1 = nn.Linear(self.input_fc1_count, 64)
        self.fc2 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32,1)
        
        self.relu = nn.ReLU()  
            
    def forward(self, x, batchsize):
        embed_name = self.name_embedding(x['name'])
        #print ("embed_name size",embed_name.size())
        #embed_name = (self.conv1_name(embed_name))
        
        # I am swapping the Embedding size and the sequence length so that convolution is done across multiple words
        # using all the embeddings.  Without this, the 1-D convolution was doing convolution using all the words but
        # a slice of embeddings.  I don't think that is the correct way to do 1D convolution.  
        embed_name = F.relu(self.conv1_name(embed_name.transpose(1,2)))
        #print ("embed_name after 1st conv",embed_name.size())
        #embed_name = F.relu(self.conv2_name(embed_name))
        #print ("embed_name after 2nd conv",embed_name.size())
        #embed_name = self.conv3_name(embed_name)
        #print ("embed_name after 3rd conv",embed_name.size())
        
        embed_item = self.item_embedding(x['item_desc'])
        #print ("embed_item size",embed_item.size())
        #embed_item = (self.conv1_item_desc(embed_item))
        embed_item = F.relu(self.conv1_item_desc(embed_item.transpose(1,2)))
        #print ("embed_item after 1 conv",embed_item.size())
        #embed_item = F.relu(self.conv2_item_desc(embed_item))
        #print ("embed_item after 2 conv",embed_item.size())
        #embed_item = self.conv3_item_desc(embed_item)
        #print ("embed_item after 3rd conv",embed_item.size())
        
        embed_brand = self.brand_embedding(x['brand_name'])
        embed_gencat = self.gencat_embedding(x['general_category'])
        embed_subcat1 = self.subcat1_embedding(x['subcat1_category'])
        embed_subcat2 = self.subcat2_embedding(x['subcat2_category'])
        embed_condition = self.condition_embedding(x['item_condition'])
        embed_catname = self.catname_embedding(x['cat_name'])
        
        #out = torch.cat((embed_brand.view(batchsize,-1),embed_gencat.view(batchsize,-1), \
        #                 embed_subcat1.view(batchsize,-1), embed_subcat2.view(batchsize,-1), \
        #                 embed_condition.view(batchsize,-1),embed_name.view(batchsize,-1), \
        #                 embed_item.view(batchsize,-1),x['shipping']),1)
        out = torch.cat((embed_brand.view(batchsize,-1), embed_catname.view(batchsize,-1),                          embed_condition.view(batchsize,-1),embed_name.view(batchsize,-1),                          embed_item.view(batchsize,-1),x['shipping']),1)
        #out = self.dropout(out)
        
        out = (self.fc1(out))
        out = F.relu(self.dropout(out))
        out = (self.fc2(out))
        out = (self.dropout(out))
        out = self.fc3(out)
        return out

max_sizes = {'max_text':MAX_TEXT,'max_name_seq':MAX_NAME_SEQ,'max_item_desc_seq':MAX_ITEM_DESC_SEQ,              'max_brand':MAX_BRAND,'max_cat_name':MAX_CATEGORY_NAME,'max_gen_category':MAX_GEN_CATEGORY,             'max_subcat1_category':MAX_SUB_CAT1_CATEGORY,'max_subcat2_category':MAX_SUB_CAT2_CATEGORY,             'max_condition':MAX_CONDITION} 

deep_learn_model = RegressionNeural(max_sizes)


# In[ ]:


# Training model function that uses the dataloader to load the data by Batch
def train_model(model, criterion, optimizer, num_epochs=1, print_every = 100):
    start = time.time()

    best_acc = 0.0
    print_loss_total = 0  # Reset every print_every

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            num_batches = dataset_sizes[phase]/50.
            #running_corrects = 0

            # Iterate over data.
            for i_batch, sample_batched in enumerate(mercari_dataloaders[phase]): 
            # get the inputs
                inputs = {'name':Variable(sample_batched['name']), 'item_desc':Variable(sample_batched['item_desc']),                     'brand_name':Variable(sample_batched['brand_name']),                     'cat_name':Variable(sample_batched['cat_name']),                     'general_category':Variable(sample_batched['general_category']),                     'subcat1_category':Variable(sample_batched['subcat1_category']),                     'subcat2_category':Variable(sample_batched['subcat2_category']),                     'item_condition':Variable(sample_batched['item_condition']),                     'shipping':Variable(sample_batched['shipping'].float())}
                prices = Variable(sample_batched['target'].float())   
                batch_size = len(sample_batched['shipping'])   
                

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs, batch_size)
                #_, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, prices)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                print_loss_total += loss.data[0]
                #running_corrects += torch.sum(preds == labels.data)
                
                
                if (i_batch+1) % print_every == 0:
                    print_loss_avg = print_loss_total / print_every
                    print_loss_total = 0
                    #print (i_batch / num_batches, i_batch, num_batches)
                    print('%s (%d %d%%) %.4f' % (timeSince(start, i_batch / num_batches),                                                  i_batch, i_batch / num_batches*100, print_loss_avg))
                
                # I have put this just so that the Kernel will run and allow me to publish
                if (i_batch) > 500:
                    break

            epoch_loss = running_loss / num_batches
            #epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            
        print()

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # load best model weights
    #model.load_state_dict(best_model_wts)
    return model                         


# In[ ]:


# Set the optimizer Criterion and train the model
criterion = nn.MSELoss()

optimizer_ft = optim.SGD(deep_learn_model.parameters(), lr=0.001, momentum=0.9)
#optimizer_ft = optim.SGD(deep_learn_model.parameters(), lr=0.005)

# Decay LR by a factor of 0.1 every 7 epochs
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
train_model(deep_learn_model,criterion,optimizer_ft)
# I have run the model a lot of times with different combination of deep learning configs and I am not able to
# get a loss below 0.0341.  


# In[ ]:


# Function to calculate the RMSLE on the validation data
def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    return (sum(to_sum) * (1.0/len(y))) ** 0.5


# In[ ]:


# Validate the model results against validation data
def validate(model, print_every = 20, phase = 'val'):
    start = time.time()
    running_loss = 0
    print_loss_total = 0
    num_batches = dataset_sizes[phase]/50.
    y_pred_full = np.array([])
    y_true_full = np.array([])
    for i_batch, sample_batched in enumerate(mercari_dataloaders[phase]): 
    # get the inputs
        inputs = {'name':Variable(sample_batched['name']), 'item_desc':Variable(sample_batched['item_desc']),             'brand_name':Variable(sample_batched['brand_name']),             'cat_name':Variable(sample_batched['cat_name']),             'general_category':Variable(sample_batched['general_category']),             'subcat1_category':Variable(sample_batched['subcat1_category']),             'subcat2_category':Variable(sample_batched['subcat2_category']),             'item_condition':Variable(sample_batched['item_condition']),             'shipping':Variable(sample_batched['shipping'].float())}
        prices = Variable(sample_batched['target'].float())   
        batch_size = len(sample_batched['shipping'])

        # forward
        outputs = model(inputs,batch_size)
        val_preds = target_scaler.inverse_transform(outputs.data.numpy())
        val_preds = np.exp(val_preds)-1
        val_true =  target_scaler.inverse_transform(prices.data.numpy())
        val_true = np.exp(val_true)-1

        #mean_absolute_error, mean_squared_log_error
        y_true = val_true[:,0]
        y_pred = val_preds[:,0]
        y_true_full = np.append(y_true_full,y_true)
        y_pred_full= np.append(y_pred_full,y_pred)
        
        loss = criterion(outputs, prices)
        #print ("output size", val_preds.shape)
        #print ("ypred_full",len(y_pred_full))

        # statistics
        running_loss += loss.data[0]
        print_loss_total += loss.data[0]
        #print("loss data shape", loss.data.size())
        #print("running loss", running_loss)
        #running_corrects += torch.sum(preds == labels.data)


        if (i_batch+1) % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            #print (i_batch / num_batches, i_batch, num_batches)
            print('%s (%d %d%%) %.4f' % (timeSince(start, i_batch / num_batches),                                          i_batch, i_batch / num_batches*100, print_loss_avg))

    v_rmsle = rmsle(y_true_full, y_pred_full)
    print(" RMSLE error on dev validate: "+str(v_rmsle))
    print("total loss", running_loss / num_batches ) 
    return y_pred_full, y_true_full


# In[ ]:


# You can see the RMSE loss on validation data is very poor.  
y_pred_val, y_true_val = validate(deep_learn_model)


# In[ ]:


axes = plt.gca()
axes.set_ylim([0,100])
plt.scatter(y_pred_val,y_true_val)


# In[ ]:


y_pred_val.mean()


# In[ ]:


y_true_val.mean()

