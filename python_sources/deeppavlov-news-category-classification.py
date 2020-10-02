#!/usr/bin/env python
# coding: utf-8

# # News Category Classification

# <img src="http://static.minne.com/productimages/41458659/large/5fd6279befc739c4a7b59269397cdeaba3e2f6e0.jpg?1506897964" width="700px">

# # Notebook Outline

# 1. [**Importing Libraries**](#Importing-Libraries)   
# 2. [**Data Load**](#Data-Load)  
# 3. [**Data Preprocessing**](#Data-Preprocessing)  
#     3-1 [**Lowercasing**](#Lowercasing)  
#     3-2 [**Tokenizer**](#Tokenizer)  
#     3-3 [**Vocabulary**](#Vocabulary)  
#     3-4 [**Bag-of-words**](#Bag-of-words)  
#     3-5 [**GloVe Embedder**](#GloVe-Embedder)  
# 4. [**Model Build**](#Model-Build)  
# 5. [**Train**](#Train)  
# 6. [**Check Result**](#Check-Result)  

# # Importing Libraries

# In[ ]:


get_ipython().system('pip install deeppavlov')


# In[ ]:


# Import Libraries

import numpy as np 
import pandas as pd
from tqdm import tqdm

# deeppavlov
from deeppavlov.dataset_readers.basic_classification_reader import BasicClassificationDatasetReader
from deeppavlov.dataset_iterators.basic_classification_iterator import BasicClassificationDatasetIterator
from deeppavlov.models.preprocessors.str_lower import StrLower
from deeppavlov.models.tokenizers.nltk_moses_tokenizer import NLTKMosesTokenizer
from deeppavlov.core.data.simple_vocab import SimpleVocabulary
from deeppavlov.models.embedders.bow_embedder import BoWEmbedder
from deeppavlov.core.data.utils import simple_download
from deeppavlov.models.embedders.glove_embedder import GloVeEmbedder
from deeppavlov.metrics.accuracy import sets_accuracy
from deeppavlov.models.classifiers.keras_classification_model import KerasClassificationModel
from deeppavlov.models.preprocessors.one_hotter import OneHotter
from deeppavlov.models.classifiers.proba2labels import Proba2Labels

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Data Load

# In[ ]:


data = pd.read_json('../input/News_Category_Dataset_v2.json', lines=True)
data.head()


# In[ ]:


data.shape


# In[ ]:


data.isnull().sum()


# In[ ]:


# connect 'headline' and 'short_description'
data['text'] = data['headline'] + " " + data['short_description']


# In[ ]:


# export csv
data.to_csv('../News_Category_Dataset_v2.csv')


# In[ ]:


# read data from particular columns of `.csv` file
dr = BasicClassificationDatasetReader().read(
    data_path='../',
    train='News_Category_Dataset_v2.csv',
    x = 'text',
    y = 'category'
)


# In[ ]:


# initialize data iterator splitting `train` field to `train` and `valid` in proportion 0.8/0.2
train_iterator = BasicClassificationDatasetIterator(
    data=dr,
    field_to_split='train',  # field that will be splitted
    split_fields=['train', 'valid'],   # fields to which the fiald above will be splitted
    split_proportions=[0.8, 0.2],  #proportions for splitting
    split_seed=23,  # seed for splitting dataset
    seed=42)  # seed for iteration over dataset


# In[ ]:


# one can get train instances (or any other data type including `all`)
x_train, y_train = train_iterator.get_instances(data_type='train')
for x, y in list(zip(x_train, y_train))[:5]:
    print('x:', x)
    print('y:', y)
    print('=================')


# # Data Preprocessing

# ## Lowercasing

# In[ ]:


str_lower = StrLower()
# check
str_lower(['Kaggle is the best place to study machine learning.'])


# ## Tokenizer

# In[ ]:


tokenizer = NLTKMosesTokenizer()
# check
tokenizer(['Kaggle is the best place to study machine learning.'])


# In[ ]:


train_x_lower_tokenized = str_lower(tokenizer(train_iterator.get_instances(data_type='train')[0]))


# ## Vocabulary

# In[ ]:


# initialize simple vocabulary to collect all appeared in the dataset classes
classes_vocab = SimpleVocabulary(
    save_path='./tmp/classes.dict',
    load_path='./tmp/classes.dict')


# In[ ]:


classes_vocab.fit((train_iterator.get_instances(data_type='train')[1]))
classes_vocab.save()


# In[ ]:


# show classes
list(classes_vocab.items())


# In[ ]:


# also one can collect vocabulary of textual tokens appeared 2 and more times in the dataset
token_vocab = SimpleVocabulary(
    save_path='./tmp/tokens.dict',
    load_path='./tmp/tokens.dict',
    min_freq=2,
    special_tokens=('<PAD>', '<UNK>',),
    unk_token='<UNK>')


# In[ ]:


token_vocab.fit(train_x_lower_tokenized)
token_vocab.save()


# In[ ]:


# number of tokens in dictionary
len(token_vocab)


# In[ ]:


# 10 most common words and number of times their appeared
token_vocab.freqs.most_common()[:10]


# ## Bag-of-words

# In[ ]:


# initialize bag-of-words embedder giving total number of tokens
bow = BoWEmbedder(depth=token_vocab.len)
# it assumes indexed tokenized samples
bow(token_vocab(str_lower(tokenizer(['Kaggle is the best place to study machine learning.']))))


# In[ ]:


# all 10 tokens are in the vocabulary
sum(bow(token_vocab(str_lower(tokenizer(['Kaggle is the best place to study machine learning.']))))[0])


# ## GloVe Embedder

# In[ ]:


# Glove : https://nlp.stanford.edu/projects/glove/
simple_download(url="http://files.deeppavlov.ai/embeddings/glove.6B.100d.txt", destination="./glove.6B.100d.txt")


# In[ ]:


embedder = GloVeEmbedder(load_path='./glove.6B.100d.txt',dim=100, pad_zero=True)


# # Model Build

# In[ ]:


# get all train and valid data from iterator
x_train, y_train = train_iterator.get_instances(data_type="train")
x_valid, y_valid = train_iterator.get_instances(data_type="valid")


# In[ ]:


# Intialize `KerasClassificationModel` that composes CNN shallow-and-wide network 
# (name here as`cnn_model`)
cls = KerasClassificationModel(save_path="./cnn_model_v0", 
                               load_path="./cnn_model_v0", 
                               embedding_size=embedder.dim,
                               n_classes=classes_vocab.len,
                               model_name="cnn_model",
                               text_size=15, # number of tokens
                               kernel_sizes_cnn=[3, 5, 7],
                               filters_cnn=128,
                               dense_size=100,
                               optimizer="Adam",
                               learning_rate=0.1,
                               learning_rate_decay=0.01,
                               loss="categorical_crossentropy")


# In[ ]:


onehotter = OneHotter(depth=classes_vocab.len, single_vector=True)


# # Train

# In[ ]:


for ep in range(10):
    for x, y in tqdm(train_iterator.gen_batches(batch_size=64, 
                                           data_type="train")):
        x_embed = embedder(tokenizer(str_lower(x)))
        y_onehot = onehotter(classes_vocab(y))
        cls.train_on_batch(x_embed, y_onehot)


# In[ ]:


cls.save()


# # Check Result

# In[ ]:


# Infering on validation data we get probability distribution on given data.
y_valid_pred = cls(embedder(tokenizer(str_lower(x_valid))))


# In[ ]:


prob2labels = Proba2Labels(max_proba=True)


# In[ ]:


# Let's look into obtained result
print("Text sample: {}".format(x_valid[0]))
print("True label: {}".format(y_valid[0]))
print("Predicted probability distribution: {}".format(dict(zip(classes_vocab.keys(), 
                                                               y_valid_pred[0]))))
print("Predicted label: {}".format(classes_vocab(prob2labels(y_valid_pred))[0]))


# In[ ]:


# calculate sets accuracy
sets_accuracy(y_valid, classes_vocab(prob2labels(y_valid_pred)))


# **If you like it, please upvote ;)**
