#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# 

# In[ ]:


train_df=pd.read_csv('/kaggle/input/twitter-sentiment-analysis-hatred-speech/train.csv')
test_df=pd.read_csv('/kaggle/input/twitter-sentiment-analysis-hatred-speech/test.csv')


# In[ ]:


train_df.describe()


# In[ ]:


train_df.head(15)


# In[ ]:


import spacy


# In[ ]:


pd.crosstab(index=train_df['label'],columns='count')


# In[ ]:


import seaborn as sns
sns.countplot(x='label',data=train_df)


#  Creating an empty model with Spacy and using TextCategorizer within the model

# In[ ]:


nlp=spacy.blank('en')
classifier = nlp.create_pipe('textcat',config={'exclusive_classes':True,'architecture' : 'bow'})
nlp.add_pipe(classifier)
classifier.add_label('RACIST/SEXIST')
classifier.add_label('NON_RACIST/SEXIST')


# Converting tweet text data to the input form for the TextCategorizer and setting seperate labels for the two different classes

# In[ ]:


train_data=train_df['tweet'].values
train_label=[{'cats':{'RACIST/SEXIST':label==1,
                    'NON_RACIST/SEXIST':label==0}} for label in train_df['label']]
train_list=list(zip(train_data,train_label))


# In[ ]:


from spacy.util import minibatch
import random
random.seed(1)
spacy.util.fix_random_seed(1)
optimizer=nlp.begin_training()
losses={}


# Training the model in small batches for 10 epochs using spacy nlp optimizer

# In[ ]:


for epoch in range(10):
    random.shuffle(train_list)
    batches=minibatch(train_list,size=24)
    for batch in batches:
        text,labels=zip(*batch)
        nlp.update(text,labels,sgd=optimizer,losses=losses)
        print(losses,batch,epoch)


# In[ ]:


test_df.head(10)
test_df['tweet']


# In[ ]:


test=[nlp.tokenizer(text) for text in test_df['tweet']]


# Once the model is trained, it is used to predict resuts on the test dataset

# In[ ]:


classifier=nlp.get_pipe('textcat')
scores,_=classifier.predict(test)
print(scores)


# The prediction gives result in probability with the class having the higher probability has the predominant sentiment

# In[ ]:


predict_label=scores.argmax(axis=1)
count=0
for pred in predict_label:
  print(test_df['tweet'][count],'racist/sexist'if pred==0 else "non_racist/sexist")
  count+=1

