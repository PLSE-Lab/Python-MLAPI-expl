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


# We need nvidia apex for 16 bit precision

# In[ ]:


get_ipython().system(' git clone https://github.com/NVIDIA/apex')
get_ipython().system(' cd apex')
get_ipython().system(' pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" /kaggle/working/apex/')


# # Text Classification with BERT for Turkish Language

# ## Let's Download simpletransformers first
# Simpletransformers is a library what has some wrappers around huggingface transformers.
# You can easily fine-tune and do some NLP stuff with it like NER or in this case Classification

# In[ ]:


get_ipython().system('pip install simpletransformers')


# In[ ]:


from simpletransformers.classification import ClassificationModel


# We will use bert base multilingual uncased what has around 100+ language support. So Turkish is also supported

# In[ ]:


# Lets import the csv file in pandas dataframe first
train_df = pd.read_csv('/kaggle/input/ttc4900/7all.csv', encoding='utf-8', header=None, names=['cat', 'text'])


# In[ ]:


# Check the df
train_df.head()


# In[ ]:


# unique categories
print(train_df.cat.unique())
print("Total categories",len(train_df.cat.unique()))


# In[ ]:


# convert string labels to integers
train_df['labels'] = pd.factorize(train_df.cat)[0]

train_df.head()


# In[ ]:


# Let's create a train and test set
from sklearn.model_selection import train_test_split

train, test = train_test_split(train_df, test_size=0.2, random_state=42)


# In[ ]:


train.shape, test.shape


# In[ ]:


# Lets define the model with the parameters (important here is the number of labels and nr of epochs)

model = ClassificationModel('bert', 'bert-base-multilingual-uncased', num_labels=7, 
                            args={'reprocess_input_data': True, 'overwrite_output_dir': True, 'num_train_epochs': 3})


# In[ ]:


# Now lets fine tune bert with the train set
model.train_model(train)


# In[ ]:


# Let's evaluate this finetuned model with the test set
result, model_outputs, wrong_predictions = model.eval_model(test)


# In[ ]:


predictions = model_outputs.argmax(axis=1)


# In[ ]:


predictions[0:10]


# In[ ]:


actuals = test.labels.values
actuals[0:10]


# In[ ]:


# Now lets see the accuracy one the test set
from sklearn.metrics import accuracy_score
accuracy_score(actuals, predictions)


# An accuracy of 90.3%. Not bad!!! 

# In[ ]:


sample_text = test.iloc[10]['text']
print(sample_text)


# In[ ]:


# Lets predict the text of sample_text:
model.predict([sample_text])


# In[ ]:


# Lets see what the truth was
test.iloc[10]['labels']


# In[ ]:


# And this was category: 
test.iloc[10]['cat']


# In[ ]:




