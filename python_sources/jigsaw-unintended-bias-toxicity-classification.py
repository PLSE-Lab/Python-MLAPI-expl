#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import string
from collections import Counter, defaultdict
#import nltk
#from nltk.stem import PorterStemmer
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.preprocessing import MultiLabelBinarizer

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_df = pd.read_csv("../input/train.csv")
print(train_df.columns)


# In[ ]:


print(train_df[["comment_text","target"]].head(10))


# In[ ]:


train_df = train_df[["comment_text","target","male","female","homosexual_gay_or_lesbian","christian","jewish","muslim","black","white","psychiatric_or_mental_illness"]]


# In[ ]:


print(train_df.columns)


# # Cleaning the Data
# Start to clean the data. We will be removing all the unwanted words like 'hahahahaahahaha' and 'FFFFFFUUUUUUUUUU' as seen above. We will also be removing all the punctuations from the sentences.

# In[ ]:


#### The text is cleaned of punctuations and is converted to lower case
train_df["comment_text"] = train_df["comment_text"].str.replace('[{}]'.format(string.punctuation), '')
train_df["comment_text"] = train_df["comment_text"].str.lower()
print(train_df["comment_text"].head(10))


# In[ ]:


#### This block of code aims to find out the rarest words to remove them out from the data.
vocab = defaultdict(int)
for i in range(len(train_df["comment_text"])):
    words = train_df["comment_text"][i].split()
    for word in words:
        if word not in vocab.keys():
            vocab[word]=1
        else:
            vocab[word]+=1
            
#print(vocab)


# In[ ]:


#### Code to drop the comments with rare words
print(len(train_df))
drop_index = []
for i in range(len(train_df)):
    words = train_df["comment_text"][i].split()
    for word in words:
        if vocab[word]<5:  # Here, 5 is a hyperparameter
            drop_index.append(i)
            break
train_df.drop(train_df.index[drop_index], inplace=True)
print(len(train_df))


# # Data is Cleaned
# After cleaning the data, now we are having around approx 1350000 rows out of 1804874 rows. These remaining row numbers may change by changing the hyperparameter '5'. We may try different parameters and see the change in the accuracy. Now we will stem the words in the data for better processing. We may also try the model without stemming.

# In[ ]:


train_x = train_df["comment_text"]

def make_y(y):
    if y>=0.5:
        return '1'
    else:
        return '0'

train_y = train_df["target"].apply(make_y)


# In[ ]:


encoder = MultiLabelBinarizer()
encoder.fit_transform(train_y)
train_encoded = encoder.transform(train_y)
num_classes = len(encoder.classes_)
print(encoder.classes_)


# In[ ]:


print(len(train_x), len(train_y))


# In[ ]:


comment_embeddings = hub.text_embedding_column("toxicity_comments", module_spec="https://tfhub.dev/google/universal-sentence-encoder/2")


# In[ ]:


multi_label_head = tf.contrib.estimator.multi_label_head(num_classes, loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)


# In[ ]:


estimator = tf.estimator.DNNEstimator(head=multi_label_head, hidden_units=[64,10], feature_columns=[comment_embeddings])


# In[ ]:


features = {
  "toxicity_comments": np.array(train_x).astype(np.str)
}
labels = np.array(train_encoded).astype(np.int32)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    features, 
    labels, 
    shuffle=True, 
    batch_size=32, 
    num_epochs=20
)


# In[ ]:


estimator.train(input_fn=train_input_fn)


# In[1]:


predict_df = pd.read_csv("../input/test.csv")
predict_x = predict_df["comment_text"]
predict_x = predict_x.str.replace('[{}]'.format(string.punctuation), '')
predict_x = predict_x.str.lower()

predict_submit = predict_df["id"]

predict_input_fn = tf.estimator.inputs.numpy_input_fn({"toxicity_comments": np.array(predict_x).astype(np.str)}, shuffle=False)
results = estimator.predict(predict_input_fn)

classes = {'0':0, '1':1}
for result, index in zip(results, range(len(predict_df))):
    answer = result["probabilities"].argsort()[-1]
    final = classes[encoder.classes_[answer]]
    predict_submit["target"][index] = final
    


# In[ ]:


predict_submit.to_csv("../output/submit.csv")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




