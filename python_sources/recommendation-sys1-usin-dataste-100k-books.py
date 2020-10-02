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


# In[ ]:


import pandas as pd
import tensorflow as tf
print(tf.test.gpu_device_name())
# See https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True


# In[ ]:


import csv
rating = csv.DictReader(open("/kaggle/input/ratings.csv"), delimiter=",")

rating1 = pd.read_csv('/kaggle/input/ratings.csv')
# book = pd.read_csv('/kaggle/input/books.csv')
book = csv.DictReader(open("/kaggle/input/books.csv"), delimiter=",")


# In[ ]:


import json
from itertools import islice
for line in islice(rating, 3 ):
    print(json.dumps(line, indent=4))


# In[ ]:


for line in islice(book, 1):
    print(json.dumps(line, indent=4))


# In[ ]:


# from lightfm.data import Dataset
# dataset = Dataset()
# dataset.fit((rating['user_id'].values),
#             (rating['book_id'].values))


# In[ ]:


from lightfm.data import Dataset

dataset = Dataset()

with tf.device('/gpu:0'):
    dataset.fit((rating1['user_id']),
            (rating1['book_id']),
          )


# In[ ]:



num_users, num_items = dataset.interactions_shape()
print('Num users: {}, num_items {}.'.format(num_users, num_items))


# In[ ]:



dataset.fit_partial(items=(x['book_id'] for x in book),
                    item_features=(x['isbn'] for x in book))


# In[ ]:


with tf.device('/gpu:0'):
    (interactions, weights) = dataset.build_interactions([tuple(i) for i in rating1.drop(['rating'], axis = 1).values])


# In[ ]:


weights


# In[ ]:


item_features = dataset.build_item_features(((x['book_id'], [x['isbn']])
                                              for x in book))
print(repr(item_features))


# In[ ]:


from lightfm import LightFM

with tf.device('/gpu:0'):
    model = LightFM(loss='bpr')
    model.fit(interactions,item_features = item_features)


# In[ ]:


from lightfm.evaluation import precision_at_k


# In[ ]:


with tf.device('/gpu:0'):
    print("Train precision: %.2f" % precision_at_k(model,interactions, k=5).mean())


# In[ ]:


import pandas as pd
import numpy as np
b = pd.read_csv('/kaggle/input/books.csv')


# In[ ]:


def sample_recommendation(model, data, user_ids):


    n_users, n_items = interactions.shape

    for user_id in user_ids:
        known_positives = b['original_title'][interactions.tocsr()[user_id].indices]

        scores = model.predict(user_id, np.arange(n_items))
        top_items = b['original_title'][np.argsort(-scores)]

        print("User %s" % user_id)
        print("     Known positives:")

        for x in known_positives[:3]:
            print("        %s" % x)

        print("     Recommended:")

        for x in top_items[:3]:
            print("        %s" % x)


# In[ ]:


sample_recommendation(model, b, [45, 4652, 6532])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




