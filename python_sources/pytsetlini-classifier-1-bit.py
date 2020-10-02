#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Install PyTsetlini from github
# (remember to enable 'Internet' access in the notebook's settings)

# In[ ]:


get_ipython().system('pip install git+https://github.com/WojciechMigda/PyTsetlini@v0.0.5')


# Import necessary packages

# In[ ]:


from pytsetlini import TsetlinMachineClassifier # Tsetlin Machine Classifier we will be training


# Let's read input datasets

# In[ ]:


train_df = pd.read_csv('../input/digit-recognizer/train.csv', header=0, dtype='uint8')
print('Read train data: {}'.format(train_df.shape))

test_df = pd.read_csv('../input/digit-recognizer/test.csv', header=0, dtype='uint8')
print('Read test data: {}'.format(test_df.shape))

train_y = train_df.label.values
train_X = train_df.values[:, 1:]

test_X = test_df.values


# ##### Tsetlin Machine Classifier requires binarized input data, specifically, each value can be either 0 or 1.
# Our input images have 28x28 pixels, each pixel encoded as an 8-bit integer. We will achieve that comparing individual pixel against a threshold of 75

# In[ ]:


train_X = (train_X > 75).astype('uint8')
test_X = (test_X > 75).astype('uint8')


# Time to build the estimator. For `TsetlinMachineClassifier` I am using parameters from one of the examples used in the original Tsetlin Machine paper.

# In[ ]:


clf = TsetlinMachineClassifier(random_state=1234,
                               clause_output_tile_size=64,
                               number_of_pos_neg_clauses_per_label=2000,
                               number_of_states=127,
                               boost_true_positive_feedback=1,
                               n_jobs=2,
                               s=10.,
                               threshold=50,
                               verbose=True)
print(clf)


# Now we can train the pipeline classifier using input data. We will train it for N epochs.

# In[ ]:


NEPOCHS = 1500
clf.fit(train_X, train_y, n_iter=NEPOCHS)


# Infer predictions when complete

# In[ ]:


y_hat = clf.predict(test_X)


# And save results into CSV

# In[ ]:


pred = pd.DataFrame(y_hat)
pred.index += 1
pred.index.name = 'ImageId'
pred.columns = ['Label']
pred.to_csv('results_{}.csv'.format(NEPOCHS), header=True)

