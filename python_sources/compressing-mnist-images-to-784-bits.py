#!/usr/bin/env python
# coding: utf-8

# # Compressing MNIST Images

# In this kernel we will compress MNIST images as far as we can without losing information. We will calculate the byte count for each manipulation and compare them with each other.
# 
# This kernel is the 2nd in a series I am doing on MNIST. The first is a kernel on the power of dimensionality reduction combined with simple algorithms such as decision trees. [You can check out the 1st kernel in this series here.](https://www.kaggle.com/carlolepelaars/97-on-mnist-with-a-single-decision-tree-t-sne)
# 
# MNIST series:
# 
# 1. [97% on MNIST with a single decision tree (+ t-SNE)](97% on MNIST with a single decision tree (+ t-SNE))
# 2. [Compressing MNIST Images (to 784 bits)](https://www.kaggle.com/carlolepelaars/compressing-mnist-images-to-784-bits)
# 3. An end-to-end neural network using only NumPy (Work in progress)

# ## Table of contents

# - [Dependencies](#1)
# - [Preparation](#2)
# - [Baseline](#3)
# - [Uint8](#4)
# - [TSVD](#5)
# - [Bitstring and ASCII Encoding](#6)
# - [Conclusion](#7)

# ## Dependencies <a id="1"></a>

# In[ ]:


# Standard libraries
import os
import sys
import pickle
import numpy as np
import random as rn
import pandas as pd
from tqdm import tqdm # Progress Bar

# Dimensionality reduction
from sklearn.decomposition import TruncatedSVD

# Specify Paths
BASE_PATH = '../input/digit-recognizer/'
TRAIN_PATH = BASE_PATH + 'train.csv'
TEST_PATH = BASE_PATH + 'test.csv'

# Seed for reproducability
seed = 1234
np.random.seed(seed)
rn.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)


# In[ ]:


# File sizes and specifications
print('\n# Files and file sizes')
for file in os.listdir(BASE_PATH):
    print('{}| {} MB'.format(file.ljust(30), 
                             str(round(os.path.getsize(BASE_PATH + file) / 1000000, 2))))


# ## Preparation <a id="2"></a>

# In[ ]:


# Load in training and testing data
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(BASE_PATH + 'sample_submission.csv')

features = [col for col in train_df.columns if col.startswith('pixel')]


# ## Baseline <a id="3"></a>

# We will try to reduce the byte count as much as we can before training a model on MNIST. Below are the baselines from where we will be working. Pandas stores every integer automatically as int64 so the initial byte count is quite large.

# In[ ]:


# The byte count for our baseline (Standard Pandas Dataframe with Int64 types)
base_byte_count_train = sys.getsizeof(train_df)
base_byte_count_test = sys.getsizeof(test_df[features])
print(f'The standard byte count for train_df is: {base_byte_count_train}')
print(f'The standard byte count for test_df is: {base_byte_count_test}')


# In[ ]:


print("Distribution description for select pixel values:")
train_df[['pixel100', 'pixel200', 'pixel300', 'pixel500', 'pixel600', 'pixel700']].describe()


# ## Uint8 <a id="4"></a>

# Since all values in this DataFrame are between 0 and 255 we can store the data as unsigned integers of 8 bytes (uint8). This will save 56 bytes on every value.

# In[ ]:


print("Training Dataframe:")
train_df.head(3)


# In[ ]:


# Convert all values to unsigned 8-bit integers
train_df = train_df.astype(np.uint8)
test_df = test_df.astype(np.uint8)
labels = list(train_df['label'])
label_byte_count = sys.getsizeof(labels)


# In[ ]:


uint8_byte_count_train = sys.getsizeof(train_df)
uint8_byte_count_test = sys.getsizeof(test_df[features])

print(f'Once we convert all integers to uint8 we are left with a byte count of {uint8_byte_count_train} for train_df and\n{uint8_byte_count_test} bytes for test_df.\n\nThis is a reduction of (compared to a basic Pandas DataFrame):\n{round(((uint8_byte_count_train - base_byte_count_train)/base_byte_count_train)*100,2)}% for the train_df and\n{round(((uint8_byte_count_test - base_byte_count_test)/base_byte_count_test)*100,2)}% for the test_df.')


# In[ ]:


print(f'The cost of storing the index in a Pandas DataFrame is: {test_df.index.memory_usage()} bytes')


# ## TSVD <a id="5"></a>

# TSVD is a technique that can be used to compress sparse data, like MNIST Images. We can decide ourselves how much we want to reduce the dimensionality. In this example we reduce the 28x28 images to 15x15 (15 components)
# 
# For more information about dimensionality reduction on MNIST I suggest you check out [this Kaggle kernel](https://www.kaggle.com/carlolepelaars/97-on-mnist-with-a-single-decision-tree-t-sne).

# In[ ]:


concat_uint8 = pd.concat([train_df, test_df])
n_components = 15
# Perform Truncated Singular Value Decomposition (TSVD) on all features
tsvd = TruncatedSVD(n_components=n_components).fit_transform(concat_uint8[features])
# Split up the t-SNE results in training and testing data
components = [f"Component_{_}" for _ in range(n_components)]
tsvd_train = pd.DataFrame(tsvd[:len(train_df)], columns=components)
tsvd_test = pd.DataFrame(tsvd[len(train_df):], columns=components)
tsvd_train['label'] = labels


# In[ ]:


print('TSVD features: ')
tsvd_train.head(3)


# In[ ]:


print(f'Once we perform TSVD we are left with a byte count of\n{sys.getsizeof(tsvd_train)} for train_df and\n{sys.getsizeof(tsvd_test)} bytes for test_df.\n\nThis is a reduction of (compared to Pandas DataFrame with uint8):\n{round(((sys.getsizeof(tsvd_train) - uint8_byte_count_train)/uint8_byte_count_train)*100,2)}% for the train_df and\n{round(((sys.getsizeof(tsvd_test) - uint8_byte_count_test)/uint8_byte_count_test)*100,2)}% for the test_df.')


# ## Bitstring and ASCII encoding <a id="6"></a>

# Since different grayscale values don't play a very large role in MNIST we can round all numbers smaller than 127 to 0 and clip all values larger or equal to 127 to 1. In this way we can reduce every image to an array of 784 bytes (0s and 1s). We will also need 42128 bytes more for the training set because we have to store the labels as uint8. From here on out we all lose the Pandas DataFrame format. We save another 128 bytes because we don't have to store the index anymore.
# 
# To reduce the byte count we store every image as a bitstring and then encode it to ASCII.

# In[ ]:


# Convert all values to 0 and 1
for df in [train_df, test_df]:
    for col in features:
        df[col] = np.where(df[col]<127, 0, 1)


# In[ ]:


train_array = []
test_array = []
# Encode every image to a ascii bitstring
for i in tqdm(range(len(train_df))):
    train_array.append(train_df.iloc[i][features].astype(str).sum().encode('ascii'))
for i in tqdm(range(len(test_df))):
    test_array.append(test_df.iloc[i][features].astype(str).sum().encode('ascii'))


# In[ ]:


bitcount_train = sys.getsizeof(train_array)
bitcount_test = sys.getsizeof(test_array)
print(f'Once we convert all integers to bits we are left with a byte count of \n{bitcount_train} for train_df and\n{bitcount_test} bytes for test_df.\n\nThis is a further reduction (compared to Pandas DataFrame with uint8) of:\n{round((((bitcount_train+label_byte_count) - uint8_byte_count_train)/uint8_byte_count_train)*100,2)}% for the train_df and\n{round(((bitcount_test - uint8_byte_count_test)/uint8_byte_count_test)*100,2)}% for the test_df.')


# Wow! Now we're talking. Note that we can still visualize the digit even though it is stored in an unintuitive way. Here is how we can easily print the representation as a 28x28 representation.

# In[ ]:


def display_digit_bytes(text, lineLength):
    """
    Visualize a 784-bit representation of an MNIST Image
    
    text: The bitstring that you want to represent
    lineLength: The length at which to jump to a new line 
    (28 in the case of MNIST images)
    """
    if len(text) <= lineLength:
        return text
    else:
        # Print line and next line recursively
        print(text[:lineLength])
        return display_digit_bytes(text[lineLength:], lineLength)


# In[ ]:


print(f'Label = {labels[3]}\n')
display_digit_bytes(train_array[3], 28)


# We store we smallest data format we have as a [pickle file](https://docs.python.org/3/library/pickle.html).

# In[ ]:


# Add labels to training data
train_array.append(labels)
# Save data as pickle files
with open('training_data.pkl','wb') as file:
    pickle.dump(train_array, file)
with open('test_data.pkl','wb') as file:
    pickle.dump(test_array, file)


# In[ ]:


print(f"File size for pickled training data: {os.path.getsize('training_data.pkl')}")
print(f"File size for pickled test data: {os.path.getsize('training_data.pkl')}")


# ## Conclusion <a id="7"></a>

# TL;DR: Use NumPy arrays with uint8. Consider dimensionality reduction to compress data.

# As you can see there are many ways to compress grayscale images without losing much information. However, a bitstring is not really suitable for training machine learning models. NumPy arrays are very practical and therefore I suggest compressing the images so they can be used with the uint8 type (values between 0 and 255). It is also beneficial to consider dimensionality reduction techniques if you need to compress the data even further. [Dimensionality reduction techniques can even increase performance of simple models as illustrated in this Kaggle kernel](https://www.kaggle.com/carlolepelaars/97-on-mnist-with-a-single-decision-tree-t-sne).

# And we are done! Please let me know if you know more tricks to compress (MNIST) images or to store image data more efficiently.
# 
# If you like this Kaggle kernel, feel free to give an upvote and leave a comment! I will try to implement your suggestions in this kernel!
