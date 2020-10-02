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


# ### Description
# 
# This notebooks runs the public [utility script](https://www.kaggle.com/xhlulu/tf-qa-jsonl-to-dataframe) to convert original dataset to DataFrame and saves it as an output. The output was used to create a new [simplified verision of the dataset](https://www.kaggle.com/feanorpk/tf-20-qa-simplified-dataframe).

# In[ ]:


# Load data and convert to DataFrame

from tf_qa_jsonl_to_dataframe import jsonl_to_df

tf_qa_input_folder = '/kaggle/input/tensorflow2-question-answering/'
train = jsonl_to_df(tf_qa_input_folder + 'simplified-nq-train.jsonl', truncate=True)
test = jsonl_to_df(tf_qa_input_folder + 'simplified-nq-test.jsonl', truncate=True, load_annotations=False)


# ### Save as output
# This output was also saved as a [dataset](https://www.kaggle.com/feanorpk/tf-20-qa-simplified-dataframe)

# In[ ]:


train.to_csv('train.csv')
test.to_csv('test.csv')

