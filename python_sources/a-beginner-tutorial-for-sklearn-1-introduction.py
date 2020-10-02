#!/usr/bin/env python
# coding: utf-8

#  # **1. Introdcution**
# 
# The main objective of this serie of kernels is to introduce the reader to the scikit-learn library. The scikit-learn (ofter called sklearn) is a Python library destinated to developing machine learning applications.
# 
# After carefully going through this, you will undertand every aspect of this library from preprocessing to dimensionality reduction.
# 
# I decided to use the [Medical Cost Personal Dataset](https://www.kaggle.com/mirichoi0218/insurance) since I found it combines both qualitative and quantitative independent features. The dataset might change according to the points to outline.
# 
# As a beginer, a first step is to visit [the Scikit learn website](https://scikit-learn.org/stable/index.html) in order to have first insights about what you can do using this library.
# 
# It should be mentionned that there exist other librarires that allow similar or slighlty different functions (with similar or different performance) such as the [Vowpal Wabbit](https://vowpalwabbit.org/) library.

# # **2. The scikit-learn library - a first look**
# 
# When you visit [the scikit-learn website](https://scikit-learn.org/stable/), you can have access to a comprehensive overview of all the things you can do with this library, you can also find various examples. I advice to visit the user guide page that you can access from the top on the home page or you can directly click [here](https://scikit-learn.org/stable/user_guide.html).

# ![image.png](attachment:image.png)

# The classification shown on the top is the same as the way the functions are classified inside the library itself. To give an exmaple, in the user guide the Logisitc regression belongs to the Linear models. If you would like to perform a Logisitc regression here is what you need to import. You can see that LogisticRegression is part of the linear_models (the same as the classification of the user guide).

# In[ ]:


from sklearn.linear_model import LogisticRegression


# The code you see below is the one that is automatically generated once you launch a kaggle kernel from an available dataset. These lines allow the preparation of the notebook and the attachement of data.
# 
# **Note:** It is evident that the understanding of the scikit-learn library could not be achieved without a good grasp of a other libraries such as numpy, pandas or seaborn (that are supposed to be an asset).
# 
# **Note 2:** If you code using Jupyter Notebook, you have to make use of a number of tools that make the work easy for you. Here are some of them:
# 
# 1- Use an Auto completer, you can add it by executing this command: %config IPCompleter.greedy=True
# 
# 2- Use a variable inspector, this is important since it shows the type of the variable and its dimension which could help you a lot for undertanding some error messages.
# 
# 3- Use a formatting tool that makes your code easily readible and facilitates the coordination with your colleagues, you can stick to the PEP8 norms for instance. An automatic tool could be embedded to your Jupyter notebook. See [here](https://github.com/hhatto/autopep8).

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


# Now that the environment is set, Let us move to the first topic: Preprocessing data.
