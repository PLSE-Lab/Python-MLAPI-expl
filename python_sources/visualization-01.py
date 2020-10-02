#!/usr/bin/env python
# coding: utf-8

# In[5]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

import pyximport; pyximport.install()
import gc
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xlrd import open_workbook
import seaborn as sns #for making plots
#import matplotlib.pyplot as plt # for plotting
import os                # for os commands
import nltk
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import ensemble, metrics, model_selection, naive_bayes
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.stats import norm
from scipy import stats
from time import time
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)
import plotly.tools as tls

train =pd.read_csv('../input/train.tsv', sep='\t')
train.dtypes
def data_description(df):
    """
    Returns a dataframe with some informations about the variables of the input dataframe.
    """
    data = pd.DataFrame(index=df.columns)
    
    # the numeric data types
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
def countplot(x, data, figsize=(10,5)):
    """
    Wraps the countplot function of seaborn and allow to specify the size of the figure.
    """ 
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.countplot(x=x, data=data, ax=ax, order=data[x].value_counts().index)
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
def split_cat(text):
    try: return text.split("/")
    except: return ("No Label", "No Label", "No Label")
    
train['general_category'], train['subcat_1'], train['subcat_2'] = zip(*train['category_name'].apply(lambda x: split_cat(x)))
train.head()
#checking train_id w.r.t general _category
import matplotlib
f,ax = plt.subplots(1,1,figsize=(10,10))
hist = train.groupby(['general_category'],as_index=False).count().sort_values(by='train_id',ascending=False)[0:13]
sns.barplot(y=hist['general_category'],x=hist['train_id'],orient='h')
matplotlib.rcParams.update({'font.size': 14})
plt.show()
import matplotlib
f,ax = plt.subplots(1,1,figsize=(12,12))
hist = train.groupby(['brand_name'],as_index=False).count().sort_values(by='train_id',ascending=False)[0:25]
sns.barplot(y=hist['brand_name'],x=hist['train_id'],orient='h')
matplotlib.rcParams.update({'font.size': 12})
plt.show()
#start = time()
sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(1, 1, figsize=(11, 7), sharex=True)
sns.despine(left=True)
sns.distplot(np.log(train['price'].values+1), axlabel = 'Log(price)', label = 'log(trip_duration)', bins = 50, color="y")
plt.setp(axes, yticks=[])
plt.tight_layout()
#end = time.time()
#print("Time taken by above cell is {}.".format((end-start)))
plt.show()
countplot('item_condition_id', train, figsize=(8,4))
plt.show()
plt.figure(figsize=(5,5))
sns.countplot(x='shipping', data=train)
plt.show()
train['log(price+1)'] = np.log(train['price']+1)
fig, (ax1, ax2) = plt.subplots(1,2,figsize = (18,6))
current_palette = sns.color_palette()
start1 = time()
train.boxplot(by = 'general_category', column = 'log(price+1)', ax = ax1, rot = 45)
ax1.set_ylabel('log(price+1)')
ax1.set_xlabel('Category')
ax1.set_title('log(price+1) of items by Category1')
end1 = time()
start2 = time()
sns.boxplot(y = 'general_category', x = 'log(price+1)',ax = ax2, data = train)
ax2.set_xlabel('log(price+1)')
ax2.set_ylabel('Category')
ax2.set_title('log(price+1) of items by Category1')
end2 = time()

# Any results you write to the current directory are saved as output.

