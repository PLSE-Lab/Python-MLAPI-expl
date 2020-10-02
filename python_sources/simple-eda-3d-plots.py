#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))


# In[ ]:


get_ipython().run_cell_magic('time', '', "train_transaction = pd.read_csv('../input/train_transaction.csv')\ntest_transaction = pd.read_csv('../input/test_transaction.csv')\n\ntrain_identity = pd.read_csv('../input/train_identity.csv')\ntest_identity = pd.read_csv('../input/test_identity.csv')\n\nsample_submission = pd.read_csv('../input/sample_submission.csv')")


# In[ ]:


train_transaction.head()


# In[ ]:


test_transaction.head()


# In[ ]:


train_identity.head()


# In[ ]:


test_identity.head()


# In[ ]:


train_transaction.shape , test_transaction.shape


# In[ ]:


train_identity.shape , test_identity.shape


# In[ ]:


l1 = train_transaction.columns
l2= train_identity.columns
list(set(l1) & set(l2)) 


# In[ ]:


train = train_transaction.merge(train_identity , how = 'left' , on = 'TransactionID')
test = test_transaction.merge(test_identity , how = 'left' , on = 'TransactionID')
print(train.shape)
print(test.shape)


# In[ ]:


import gc

del train_transaction, train_identity, test_transaction, test_identity
gc.collect()


# Let's see missing values in the train data variable wise 

# In[ ]:


train.isnull().sum()


# We have lots of variables with missing values, now let's see the data types of each column

# In[ ]:


train.dtypes


# What are the categorical columns do we have in this data?, let's see

# In[ ]:


cat_cols = [c for c in train.columns if train[c].dtype == object ]
cat_cols


# Let's see the number of different entries for those categorical columns 

# In[ ]:


for c in cat_cols:
    print('number of unique entries for column' , c , '=' , train[c].nunique())


# Let's quickly check the distribution of the target variable here

# In[ ]:


train.isFraud.value_counts()


# In[ ]:


train.isFraud.value_counts().plot('bar')
print('target ratio is', round(20663/len(train)*100,2) , 'percent')


# It seems that we have highly imbalanced binary target distribution.
# Now, from [Bojan's public kernel](https://www.kaggle.com/tunguz/adversarial-ieee) we know that TransactionDT is the variable which has different distribution in both train and test data, let's re-confirm that quickly. 

# In[ ]:


pd.options.display.float_format = '{:.2f}'.format
train.TransactionDT.describe()


# In[ ]:


test.TransactionDT.describe()


# In[ ]:


train.TransactionDT.max() < test.TransactionDT.min()


# So, we see that the data was splot by this variable and we can not use this variable readily, probably creating other variables from this variable like - time or day or weekend or not kind of variable can be useful from this one (If possible). Also, this variable can help us in creating different effective validation strategy.
# 
# Anyway, let's move on and see the distribution of transaction amount.

# In[ ]:


import matplotlib.pyplot as plt

plt.hist(train['TransactionAmt'] , bins = 100)
plt.title('transaction amount for train set')
plt.show()

plt.hist(test['TransactionAmt'] , bins = 100)
plt.title('transaction amount for test set')
plt.show()


# In[ ]:


train.TransactionAmt.describe()


# In[ ]:


test.TransactionAmt.describe()


# We can see that the transaction amount in the train data is ranging till 32K where as the same ranges till 10K in test, let's see their distribution in the log scale.

# In[ ]:


plt.hist(np.log(train['TransactionAmt']) , bins = 100)
plt.title('Log scale transaction amount for train set')
plt.show()

plt.hist(np.log(test['TransactionAmt']), bins = 100)
plt.title('Log scale transaction amount for test set')
plt.show()


# What about the numeric columns? Let's look at them quickly.

# In[ ]:


## Let's subset the numerical columns in train data ##

num_cols = [c for c in train.columns if train[c].dtype != object ]
train_num = train[num_cols]
#print(train_num.shape)
train_num.head()
missing_cols = [c for c in train_num.columns if train_num[c].isnull().sum()/len(train_num) >0.80 ]
len(missing_cols)


# Wow, as we can see that, there are 69 columns in the train data which have more than 80% missing entries.
# But, are all of them really numeric?

# In[ ]:


for c in train_num.columns:
    print('number of unique entries for column' , c , '=' , train_num[c].nunique())


# There are lot's of columns which have very less number of unique values altogether, probably treating them as categorical can help.
# Let's subset our train data with columns which have more unique numbers, so probably the numerical columns. Also, let's impute the missing values by their column means

# In[ ]:


num_cols = [c for c in train_num.columns if train_num[c].nunique()>5000 ]
len(num_cols) ## 40 columns
#num_cols
train_num = train_num[num_cols]
train_num = train_num.fillna(train_num.mean())
train_num['target'] = train['isFraud']


# Let's for the sake of simplicity and time we randomly select the 10% of the data and run a PCA on that, after that, we will take 3 PCA components to plot.

# In[ ]:


import random

data1 = train_num.sample(frac= 0.1 , random_state=10)
data1.head()


# In[ ]:


data1.target.value_counts()


# Let's check if we preserve the target ratio in our sample data or not

# In[ ]:


print('target ratio in the sample data is' , round(2060/len(data1)*100,2) , 'which seems okay')


# In[ ]:


target = data1['target']
del data1['target'], data1['TransactionDT'], data1['TransactionID']


# In[ ]:


## Let's try PCA on this dataset ##

from sklearn.preprocessing import StandardScaler
data_pca = StandardScaler().fit_transform(data1)

#data_pca = pd.DataFrame(data_pca)
#data_pca.head()
#data_pca.describe()

from sklearn.decomposition import PCA

pca = PCA(n_components=3)
comps = pca.fit_transform(data_pca)

final_pca_data = pd.DataFrame(data = comps , columns=['pc1' , 'pc2' , 'pc3'])

final_pca_data.head()


# Now a 3D plot using these PCA data. I learned this 3D plotting from [this kernel](https://www.kaggle.com/chechir/molecular-eda-3d-plots)

# In[ ]:


import plotly
import plotly.graph_objs as go
from plotly.graph_objs import FigureWidget

traces = go.Scatter3d(
    x=final_pca_data['pc1'],
    y=final_pca_data['pc2'],
    z=final_pca_data['pc3'],
    mode='markers',
    marker=dict(
        size=4,
        opacity=0.2,
        color=target,
        colorscale='Viridis',
     )
)

layout = go.Layout(
    autosize=True,
    showlegend=True,
    width=800,
    height=1000,
)

FigureWidget(data=[traces], layout=layout)


# From the above plot it seems that the yellow points (which are target = 1) are quite mixed up with the purple (target = 0) ones, which may indicate not to do oversampling of target = 1s blindly. So, we may need to be careful incase we try oversampling. A better perspective is possible if we plot t-sne components, which I have commented below. However, please note that the results are based on a very small subset so results colud be very much inconclusive. 

# In[ ]:


get_ipython().run_cell_magic('time', '', "\n## same with T-SNE ##\n\nfrom sklearn.manifold import TSNE\n\ntsne = TSNE(n_components=3 , random_state=0)\ndata_tsne = tsne.fit_transform(data1)\n\ndata_tsne\n\ndata_tsne = pd.DataFrame(data_tsne , columns=['tsne1' , 'tsne2' , 'tsne3'])\ndata_tsne.head()\n\n## 3D plot with TSNE components ##\n\ntraces = go.Scatter3d(\n    x=data_tsne['tsne1'],\n    y=data_tsne['tsne2'],\n    z=data_tsne['tsne3'],\n    mode='markers',\n    marker=dict(\n        size=4,\n        opacity=0.2,\n        color=target,\n        colorscale='Viridis',\n     )\n)\n\nlayout = go.Layout(\n    autosize=True,\n    showlegend=True,\n    width=800,\n    height=1000,\n)\n\nFigureWidget(data=[traces], layout=layout)")

