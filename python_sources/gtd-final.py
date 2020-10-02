#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# SVC try 
# OneHotEncoding
# GridSearch and class_weights
# grid search for the best kernel
# limiting data to 10,000
# run with the results from grid search
# using best results {'C': 1.0, 'degree': 3.0, 'gamma': 4.0}
# Reducing the sample size to 50,000 entries for Grid Search
# accuracy round 4
# final


# In[ ]:


get_ipython().run_line_magic('env', 'JOBLIB_TEMP_FOLDER=/tmp')
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb

from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score

from fastai.structured import *
from fastai.column_data import *
from sklearn.svm import SVC
from sklearn import preprocessing

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# Importing the data
data = pd.read_csv('../input/globalterrorismdb_0718dist.csv',engine = 'python')


# In[ ]:


# There appears to be a lot of string and binary data
data.head(2)


# In[ ]:


# So there are close to 2 Lakh entries and 135 features
data.shape


# # List of Steps
# 1. Pre Processing
# 2. Visualizing 
# 3. Removing outliers
# 4. Validation split
# 5. Training the model
# 6. Making the final prediction

# # Pre Processing 

# In[ ]:


# There are a lot of null values which will be eliminated.
print(data.isnull().sum().to_frame().sort_values(0,ascending=False))


# In[ ]:


# Get the count of attacks from that event
data['count'] = data['eventid'].astype(str).str.slice(-2,).map(lambda x : int(x))

# Standardizing the year
data.iyear = data.iyear.apply(lambda x: 2018-x)


# In[ ]:


# Categorical variable list : Nomial - 0 or 1 answer
cat_no = ['extended','crit1','crit2','crit3','doubtterr','multiple','country','region','specificity','attacktype1','success',
          'suicide','weaptype1','weapsubtype1','targtype1','targsubtype1','natlty1','guncertain1',
         'claimed','property','ishostkid','ransom','INT_LOG','INT_IDEO']

#Continous variable list
#nperpcap is removed due to sparsity 
cont = ['count', 'iyear','imonth','iday','latitude','longitude','nperps','nkill','nkillus']

target = ['gname']

ids = ['eventid']

# Converting it into int
# -9 is replacing NA values as the same was followed when fillig the data
for v in cat_no:
    data[v] = data[v].fillna(-9).astype('int32')

# Replacing the negative categories
data[cat_no] = data[cat_no].replace(-9,0).astype('int32')

# Coverting it into continous variables
for v in cont:
    data[v] = data[v].fillna(0).astype('float32')

# Replacing missing values with zero, the unknown values will be replaced with 0
data[['nperps','nkill','nkillus']] = data[['nperps','nkill','nkillus']].replace(to_replace=[-99,-9],value=0).astype('float32')

# Consolidating it
data = data[ids+cat_no+cont+target]


# In[ ]:


# Test set split
test = data[data['gname']=='Unknown']
test['gname'] = -999

# Training set split
train = data[data['gname']!='Unknown']

# Label encoding gname
le = LabelEncoder()
le.fit(train['gname'])
train['gname'] = le.transform(train['gname'])


# In[ ]:


train = pd.concat([train,test],axis=0)
train = train.reset_index(drop=True)


# # Visualizing 

# In[ ]:


# Let's look at the distribution 
# The class distribution is not even and most of the data is missing
sns.distplot(train.gname)


# In[ ]:


# Lets look at the correlation plot
# We can see that the variables iyear and count have multicollinearity and so does nkill and nkillus.
# The correlation index is not high so we can keep them.
corrmat = train.corr()
sns.heatmap(corrmat,vmin=0)


# ## Removing Outliers
# 

# In[ ]:


# 10000 nperps is a large value by any stretch of imagination, so we will be dropping it
# removing entries with nkill > 500 Y& nkillus > 500
train[['nkill','nkillus','nperps']].describe()


# In[ ]:


train = train[(train.nperps<10000)]
test = test[(test.nperps<10000)]

train = train[(train.nkill<500)]
test = test[(test.nkill<500)]

train = train[(train.nkillus<500)]
test = test[(test.nkillus<500)]


# In[ ]:


# Creating a df to get the size
size_train = train.groupby('gname').size().to_frame().sort_values(0,ascending=False)
size_train.describe()


# In[ ]:


# Keeping only the values that have incidents below 50
train = train.drop(list(size_train[size_train[0]<50].index),axis=0)


# In[ ]:


# The data is highly imbalanced, if the compute limit for kaggle was not limited to 6 hrs then Oversampling would have
# been performed for the minority class
train[train.gname != -999].gname.hist(bins=100)


# # Validation split

# In[ ]:


# Using the proc_df to scales the continous variables, they will be combined again
train, train_y, nas, mapper = proc_df(train, 'gname', do_scale=True,ignore_flds=['eventid','extended','crit1','crit2',
                                                                                 'crit3','doubtterr','multiple','country','region',
                                                                                 'specificity','attacktype1','success','suicide',
                                                                                 'weaptype1','weapsubtype1','targtype1','targsubtype1',
                                                                                 'natlty1','guncertain1','claimed','property','ishostkid',
                                                                                 'ransom','INT_LOG','INT_IDEO'])

train = train.reset_index(drop=True)
train = pd.concat([train,pd.DataFrame(train_y)],axis=1,join_axes=[train.index])
train=train.rename(index=str,columns={0:'gname'})
train = train.reset_index(drop=True)


# In[ ]:


# Creating a function to oneHotEncode all the categorical variables since there are not ordinal
def ohe(train,features):
    """
    The functions takes the df with the list of arguments to be one hot encoded and returns a df
    train is the df and features is a list
    """
    for v in features:
        df = train[v].values
        train = train.drop([v],axis=1)
        oh = OneHotEncoder(sparse=False)
        df = df.reshape(len(df),1)
        df = oh.fit_transform(df)
        df = df[:,1:]
        train = pd.concat([train,pd.DataFrame(df)],axis=1,sort=False)
    return train


# In[ ]:


train = ohe(train,['country','region','specificity','attacktype1','weaptype1','weapsubtype1','targtype1','targsubtype1','natlty1'])


# In[ ]:


train.shape


# In[ ]:


# Looking at the final DataFrame
train.head()


# In[ ]:


# Test set split
test = train[train.gname == -999]

# Training set split
train = train[train.gname!= -999]


# In[ ]:


test.shape


# In[ ]:


# Limiting the data so as to keep the compute time low
# Kaggle kernel has a limit of 6 hrs.
train = train.iloc[:50000,:]


# In[ ]:


# To check if any null values have crept in
train.isnull().sum().sum()


# In[ ]:


# Dropping the event ID
train.drop(['eventid'],axis='columns',inplace=True)

# Creating the split for validation
trainData, validData = train_test_split(train,test_size=0.3)

train_X, train_y, nas = proc_df(trainData, 'gname', do_scale=False)
valid_X, valid_y, nas = proc_df(validData, 'gname', do_scale=False)


# In[ ]:


train_X.shape,train_y.shape,valid_X.shape,valid_y.shape


# # Training the model

# In[ ]:


# The SVC model had the best overall accruaracy score among the models tested
# Using the best results from GridSearch
est = SVC(C=1, gamma=4, kernel='poly', degree=3, max_iter=10000, class_weight='balanced')


# In[ ]:


# Fitting the training data
est.fit(train_X,train_y)


# In[ ]:


# Making the prediction
y_pred = est.predict(valid_X)


# In[ ]:


# Checking the accruracy and F1 score
# The F1 score is considered as it paints a better picture when compared to accuracy as accuracy can be high for a highly imbalanced
# problem like ours.
accuracy_score(valid_y,y_pred)


# In[ ]:


f1_score(valid_y,y_pred,average='macro')


# In[ ]:


# The model was run 4 times to get the average accuracy and f1 score, this is done as the time complexity would be high for performing 
# K-fold cross validation. The limit set for Kaggle notebooks in 6 hrs
# avg accuracy = 68 % which is the best for among the models tested like lightGBM and KNN & Random forest.


# # Making the final prediction

# In[ ]:


test = test.reset_index(drop=True) # resetting the index of the df
test_X = test.drop(['eventid','gname'],axis='columns') # Dropping the event ID
test_pred = est.predict(test_X) # making the prediction
test_pred = le.inverse_transform(test_pred) # inverse transform
pred = pd.DataFrame(test_pred) # converting the array into a df


# In[ ]:


# The indexes are same so they can be combined.
test.index,pred.index


# In[ ]:


final = pd.concat([test.eventid,pred],axis=1,join_axes=[test.index]) # combining the final data

final.to_csv('Final_pred.csv') # Exporting it to csv file 

