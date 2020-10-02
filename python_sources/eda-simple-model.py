#!/usr/bin/env python
# coding: utf-8

# # TOC TOC
# 1. [Importing neccesary modules ](#importing-necessary-modules)
# 2. [Loading data](#loading-data)
# 3. [Exploring data](#exploring-data)
#     1. [Target](#target)
#     2. [Sparsity of the dataset](#sparsity-of-the-dataset)
#     3. [Zero variance and dupllicate date](#zero-variance-and-duplicate)
# 4. [Dimensionality Reduction](#dimensionality-reduction)
# 5. [Modeling](#modeling)
#     1. [OLS](#ols)

# # Importing necesary modules<a name="importing-neccesary-modules"></a>
# In this sections will import necessary modules for the kernel
# 

# In[ ]:


#%matplotlib inline

# for seaborn issue:
import warnings
warnings.filterwarnings("ignore")
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model

print(os.listdir("../input"))


# # Loading data <a name="loading-data"></a>
# In this section we will load the data to use.  We just have 2 file, the train file and the test file (also the sample submission file).
# 

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# # Exploring data <a name="exploring-data"></a>
# I will start exploring the train data file. The only data. 
# ### Train
# Let's see the train data

# In[ ]:


train.head(5)


# In[ ]:


print(train.columns)
print(train.shape)


# Ok, the columns names are wired. Let's the more information about the data set.

# In[ ]:


print(train.info())


# In[ ]:


# Null values?
nulls_data = train.isnull().sum().sum()
print("There are {} null data on the dataset".format(nulls_data))


# We see that the dataset have 4993 columns and 4459 entries. We don't have more information about the features, just that 1845 are float, 3147 are int and just 1 is string (the ID).  How we could explore this data if we don't know what relationship could are between columns?
# 
# ### Test 
# Let's see the test data

# In[ ]:


test.head(5)


# In[ ]:


print(test.columns)
print(test.shape)
print('test info:')
print(test.info())
# Null values?
nulls_data = test.isnull().sum().sum()

print("There are {} null data on the dataset".format(nulls_data))


# ## Target <a name="target"></a>

# In[ ]:


print(train.target.describe())


# In[ ]:


train.target.plot.hist()


# The target on train dataset is right-skewed

# In[ ]:


target_log = np.log(train.target)
target_subx = 1/train.target
target_square = np.square(train.target)
print(target_log.skew())

print(target_square.skew())
target_log.plot.hist()
train.target = target_log


# ## Sparsity of the dataset <a name="sparsity-of-the-dataset"></a>
# Various public kernels show that there are many zeros on the dataset. 

# In[ ]:


columns = train.columns
print(len(train[train[columns[2]] == 0])/len(train[columns[2]]))
print(len(train[columns[2]]))


# In[ ]:


list_zeros = [len(train[train[d] == 0])/4459. for d in columns]
# list_zeros = []
#for d in columns:
#    zeros = len(train[train[d] == 0])
#    total = 4459.
#    list_zeros.append(zeros/total)


# In[ ]:


sns.distplot(list_zeros, bins=100)


# ## Zero variance and duplicate features<a name="#zero-variance-and-duplicate"></a>
# I will remove  the zero variance features and duplicated columns.

# In[ ]:


# df = df.loc[:, df.var() == 0.0]
# obj_df = train.select_dtypes(include=['object'])
obj_df = train.iloc[:, :2]
# num_df = train.select_dtypes(exclude=['object'])
num_df = train.iloc[:,2:]
var = num_df.var()
l_keys_notzeros = []
l_values_notzeros = []
for k, v in var.items():
    if v != 0.0:
        l_keys_notzeros.append(k)
        l_values_notzeros.append(v)
# foo = num_df.loc[:, num_df.var() != 0.0]
foo = num_df[l_keys_notzeros]
new_train_without_zeros = pd.concat([obj_df, foo], axis=1) # new data without zero variance
print(new_train_without_zeros.shape)


# In[ ]:


obj_df = test.iloc[:, :1]
num_df = test.iloc[:,1:]
foo = num_df[l_keys_notzeros]
new_test_without_zeros = pd.concat([obj_df, foo], axis=1) # new data without zero variance
print(new_test_without_zeros.shape)


# In[ ]:


del obj_df
del num_df
del foo


# In[ ]:


# Remove duplicated columns
col_to_remove = list()
col_scanned = list()
dup_list = dict()

cols = new_train_without_zeros.columns

for i in range(len(cols) - 1):
    v = new_train_without_zeros[cols[i]].values
    dup_cols = list()
    for j in range(i+1, len(cols)):
        if np.array_equal(v, new_train_without_zeros[cols[j]].values):
            col_to_remove.append(cols[j])
            if cols[j] not in col_scanned:
                dup_cols.append(cols[j]) 
                col_scanned.append(cols[j])
                dup_list[cols[i]] = dup_cols
print(col_to_remove)    


# In[ ]:


cols = [c for c in cols if c not in col_to_remove]
cols_test = [c for c in cols if c != 'target']


# In[ ]:


new_train = new_train_without_zeros[cols]
new_test = new_test_without_zeros[cols_test]

print(new_train.shape)
print(new_test.shape)


# In[ ]:


del new_train_without_zeros
del new_test_without_zeros
del train
del test


# In[ ]:


del col_to_remove
del col_scanned
del dup_list
del cols


# # Dimensionality Reduction <a name="dimensionality-reduction"></a>
# I will try to reduce the dataset dimension

# In[ ]:


id_target_train = new_train.iloc[:,:2]
new_train = new_train.iloc[:,2:].values


# In[ ]:


id_test = new_test.iloc[:,:1]
new_test = new_test.iloc[:, 1:].values


# In[ ]:


print('Shape of train: ',new_train.shape)
print('Shape of test: ', new_test.shape)
#print('Shape of target: ',log_target.shape)
#print('Shape of test: ',test.shape)


# In[ ]:


def transform (dataframe):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(dataframe)
    return pd.DataFrame(scaled_data)


# In[ ]:


new_train = transform(new_train)


# In[ ]:


new_test = transform(new_test)


# In[ ]:


# num_data = ttrain.select_dtypes(exclude='object')
num_data = new_train
pca = PCA(copy=True, n_components=2000, whiten=False)
new = pca.fit(num_data).transform(num_data)
print(pca.explained_variance_ratio_) 
len_pca = len(pca.explained_variance_ratio_)
print("The first {} PCA explain {}".format(len_pca, pca.explained_variance_ratio_.sum()*100))


# In[ ]:


# var=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3)*100)
var = pca.explained_variance_ratio_.cumsum()
plt.ylabel('% Variance Explained')
plt.xlabel('# of Features')
plt.title('PCA Analysis')
plt.style.context('seaborn-whitegrid')

plt.plot(var)
plt.show()


# In[ ]:


pca_train = pd.DataFrame(data=new, columns=['pca{}'.format(i) for i in range(2000)])
pca_train = pd.concat([id_target_train[['ID','target']], pca_train], axis = 1)
print(pca_train.head(1))


# In[ ]:


num_data = new_test
new = pca.transform(num_data)
pca_test = pd.DataFrame(data=new, columns=['pca{}'.format(i) for i in range(2000)])
pca_test = pd.concat([id_test[['ID']], pca_test], axis=1)
print(pca_test.head(1))


# # Modeling <a name="modeling"></a>
# Now, I think that we could start create some models
# 
# ## OLS <a name="ols"></a>
# We will start with a very simply model: _Ordinary Least Squares_

# In[ ]:


x_train = pca_train.iloc[:, 2:]
y_train = pca_train.iloc[:, 1:2]

x_test = pca_test.iloc[:, 1:]


# In[ ]:


linear_regression = linear_model.LinearRegression()
linear_regression.fit(x_train, y_train)
print(linear_regression.coef_)


# In[ ]:


target_test  = linear_regression.predict(x_test)


# In[ ]:


target_test = pd.DataFrame(data=target_test, columns=['target'])
print(target_test.head(1))


# In[ ]:


to_submit = pd.concat([pca_test['ID'], target_test['target']], axis=1)
print(to_submit.head(1))


# In[ ]:


to_submit.to_csv('ols.csv', columns=['ID','target'], index=False)

