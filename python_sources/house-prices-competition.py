#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


import matplotlib.pyplot as plt
import seaborn as sns
import missingno
from collections import defaultdict

pd.set_option('display.max_columns',1000)

from IPython.core.interactiveshell import InteractiveShell # print everything within a cell
InteractiveShell.ast_node_interactivity = "all";


# In[ ]:


# Import files
df = pd.read_csv('../input/train.csv')
d_test = pd.read_csv('../input/test.csv')


# In[ ]:


df.head()


# # Data Prep

# Let's check missing values

# In[ ]:


missingno.matrix(df, figsize = (30,5))


# In[ ]:


df.isna().sum().sort_values(ascending = False)[:20]


# All "NA"s can be replaced with the word "None" if the type of the value is categorical, or the integer/float 0 if the value type is numerical.
# 
# 

# In[ ]:


for c in df.columns:
    if df[c].dtypes=='O':
        df[c].fillna(value='none',inplace=True)
    else:
        df[c].fillna(value=0,inplace=True)


# # EDA

# In[ ]:


df.head()


# ## Check Correlations

# # Transform data based on type

# In[ ]:


# Numerical types which should be categorical
objects_list = ['MSSubClass']

# Categorical types which can be taken as Linkert scale (rating of e.g. 1-5 where 1 is better than 2, 2 is better than 3 etc)
linkert_list = ['ExterQual', 'ExterCond','BsmtQual', 'BsmtCond', 'BsmtExposure', 'HeatingQC','KitchenQual',
              'Functional','FireplaceQu','GarageQual','GarageCond','PavedDrive','Fence' ]

# onehot_list = 


# In[ ]:


fig, ax = plt.subplots(int((df.shape[1]/3))+1, 3,figsize = (20,4* int((df.shape[1]/3))+1))
for var, subplot in zip(df.columns, ax.flatten()):
    # if the value is categorical, plot boxes for each categorical type
    if df[var].dtypes == 'O':
        _=sns.boxplot(y=df[var], x=df['SalePrice'],orient='h', ax=subplot)
    else:
        _=sns.scatterplot(x=df[var],y=df['SalePrice'],ax=subplot)


# In[ ]:


for c in df.columns:
    if df[c].dtypes == 'O':
        # sort by categorical values and check saleprice
        d = df.set_index(c)['SalePrice'].groupby(axis=0,level=0).mean()
        d = round(d/df['SalePrice'].mean()*100,1)
        d.name='Mean'
        b = df.set_index(c)['SalePrice'].groupby(axis=0,level=0).std()
        b = round(b/df['SalePrice'].std()*100,1)
        b.name='std'
        pd.concat([d,b],axis=1)


# # Data Prep

# ## Int to Obj

# In[ ]:


df[objects_list] = df[objects_list].astype('O')


# ## Linkert Scales

# In[ ]:


a = []
for c in linkert_list:
    a.append(df[c].unique().tolist())

label_values = pd.DataFrame(a).T
label_values.columns = linkert_list
label_values


# In[ ]:


# Choose unique set of labels. We can see 4 different 
label_loc = [4,7,8,11,12]
for i in label_loc:
    label_values.iloc[:,i].dropna().tolist()


# In[ ]:


# create map
sortlabels = defaultdict()

sortlabels['Exposure'] = dict(zip(['No', 'Gd', 'Mn', 'Av', 'none'],
                                  ['1.No', '4.Gd', '2.Mn', '3.Av', '0.none']))

sortlabels['Quality'] = dict(zip(['none', 'TA', 'Gd', 'Fa', 'Ex', 'Po'],
                                 ['0.none', '3.TA', '4.Gd', '2.Fa', '5.Ex', '1.Po']))

sortlabels['Paved'] = dict(zip(['Y', 'N', 'P'],
                               ['3.Y', '1.N', '2.P']))

sortlabels['Fence'] = dict(zip(['none', 'MnPrv', 'GdWo', 'GdPrv', 'MnWw'],
                               ['0.none', '3.MnPrv', '2.GdWo', '4.GdPrv', '1.MnWw']))

sortlabels['Functional'] = dict(zip(['Typ', 'Min1', 'Maj1', 'Min2', 'Mod', 'Maj2', 'Sev', 'Sal'],
                                    ['1.Typ', '2.Min1', '5.Maj1', '3.Min2', '4.Mod', '6.Maj2', '7.Sev', '8.Sal']))


# set corresponding label values for linkert_list
linkert_map = dict(zip(linkert_list,['Quality', 'Quality','Quality', 'Quality', 'Exposure', 'Quality','Quality',
              'Functional','Quality','Quality','Quality','Paved','Fence' ]))


# In[ ]:


# apply map
for c in linkert_list:
    df[c] = df[c].replace(sortlabels[linkert_map[c]])


# ## Categorical encoding

# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


# find new categorical cols
categorical_cols = objects_list + linkert_list

# create label dict
labeldict = defaultdict(LabelEncoder)

# encode categorical values
labelfit = df[categorical_cols].apply(lambda x: labeldict[x.name].fit_transform(x))
df[categorical_cols] = labelfit


# In[ ]:


# Check encoded classes if required
labeldict['ExterQual'].classes_


# ## OneHotEncoding
# for categorical values which cannot be sorted in a scale from best to worst

# after the Label Encoding, object values remaining:

# In[ ]:


onehot_cols = df.columns[df.dtypes=='O']
df.loc[:,onehot_cols].head()


# In[ ]:


# plot relationship to SalePrice

fig, ax = plt.subplots(int((df[onehot_cols].shape[1]/3))+1, 3,figsize = (20,4* int((df[onehot_cols].shape[1]/3))+1))
for var, subplot in zip(onehot_cols, ax.flatten()):
    _=sns.boxplot(y=df[var], x=df['SalePrice'],orient='h', ax=subplot)


# There seem to be differences that would play a role. On the other hand, we should not increase the number of columns too much as we only have 1460 rows of data to train our model. We will one-hot encode the values we have, and perhaps we'll implement PCA to reduce dimensionality.

# In[ ]:


# save numerical vars (including linkert scale vars for future reference)
numerical_cols = df.iloc[:,:-1].loc[:,df.iloc[:,:-1].dtypes!='O'].columns


# ## Apply OneHotEncoding

# In[ ]:


from sklearn.preprocessing import OneHotEncoder


# In[ ]:


enc = OneHotEncoder(categories = 'auto', sparse=False)
a = enc.fit_transform(df[onehot_cols])
# create dataframe with OneHotEncoded values
df_onehot = pd.DataFrame(data=a,columns =enc.get_feature_names() ,index = df.index)


# # Model tests

# In[ ]:


from sklearn.model_selection import train_test_split


# ## 1st test: Add all values

# In[ ]:


x = pd.concat([df[numerical_cols],df_onehot],axis=1)
y = df['SalePrice']


# ### Data scaling

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
y_scaled = scaler.fit_transform(y.values.reshape(-1,1))
y_scaled = y_scaled.ravel()


# ### Train test split

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split (x_scaled,y_scaled,test_size = 0.1, random_state = 19)


# ### Lasso Regression
# Lasso regression to avoid overfitting while minimizing features which don't play a significant role.

# In[ ]:


from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score as r2


# In[ ]:


lr = LassoCV(random_state = 42,cv=5,max_iter=10000,eps=0.001)
_=lr.fit(x_train,y_train)

print ('Lasso training score:')
lr.score(x_train,y_train)

y_pred = lr.predict(x_test)
print ('Lasso test score:')
r2(y_test,y_pred)


# ### Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

clf = RandomForestRegressor(n_estimators = 1000, max_depth = 15,min_samples_split = 50, random_state = 42)
clf.fit(x_train, y_train)

print ('training score:')
clf.score(x_train,y_train)

print ('test score:')
clf.score(x_test,y_test)


# It seems that, while RandomForest has a reduced score in the training set, overfitting is avoided and as such a 73.7% score in the test set is achieved.

# # 2nd test: Perform PCA for OneHotEncoded variables

# In[ ]:


from sklearn.decomposition import PCA
df_onehot.shape


# since all values are 1 or 0, we will not standardize them further.

# In[ ]:


pca = PCA(n_components = 60)
onehot_pca = pca.fit_transform(df_onehot)
print ('Total % variance explained by the new dimensions')
pca.explained_variance_ratio_.sum()

df_onehot_pca = pd.DataFrame(data = onehot_pca)


# In[ ]:


x = pd.concat([df[numerical_cols],df_onehot_pca],axis=1)
y = df['SalePrice']


# ### Data scaling

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
y_scaled = scaler.fit_transform(y.values.reshape(-1,1))
y_scaled = y_scaled.ravel()


# ### Train test split

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split (x_scaled,y_scaled,test_size = 0.1, random_state = 19)


# ### Lasso Regression
# Lasso regression to avoid overfitting while minimizing features which don't play a significant role.

# In[ ]:


from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score as r2


# In[ ]:


lr = LassoCV(random_state = 42,cv=5,max_iter=10000,eps=0.001)
_=lr.fit(x_train,y_train)

print ('Lasso training score:')
lr.score(x_train,y_train)

y_pred = lr.predict(x_test)
print ('Lasso test score:')
r2(y_test,y_pred)


# ### Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

clf = RandomForestRegressor(n_estimators = 1000, max_depth = 15,min_samples_split = 50, random_state = 42)
clf.fit(x_train, y_train)

print ('training score:')
clf.score(x_train,y_train)

print ('test score:')
clf.score(x_test,y_test)


# Results improved for Lasso Regression (as expected, due to reduced number of variables) but slightly deteriorated for Random Forest.

# # 3rd test: Perform PCA for all variables

# In[ ]:


from sklearn.decomposition import PCA


# In[ ]:


d_pca = pd.concat([df[numerical_cols],df_onehot],axis=1)

# scale data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
d_pca_scaled = scaler.fit_transform(d_pca)


# In[ ]:


pca = PCA(n_components = 160)
reduced_pca = pca.fit_transform(d_pca_scaled)
print ('Total % variance explained by the new dimensions')
pca.explained_variance_ratio_.sum()

df_reduced_pca = pd.DataFrame(data = reduced_pca)


# In[ ]:


x = df_reduced_pca
y = df['SalePrice']


# ### Data scaling

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
y_scaled = scaler.fit_transform(y.values.reshape(-1,1))
y_scaled = y_scaled.ravel()


# ### Train test split

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split (x_scaled,y_scaled,test_size = 0.1, random_state = 19)


# ### Lasso Regression
# Lasso regression to avoid overfitting while minimizing features which don't play a significant role.

# In[ ]:


from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score as r2


# In[ ]:


lr = LassoCV(random_state = 42,cv=5,max_iter=10000,eps=0.001)
_=lr.fit(x_train,y_train)

print ('Lasso training score:')
lr.score(x_train,y_train)

y_pred = lr.predict(x_test)
print ('Lasso test score:')
r2(y_test,y_pred)


# ### Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

clf = RandomForestRegressor(n_estimators = 1000, max_depth = 15,min_samples_split = 50, random_state = 42)
clf.fit(x_train, y_train)

print ('training score:')
clf.score(x_train,y_train)

print ('test score:')
clf.score(x_test,y_test)


# Results further improved for Lasso Regression with an r2 score of 70.4%, and deteriorated for Random Forest with an r2 score of 69.4%.

# # 4th test: Remove OneHotEncoded variables

# In[ ]:


x = df[numerical_cols]
y = df['SalePrice']


# ### Data scaling

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
y_scaled = scaler.fit_transform(y.values.reshape(-1,1))
y_scaled = y_scaled.ravel()


# ### Train test split

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split (x_scaled,y_scaled,test_size = 0.1, random_state = 19)


# ### Lasso Regression
# Lasso regression to avoid overfitting while minimizing features which don't play a significant role.

# In[ ]:


from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score as r2


# In[ ]:


lr = LassoCV(random_state = 42,cv=5,max_iter=10000,eps=0.001)
_=lr.fit(x_train,y_train)

print ('Lasso training score:')
lr.score(x_train,y_train)

y_pred = lr.predict(x_test)
print ('Lasso test score:')
r2(y_test,y_pred)


# ### Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

clf = RandomForestRegressor(n_estimators = 1000, max_depth = 15, min_samples_split = 50, random_state = 42)
clf.fit(x_train, y_train)

print ('training score:')
clf.score(x_train,y_train)

print ('test score:')
clf.score(x_test,y_test)


# Maximum score so far was achieved for Random Forests without One-Hot encoded variables (r2 of 73.97%)

# # Tensorflow NN

# In[ ]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# In[ ]:


x = pd.concat([df[numerical_cols],df_onehot],axis=1)
y = df['SalePrice']


# ### Data scaling

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
y_scaled = scaler.fit_transform(y.values.reshape(-1,1))
y_scaled = y_scaled.ravel()


# ### Train test split

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split (x_scaled,y_scaled,test_size = 0.1, random_state = 19)


# In[ ]:


def build_model():
  model = keras.Sequential([
    layers.Dense(128, activation=tf.nn.relu, input_shape=[x_train.shape[1]]),
    layers.Dropout(0.2),
    layers.Dense(256, activation=tf.nn.relu), 
    layers.Dropout(0.2),
    layers.Dense(64, activation=tf.nn.relu), 
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.Adam(0.00001)

  model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_squared_error','mean_absolute_error'])
  return model


def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.ylim([0,20])
  plt.legend()
  plt.show()


# In[ ]:


# initialize the model
model = build_model()
model.summary()


# In[ ]:


# Test run
example_batch = x_train[:10,:]
example_result = model.predict(example_batch)
example_result


# Our output is of the correct size.

# In[ ]:


# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 300

history = model.fit(
  x_train, y_train,
  epochs=EPOCHS, batch_size = 8,
    validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])

plot_history(history)


# In[ ]:


y_pred = model.predict(x_test)
r2(y_test,y_pred)


# In[ ]:


y_train_pred = model.predict(x_train)
r2(y_train,y_train_pred)


# ## Next steps:
# * Build NN with tensorflow.
# 
# check https://www.tensorflow.org/tutorials/keras/basic_regression <br>
# https://medium.com/mlreview/a-simple-deep-learning-model-for-stock-price-prediction-using-tensorflow-30505541d877

# In[ ]:





# In[ ]:




