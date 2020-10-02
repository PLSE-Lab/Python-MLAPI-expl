#!/usr/bin/env python
# coding: utf-8

# Examine the columns and convert the categorical data.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt


# In[ ]:


data_train_raw = pd.read_csv('../input/train.csv')
data_test_raw = pd.read_csv('../input/test.csv')
# print(data_train_raw.dtypes)


# How many unique values are in each categorical column?

# In[ ]:


col_uniques=[]
for col in data_train_raw.columns:
    if (col.find('cat') !=-1):
        col_uniques.append([col, len(data_train_raw[col].unique())])
print(col_uniques)


# Now convert the data using the label encoder.

# In[ ]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()

for col in data_train_raw.columns:
    if (col.find('cat') !=-1):
      #  print(col)
        data_train_raw[str(col+'_numerical')]=le.fit_transform(data_train_raw[col])
        data_test_raw[col] = data_test_raw[col].map(lambda s: '<unknown>' if s not in le.classes_ else s)
        le.classes_ = np.append(le.classes_, '<unknown>')
        data_test_raw[str(col+'_numerical')]=le.transform(data_test_raw[col])
print(data_train_raw.columns)


# There are many different columns now, so it will be difficult to view all of the data. While we could loop through all of the categories and look for interesting things, lets first do a principle component analysis. First I will normalize all of the numerical columns (after splitting into a training and validation sample).

# In[ ]:


XCols =[0]
datacols=data_train_raw.columns
for c in range(len(datacols)):
    if(datacols[c].find('cont')!=-1 or datacols[c].find('numerical')!=-1):
        XCols.append(c)
X_total = data_train_raw[XCols]
Y_total = data_train_raw['loss']


# In[ ]:


XColst =[0]
datacols=data_test_raw.columns
for c in range(len(datacols)):
    if(datacols[c].find('cont')!=-1 or datacols[c].find('numerical')!=-1):
        XColst.append(c)
X_test = data_test_raw[XColst]


# In[ ]:


# data_test_raw[XColst]


# Let's look at the loss column. This will show that it will be better to predict the log of the loss and then take the exponent of the prediction later.

# In[ ]:


plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.hist(Y_total,100)
plt.title('loss')

plt.subplot(1,2,2)
plt.hist(np.log(Y_total),100)
plt.title('log(loss)')

plt.show()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X_total, Y_total, test_size=0.4, random_state=0)


# Now we normalize all of the columns, we will save the transformation parameters so they can be used by the validation and test samples. Note that the 'cont' columns are already in the range (0,1).

# In[ ]:


stds=[1]
means=[0]
xcols=list(X_train.columns)


for c in range(1,len(xcols)):
    mm = X_train[xcols[c]].mean()
    ss = X_train[xcols[c]].std()
    
    means.append(mm)
    stds.append(ss)
    
#    print(xcols[c],r)


# In[ ]:


X_train = (X_train[xcols] - means) / stds
X_valid = (X_valid[xcols] - means) / stds
X_test = (X_test[xcols] - means) / stds


# In[ ]:


xcols.remove('id')

print("Train")
print(X_train[xcols[100]].describe())
print("Valid")
print(X_valid[xcols[100]].describe())
print("Test")
print(X_test[xcols[100]].describe())


# In[ ]:


from sklearn.cluster import KMeans
km = KMeans(n_clusters=8,n_jobs=-1)
km.fit(X_train[xcols])

X_train['km']=km.predict(X_train[xcols])
X_valid['km']=km.predict(X_valid[xcols])
X_test['km']=km.predict(X_test[xcols])
xcols.append('km')


# In[ ]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X_train[xcols])


# In[ ]:


X_train_transformed = pca.transform(X_train[xcols])


# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.scatter(X_train_transformed[:,0],y_train)
plt.title('First Axis')
plt.ylabel('loss')
plt.yscale('log')

plt.subplot(1,2,2)
plt.scatter(X_train_transformed[:,1],y_train)
plt.title('Second Axis')
plt.yscale('log')
plt.show()


# In[ ]:


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train_transformed[:,0], X_train_transformed[:,1], np.log(y_train))

plt.show()


# There may be some correlations in the data here, but it appears that the PCA here is loosing much of the variance.

# ## Attempt some simple models

# ### Random forrest

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

rfr = RandomForestRegressor(n_estimators= 400, n_jobs=-1)


# In[ ]:


rfr.fit(X_train[xcols],np.log(y_train))


# In[ ]:


X_train['rfr']=np.exp(rfr.predict(X_train[xcols]))
X_valid['rfr']=np.exp(rfr.predict(X_valid[xcols]))
X_test['rfr']=np.exp(rfr.predict(X_test[xcols]))


# In[ ]:


trainscore=mean_squared_error(X_train['rfr'],y_train)
validscore=mean_squared_error(X_valid['rfr'],y_valid)


# In[ ]:


plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title(r'Training score='+str(trainscore))
plt.scatter(X_train['rfr'], y_train)
plt.xlabel('Prediction')
plt.ylabel('Truth')

plt.subplot(1,2,2)
plt.title(r'Validation score='+str(validscore))
plt.scatter(X_valid['rfr'], y_valid)
plt.xlabel('Prediction')
plt.ylabel('Truth')

plt.show()


# We see that the validation set is preforming much worse with the default random forrest hyper parameters. We could try fixing this for overtraining. However, lets first run this on the test set to get a first score.

# In[ ]:


X_test.columns


# In[ ]:


rfrpred=pd.DataFrame(list(zip(X_test['id'],X_test['rfr'])),columns=('id','loss'))
rfrpred['id']=rfrpred['id'].astype('int')
                     


# In[ ]:


rfrpred.to_csv('submit_RFR_' +str(validscore) +'.csv', index=False)


# In[ ]:


list(enumerate(sorted(list(zip(xcols,rfr.feature_importances_)), key=lambda l:l[1], reverse=True)))


# ### Linear Regression

# In[ ]:


from sklearn.linear_model import Ridge


# In[ ]:


rid = Ridge(alpha=1e-6, fit_intercept=True, normalize=False, 
                  copy_X=True, max_iter=None, tol=0.001, solver='auto', random_state=None)


# In[ ]:


rid.fit(X_train[xcols],np.log(y_train))


# In[ ]:


X_train['rid'] = np.exp(rid.predict(X_train[xcols]))
X_valid['rid'] = np.exp(rid.predict(X_valid[xcols]))
X_test['rid'] = np.exp(rid.predict(X_test[xcols]))

trainscore=mean_squared_error(X_train['rid'],y_train)
testscore=mean_squared_error(X_valid['rid'],y_valid)
print(trainscore,testscore)


# In[ ]:


plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title(r'Training score='+str(trainscore))
plt.scatter(X_train['rid'], y_train)
plt.xlabel('Prediction')
plt.ylabel('Truth')

plt.subplot(1,2,2)
plt.title(r'Validation score='+str(testscore))
plt.scatter(X_valid['rid'], y_valid)
plt.xlabel('Prediction')
plt.ylabel('Truth')

plt.show()


# In[ ]:


ridpred=pd.DataFrame(list(zip(X_test['id'],X_train['rid'])),columns=('id','loss'))
ridpred['id']=ridpred['id'].astype('int')
ridpred.head()


# In[ ]:


rid.feature_importances_


# In[ ]:


ridpred.to_csv('submit_ridge_' +str(testscore) +'.csv', index=False)


# ## Basic NN

# In[ ]:


from sklearn.neural_network import MLPRegressor

mlpnnR = MLPRegressor(hidden_layer_sizes=(int(X_train.shape[1]/2),int(X_train.shape[1]/2), int(X_train.shape[1]/2),int(X_train.shape[1]/2)), 
                       activation='logistic', 
                       solver='adam', 
                       alpha=0.1, 
                       batch_size='auto',
#                        learning_rate='adaptive',
                       learning_rate_init=0.0001,
                       power_t=0.5, max_iter=200,
                       shuffle=True, 
                       random_state=None, 
                       tol=0.0001, 
                       verbose=True,
                       warm_start=False,
                       momentum=0.9,
                       nesterovs_momentum=True, early_stopping=False, 
                       validation_fraction=0.1, beta_1=0.9, beta_2=0.999, 
                       epsilon=1e-08)


# In[ ]:


mlpnnR.fit(X_train[xcols],np.log(y_train))


# In[ ]:


X_train['nn'] = np.exp(mlpnnR.predict(X_train[xcols]))
X_valid['nn'] = np.exp(mlpnnR.predict(X_valid[xcols]))
X_test['nn'] = np.exp(mlpnnR.predict(X_test[xcols]))


# In[ ]:


trainnnpred=X_train['nn']
validnnpred=X_valid['nn']

nnscoret=mean_squared_error(trainnnpred,y_train)
nnscorev=mean_squared_error(validnnpred,y_valid)

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title(r'Training score='+str(nnscoret))
plt.scatter(trainnnpred, y_train)
plt.xlabel('Prediction')
plt.ylabel('Truth')

plt.subplot(1,2,2)
plt.title(r'Validation score='+str(nnscorev))
plt.scatter(validnnpred, y_valid)
plt.xlabel('Prediction')
plt.ylabel('Truth')

plt.show()


# In[ ]:


mlpout=pd.DataFrame(
    list(zip(X_test['id'],X_test['nn'])),
    columns=('id','loss'))
mlpout['id']=mlpout['id'].astype('int')
mlpout.head()
mlpout.to_csv('submit_nnet_' +str(nnscorev) +'.csv', 
               index=False)


# ## Make an average

# In[ ]:


Ave2=pd.DataFrame()
Ave2['id']=ridpred['id']
Ave2['loss']=(1/testscore*ridpred['loss'] + 1/validscore*rfrpred['loss'] + 1/nnscorev*mlpout['loss'])/(1/testscore+1/validscore+1/nnscorev)

Ave2.head()
Ave2.to_csv('Average.csv', index=False)

