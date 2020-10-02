#!/usr/bin/env python
# coding: utf-8

# # [Nomad2018 Predicting Transparent Conductors](#https://www.kaggle.com/c/nomad2018-predict-transparent-conductors)
# Predict the key properties of novel transparent semiconductors
# ***
# The diffrent properties of **Aluminum,Gallium,Indium** is given in data set. In order to reduce electric transmission loss,discovery of new **transparent conductor** alloy is important. The transparent conductor having characteristic **good conductivity** and have a **low absorption**. 
# 
# The aim is to prediction of two target properties: the formation energy (which is an indication of the stability of a new material) and the bandgap energy (which is an indication of the potential for transparency over the visible range) to facilitate the discovery of new transparent conductors
# ***

# ## Import library

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from sklearn.feature_selection import VarianceThreshold
get_ipython().run_line_magic('matplotlib', 'inline')
seed=2300


# ## Read data set

# In[ ]:


#path = 'file/'
path = '../input/'
train = pd.read_csv(path+'train.csv')
test = pd.read_csv(path+'test.csv')
print('Number of rows and columns in train data set:',train.shape)
print('Number of rows and columns in test data  set:',test.shape)


# ### Evaluvation metric

# In[ ]:


def rmsle(y_true,y_pred):
    assert len(y_true) == len(y_pred)
    return np.square(np.log(y_pred + 1) - np.log(y_true + 1)).mean() ** 0.5


# ## Explore data set

# In[ ]:


train.head()


# In[ ]:


train.describe()


# ## Dependant variable distribution

# In[ ]:


fig,ax = plt.subplots(1,2,figsize=(14,4))
ax1,ax2 = ax.flatten()
sns.distplot(train['formation_energy_ev_natom'],bins=50,ax=ax1,color='b')
sns.distplot(train['bandgap_energy_ev'],bins=50,ax=ax2,color='r')


# Distribution of data is not normal, both are right skewed 

# In[ ]:


plt.figure(figsize=(14,8))
plt.scatter(train['formation_energy_ev_natom'],train['bandgap_energy_ev'],color=['r','b'])


# In[ ]:


# Lattice angle
f,ax = plt.subplots(2,3,figsize=(14,4))
feat = train.columns[train.columns.str.startswith('lattice')]
train[feat].plot(kind='hist',subplots=True,figsize=(6,6),ax=ax)
plt.tight_layout()


# In[ ]:


# Lattice angle
f,ax = plt.subplots(1,3,figsize=(14,4))
feat = train.columns[train.columns.str.startswith('percent')]
train[feat].plot(kind='kde',subplots=True,figsize=(6,6),ax=ax)
plt.tight_layout()


# In[ ]:


fig,ax = plt.subplots(1,2, figsize=(14,4))
ax1, ax2 = ax.flatten()
sns.countplot(train['spacegroup'], palette = 'magma', ax = ax1)
sns.countplot(x = train['number_of_total_atoms'], palette = 'viridis', ax = ax2)


# In[ ]:


pd.crosstab(train['number_of_total_atoms'],train['spacegroup'])


# ## Co relation plot

# In[ ]:


cor = train.corr()
plt.figure(figsize=(12,8))
sns.heatmap(cor,cmap='Set1',annot=True)


# In[ ]:


# Degree to radian
train['alpha_rad'] = np.radians(train['lattice_angle_alpha_degree'])
train['beta_rad'] = np.radians(train['lattice_angle_beta_degree'])
train['gamma_rad'] = np.radians(train['lattice_angle_gamma_degree'])

test['alpha_rad'] = np.radians(test['lattice_angle_alpha_degree'])
test['beta_rad'] = np.radians(test['lattice_angle_beta_degree'])
test['gamma_rad'] = np.radians(test['lattice_angle_gamma_degree'])


# ## Volumn
# [Soure](#https://www.kaggle.com/cbartel/random-forest-using-elemental-properties/notebook)

# In[ ]:


def vol(df):
    """
    Args:
        a (float) - lattice vector 1
        b (float) - lattice vector 2
        c (float) - lattice vector 3
        alpha (float) - lattice angle 1 [radians]
        beta (float) - lattice angle 2 [radians]
        gamma (float) - lattice angle 3 [radians]
    Returns:
        volume (float) of the parallelepiped unit cell
    """
    volumn = df['lattice_vector_1_ang']*df['lattice_vector_2_ang']*df['lattice_vector_3_ang']*np.sqrt(
    1 + 2*np.cos(df['alpha_rad'])*np.cos(df['beta_rad'])*np.cos(df['gamma_rad'])
    -np.cos(df['alpha_rad'])**2
    -np.cos(df['beta_rad'])**2
    -np.cos(df['gamma_rad'])**2)
    df['volumn'] = volumn


# In[ ]:


vol(train)
vol(test)


# In[ ]:


# Atomic density
train['density'] = train['number_of_total_atoms'] / train['volumn']
test['density'] = test['number_of_total_atoms'] / test['volumn']


# ## Mean & Median range

# In[ ]:


def mean_median_feature(df):
        print('# Mean & Median range')
        dmean = df.mean()
        dmedian = df.median()
        #q0_1 = df.quantile(0.1)
        #q0_99 = df.quantile(0.99)
        q1 = df.quantile(0.25)
        d2 = df.quantile(0.5)
        q3 = df.quantile(0.75)
        col = df.columns
        del_col = ['id','formation_energy_ev_natom','bandgap_energy_ev']
        col = [w for w in col if w not in del_col]
        
        for c in col:
            df['mean_'+c] = (df[c] > dmean[c]).astype(np.uint8)
            df['median_'+c] = (df[c] > dmedian[c]).astype(np.uint8)
            #df['q0_1_'+c] = (df[c] < q0_1[c]).astype(np.uint8)
            #df['q0_99_'+c] = (df[c] > q0_99[c]).astype(np.uint8)
            df['q1_'+c] = (df[c] < q1[c]).astype(np.uint8)
            df['q2_'+c] = (df[c] < q1[c]).astype(np.uint8)
            df['q3_'+c] = (df[c] > q3[c]).astype(np.uint8)
            
        print('Shape',df.shape)


mean_median_feature(train)
mean_median_feature(test) 


# ### One Hot encoding

# In[ ]:


def OHE(df1,df2,columns):
    len = df1.shape[0]
    df = pd.concat([df1,df2],axis=0)
    c2,c3 = [], {}
    print('Categorical variables',columns)
    for c in columns:
        c2.append(c)
        c3[c] = 'ohe_'+c
        
    df = pd.get_dummies(data = df, columns = c2, prefix = c3)
    df1 = df.iloc[:len,:]
    df2 = df.iloc[len:,:]
    print('Data size',df1.shape,df2.shape)
    return df1,df2


# In[ ]:


col = ['spacegroup','number_of_total_atoms']
train1,test1 = OHE(train,test,col)


# ##  Feature selection 
# 

# In[ ]:


col = ['formation_energy_ev_natom','bandgap_energy_ev']
X = train1.drop(['id']+col,axis=1)
y = train1[col]
x_test = test1.drop(['id']+col,axis=1)

selector = VarianceThreshold(threshold=0)
selector.fit(X) # Fit to train without id and target variables

f = np.vectorize(lambda x : not x) # Function to toggle boolean array elements

v = X.columns[f(selector.get_support())]
print('{} variables have too low variance.'.format(len(v)))
print('These variables are {}'.format(list(v)))
selected_feat = X.columns.drop(v)

#update 
X = X[selected_feat]
x_test = x_test[selected_feat]


# ## Model

# In[ ]:


kf = KFold(n_splits=5,random_state=seed,shuffle=True)
cv_score =[]
pred_test_full_1 = np.zeros((x_test.shape[0],kf.n_splits))
pred_test_full_2 = np.zeros((x_test.shape[0],kf.n_splits))
lr = LinearRegression()

for i, (train_index, valid_index) in enumerate(kf.split(X)):
    print('{} of Kfold {}'.format(i+1,kf.n_splits))
    xtrain, xvalid = X.loc[train_index], X.loc[valid_index]
    ytrain, yvalid = y.loc[train_index], y.loc[valid_index]
    
    ##Building model for ',col[0]
    lr.fit(xtrain,ytrain[col[0]])
    pred_test_full_1[:,i] = lr.predict(x_test)
    y_pred = lr.predict(xvalid)
    score = rmsle(yvalid[col[0]],y_pred)
    cv_score.append(score)
    print('R square for {} is {''} :'.format(col[0],score))
    
    ##Building model for ',col[1]
    lr.fit(xtrain,ytrain[col[1]])
    pred_test_full_2[:,i] = lr.predict(x_test)
    y_pred = lr.predict(xvalid)
    score = rmsle(yvalid[col[1]],y_pred)
    print('R square for {} is {}:'.format(col[1],score))
    cv_score.append(score)


# In[ ]:


print(cv_score)
np.mean(cv_score)


# ## Submit prediction

# In[ ]:


y_pred = np.zeros((x_test.shape[0],len(col)))
y_pred[:,0],y_pred[:,1] = pred_test_full_1.mean(axis=1), pred_test_full_2.mean(axis=1)
y_pred[y_pred <= 0] = 1e-5

submit = pd.DataFrame({'id':test['id'],'formation_energy_ev_natom':y_pred[:,0],'bandgap_energy_ev':y_pred[:,1]})
submit.to_csv('lr_conductor.csv',index=False)


# In[ ]:


submit.head()


# ### Thank you for visiting
