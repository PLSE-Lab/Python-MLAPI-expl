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
from sklearn.metrics import log_loss,mean_squared_error
get_ipython().run_line_magic('matplotlib', 'inline')
seed=2390


# ## Read data set

# In[ ]:


#path = ''
path = '../input/'
train = pd.read_csv(path+'train.csv')
test = pd.read_csv(path+'test.csv')
print('Number of rows and columns in train data set:',train.shape)
print('Number of rows and columns in test data  set:',test.shape)


# ## Explore data set

# In[ ]:


train.head()


# ### RMSLE

# In[ ]:


def rmsle(y_true,y_pred):
    return np.sqrt(np.square(np.log(y_pred + 1) - np.log(y_true + 1)).mean())


# ## Dependant variable distribution

# In[ ]:


fig,ax = plt.subplots(2,1,figsize=(12,8))
ax1,ax2 = ax.flatten()
sns.distplot(train['formation_energy_ev_natom'],bins=50,ax=ax1,color='b')
sns.distplot(train['bandgap_energy_ev'],bins=50,ax=ax2,color='r')


# Distribution of data is not normal, both are right skewed 

# In[ ]:


plt.figure(figsize=(14,8))
plt.scatter(train['formation_energy_ev_natom'],train['bandgap_energy_ev'],color=['r','b'])


# In[ ]:


train.describe()


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
# (Soure)(#https://www.kaggle.com/cbartel/random-forest-using-elemental-properties/notebook)

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


# ## Mean & Median

# In[ ]:


def mean_median_feature(df):
        print('# Mean & Median range')
        dmean = df.mean()
        dmedian = df.median()
        #q0_1 = df.quantile(0.1)
        #q0_99 = df.quantile(0.99)
        q1 = df.quantile(0.25)
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
            df['q3_'+c] = (df[c] > q3[c]).astype(np.uint8)
            
        print('Shape',df.shape)


mean_median_feature(train)
mean_median_feature(test) 


# In[ ]:


test.head()


# In[ ]:


col = ['formation_energy_ev_natom','bandgap_energy_ev']
X = train.drop(['id']+col,axis=1)
y = train[col]
x_test = test.drop(['id'],axis=1)


# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,BatchNormalization


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_1 = sc.fit_transform(X)
x_test_1 = sc.fit_transform(x_test)
y = y.values
#y_1 = sc.fit_transform(y)


# In[ ]:


X.shape


# ## ANN

# In[ ]:


regressor = Sequential()
#1 and hidden layer
regressor.add(Dense(units = 1024, activation = 'relu', kernel_initializer = 'glorot_uniform',input_dim = X.shape[1]))
regressor.add(Dropout(0.1))
regressor.add(Dense(units = 512, activation = 'relu', kernel_initializer = 'uniform'))
regressor.add(Dropout(0.1))
regressor.add(Dense(units = 64, activation = 'relu', kernel_initializer = 'uniform'))
regressor.add(Dropout(0.1))
regressor.add(Dense(units = 2, activation = 'relu', kernel_initializer = 'uniform'))

#compile ANN
regressor.compile(optimizer = 'adam', loss = 'mse', metrics =['accuracy'])
regressor.fit(X_1,y,batch_size = 3, epochs = 50, validation_split=0.1)


# In[ ]:


#Local CV
rmsle(y,regressor.predict(X_1))


# In[ ]:


y_pred = regressor.predict(x_test_1)
y_pred 


# ## Submit prediction

# In[ ]:


submit = pd.DataFrame({'id':test['id'],'formation_energy_ev_natom':y_pred[:,0],'bandgap_energy_ev':y_pred[:,1]})
submit.to_csv('NN_conductor.csv',index=False)


# In[ ]:


submit.head()


# ## Thank you
