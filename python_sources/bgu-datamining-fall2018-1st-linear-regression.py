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


# **This notebook format submitted according to the guidelines of a data mining course belonging to Ben-Gurion University of the Negev, Winter 2018**
# 
# **CONTENTS**
# 1. Physical introduction
# 2. Data mining concepts
# 3.  Data explore and visualization
# 4. Data processing: using physical methods for "atoms number" normalization, add new featuers by quantiles (dividing the range of a probability distribution into continuous intervals with equal probabilities)and add new featuers via convert the categorial features into binary using    "One Hot encoding" function
# 5. Fitting the targets to Gaussian distribution by log(x+1): using "numpy.log1p()
# 6. hold-out methods  cross validation: using "train_test_split"
# 7. Linear Regression 
# 8. RMSLE calculation
# 

# **1. Physical Introduction:**
# 
# In order to find the optimum composition for transparent crystal, some basic principle should
# be the basic rule for computational approach. The alloys for transparent crystal should be (AlxGayInz)2NO3N, where x+y+z = 1 and N is an integer (usually between 5 and 100). There are infinite possible combinations for the values of x,y and z so the choice of computational method is the pivotal issue for transparent conductor materials design efficiency. There exist one primary computational method for solid-state physics caled **Density Functional Theory**, which is able to get high accuracy result but requires much computing time even for supercomputers. In this way, the data-driven method will be an alternative way to improve the efficiency for the transparent crystal design process.
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 

# **2.Data mining comcepts**
# 
# The database consists of 11 features with 2400 observations for *training*, another 400 observations for *test* and two *target* vectors. The ultimate goal is to predict the values of the two vectors based on the 11th features.
# 
# A. For training: we will use some of the 2400 observations 
# B. For validation : We will use the remaining part.
# C. For prediction: We used 400 test observation.
# 

# In[ ]:


import numpy as np
import pandas as pa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression as LR
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# **3.  Data explore and visualization**

# In[ ]:


train = pa.read_csv('../input/train.csv')
test = pa.read_csv('../input/test.csv')


# In[ ]:


print ('train shape:'+ str(train.shape))
print ('test shape:'+ str(test.shape))
train.head()


# In[ ]:


test.head()


# In[ ]:


train.describe()


# Categorial data

# In[ ]:


fig,ax = plt.subplots(1,2, figsize=(14,4))
ax1, ax2 = ax.flatten()
sns.countplot(train['spacegroup'], palette = 'viridis', ax = ax1)
sns.countplot(x = train['number_of_total_atoms'], palette = 'magma', ax = ax2)


# precintage data

# In[ ]:


f,ax = plt.subplots(1,3,figsize=(14,4))
feat = train.columns[train.columns.str.startswith('percent')]
train[feat].plot(kind='hist',subplots=True,figsize=(6,6),ax=ax)
plt.tight_layout()


# discrit data

# In[ ]:


f,ax = plt.subplots(2,3,figsize=(14,4))
feat = train.columns[train.columns.str.startswith('lattice')]
train[feat].plot(kind='hist',subplots=True,figsize=(6,6),ax=ax)
plt.tight_layout()


# Correlation plot

# In[ ]:


corr = train.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# **3. Data processing**
# 
# First of all, we want to chick if there are no correlations between the two tearget vectors using scattring plot:
# 

# In[ ]:


plt.figure(figsize=(14,8))
plt.scatter(train['formation_energy_ev_natom'],train['bandgap_energy_ev'],color=['r','b'])


# As we see, there is a great overlap between the targets, so we will use each target separately.
# After testing the model for the first time,we also tried some other methods for optimization. The first one is that we add a new feature **volume** to the 3000 (training and test combined) crystal by following a general **volume** equation for lattice regardless of special length or angle as,
# 
#  "number_of_total_atoms"  to the crystal *volume*, this crystal is Triclinic type, where the valuem is :
# V=(lattice_vector_1_ang*lattice_vector_2_ang*lattice_vector_3_ang)*[1+2*cos(alpha)*cos(beta)*cos(gama)-cos(alpha)^2-cos(beta)^2-cos(gama)^2)^0.5
# ![http://geoclasses.com/wp-content/uploads/2018/02/volume-of-crystal-system-.jpg](http://)
# 
# **BUT!! the angle shuld be in RADIANS**
# 
# see linke: http://geoclasses.com/notes/volume-of-crystal-system/
# 
# **Note that this is not simple normalization, normalization factor is not equal for  each  crystal !!**

# Transform the angles from "Deg" into "Radians"

# In[ ]:


train['alpha_rad'] = np.radians(train['lattice_angle_alpha_degree'])
train['beta_rad'] = np.radians(train['lattice_angle_beta_degree'])
train['gamma_rad'] = np.radians(train['lattice_angle_gamma_degree'])

test['alpha_rad'] = np.radians(test['lattice_angle_alpha_degree'])
test['beta_rad'] = np.radians(test['lattice_angle_beta_degree'])
test['gamma_rad'] = np.radians(test['lattice_angle_gamma_degree'])


# Calculate the valume of the crystal

# In[ ]:





# In[ ]:


def vol(df):
    volumn = df['lattice_vector_1_ang']*df['lattice_vector_2_ang']*df['lattice_vector_3_ang']*np.sqrt(
    1 + 2*np.cos(df['alpha_rad'])*np.cos(df['beta_rad'])*np.cos(df['gamma_rad'])
    -np.cos(df['alpha_rad'])**2
    -np.cos(df['beta_rad'])**2
    -np.cos(df['gamma_rad'])**2)
    df['volumn'] = volumn
vol(train)
vol(test)


# the density:

# In[ ]:


train['density'] = train['number_of_total_atoms'] / train['volumn']
test['density'] = test['number_of_total_atoms'] / test['volumn']


# 
#     
#   second,    add another features using quantile function in python, where we can divide the data into 3 levels , qun0.25, qun 0.5 and qun 0.75,
#     

# 

# In[ ]:


def mean_median_feature(df):
        dmean = df.mean()
        dmedian = df.median()
        q1 = df.quantile(0.25)
        d2 = df.quantile(0.5)
        q3 = df.quantile(0.75)
        col = df.columns
        del_col = ['id','formation_energy_ev_natom','bandgap_energy_ev']
        col = [w for w in col if w not in del_col]
        
        for c in col:
            df['mean_'+c] = (df[c] > dmean[c]).astype(np.uint8)
            df['median_'+c] = (df[c] > dmedian[c]).astype(np.uint8)
            df['q1_'+c] = (df[c] < q1[c]).astype(np.uint8)
            df['q2_'+c] = (df[c] < q1[c]).astype(np.uint8)
            df['q3_'+c] = (df[c] > q3[c]).astype(np.uint8)
            
        print('Shape',df.shape)


mean_median_feature(train)
mean_median_feature(test)
train.shape


# Thered, another featuers  added by   covnert  the categorial features ''spacegroup' and ''number_of_total_atoms'' into  binary data using One Hot encoding:
# 
# 

# In[ ]:


def OHE(df1,df2,columns):
    len = df1.shape[0]
    df = pa.concat([df1,df2],axis=0)
    c2,c3 = [], {}
    print('Categorical variables',columns)
    for c in columns:
        c2.append(c)
        c3[c] = 'ohe_'+c
        
    df = pa.get_dummies(data = df, columns = c2, prefix = c3)
    df1 = df.iloc[:len,:]
    df2 = df.iloc[len:,:]
    print('Data size',df1.shape,df2.shape)
    return df1,df2
col = ['spacegroup','number_of_total_atoms']
train1,test1 = OHE(train,test,col)


# Data after processing:

# In[ ]:


col = ['formation_energy_ev_natom','bandgap_energy_ev']
X = train1.drop(['id']+col,axis=1)
y = train1[col]
valid = test1.drop(['id']+col,axis=1)


# **5. Fitting the targets**
# 
# For accurate scores, the target most be normal (Gaussian) distributed, thefore, we can use log(x+1) function in order to transform the targets as Gaussian distributed:

# In[ ]:


# formation_energy_ev_natom
fig,ax = plt.subplots(1,2,figsize=(14,4))
ax1,ax2 = ax.flatten()
sns.distplot(train['formation_energy_ev_natom'],bins=50,ax=ax1,color='b')
formation_energy_ev_log1 = np.log1p(train['formation_energy_ev_natom'])
formation_energy_ev_log1 = pa.DataFrame({'log(1+formation_energy_ev)': formation_energy_ev_log1})
sns.distplot(formation_energy_ev_log1['log(1+formation_energy_ev)'],bins=50,ax=ax2,color='r')


# We can see for the "formation_energy_ev" target, almost there are no changes.
# let see what about "bandgap_energy_ev":

# In[ ]:



fig,ax = plt.subplots(1,2,figsize=(14,4))
ax1,ax2 = ax.flatten()
sns.distplot(train['bandgap_energy_ev'],bins=50,ax=ax1,color='b')
bandgap_energy_ev_log1 = np.log1p(train['bandgap_energy_ev'])
bandgap_energy_ev_log1 = pa.DataFrame({'log(1+ bandgap_energy_ev_log1)': bandgap_energy_ev_log1})
sns.distplot(bandgap_energy_ev_log1['log(1+ bandgap_energy_ev_log1)'],bins=50,ax=ax2,color='r')


# Grate!! in "gap_energy_ev" target we can see "skewed"!!

# In[ ]:



y1 = y.formation_energy_ev_natom
y1 = np.log1p(y.formation_energy_ev_natom)
y1 = pa.DataFrame(y1.values, columns=['formation_energy_ev_natom'])
y2 = y.bandgap_energy_ev
y2 = np.log1p(y2)
y2 = pa.DataFrame(y2.values, columns=['bandgap_Eev'])
y = pa.concat([y.formation_energy_ev_natom, y2], axis=1)


# **6. using train-test-split for crros validation**
# 
# in order to avoid overfitting, we used hold-out  cross validation!

# In[ ]:


X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y1,  train_size=0.8, test_size=0.2, shuffle=True, random_state=400000)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y2,  train_size=0.8, test_size=0.2, shuffle=True, random_state=400000)


# **7. using Linear Regression for eache "target":**

# In[ ]:


clf1 = LR().fit(X_train1, y_train1)
clf2 = LR().fit(X_train2, y_train2)
predict_test1 = clf1.predict(X_test1)
predict_test2 = clf2.predict(X_test2)
predict_test1 = np.array(predict_test1)
predict_test2 = np.array(predict_test2)
y_test1 = y_test1.values
y_test1 = np.array(y_test1)
y_test2 = y_test2.values
y_test2 = np.array(y_test2)


# 3. transform the target (for caluclate the RMS) into original units (without log(1+x)), using :
#     exp(x)-1:

# In[ ]:


y_org1 = np.exp(y_test1)-1
predic_org1 = np.exp(predict_test1)-1
y_org2 = np.exp(y_test2)-1
predic_org2 = np.exp(predict_test2)-1


# **8. Calculate the  RMSLE:**

# In[ ]:


def rmsle(y_true,y_pred):
    assert len(y_true) == len(y_pred)
    return np.square(np.log(y_pred + 1) - np.log(y_true + 1)).mean() ** 0.5
rmsle1 = rmsle(y_org1, predic_org1)
rmsle2 = rmsle(y_org2, predic_org2)

RMSLE = (float(rmsle1 + rmsle2) / 2)
print('RMSLE:' + str(RMSLE))


# we can se the relation between the true values and the predicted as linear fitting :

# In[ ]:


plt.scatter(y_org1, predic_org1)
plt.xlabel('True Values (formation_energy_ev_natom)')
plt.ylabel('Predictions (formation_energy_ev_natom)')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])


# In[ ]:


plt.scatter(y_org2, predic_org2)
plt.xlabel('True Values (bandgap_Eev)')
plt.ylabel('Predictions (bandgap_Eev)')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])


# In[ ]:




