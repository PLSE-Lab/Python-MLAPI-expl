#!/usr/bin/env python
# coding: utf-8

# I am using only 5% train data set. I have not find out any missing values and duplicate rows (All Columns). There are duplicate values in single column. 
# 
# 
# I am just start yet kaggle compitition so i don't wish to win this compition but if you like this karnel please up vote me or any suggestion please comment me also notifiy my misstake.
# 
# ### This kernel is not completed !
# 
# ## Workflow stages
# <ol>
#     <li>Visualize, Explore the data </li>
#     <li> Model, predict </li>
#     <li> Supply or submit the results.</li>
# <ol>
# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
lbl = LabelEncoder()
color = sns.color_palette()
sns.set_style('darkgrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


path = '../input/champs-scalar-coupling/'


# In[ ]:


train = pd.read_csv(path+'/train.csv')
test = pd.read_csv(path+'test.csv')


# In[ ]:


train.head()


# In[ ]:


train.shape


# In[ ]:


plt.figure(figsize=(8,6))
sns.countplot(train['type'])
plt.show()


# In[ ]:


plt.figure(figsize=(8,6))
sns.countplot(train['atom_index_0'])
plt.show()


# In[ ]:


plt.figure(figsize=(8,6))
sns.countplot(train['atom_index_1'])
plt.show()


# In[ ]:


train = train.sample(frac=0.09, random_state=5)


# In[ ]:


test = pd.read_csv(path+'/test.csv')


# In[ ]:


potential_energy = pd.read_csv(path+'/potential_energy.csv')


# In[ ]:


train.head(5)


# In[ ]:


pd.isnull(train).sum()


# In[ ]:


train['type'].unique()


# In[ ]:


train[['type','scalar_coupling_constant']].groupby(['type'], as_index=False ).mean().sort_values(by='scalar_coupling_constant', ascending=True)


# In[ ]:


train[['atom_index_0','scalar_coupling_constant']].groupby(['atom_index_0'], as_index=False ).mean().sort_values(by='scalar_coupling_constant', ascending=True)


# In[ ]:


train[['atom_index_1','scalar_coupling_constant']].groupby(['atom_index_1'], as_index=False ).mean().sort_values(by='scalar_coupling_constant', ascending=True)


# In[ ]:


for col in train['type'].unique():
    sns.distplot(train[train['type'] == col]['scalar_coupling_constant'])
    plt.show()
    #print(col)


# In[ ]:


potential_energy.head(5)


# In[ ]:


potential_energy.isnull().sum()


# In[ ]:


train  = pd.merge(train, potential_energy, how='left', on='molecule_name', right_index=False)
test  = pd.merge(test, potential_energy, how='left', on='molecule_name', right_index=False )


# In[ ]:


#test  = pd.merge(test, potential_energy, how='left', on='molecule_name')


# In[ ]:


train.head(5)


# In[ ]:


train = train[['id','molecule_name','atom_index_0','atom_index_1','type','potential_energy','scalar_coupling_constant']]


# In[ ]:


structures = pd.read_csv(path+'/structures.csv')


# In[ ]:


train.columns


# In[ ]:


train['atom1'] = train['type'].str[2]
test['atom1'] = test['type'].str[2]


# In[ ]:


train['atom2'] = train['type'].str[3]
test['atom2'] = test['type'].str[3]


# In[ ]:


train['coupling_type'] = train['type'].str[0:2]
test['coupling_type'] = test['type'].str[0:2]


# In[ ]:


train.isnull().sum()


# In[ ]:


structures.rename(columns={'x':'x1','y':'y1','z':'z1'}, inplace=True)
structures.head()


# In[ ]:


train = pd.merge(train, structures, how ='left', left_on=['molecule_name', 'atom_index_0','atom1'], right_on=['molecule_name', 'atom_index','atom'], right_index=False)
test = pd.merge(test, structures, how ='left', left_on=['molecule_name', 'atom_index_0','atom1'], right_on=['molecule_name', 'atom_index','atom'], right_index=False)


# In[ ]:


structures.rename(columns={'x1':'x2','y1':'y2','z1':'z2'}, inplace=True)


# In[ ]:


train = pd.merge(train, structures, how ='left', left_on=['molecule_name', 'atom_index_1','atom2'], right_on=['molecule_name', 'atom_index','atom'], right_index=False)
test = pd.merge(test, structures, how ='left', left_on=['molecule_name', 'atom_index_1','atom2'], right_on=['molecule_name', 'atom_index','atom'], right_index=False)


# In[ ]:


train['x2-x1'] = train['x2']-train['x1']
train['y2-y1'] = train['y2']-train['y1']
train['z2-z1'] = train['z2']-train['z1']

test['x2-x1'] = test['x2']-test['x1']
test['y2-y1'] = test['y2']-test['y1']
test['z2-z1'] = test['z2']-test['z1']


# In[ ]:


train['pow(x2-x1)'] = train['x2-x1']**2
train['pow(y2-y1)'] = train['y2-y1']**2
train['pow(z2-z1)'] = train['z2-z1']**2

test['pow(x2-x1)'] = test['x2-x1']**2
test['pow(y2-y1)'] = test['y2-y1']**2
test['pow(z2-z1)'] = test['z2-z1']**2


# In[ ]:


train['pow(x2-x1)+pow(y2-y1)+pow(z2-z1)'] = train['pow(x2-x1)']+train['pow(y2-y1)']+train['pow(z2-z1)']

test['pow(x2-x1)+pow(y2-y1)+pow(z2-z1)'] = test['pow(x2-x1)']+test['pow(y2-y1)']+test['pow(z2-z1)']


# In[ ]:


train['distance'] = np.sqrt(train['pow(x2-x1)+pow(y2-y1)+pow(z2-z1)'])

test['distance'] = np.sqrt(test['pow(x2-x1)+pow(y2-y1)+pow(z2-z1)'])


# In[ ]:


train.columns


# In[ ]:


print(os.listdir(path))


# In[ ]:


mulliken_charges = pd.read_csv(path+'/mulliken_charges.csv')


# In[ ]:


mulliken_charges.head()


# In[ ]:


train = pd.merge(train, mulliken_charges, how='left', left_on=['molecule_name', 'atom_index_0'], right_on=['molecule_name','atom_index'], right_index=False)

test = pd.merge(test, mulliken_charges, how='left', left_on=['molecule_name', 'atom_index_0'], right_on=['molecule_name','atom_index'], right_index=False)


# In[ ]:


train.rename(columns ={'mulliken_charge':'mulliken_charge_0'}, inplace=True)


# In[ ]:


train = pd.merge(train, mulliken_charges, how='left', left_on=['molecule_name', 'atom_index_1'], right_on=['molecule_name','atom_index'], right_index=False)

test = pd.merge(test, mulliken_charges, how='left', left_on=['molecule_name', 'atom_index_1'], right_on=['molecule_name','atom_index'], right_index=False)


# In[ ]:


train.rename(columns ={'mulliken_charge':'mulliken_charge_1'}, inplace=True)


# In[ ]:


sc_contributions = pd.read_csv(path+'/scalar_coupling_contributions.csv')


# In[ ]:


train = pd.merge(train, sc_contributions, how='left', left_on=['molecule_name', 'atom_index_0', 'atom_index_1', 'type'], right_on=['molecule_name', 'atom_index_0', 'atom_index_1', 'type'], right_index=False)

test = pd.merge(test, sc_contributions, how='left', left_on=['molecule_name', 'atom_index_0', 'atom_index_1', 'type'], right_on=['molecule_name', 'atom_index_0', 'atom_index_1', 'type'], right_index=False)


# In[ ]:


dipole_moments = pd.read_csv(path+'/dipole_moments.csv')


# In[ ]:


dipole_moments.head()


# In[ ]:


train = pd.merge(train, dipole_moments, how='left', left_on=['molecule_name'], right_on=['molecule_name'], right_index=False)

test = pd.merge(test, dipole_moments, how='left', left_on=['molecule_name'], right_on=['molecule_name'], right_index=False)


# In[ ]:


train.head()


# In[ ]:


ms_tensors = pd.read_csv(path+'/magnetic_shielding_tensors.csv')


# In[ ]:


ms_tensors.head()


# In[ ]:


train[['atom1','scalar_coupling_constant']].groupby(['atom1'], as_index=False ).mean().sort_values(by='scalar_coupling_constant', ascending=True)


# In[ ]:


train[['atom2','scalar_coupling_constant']].groupby(['atom2'], as_index=False ).mean().sort_values(by='scalar_coupling_constant', ascending=True)


# In[ ]:


train[['coupling_type','scalar_coupling_constant']].groupby(['coupling_type'], as_index=False ).mean().sort_values(by='scalar_coupling_constant', ascending=True)


# In[ ]:


train1 = train.copy()


# In[ ]:


train1['atom1'] = lbl.fit_transform(train1['atom1'])

test['atom1'] = lbl.fit_transform(test['atom1'])


# In[ ]:


train1['atom2'] = lbl.fit_transform(train1['atom2'])

test['atom2'] = lbl.fit_transform(test['atom2'])


# In[ ]:


train1['coupling_type'] = lbl.fit_transform(train1['coupling_type'])

test['coupling_type'] = lbl.fit_transform(test['coupling_type'])


# In[ ]:


train1['potential_energy'] = lbl.fit_transform(train1['potential_energy'])

test['potential_energy'] = lbl.fit_transform(test['potential_energy'])


# In[ ]:


#train1['scalar_coupling_constant'] = lbl.fit_transform(train1['scalar_coupling_constant'])


# In[ ]:


train1.head()


# In[ ]:


X = np.array(train1[['atom_index_0', 'atom_index_1', 'atom1','atom2', 'coupling_type', 'potential_energy']])
Y = np.array(train1['scalar_coupling_constant'])


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.999900, random_state=52)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb


# In[ ]:


model_name = []
model_score =[]


# In[ ]:


'''linsvc = LinearSVC()
linsvc.fit(X_train,Y_train)
linsvc_score = round(linsvc.score(X_train,Y_train)*100, 2)
model_name.append('LinearSVC')
model_score.append(linsvc_score)
linsvc_score'''


# In[ ]:


'''svc = SVC()
svc.fit(X_train,Y_train)
svc_score = round(svc.score(X_train,Y_train)*100, 2)
model_name.append('SVC')
model_score.append(svc_score)
svc_score'''


# In[ ]:


'''kneighbors = KNeighborsClassifier()
kneighbors.fit(X_train,Y_train)
kneighbors_score = round(kneighbors.score(X_train,Y_train)*100, 2)
model_name.append('KNeighborsClassifier')
model_score.append(kneighbors_score)
kneighbors_score'''


# In[ ]:


randomforest = RandomForestRegressor()
randomforest.fit(X_train,Y_train)
randomforest_score = round(randomforest.score(X_train,Y_train)*100, 2)
model_name.append('RandomForestRegressor')
model_score.append(randomforest_score)
randomforest_score


# In[ ]:


gradient = GradientBoostingRegressor()
gradient.fit(X_train,Y_train)
gradient_score = round(gradient.score(X_train,Y_train)*100, 2)
model_name.append('GradientBoostingRegressor')
model_score.append(gradient_score)
gradient_score


# In[ ]:


all_score = pd.DataFrame({'model_name':model_name, 'model_score':model_score})
all_score


# In[ ]:


selected_col = ['atom_index_0', 'atom_index_1', 'atom1','atom2', 'coupling_type', 'potential_energy']

predict_result = randomforest.predict(test[selected_col])


# In[ ]:


submission = pd.DataFrame({'id':test['id'], 'scalar_coupling_constant':predict_result})
submission.to_csv('my_submission.csv', index=False)

