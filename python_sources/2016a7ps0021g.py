#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


data_tr = pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/train.csv", sep=',')
data_te = pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/test.csv", sep=',')


# In[ ]:


data_tr.head()


# In[ ]:


data_te.head()


# In[ ]:


data_tr= pd.get_dummies(data_tr, prefix='type', columns=['type'])
data_te= pd.get_dummies(data_te,prefix='type', columns=['type'])


# In[ ]:


#print(data_tr.replace(r'^\s*$', np.nan, regex=True))
#data_tr.replace('', np.nan, inplace=True)
data_tr=data_tr.fillna(data_tr.mean())


# In[ ]:


data_te=data_te.fillna(data_te.mean())


# In[ ]:


idd=data_te['id']


# In[ ]:


import seaborn as sns
f, ax = plt.subplots(figsize=(10, 8))
corr = data_tr.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax, annot = True);


# In[ ]:


data_tr=data_tr.drop(['id'],1)
data_tr=data_tr.drop(['type_new'],1)
#data_tr=data_tr.drop(['feature4'],1)


# In[ ]:


data_te=data_te.drop(['id'],1)
data_te=data_te.drop(['type_new'],1)
#data_te=data_te.drop(['feature4'],1)


# In[ ]:


y=data_tr['rating']


# In[ ]:


data_tr=data_tr.drop(['rating'],1)
data_tes=data_te


# In[ ]:


column= list(data_tr.columns)
#Preprocess
#columns
column.remove('type_old')


# In[ ]:


x=data_tr


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.20, random_state=42,stratify=y)


# In[ ]:


X_train.head()


# # ExtraTree

# In[ ]:


from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor


# In[ ]:


score_train_RF = []
score_test_RF = []

for i in range(1,18,1):
    rf = ExtraTreesRegressor(n_estimators=i, random_state = 42)
    rf.fit(X_train, y_train)
    sc_train = rf.score(X_train,y_train)
    score_train_RF.append(sc_train)
    sc_test = rf.score(X_val,y_val)
    score_test_RF.append(sc_test)


# In[ ]:


param_grid = { 
    'n_estimators': [1000,2000,3000,4000],
    'max_depth' : [20,30,40,50],
}


# In[ ]:


#GRID_SEARCH
#from sklearn.model_selection import GridSearchCV
#CV = GridSearchCV(estimator=rf, param_grid=param_grid, cv= 5, verbose=20,n_jobs=-1)
#CV.fit(x,y)


# In[ ]:


#CV.best_params_


# In[ ]:


rf=ExtraTreesRegressor(random_state=42, n_estimators= 2500, max_depth=27, max_features='auto')


# In[ ]:


rf.fit(x,y)


# In[ ]:


predy = rf.predict(data_tes)
predy


# In[ ]:


data_te.shape


# In[ ]:


dframe = pd.DataFrame(predy)


# In[ ]:


dframe=dframe.round(0)


# In[ ]:


dframe[0].value_counts()


# In[ ]:


dff=pd.DataFrame(idd)


# In[ ]:


final_data=dff.join(dframe,how='left')
final_data= final_data.rename(columns={0: "rating"})
final_data['rating']=final_data['rating'].apply(int)


# In[ ]:


final_data.head()


# In[ ]:


final_data['rating'].value_counts()


# In[ ]:


final_data.to_csv('ML_DhruvKhetarpal.csv', index = False)


# In[ ]:




