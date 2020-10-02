#!/usr/bin/env python
# coding: utf-8

# # Diagnosis of COVID-19 and its clinical spectrum
# > ## AI and Data Science supporting clinical decisions
# > #### Rodrigo Fragoso

# ##### The main objective on this notebook is to explore and find how to deal with the NaN values and implement a vanilla Classifier to understand how the balance between positive and negative results can affect our models.

# ## Importings

# In[ ]:


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm


# In[ ]:


data=pd.read_excel('/kaggle/input/covid19/dataset.xlsx')
data.head(7)


# ### __With a brief look, we can see that are lots of NaN Values on this dataset__

# In[ ]:


shape=data.shape
print(shape[1],'columns')
print(shape[0],'rows')


# #### Transforming the target variable in binary

# In[ ]:


def positive_bin(x):
    if x == 'positive':
        return 1
    else:
        return 0
data['SARS-Cov-2 exam result_bin']=data['SARS-Cov-2 exam result'].map(positive_bin)


# In[ ]:


tg_values=data['SARS-Cov-2 exam result'].value_counts()
tg_values.plot.barh(color='red')
print("Negative exam results: "+"{:.2%}".format(tg_values[0]/tg_values.sum())+' ('+str(tg_values[0])+' records)')
print("Positive exam results: "+"{:.2%}".format(tg_values[1]/tg_values.sum())+'  ('+str(tg_values[1])+' records)')
print('')


# ### We can see a very unbalanced dataset, maybe over or undersampling will be a good idea
# > #### Before that, Assuming all results as negative will be a good baseline

# In[ ]:


data['SARS-Cov-2 exam result_Baseline']=0
print("Baseline accuracy: "+"{:.2%}".format((data['SARS-Cov-2 exam result_Baseline']==data['SARS-Cov-2 exam result_bin']).sum()/len(data['SARS-Cov-2 exam result_Baseline'])))


# ## Exploring missing/NaN data 

# In[ ]:


if data.isnull().values.any() == True:
    print('Found NaN values!!:')
else:
    print('No NaN values =):')
print(' ')

nulls=(data.isnull().sum()/len(data))*100


# In[ ]:


nulls.sort_values(ascending=False)


# In[ ]:


ax=nulls.hist(bins=90, grid=False, figsize=(10,6), color='red')
ax.set_xlabel("% of Nulls")
ax.set_ylabel("Number of variables")
print('')


# In[ ]:


pos=data[data['SARS-Cov-2 exam result_bin']==1]
neg=data[data['SARS-Cov-2 exam result_bin']==0]


# In[ ]:


if pos.isnull().values.any() == True:
    print('Found NaN values!!:')
else:
    print('No NaN values =):')
print(' ')

nulls_pos=(pos.isnull().sum().sort_values(ascending=False)/len(pos))*100
nulls_pos


# In[ ]:


ax=nulls_pos.hist(bins=80, grid=False, figsize=(10,6), color='black')
ax.set_xlabel("% of Nulls")
ax.set_ylabel("Number of variables")
print('')


# In[ ]:


if neg.isnull().values.any() == True:
    print('Found NaN values!!:')
else:
    print('No NaN values =):')
print(' ')

nulls_neg=(neg.isnull().sum().sort_values(ascending=False)/len(neg))*100
nulls_neg


# In[ ]:


ax=nulls_neg.hist(bins=80, grid=False, figsize=(10,6), color='blue')
ax.set_xlabel("% of Nulls")
ax.set_ylabel("Number of variables")
print('')


# ### Conclusions from this session:
# - As we can see there a lot of missing records;
# - Most of the variables have at leats 80% of NaNs;
# - We are dealing with a very unbalanced dataset 9:1 negative/positive results.

# ## Feature Selection

# > #### Based on the histograms, I'm selecting features that have a maximum of 90% as missing values

# In[ ]:


nulls.drop(['SARS-Cov-2 exam result','Patient ID','SARS-Cov-2 exam result_bin','SARS-Cov-2 exam result_Baseline'],inplace=True)


# In[ ]:


selecting_variables=nulls.loc[nulls<90]
selecting_variables


# In[ ]:


variables=selecting_variables.index.tolist()
variables.append('Patient ID')
variables.append('SARS-Cov-2 exam result_bin')


# In[ ]:


df=data[variables]
df[df['Parainfluenza 2'].notnull()].head()


# > ####  Labeling categorical values

# In[ ]:


import warnings
warnings.filterwarnings("ignore")

def bins(x):
    if x == 'detected' or x=='positive':
        return 1
    elif x=='not_detected' or x=='negative':
        return 0
    else:
        return x
    
for col in df.columns:
    df[col]=df[col].apply(lambda row: bins(row))


# > #### Dealing with NaNs

# In[ ]:


pd.set_option('display.max_columns', None)
df.describe()


# > #### Variables with too long range, not good to substitute NaNs with 0
# > #### Using KNN imputer

# In[ ]:


variables_imputer=variables[4:18]
teste=df[variables_imputer]


# In[ ]:


from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5,missing_values=np.nan)
imputer.fit(teste)
teste[:]=imputer.transform(teste)


# In[ ]:


df.drop(variables_imputer,axis=1,inplace=True)


# > #### For categorical variables, NaN will be replaced with -1

# In[ ]:


data_final= pd.concat([teste,df],axis=1)
data_final.fillna(-1,inplace=True)
data_final.head()


# #### Finally, I'll be using this dataset

# In[ ]:


X=data_final.drop(['SARS-Cov-2 exam result_bin','Patient ID'],axis=1)
y=data_final['SARS-Cov-2 exam result_bin']


# ### Model Selection

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2,random_state=5)
print(X_train.shape, X_test.shape)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import log_loss, accuracy_score

resultados=[]
kf=RepeatedKFold(n_splits=10, n_repeats=1, random_state=5)
for train,valid in tqdm(kf.split(X_train)):
    
    Xtr, Xvld = X_train.iloc[train], X_train.iloc[valid]
    ytr, yvld = y_train.iloc[train], y_train.iloc[valid]
    
    rf= RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
    rf.fit(Xtr,ytr)
    
    p=rf.predict(Xvld)
    acc=accuracy_score(yvld,p)
    resultados.append(acc)


# In[ ]:


print("Vanilla RandomForest Train accuracy: "+"{:.2%}".format(np.mean(resultados)))


# In[ ]:


from sklearn.linear_model import LogisticRegression

resultados=[]
kf=RepeatedKFold(n_splits=10, n_repeats=1, random_state=5)
for train,valid in tqdm(kf.split(X_train)):
    
    Xtr, Xvld = X_train.iloc[train], X_train.iloc[valid]
    ytr, yvld = y_train.iloc[train], y_train.iloc[valid]
    
    lr= LogisticRegression(max_iter=300)
    lr.fit(Xtr,ytr)
    
    p=lr.predict(Xvld)
    acc=accuracy_score(yvld,p)
    resultados.append(acc)


# In[ ]:


print("Vanilla Logistic Regression Train accuracy: "+"{:.2%}".format(np.mean(resultados)))


# In[ ]:


p2=rf.predict(X_test)
p2[:]=rf.predict(X_test)
acc=accuracy_score(y_test,p2)
print("Vanilla RandomForest Test accuracy: "+"{:.2%}".format(acc))
p2=lr.predict(X_test)
p2[:]=lr.predict(X_test)
acc=accuracy_score(y_test,p2)
print("Vanilla Logistic Regression Test accuracy: "+"{:.2%}".format(acc))


# In[ ]:


visual=pd.concat([X_test,y_test],axis=1)
visual['predict']=p2
visual2=visual[visual['SARS-Cov-2 exam result_bin']==visual['predict']]

print('Positive results in the test sample: ',visual[visual['SARS-Cov-2 exam result_bin']==1].shape[0])
print('Positive results correctly predicted: ',visual2[visual2['predict']==1].shape[0])
print('Positive accuracy: ',"{:.2%}".format(visual2[visual2['predict']==1].shape[0]/visual[visual['SARS-Cov-2 exam result_bin']==1].shape[0]))


# #### We can see low accuracy on positive results

# In[ ]:


sns.set(font_scale=2)


# In[ ]:


pred_prob=lr.predict_proba(X_test)

from sklearn.metrics import precision_recall_curve

scores=pred_prob[:,1]
precision, recall, thresholds = precision_recall_curve(y_test, scores)
fig, ax = plt.subplots(figsize=(10,7))
plt.plot(recall[:-1],precision[:-1],label="logistic regeression",color='red')
plt.legend(loc="center right")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()


# In[ ]:


from sklearn.metrics import roc_curve

scores=pred_prob[:,1]
fpr, tpr, thresholds = roc_curve(y_test,scores)
fig, ax = plt.subplots(figsize=(10,7))
plt.plot(fpr,tpr,color='red')
plt.xlabel("False Positive Rate",size=15)
plt.ylabel("True Positive Rate",size=15)

plt.show()


# ### Conclusions from this session:
# - In spite of having a "high" accuracy, we are only matching on negative exam results, the 7.55% for positive results accuracy explains it;
# - Probably,this is happening because the data is too unbalanced;
# - Looking at the precision/recall curve, we can see that changing the threshold won't bring better results.

# ### Next steps:
# - Test another imputer methods for NaNs;
# - Test another classifiers;
# - Do some over and undersampling to improve our classifiers;
# - Do some hyperparameter tuning

# In[ ]:




