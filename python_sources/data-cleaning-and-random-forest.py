#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## This Python 3 environment comes with many helpful analytics libraries installed
## It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
## For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

## Input data files are available in the "../input/" directory.
## For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

## Any results you write to the current directory are saved as output.


# In[ ]:


data=pd.read_csv('../input/train.csv')


# In[ ]:


data.info()


# The data have 9557 entries, each entry has 143 columns.
# 
# Most of the data are floats and integers, a few objects. Let's take a look at the objects.

# In[ ]:


data.columns[data.dtypes==object]


# * Id,  idhogar - no problem, they are just identifications 
# * dependency - dependency rate 
# * edjefe, edjefa - years of education of head of household

# Let's look at the dependency rate.

# In[ ]:


data['dependency'].unique()


# Lots of numbers as string, plus 'yes' and 'no'. However, we have a column containing the square values if the dependency, 'SQBdependency'. Maybe that can help us.

# In[ ]:


data[(data['dependency']=='no') & (data['SQBdependency']!=0)]


# So the "square" of no is 0.

# In[ ]:


data[(data['dependency']=='yes') & (data['SQBdependency']!=1)]


# In[ ]:


data[(data['dependency']=='3') & (data['SQBdependency']!=9)]


# Seems like we can just derive the dependency from the SQBdependency.

# In[ ]:


data['dependency']=np.sqrt(data['SQBdependency'])


# Now let's  look at the jefe/jefa education.

# In[ ]:


data['edjefe'].unique()


# In[ ]:


data['edjefa'].unique()


# Again, numbers, 'yes' and 'no'. Here there's also 'SQBedjefe', yay!

# In[ ]:


data['SQBedjefe'].unique()


# In[ ]:


data[['edjefe', 'edjefa', 'SQBedjefe']][:20]


# Hmmm, 'SQBedjefe is just the square of 'edjefe', it's 0 if the head of the household is a woman.

# In[ ]:


data[['edjefe', 'edjefa', 'SQBedjefe']][data['edjefe']=='yes']


# In[ ]:


data[(data['edjefe']=='yes') & (data['edjefa']!='no')]


# In[ ]:


data[(data['edjefa']=='yes') & (data['parentesco1']==1)][['edjefe', 'edjefa', 'parentesco1', 'escolari']]


# In[ ]:


data[data['edjefe']=='yes'][['edjefe', 'edjefa','age', 'escolari', 'parentesco1','male', 'female', 'idhogar']]


# In[ ]:


data[(data['edjefe']=='no') & (data['edjefa']=='no')][['edjefe', 'edjefa', 'age', 'escolari', 'female', 'male', 'Id', 'parentesco1', 'idhogar']]


# In[ ]:


data[(data['edjefe']=='yes') & data['parentesco1']==1][['escolari']]


# Basically:
# * 'edjefe' and 'edjefa' are both 'no' when the head of the household had 0 years of school
# * there's 'edjefe'= 'yes' and 'edjefa'='no' in some cases, all these cases the head of the household had 1 year of school
# * there's 'edjefe'= 'no' and 'edjefa'='yes' in some cases, all these cases the head of the household had 1 year of school
# * most of the time either 'edjefe' or 'edjefa' is a number while the other is a 'no'
# 
# Let's merge the jefe and jefa education into one, undependent of gender

# In[ ]:


conditions = [
    (data['edjefe']=='no') & (data['edjefa']=='no'), #both no
    (data['edjefe']=='yes') & (data['edjefa']=='no'), # yes and no
    (data['edjefe']=='no') & (data['edjefa']=='yes'), #no and yes 
    (data['edjefe']!='no') & (data['edjefe']!='yes') & (data['edjefa']=='no'), # number and no
    (data['edjefe']=='no') & (data['edjefa']!='no') # no and number
]
choices = [0, 1, 1, data['edjefe'], data['edjefa']]
data['edjefx']=np.select(conditions, choices)
data['edjefx']=data['edjefx'].astype(int)
data[['edjefe', 'edjefa', 'edjefx']][:15]


# In[ ]:


data.describe()


# Let's figure out if there are missing values.

# In[ ]:


data.columns[data.isna().sum()!=0]


# Columns with nans:
# * v2a1 - monthly rent
# * v18q1 - number of tablets
# * rez_esc - years behind school
# * meaneduc - mean education for adults
# * SQBmeaned - square of meaned  

# 'meaneduc' and 'SQBmeaned' are related, let's start with those.

# In[ ]:


data[data['meaneduc'].isnull()]


# Not a lot of rows

# In[ ]:


data[data['meaneduc'].isnull()][['Id','idhogar','edjefe','edjefa', 'hogar_adul', 'hogar_mayor', 'hogar_nin', 'age', 'escolari']]


# In[ ]:


print(len(data[data['idhogar']==data.iloc[1291]['idhogar']]))
print(len(data[data['idhogar']==data.iloc[1840]['idhogar']]))
print(len(data[data['idhogar']==data.iloc[2049]['idhogar']]))


# So, the 5 rows with Nan for 'meaneduc' is just 3 households, where 18-19 year-olds live.  No other people live in these households. Then we can just take the education levels of these kids ('escolari') and put them into 'meaneduc' and 'SQBmeaned'.

# In[ ]:


meaneduc_nan=data[data['meaneduc'].isnull()][['Id','idhogar','escolari']]


# In[ ]:


me=meaneduc_nan.groupby('idhogar')['escolari'].mean().reset_index()


# In[ ]:


me


# In[ ]:


for row in meaneduc_nan.iterrows():
    idx=row[0]
    idhogar=row[1]['idhogar']
    m=me[me['idhogar']==idhogar]['escolari'].tolist()[0]
    data.at[idx, 'meaneduc']=m
    data.at[idx, 'SQBmeaned']=m*m
    


# 

# Next, let's look at 'v2a1', the monthly rent payment, that also has missing values.

# In[ ]:


data['v2a1'].isnull().sum()


# That's a lot of missing values.
# 
# But maybe they own their houses. We can look what type of housing these people with missing values live..

# In[ ]:


norent=data[data['v2a1'].isnull()]
print("Owns his house:", norent[norent['tipovivi1']==1]['Id'].count())
print("Owns his house paying installments", norent[norent['tipovivi2']==1]['Id'].count())
print("Rented ", norent[norent['tipovivi3']==1]['Id'].count())
print("Precarious ", norent[norent['tipovivi4']==1]['Id'].count())
print("Other ", norent[norent['tipovivi5']==1]['Id'].count())
print("Total ", 6860)


# The majority in fact owns their houses, only a few have odd situations. We can probably just assume they don't pay rent, and put 0 in these cases.

# In[ ]:


data['v2a1']=data['v2a1'].fillna(0)


# Now, let's look at 'v18q1', which indicates how many tablets the household owns.

# In[ ]:


data['v18q1'].isna().sum()


# That's also a lot rows with missing values... However, there's a column, 'v18q', which indicates whether there's a tablet in the household at all, that might help!

# In[ ]:


tabletnan=data[data['v18q1'].isnull()]
tabletnan[tabletnan['v18q']==0]['Id'].count()


# In[ ]:


data['v18q1'].unique()


# That's exactly the number of rows with missing values! There's also no 0 among the values of 'v18q1'. So all the nans in 'v18q1' just means they don't own a tablet! So we can just change them to 0.

# In[ ]:


data['v18q1']=data['v18q1'].fillna(0)


# Next up is 'rez_esc', which indicates if a person is behind in school.

# In[ ]:


data['rez_esc'].isnull().sum()


# That's also a crazy lot of rows..

# In[ ]:


data['rez_esc'].describe()


# In[ ]:


data['rez_esc'].unique()


# In[ ]:


data[data['rez_esc']>1][['age', 'escolari', 'rez_esc']][:20]


# Hmmm, these are all schoolchildren...

# In[ ]:


rez_esc_nan=data[data['rez_esc'].isnull()]
rez_esc_nan[(rez_esc_nan['age']<18) & rez_esc_nan['escolari']>0][['age', 'escolari']]


# So all the nans here are either adults or children before school age. We can input  0 again.

# In[ ]:


data['rez_esc']=data['rez_esc'].fillna(0)


# Someone commented in the discussions that the same household can have different target values. Let's look at it.

# In[ ]:


d={}
weird=[]
for row in data.iterrows():
    idhogar=row[1]['idhogar']
    target=row[1]['Target']
    if idhogar in d:
        if d[idhogar]!=target:
            weird.append(idhogar)
    else:
        d[idhogar]=target


# In[ ]:


len(set(weird))


# There are 85 households like that.

# In[ ]:


data[data['idhogar']==weird[2]][['idhogar','parentesco1', 'Target']]


# In the discussion we were told that the correct target value is the one belonging to the head of the household. So we should set the correct value each time.

# In[ ]:


for i in set(weird):
    hhold=data[data['idhogar']==i][['idhogar', 'parentesco1', 'Target']]
    target=hhold[hhold['parentesco1']==1]['Target'].tolist()[0]
    for row in hhold.iterrows():
        idx=row[0]
        if row[1]['parentesco1']!=1:
            data.at[idx, 'Target']=target
    


# In[ ]:


data[data['idhogar']==weird[1]][['idhogar','parentesco1', 'Target']]


# In[ ]:


def data_cleaning(data):
    data['dependency']=np.sqrt(data['SQBdependency'])
    data['rez_esc']=data['rez_esc'].fillna(0)
    data['v18q1']=data['v18q1'].fillna(0)
    data['v2a1']=data['v2a1'].fillna(0)
    
    conditions = [
    (data['edjefe']=='no') & (data['edjefa']=='no'), #both no
    (data['edjefe']=='yes') & (data['edjefa']=='no'), # yes and no
    (data['edjefe']=='no') & (data['edjefa']=='yes'), #no and yes 
    (data['edjefe']!='no') & (data['edjefe']!='yes') & (data['edjefa']=='no'), # number and no
    (data['edjefe']=='no') & (data['edjefa']!='no') # no and number
    ]
    choices = [0, 1, 1, data['edjefe'], data['edjefa']]
    data['edjefx']=np.select(conditions, choices)
    data['edjefx']=data['edjefx'].astype(int)
    data.drop(['edjefe', 'edjefa'], axis=1, inplace=True)
    
    meaneduc_nan=data[data['meaneduc'].isnull()][['Id','idhogar','escolari']]
    me=meaneduc_nan.groupby('idhogar')['escolari'].mean().reset_index()
    for row in meaneduc_nan.iterrows():
        idx=row[0]
        idhogar=row[1]['idhogar']
        m=me[me['idhogar']==idhogar]['escolari'].tolist()[0]
        data.at[idx, 'meaneduc']=m
        data.at[idx, 'SQBmeaned']=m*m
        
    return data


# ### Ploting

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


data['Target'].hist()


# In[ ]:


data_undersampled=data.drop(data.query('Target == 4').sample(frac=.75).index)


# In[ ]:


data_undersampled['Target'].hist()


# ### Random forest

# In[ ]:


X=data_undersampled.drop(['Id', 'idhogar', 'Target', 'edjefe', 'edjefa'], axis=1)
y=data_undersampled['Target']


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[ ]:


X_train.shape


# In[ ]:


y_train.shape


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# In[ ]:


clf = RandomForestClassifier()
params={'n_estimators': list(range(40,61, 1))}
gs = GridSearchCV(clf, params, cv=5)


# In[ ]:


gs.fit(X_train, y_train)


# In[ ]:


preds=gs.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test, preds))


# In[ ]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, preds))


# In[ ]:


print(gs.best_params_)
print(gs.best_score_)
print(gs.best_estimator_)


# In[ ]:


cvres = gs.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(mean_score), params)


# In[ ]:


test_data=pd.read_csv('../input/test.csv')


# In[ ]:


test_data=data_cleaning(test_data)


# In[ ]:


ids=test_data['Id']
test_data.drop(['Id', 'idhogar'], axis=1, inplace=True)


# In[ ]:


test_predictions=gs.predict(test_data)


# In[ ]:


test_predictions[:5]


# In[ ]:


submit=pd.DataFrame({'Id': ids, 'Target': test_predictions})


# In[ ]:


submit.to_csv('submit.csv', index=False)


# In[ ]:




