#!/usr/bin/env python
# coding: utf-8

# ## Load files and drop not used features

# In[ ]:


import pandas as pd
import numpy as np

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
test['Survived'] = np.nan
y = train['Survived']
PassId = test['PassengerId'].values
data=train.append(test,ignore_index=True,sort=False)
data.drop(['Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked','PassengerId'],axis=1,inplace=True)


# ## Some EDA with the two remainig features (Name and Ticket).
# From Name we will extract Family Name ans Title. Title grouped by sex and age.    
# From Ticket we will extract Tickect Cound..

# In[ ]:


data['Family_name']=data['Name'].str.split(', ').str[0]
data['Title']=data['Name'].str.split(', ').str[1].str.split('.').str[0]
data['Title'] = data['Title'].replace(['Ms','Mlle'], 'Miss')
data['Title'] = data['Title'].replace(['Mme','Dona','the Countess','Lady'], 'Mrs')
data['Title'] = data['Title'].replace(['Rev','Mlle','Jonkheer','Dr','Capt','Don','Col','Major','Sir'], 'Mr')


data['Ticket_count'] = data.Ticket.apply(lambda x: data[data['Ticket']==x].shape[0] )


# ## Time to overfit.
# Two new features based on tickets groups and Survided.

# In[ ]:


# Female and chil No survide.
list1=data[(data['Title']!='Mr') & (data['Survived']==0) ]['Ticket'].tolist()

# Man survive.
list2=data[(data['Title']=='Mr') & (data['Survived']==1) ]['Ticket'].tolist()

data['Ticket_wit_FC_dead']=0
data['Ticket_wit_M_alive']=0

data.loc[data['Ticket'].isin(list1),'Ticket_wit_FC_dead' ]=1
data.loc[data['Ticket'].isin(list2),'Ticket_wit_M_alive' ]=1


# ## Encode Categorical features.

# In[ ]:


import category_encoders as ce
from sklearn import preprocessing

data['Title_Encode']=np.nan
ce_target_encoder = ce.TargetEncoder(cols=['Title'], smoothing=0.3)
ce_target_encoder.fit(data[:len(y)],y)
data['Title_Encode']=ce_target_encoder.transform(data)['Title']

le = preprocessing.LabelEncoder()
data['Family_name_Encode']=le.fit_transform(data['Family_name'])
data['Ticket_Number_Encode']=le.fit_transform(data['Ticket'])


# In[ ]:


data.drop(['Survived','Name','Ticket','Title','Family_name'],axis=1,inplace=True)
train = data[:len(y)]
test = data[len(y):]


# ## Building the model, predictions and Submission File

# In[ ]:


from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='liblinear', max_iter=200)
model.fit(train, y)

pd.DataFrame({ 'PassengerId' : PassId, 'Survived': model.predict(test) }).to_csv('submission.csv', index=False)

