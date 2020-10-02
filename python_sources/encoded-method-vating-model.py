#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import lab
import missingno
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_auc_score,roc_curve
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# load data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.head()


# In[ ]:


# miss value count
missingno.bar(train)
plt.show()


# In[ ]:


missingno.bar(test)
plt.show()


# In[ ]:


train.describe(include='all')


# In[ ]:


test.describe(include='all')


# In[ ]:


# as we see,drop feature that not need
train.drop(columns=['Ticket','Cabin'], axis=1, inplace=True)
test.drop(columns=['Ticket', 'Cabin'], axis=1, inplace=True)


# In[ ]:


# fill na
train.fillna(np.nan)
test.fillna(np.nan)
age_na_fill = train.Age.median()
train.Age.fillna(age_na_fill, inplace=True)
test.Age.fillna(age_na_fill, inplace=True)
train.Embarked.fillna('S', inplace=True)
test.Embarked.fillna('S', inplace=True)
test.Fare.fillna(test.Fare.mean(), inplace=True)


# In[ ]:


train['Title'] = train.Name.str.extract("([A-Za-z]+)\.", expand=True)
test['Title'] = test.Name.str.extract('([A-Za-z]+)\.', expand=True)


# In[ ]:


train.head()


# In[ ]:


train.Title.value_counts()


# In[ ]:


title_replacement = {'Dr':'Other', 'Rev':'Other', 'Mlle':'Other', 'Major':'Other', 'Col':'Other', 'Sir':'Other', 'Mme':'Other', 
                    'Countess':'Other','Capt':'Other','Don':'Other','Lady':'Other','Ms':'Other','Jonkheer':'Other','Dona':'Other'}
train.replace({'Title':title_replacement}, inplace=True)
test.replace({'Title':title_replacement}, inplace=True)


# In[ ]:


train.drop(columns=['Name'], inplace=True)
test.drop(columns=['Name'], inplace=True)


# In[ ]:


missingno.bar(train)
plt.show()


# In[ ]:


missingno.bar(test)
plt.show()


# In[ ]:


train.describe(include='all')


# In[ ]:


# now we encoded categorical feature like 'Pclass', 'Sex', 'Embarked', 'Title'
# first encoded method
# One Hot Encoding
# we can encoding feature use pd.get_dummies function like this
# ohe = pd.get_dummies(train, columns=['Pclass', 'Sex', 'Embarked', 'Title'])
# ohe.head()
ohe = OneHotEncoder()
df = pd.DataFrame()
test_df = pd.DataFrame()
for col in ['Pclass', 'Sex', 'Embarked', 'Title']:
    ohe.fit(train[col].values.reshape(-1, 1))
    s = [f'{col}_{i}' for i in ohe.categories_[0]]
    df = pd.concat([df, pd.DataFrame(ohe.transform(train[col].values.reshape(-1,1)).todense(),columns = s, dtype=np.int32)], axis=1)
    test_df = pd.concat([test_df, pd.DataFrame(ohe.transform(test[col].values.reshape(-1,1)).todense(),columns = s, dtype=np.int32)], axis=1)
new_train_data = pd.concat([train, df], axis=1)
new_test_data = pd.concat([test, test_df], axis=1)


# In[ ]:


# LabelEncoder
label = LabelEncoder()
df = train[['Sex', 'Embarked', 'Title']].apply(label.fit_transform, axis = 0)
df.columns = ['Sex_Label_Encoded', 'Embarked_Label_Encoded', 'Title_Label_Encoded']
df.head()


# In[ ]:


new_train_data = pd.concat([new_train_data, df], axis=1)
rule = {}
rule['label'] = {}
rule['label']['Sex'] = pd.DataFrame([['male', 1],['female', 0]], columns=['Sex', 'Sex_Label_Encoded'], dtype=np.int32)
rule['label']['Embarked'] = pd.DataFrame([['S', 2],['C',0],['Q',1]], columns=['Embarked', 'Embarked_Label_Encoded'], dtype=np.int32)
rule['label']['Title'] = pd.DataFrame([['Master', 0],['Miss', 1], ['Mr', 2], ['Mrs', 3], ['Other', 4]], columns=['Title', 'Title_Label_Encoded'], dtype=np.int32)


# In[ ]:


for col in rule['label']:
    tmp = rule['label'][col]
    new_test_data = pd.merge(new_test_data, tmp, on=col, how='left')


# In[ ]:


# freq Encoding
sample_data = new_train_data[['Pclass','Sex', 'Embarked', 'Title']]
rule = {'freq':{}}
for col in sample_data.columns:
    tmp = sample_data.groupby(by=col).size().reset_index()
    tmp.columns = [col, f'{col}_Freq_Encoded']
    rule['freq'][col] = tmp
    new_train_data = pd.merge(new_train_data, tmp, on=col, how='left')
    new_test_data = pd.merge(new_test_data, tmp, on=col, how='left')


# In[ ]:


# mean/median/max/min/sum encoding
# step 1: select a categorical feature
# step 2: select a continuous feature
# step 3: Group by the categorical variable and obtain the aggregated operate function over the numeric variable
sample_data = new_train_data[['Pclass','Sex', 'Embarked', 'Title', 'Fare']]
rule['mean'] = {}
for col in ['Pclass','Sex', 'Embarked', 'Title']:
    tmp = sample_data.groupby(by=col)['Fare'].mean().reset_index()
    tmp.columns = [col, f'{col}_Mean_Encoded']
    rule['mean'][col] = tmp
    new_train_data = pd.merge(new_train_data, tmp, on=col, how='left')
    new_test_data = pd.merge(new_test_data, tmp, on=col, how='left')


# In[ ]:


# rule store the mapping
from pprint import pprint
pprint(rule)


# In[ ]:


# clear data
new_train_data.drop(columns=['PassengerId','Sex', 'Embarked', 'Title'], axis=1, inplace=True)
new_test_data.drop(columns=['PassengerId','Sex', 'Embarked', 'Title'], axis=1, inplace=True)


# In[ ]:


new_train_data.shape, new_test_data.shape


# In[ ]:


# model train
rf = RandomForestClassifier(n_estimators=1000, max_depth=3, min_samples_leaf=10, random_state=0)
clf_1 = AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=0)
clf_2 = LogisticRegression(penalty='l1', max_iter=100, random_state=0)
vc = VotingClassifier(estimators=[('rf', rf),('ada', clf_1),('lr',clf_2)])
vc.fit(new_train_data[list(set(new_train_data.columns)-set(['Survived']))], new_train_data.Survived)


# In[ ]:


test_y = pd.read_csv('../input/gender_submission.csv')
pred = pd.DataFrame({'PassengerId':test_y.PassengerId,'Survived':vc.predict(new_test_data[list(set(new_train_data.columns)-set(['Survived']))])})
pred.to_csv('submission.csv', index=False)


# In[ ]:


fpr, tpr, thresholds = roc_curve(test_y.Survived, vc.predict(new_test_data[list(set(new_train_data.columns)-set(['Survived']))]))
print((tpr-fpr).max())


# In[ ]:


plt.plot(fpr, tpr, 'r:')
plt.show()


# In[ ]:


plt.plot(np.arange(len(tpr)), tpr, 'r', label='tpr')
plt.plot(np.arange(len(tpr)), fpr, 'b', label='fpr')
plt.legend()
plt.show()


# In[ ]:




