#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings("ignore")
import os
print(os.listdir("../input"))


# In[ ]:


train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
gen_sub = pd.read_csv("../input/gender_submission.csv")


# 

# In[ ]:





# In[ ]:


train_data.shape , test_data.shape , gen_sub.shape


# In[ ]:


train = pd.concat([train_data,test_data],axis = 0).reset_index(drop = True)


# In[ ]:


train.head()


# In[ ]:


train.sample(frac = 0.01)


# In[ ]:


fig,ax = plt.subplots(figsize = (6,4))
sns.barplot(x = train['Sex'].value_counts(normalize = True).index , y = train['Sex'].value_counts(normalize  = True).values)
plt.title("Distribution of Sex")
plt.xlabel("Gender")
plt.ylabel("Counts")
print(train['Sex'].value_counts(normalize = True))


# In[ ]:


print(train['Pclass'].value_counts(normalize = True))
fig,ax = plt.subplots(figsize = (6,4))
sns.barplot(x =train['Pclass'].value_counts(normalize = True).index , y = train['Pclass'].value_counts(normalize = True).values)
plt.title("Distribution of Class in Ship")
plt.xlabel("Classes")
plt.ylabel("Counts")
sns.despine()


# In[ ]:


def missing_values(df):
    columns = list(df.columns)
    count = []
    percent = []
    column = []
    for col in columns:
        count.append(df[col].isnull().sum())
        percent.append((df[col].isnull().sum())/ df[col].isnull().count())
        column.append(col)
    data = pd.DataFrame(data = {'Columns':column,'missing_value':count,'percentage':percent})
    return(data)
        


# In[ ]:


df = missing_values(train) 
df


# In[ ]:


fig,ax = plt.subplots(figsize = (6,4))
sns.barplot(x = train['Survived'].value_counts().index , y = train['Survived'].value_counts().values)
plt.title("Survived or not")
#plt.legend()
print(train['Survived'].value_counts(normalize = True))
plt.show()


# In[ ]:


g = sns.factorplot(x = 'SibSp',y = 'Survived',hue = 'Pclass',data = train ,kind = 'bar')
g.set_ylabels("Survival Probability")


# In[ ]:


sns.heatmap(train[["Survived","SibSp","Parch","Age","Fare"]].corr(),annot=True, fmt = ".2f", cmap = "YlGnBu")


# In[ ]:


ax = sns.kdeplot(train['Age'][(train['Survived']==0)&(train['Age'].notnull())] , color = 'r',shade = True,label = 'Not Survived')
ax = sns.kdeplot(train['Age'][(train['Survived']==1)&(train['Age'].notnull())] , color = 'b',shade = True,label = 'Survived')
plt.title("Age vs Survival")
plt.xlabel("Age")
plt.ylabel("Survival Probability")
sns.despine()


# In[ ]:


sns.barplot(x = 'Sex' , y = 'Survived',data = train) # it discernible that being male reduces our chances of survival than female


# In[ ]:


sns.barplot(x = 'Pclass' , y = 'Survived' ,hue = 'Sex' ,  data = train,palette = 'Set1')


# In[ ]:


train.head()


# In[ ]:


sns.barplot(train['Parch'].value_counts().index , train['Parch'].value_counts(normalize = True).values)
plt.xlabel("Parch")
plt.ylabel("counts")
sns.despine()


# In[ ]:


sns.barplot(x = "Parch" , y = "Survived",data = train , palette = 'Set1')


# In[ ]:


sns.factorplot("Pclass", col="Embarked",  data=train,size=6, kind="count", palette="muted")
sns.despine()


# In[ ]:


sns.barplot(x = train['Embarked'].value_counts(normalize = True).index , y = train['Embarked'].value_counts(normalize = True).values)
plt.title("Embarked distribution")
plt.xlabel("Embarked")
plt.ylabel("Frequency")
sns.despine()


# In[ ]:


sns.factorplot(x = "Embarked" , y = "Survived" ,data = train ,kind = 'bar', palette = 'Set1')
sns.despine()


# In[ ]:


train.head(2)


# > **Filling the Missing Values **

# In[ ]:


age_nan_index = list(train['Age'][train['Age'].isnull()].index)
for i in age_nan_index:
    age_med = train['Age'].median()
    age_pred = train['Age'][((train['SibSp']==train.iloc[i]['SibSp'])&(train['Pclass']==train.iloc[i]['Pclass']) 
                             & (train['Parch']==train.iloc[i]['Parch']))].median()
    if not np.isnan(age_pred):
        train['Age'].iloc[i] = age_pred
    else:
        train['Age'].iloc[i] = age_med


# In[ ]:


sns.factorplot(x = 'Survived',y = 'Age',data = train , kind = 'box',palette = 'BuGn')
sns.factorplot(x = 'Survived',y = 'Age',data = train , kind = 'violin',palette = 'Set1')


# In[ ]:


train.head(5)


# ****Feature Engineering****

# In[ ]:


train['Name'].sample(5)


# In[ ]:


train['Name'][0].split(',')[1].split('.')[0] # to check 


# In[ ]:


train['Title'] = [i.split(',')[1].split('.')[0].strip() for i in train['Name']]


# In[ ]:


sns.barplot(x = train['Title'].value_counts().index , y = train['Title'].value_counts(normalize = False).values)
plt.xticks(rotation = 45)
sns.despine()
print(train['Title'].value_counts())


# In[ ]:


train['Title'].unique()


# In[ ]:


train['Title'] = train['Title'].replace(['Lady', 'the Countess','Countess','Capt', 
                                         'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],
                                        'Rare')
train['Title'].unique()


# In[ ]:


train['Title'] = train['Title'].map({'Master':0,'Miss':1,'Mrs':1,'Mme':1,'Ms':1,'Mlle':1,'Mr':2,'Rare':3,})
train['Title'].unique()


# In[ ]:


train['Title'] = train['Title'].astype(int)
ax = sns.countplot(train['Title'])
ax.set_xticklabels(["Master","Miss/Ms/Mme/Mlle/Mrs","Mr","Rare"],rotation = 45)
sns.despine()


# In[ ]:


train.head()


# In[ ]:


train['Family_Size'] = train['SibSp'] + train['Parch'] + 1
sns.countplot(train['Family_Size'])


# In[ ]:


train['Single'] = train['Family_Size'].map(lambda i:1 if i==1 else 0)
train['SmallF'] = train['Family_Size'].map(lambda i:1 if i==2 else 0)
train['MediumF'] = train['Family_Size'].map(lambda i:1 if 3<=i<=4 else 0)
train['LargeF'] = train['Family_Size'].map(lambda i:1 if i>=5 else 0)


# In[ ]:


# cabin 
train['Cabin'].isnull().sum() # 687 null values we can asssume that nan values mean they didn't get the cabin


# In[ ]:


train['Cabin']= pd.Series([i[0] if not pd.isnull(i) else 'X' for i in train['Cabin'] ])
sns.countplot(train['Cabin'])
sns.factorplot(x = 'Cabin' , y = 'Survived',data = train,kind = 'bar')


# In[ ]:


train['Ticket'] = pd.Series([i.replace('.',' ').replace('/',' ').split(' ')[0] if not i.isdigit() else 'X' for i in train['Ticket']])


# In[ ]:


train.head()


# In[ ]:


train = pd.get_dummies(train , columns = ['Cabin'],prefix = "Cab")


# In[ ]:


train = pd.get_dummies(train , columns = ["Ticket"],prefix = "T")
train = pd.get_dummies(train , columns = ["Embarked"],prefix = "Emb")
train = pd.get_dummies(train , columns = ["Title"])


# In[ ]:


train = pd.get_dummies(train,columns =["Parch"],prefix = "P")


# In[ ]:


# modelling 
train.drop(labels = ['PassengerId'],axis = 1,inplace = True)
train_df = train[:len(train_data)]
test_df = train[len(train_data):]


# In[ ]:


test_df.drop(labels = ['Survived'],axis = 1,inplace = True)


# In[ ]:


y_train = train_df['Survived']
x_train = train_df.drop(labels = ['Survived'],axis = 1)


# In[ ]:


import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.model_selection import StratifiedKFold,GridSearchCV,cross_val_score,learning_curve
from sklearn.ensemble import GradientBoostingClassifier , ExtraTreesClassifier, AdaBoostClassifier,VotingClassifier


# In[ ]:





# In[ ]:


'''model_lg = lgb.LGBMClassifier(silent = False)
param_dist = {"max_depth":[15,25,35,45,50],
              "learning_rate":[0.01,0.05,0.1,0.2,0.5,0.4],
             "num_leaves":[100,200,300,500],
             "n_estimators":[50,100,150]}
grid_search = GridSearchCV(model_lg,param_grid = param_dist,cv = 3,scoring="roc_auc"
                          ,verbose = 5)
grid_search.fit(x_train,y_train)
grid_search.best_estimator_'''


# In[ ]:


model_xg = xgb.XGBClassifier()
param_dist = {"max_depth":[5,10,15,20,30],
             "min_child_weight":[1,2,3,4,6],
             "n_estimators":[50,100,150,200],
             "learning_rate":[0.01,0.05,0.1,0.16,0.2]}
grid_search = GridSearchCV(model_xg,param_grid=param_dist,cv = 3,verbose = 10,n_jobs=-1)
grid_search.fit(x_train.drop(labels=['Name','Sex'],axis=1),y_train)
print(grid_search.best_estimator_)


# In[ ]:


model = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
       max_depth=15, min_child_weight=3, missing=None, n_estimators=50,
       n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1)
model.fit(x_train.drop(labels=['Name','Sex'],axis=1),y_train)
y_pred = model.predict(test_df.drop(labels=['Name','Sex'],axis=1))


# In[ ]:


test_Survived = pd.Series(y_pred,name = "Survived")
results = pd.concat([gen_sub['PassengerId'],test_Survived],axis=1,ignore_index = False)


# In[ ]:


results = results.astype('int')
results.to_csv("submission.csv",index=False,encoding = "utf-8")


# In[ ]:




