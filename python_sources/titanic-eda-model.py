#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


print(train.shape)
print(test.shape)


# In[ ]:


train.isna().sum()


# In[ ]:


test.isna().sum()


# In[ ]:


train.columns


# In[ ]:


train['Survived'].value_counts(normalize=True)


# In[ ]:


sns.countplot(train['Survived'])


# In[ ]:


train['Fare'].describe()


# In[ ]:


pd.crosstab(train['Sex'],train['Survived']).plot.bar(stacked=True)
plt.ylabel('Frequency')
plt.show()


# Females rate of Survival is High when compared to the Males

# In[ ]:


sns.boxplot(y=train['Age'])


# In[ ]:


plt.figure(figsize=(10,5))
sns.boxplot(train['Pclass'],train['Age'])
plt.show()


# The people in the Plcass 3(Lower Class) age median is around 25
# The people in the Plcass 2(Middle Class) age median is around 29
# The people in the Plcass 1(Upper Class) age median is around 38
# 
# 

# In[ ]:


train[train['Age']<1]


# In[ ]:


fix,ax = plt.subplots(1,3,figsize=(12,5))
sns.boxplot(y=train['Fare'],ax=ax[0])
sns.distplot(train['Fare'],ax=ax[1])
sns.distplot(train['Age'].dropna(),ax=ax[2])
plt.tight_layout()
plt.show()


# In[ ]:


train['Age'].skew()


# In[ ]:


pd.crosstab(train['Pclass'],train['Survived']).plot.bar(stacked=True)
plt.xlabel('Frequency')
plt.show()


# As we can see from that graph that ratio of people who died were from Pclass 3 (Lower class).
# The Pclass 1(Upper class) has more number of Survivals when compared to the other class.

# In[ ]:


pd.crosstab(train['Parch'],train['Survived']).plot.bar(stacked=True)
plt.show()


# The childeren who travelled with their nanny's have high deaths when compared to the
# other parch.

# In[ ]:


pd.crosstab(train['Embarked'],train['Survived'],).plot.bar(stacked=True)
plt.show()


# The passengers who were in the embarked S i.e Southampton their death rate is high when compared to the other Port of Embarkation.

# In[ ]:


sns.heatmap(pd.crosstab(train['SibSp'],train['Survived']),annot=True,cmap='Blues')
plt.show()


# The people who were travelling alone has less survival rate when compared to the people travelling with the family

# In[ ]:


from wordcloud import WordCloud
for col in ['Name','Cabin']:
    
    text = " ".join(review for review in train[col].dropna())
    word = WordCloud(width=1000,height=800,margin=0,max_font_size=150,background_color='white').generate(text)

    plt.figure(figsize=[8,8])
    plt.imshow(word,interpolation='bilinear')
    plt.axis('off')
    plt.show()


# In[ ]:


#IQR method
for col in ['Age','Fare']:
    
    q1= train[col].quantile(0.25)
    q3 = train[col].quantile(0.75)
    iqr=q3-q1
    print(col,'IQR',iqr)

    upper_limit = q3+1.5*iqr
    lower_limit = q1-1.5*iqr
    print(col,'Upper limit for age',upper_limit)
    print(col,'Lower limit for age',lower_limit)


# In[ ]:


sns.heatmap(train.corr(),cmap='Blues',annot=True)
plt.show()


# In[ ]:


train_1 = train.copy()


# In[ ]:


age_median=train_1['Age'].median()
train_1['Age'] = train_1['Age'].fillna(age_median)


# In[ ]:


for col in train.columns:
    
    print(col,'Percentage of missing values',train[col].isna().sum()/train.shape[0]*100)


# As the percentage of missing values is High for the cabin we will remove that column

# In[ ]:


train_1.drop(columns=['Cabin'],inplace=True)


# In[ ]:


from sklearn_pandas import CategoricalImputer
imputer = CategoricalImputer()
train_1['Embarked']=imputer.fit_transform(train['Embarked'])


# In[ ]:


train_1.isna().sum()


# In[ ]:


#we will drop passengerid,Ticket,Name
train_1.drop(columns=['Name','Ticket','PassengerId'],inplace=True)


# In[ ]:


train_1 = pd.get_dummies(data=train_1,columns=['Sex','Embarked'],drop_first=True)
train_1.head()


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier,VotingClassifier
from sklearn.naive_bayes import BernoulliNB,GaussianNB
from sklearn.svm import SVC


# In[ ]:


X = train_1.drop(columns=['Survived'],axis=1)
y= train_1['Survived']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[ ]:


model = DecisionTreeClassifier(max_depth=6,class_weight='balanced',random_state=0)
model.fit(X_train,y_train)
acc_decision_tree=model.score(X_train,y_train)*100
print(model.score(X_train,y_train))
print(model.score(X_test,y_test))


# In[ ]:


model.feature_importances_


# In[ ]:


sns.barplot(x=model.feature_importances_,y=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q',
       'Embarked_S'])
plt.title('Feature Importance Plot')
plt.show()


# In[ ]:


model1  = LogisticRegression(class_weight='balanced',C=4.5,random_state=0)
model1.fit(X_train,y_train)
acc_logistic=model1.score(X_train,y_train)*100
print(model1.score(X_train,y_train))
print(model1.score(X_test,y_test))


# In[ ]:


model2  = RandomForestClassifier(n_estimators=21,max_depth=6,criterion='gini',random_state=0,class_weight='balanced',
                                min_samples_split=2)
model2.fit(X_train,y_train)
acc_random_forest=model2.score(X_train,y_train)*100
print(model2.score(X_train,y_train))
print(model2.score(X_test,y_test))


# In[ ]:


rf  = RandomForestClassifier(n_estimators=1000,min_samples_split=30,min_samples_leaf=5,random_state=42,warm_start=True)
rf.fit(X_train,y_train)
acc_random_forest=rf.score(X_train,y_train)*100
print(rf.score(X_train,y_train))
print(rf.score(X_test,y_test))


# In[ ]:


model2.feature_importances_


# In[ ]:


sns.barplot(x=model2.feature_importances_,y=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q',
       'Embarked_S'])
plt.title('Feature Importance Plot')
plt.show()


# In[ ]:


#Learning rate =0.01 got from the grid serach cv
model3 = AdaBoostClassifier(base_estimator=model2,random_state=0,learning_rate=0.001)
model3.fit(X_train,y_train)
acc_ada_boost=model3.score(X_train,y_train)*100
print(model3.score(X_train,y_train))
print(model3.score(X_test,y_test))


# In[ ]:


#from the Grid Searchcv I have got the parameters as 36,1,1
model4 =GradientBoostingClassifier(random_state=1,n_estimators=36,max_depth=1,learning_rate=1)
model4.fit(X_train,y_train)
acc_gradient_boost=model4.score(X_train,y_train)*100
print(model4.score(X_train,y_train))
print(model4.score(X_test,y_test))


# In[ ]:


# By gridsearch we have got the values for the hyperparameters
model5 = SVC(kernel='rbf',C=10,gamma=0.1,random_state=1)
model5.fit(X_train,y_train)
acc_svc=model5.score(X_train,y_train)*100
print(model5.score(X_train,y_train))
print(model5.score(X_test,y_test))


# In[ ]:


model6 = VotingClassifier(estimators=[('DT',model),('LR',model1),('RF',model2),('AD',model3),('GB',model4),('SVC',model5)],
                          voting='hard')
model6.fit(X_train,y_train)
acc_voting_classifier=model6.score(X_train,y_train)*100
print(model6.score(X_train,y_train))
print(model6.score(X_test,y_test))


# In[ ]:


from sklearn.ensemble import BaggingClassifier
bc = BaggingClassifier(base_estimator=model2,n_estimators=20,random_state=1)
bc.fit(X_train,y_train)
acc_Bagging_classifier=bc.score(X_train,y_train)*100
print(bc.score(X_train,y_train))
print(bc.score(X_test,y_test))


# In[ ]:


from sklearn.model_selection import GridSearchCV

parameters = [{'learning_rate':[0.01,0.1,0.001,1,5,10,20]}]
grid_search = GridSearchCV(estimator = model3,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
                           #n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_


# In[ ]:


print('Best Accuracy',best_accuracy)
print('Best Parameters',best_parameters)               


# In[ ]:


from sklearn.model_selection import cross_val_score
for models  in [model,model1,model2,model3,model4,model5,model6,bc]:
    
    accuracies = cross_val_score(estimator = models, X = X_train, y = y_train, cv = 10)
    print(models,accuracies.mean())
    print(accuracies.std())


# In[ ]:


models = pd.DataFrame({
    'Model': ['Decision Tree','Logistic Regression','Random Forest','Adaboost Classifier','Gradient boost',
             'Support Vector Classifier','Voting Classifier','Bagging Classifier'],
    'Score': [acc_decision_tree,acc_logistic,acc_random_forest,acc_ada_boost,acc_gradient_boost,
             acc_svc,acc_voting_classifier,acc_Bagging_classifier]})
models.sort_values(by='Score', ascending=False)


# In[ ]:


test_1 = test.copy()
test_1['Age'] = test_1['Age'].fillna(test_1['Age'].median())
test_1['Fare'] = test_1['Fare'].fillna(test_1['Fare'].mean())
test_1.drop(columns=['Name','Ticket','PassengerId','Cabin'],inplace=True)


# In[ ]:


test_1.head()


# In[ ]:


test_1 = pd.get_dummies(data=test_1,columns=['Sex','Embarked'],drop_first=True)
test_1.head()


# In[ ]:


y_pred = rf.predict(test_1)
y_pred


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_pred
    })
submission.to_csv('submission.csv', index=False)


# In[ ]:




