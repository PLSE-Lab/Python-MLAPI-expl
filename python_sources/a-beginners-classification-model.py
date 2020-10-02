#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ## Reading in the nessesary Libraries

# In[ ]:


import seaborn as sns
from matplotlib import pyplot as plt


# In[ ]:


#reading in the train data
df=pd.read_csv('/kaggle/input/titanic/train.csv')
df.head()


# In[ ]:


sns.set(style="ticks", color_codes=True)
sns.pairplot(df[['Survived','Age','SibSp','Pclass']])
plt.show()


# In[ ]:





# In[ ]:


#checking for null values 
sns.heatmap(df.isnull())


# In[ ]:


#correlation heatmap
sns.heatmap(df.corr())


# In[ ]:


df=df.drop(['Cabin','Name','Ticket','Embarked','Fare'],axis=1)


# ## Some Visualisations

# In[ ]:


df.head()


# In[ ]:


#The Gender Gap Betwwen the people who survived
df_survived=df[df['Survived']==1]
sns.countplot(df_survived['Sex'])


# ### Encoding the Sex Values to binary

# In[ ]:


from sklearn.preprocessing import OneHotEncoder
# creating instance of one-hot-encoder
enc = OneHotEncoder(handle_unknown='error',drop='first')
# passing the gender column 
enc_df = pd.DataFrame(enc.fit_transform(df[['Sex']]).toarray())
# merge with main df bridge_df on key values
df =df.join(enc_df)


# In[ ]:


df['Male']=df[0]
df=df.drop(['Sex',0],axis=1)


# ## <u>Gender and survivorship plot</u>

# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(df.Survived,hue=df.Male)
plt.xticks(ticks=[0,1],labels=['Did not Survive','Survived'])
plt.xlabel('')
plt.ylabel('Count')
plt.title('Survived VS Gender')

plt.show()


# ### From the above plot it is clear that the males dies disproportionately more than the females who were aboard the HMS-Titanic 

# ## <u>Class and survivorship plot</u>

# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(df.Survived,hue=df.Pclass)
plt.xticks(ticks=[0,1],labels=['Did not Survive','Survived'])
plt.xlabel('')
plt.ylabel('Count')
plt.title('Survived VS Pclass')
plt.show()


# ### This shows us the fact that the people who formed the third class of passengers on this ship died dispropotionately more than the first and second classes 

# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(df.Male,hue=df.Pclass)
plt.xticks(ticks=[0,1],labels=['Female','Male'])
plt.xlabel('')
plt.ylabel('Count')
plt.title('Gender Split in Passenger Class')
plt.show()


# In[ ]:


df.fillna(df.Age.median(),inplace=True)


# In[ ]:


sns.heatmap(df.isnull())


# In[ ]:


from sklearn.svm import SVC
clf = SVC()


# In[ ]:


df.set_index(['PassengerId'],inplace=True)


# In[ ]:


x=df[['Pclass','Age','SibSp','Parch','Male']]
y=df['Survived']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.25,train_size=0.75)


# In[ ]:


from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 2,3,5,10], 'kernel': ['linear']},
              {'C': [1, 2,3,5,10], 'kernel': ['rbf'], 'gamma': [0.3, 0.4, 0.5, 0.6, 0.7]}]
grid_search = GridSearchCV(estimator = clf,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
accuracy = grid_search.best_score_
best_param =grid_search.best_params_


# In[ ]:





# In[ ]:


print(best_param)
print(accuracy)


# In[ ]:


clf_main= SVC(C=1, kernel='rbf',gamma=0.3)
clf_main.fit(X_train,y_train)


# In[ ]:


from sklearn.metrics import confusion_matrix

#add a classifier 
# predict the values from the classifier using test data
predicted=clf_main.predict(X_test)
confusion = confusion_matrix(y_test, predicted)
print('Confusion Matrix (SVC)\n', confusion)


# In[ ]:


clf_main.score(X_test,y_test)


# In[ ]:


from sklearn.metrics import classification_report

print(classification_report(y_test, predicted, target_names=[' Dead', 'Survived']))


# ## Using Xgboost for the Classifier

# In[ ]:


import xgboost as xgb


# In[ ]:


clf_x=xgb.XGBClassifier()


# In[ ]:


params={'max_depth':[3,5,10,20,30,40,50,100],
        'learning_rate':[0.01,0.05,0.1,0.15,0.2],
        'n_estimators':[100,500,1000],
        'min_child_weight ':[1,2,3,4,5]
        }


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
def hypertuning_rscv(clf, p_distr, nbr_iter,X,y):
    rdmsearch = RandomizedSearchCV(clf, param_distributions=p_distr,
                                  n_jobs=-1, n_iter=nbr_iter, cv=10)
    #CV = Cross-Validation ( here using Stratified KFold CV)
    rdmsearch.fit(X,y)
    ht_params = rdmsearch.best_params_
    ht_score = rdmsearch.best_score_
    return ht_params, ht_score

rf_parameters, rf_ht_score = hypertuning_rscv(clf_x, params, 40, x,y)


# In[ ]:


rf_parameters


# In[ ]:


clf_x=xgb.XGBClassifier(n_estimators= 100,
                    min_child_weight = 2,
                    max_depth= 3,
                    learning_rate= 0.1,
                    verbosity =3,
                    objective="binary:logistic",
                    n_jobs=-1
                   )
        


# In[ ]:


clf_x.fit(x,y)


# In[ ]:


x


# ## Reading in the Test data

# In[ ]:


df_test=pd.read_csv('../input/titanic/test.csv')
df_test.head()


# In[ ]:


df_test=df_test.drop(['Cabin','Name','Ticket','Embarked','Fare'],axis=1)
# passing the gender column 
enc_df_test = pd.DataFrame(enc.fit_transform(df_test[['Sex']]).toarray())
# merge with main df bridge_df on key values
df_test =df_test.join(enc_df_test)


# In[ ]:


df_test


# In[ ]:


df_test['Male']=df_test[0]
df_test=df_test.drop(['Sex',0],axis=1)


# In[ ]:


df_test.fillna(df_test.Age.median(),inplace=True)


# In[ ]:


sns.heatmap(df_test.isnull())


# In[ ]:


df_test.set_index(['PassengerId'],inplace=True)


# In[ ]:


test_predict=clf_x.predict(df_test)


# In[ ]:


test_predict=pd.Series(test_predict)


# In[ ]:


df_test.reset_index(inplace=True)


# In[ ]:


df_predict=df_test['PassengerId']


# In[ ]:


df_predict= pd.concat([df_predict,test_predict], axis=1)


# In[ ]:


df_predict.rename(columns={0: "Survived"},inplace=True)


# In[ ]:


df_predict.to_csv("submission.csv",index=False)


# In[ ]:


sns.countplot(df_predict.Survived)


# In[ ]:




