#!/usr/bin/env python
# coding: utf-8

# ## Importing Necessary Libraries 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
import xgboost as xgb


# ### Importing The Dataset

# In[ ]:


df_train = pd.read_csv('/kaggle/input/titanic/train.csv')
df_test = pd.read_csv('/kaggle/input/titanic/test.csv')


# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# In[ ]:


df_train.corr()


# In[ ]:


df_train.shape


# In[ ]:


df_train.describe()


# In[ ]:


df_test.isnull().sum()


# In[ ]:


df_train.describe(include=['O'])


# In[ ]:


df_train['Embarked'].fillna(df_train['Embarked'].mode()[0],inplace = True)


# In[ ]:


sns.countplot(x = 'Survived',hue = 'Sex',data = df_train)


# In[ ]:


df_train.Age.plot.hist(color = 'r')


# In[ ]:


sns.boxplot(data = df_train)


# In[ ]:


sns.boxplot(x = 'Pclass',y = 'Age',data = df_train)


# In[ ]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass ==1:
            return 37
        elif Pclass ==2:
            return 29
        else:
            return 24
        
    return Age    


# In[ ]:


df_train['Age'].fillna(impute_age(df_train['Age']),inplace = True)
df_test['Age'].fillna(impute_age(df_test['Age']),inplace = True)


# In[ ]:


df_test.isnull().sum()


# In[ ]:


pd.set_option('display.max_rows',None)
df_test = df_test.drop('Cabin',axis = 1)
df_test.loc[pd.isnull(df_test).any(1),:]


# In[ ]:


df_train['Sex'] = df_train['Sex'].map({'male':1,'female':0})
df_test['Sex'] = df_test['Sex'].map({'male':1,'female':0})


# In[ ]:


print(df_train['SibSp'].value_counts())
print('------'*10)
df_train['Parch'].value_counts()


# In[ ]:


df_test['Fare'].fillna(method = 'ffill',inplace = True)


# In[ ]:


df_train = pd.get_dummies(df_train,columns = ['Embarked'],drop_first = True)
df_test = pd.get_dummies(df_test,columns = ['Embarked'],drop_first = True)


# In[ ]:


df_train.columns


# In[ ]:


sns.barplot(x = 'Survived',y = 'Age',hue = 'Sex',data = df_train)
plt.show()


# In[ ]:


df_train['Survived'].value_counts()


# In[ ]:


features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked_Q','Embarked_S']
X = df_train[features]
y = df_train['Survived']


# In[ ]:


# import pandas_profiling
# df_train.profile_report(style={'full_width':True})


# ## Modelling

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,random_state = 42)


# In[ ]:


def model(algo):
    y_pred = algo.fit(X_train,y_train).predict(X_test)
    i = str(algo).split('(')[0]
    print('\n---------',i,'---------')
    print('accuracy_score : ',accuracy_score(y_test,y_pred))
    
log_reg = LogisticRegression()
dec_tree  = DecisionTreeClassifier()
gnb = GaussianNB()
knn3 = KNeighborsClassifier(n_neighbors=3)
knn5 = KNeighborsClassifier(n_neighbors=5)
knn7 = KNeighborsClassifier(n_neighbors=7)
rf = RandomForestClassifier(n_estimators=100)
bg = BaggingClassifier()

models = [log_reg,dec_tree,gnb,knn3,knn5,knn7,rf,bg]


# In[ ]:


for algo in models:
    model(algo)


# In[ ]:


from sklearn.ensemble import AdaBoostClassifier
adb = AdaBoostClassifier()
y_pred = adb.fit(X_train,y_train).predict(X_test)
accuracy_score(y_test,y_pred)


# In[ ]:


xgbc = xgb.XGBClassifier(learning_rate=0.01)
y_pred = xgbc.fit(X_train,y_train).predict(X_test)
accuracy_score(y_test,y_pred)


# In[ ]:


# !pip install graphviz


# In[ ]:


# import graphviz 
# dot_data = tree.export_graphviz(dtree, out_file=None, 
#                                 feature_names = data1_x_bin, class_names = True,
#                                 filled = True, rounded = True)
# graph = graphviz.Source(dot_data) 
# graph


# In[ ]:


from sklearn.preprocessing import StandardScaler
scX = StandardScaler() 
X_train = scX.fit_transform(X_train) 
X_test = scX.fit_transform(X_test)


# In[ ]:


model(log_reg)


# In[ ]:


model(adb)


# In[ ]:


from sklearn.decomposition import PCA
pca = PCA(n_components = None) 
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explainedvariance = pca.explained_variance_ratio_


# In[ ]:


explainedvariance


# In[ ]:


pred = log_reg.fit(X_train,y_train).predict(X_test)
accuracy_score(y_test,y_pred)


# In[ ]:





# In[ ]:


test = df_test[features]
test = scX.fit_transform(test)
test  = pca.fit_transform(test)


# In[ ]:


preds = log_reg.predict(test)


# In[ ]:


from sklearn.linear_model import SGDClassifier
predxgb = xgb.XGBClassifier(max_depth = 20,learning_rate = 0.1,n_estimators=70,gamma = 1,subsample=1).fit(X_train,y_train).predict(X_test)
accuracy_score(y_test,predxgb)


# In[ ]:


sgd_pred = SGDClassifier().fit(X_train,y_train).predict(X_test)
accuracy_score(y_test,sgd_pred)


# ## The best results were given by logistic regression

# In[ ]:


sample_submission = pd.DataFrame({'PassengerId':df_test['PassengerId'],'Survived':preds})
sample_submission.head()


# In[ ]:


sample_submission.to_csv('submission.csv',index = False)


# In[ ]:




