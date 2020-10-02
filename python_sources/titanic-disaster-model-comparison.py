#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/train.csv')


# <h2>Lets check how our data looks like</h2>

# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# <h3>Looks like we have some NULL values hiding out there.Lets check it out !</h3>

# In[ ]:


sns.heatmap(df.isnull(),cmap='viridis',cbar=False,yticklabels=
           False)


# As we can seen there is a huge band of missing data for the Cabin column and Age column.Now as the logic suggests Cabin number has no relevance with the Survival data so there is no harm if we drop this column.

# In[ ]:


df1 = df.drop('Cabin',axis=1)


# In[ ]:


sns.heatmap(df1.isnull(),cmap='viridis',cbar=False,yticklabels=
           False)


# <h3>Lets do some EDA of the data now</h3>

# In[ ]:


sns.countplot(x="Survived",data=df1)


# In[ ]:


df1.columns


# In[ ]:


sns.set_style('darkgrid')
sns.boxplot(x='Pclass',y='Age',data=df1)


# In[ ]:


sns.countplot(x='Survived',hue='Sex',data=df1)


# In[ ]:


plt.figure(figsize=(14,6))
sns.violinplot(x='SibSp',y='Age',hue='Survived',data=df1)


# In[ ]:


df1.columns


# We can also drop the PassengerId,Name and ticket number columns as they also don't have much relevance with the survived result.

# In[ ]:


df2=df1.drop(['PassengerId','Name','Ticket'],axis=1)


# In[ ]:


df2.head()


# In[ ]:


df2.info()


# We have some missing data in Embarked column.Lets try to figure it out with some eda.

# In[ ]:


sns.boxplot(x='Embarked',y='Fare',data=df2)


# As per the above plot it seems that passengers who embarked from Cherbourg have paid more fare.

# In[ ]:


sns.countplot(x='Embarked',data=df2)


# In[ ]:


df2.groupby('Embarked')['Fare'].mean()


# In[ ]:


df2[df2['Embarked'].isnull()]


# As the stats suggest the passenger paid fare of 80 and the class is 1st so they most likely have embarked from Cherbourg.

# In[ ]:


df2.loc[61,'Embarked']='C'


# In[ ]:


df2.loc[829,'Embarked']='C'


# Now lets manipulate the age column based upon the data.As the data suggest there is a great correlation between the passenger class and age because person with more wealth is older since it takes time to earn more wealth.So people who are rich are mostly older than others. (Unless there father is a Millionare :-) )

# In[ ]:


df2.groupby('Pclass')['Age'].mean()


# In[ ]:


def compute_age(cols):
    age=cols[0]
    pc=cols[1]
    if pd.isnull(age):
        if pc==1:
            return 38
        elif pc==2:
            return 30
        elif pc==3:
            return 25
    else:
        return age


# In[ ]:


df2['Age'] = df2[['Age','Pclass']].apply(compute_age,axis=1)


# In[ ]:


sns.heatmap(df2.isnull(),cmap='viridis',yticklabels=False,cbar=False)


# Now our dataset is NULL free and we are good to proceed further.

# Now its time for Label encoding to make the text readable for the Machine.

# In[ ]:


sex = pd.get_dummies(df2['Sex'],drop_first=True)
embark = pd.get_dummies(df2['Embarked'],drop_first=True)


# In[ ]:


df3=df2.drop(['Sex','Embarked'],axis=1)


# In[ ]:


df3 = pd.concat([df3,sex,embark],axis=1)


# In[ ]:


df3.head()


# In[ ]:


X=df3.drop('Survived',axis=1)
y=df3['Survived']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Now since our dataset is fully processed lets move forward and apply some Machine learning algorithms.
# We will first start with our robust machine learning algorithm  **Logistic Regression.**

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


lgr = LogisticRegression()


# In[ ]:


lgr.fit(X_train,y_train)


# In[ ]:


preds=lgr.predict(X_test)


# Lets analyze our result with some reports.

# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(y_test,preds))


# **Not bad ! we got an accuracy of 81%.**
# Lets move ahead and try some other alogorithms and see if we can further improve our accuracy.

# Lets try **Support Vector Machines **algorithm with **Grid Search** now

# In[ ]:


from sklearn.svm import SVC


# In[ ]:


param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 


# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)


# In[ ]:


grid.fit(X_train,y_train)


# In[ ]:


grid.best_params_


# In[ ]:


grid_predictions = grid.predict(X_test)


# In[ ]:


print(classification_report(y_test,grid_predictions))


# It seems like we ended up reducing our model accuracy with this.<br/>
# **No issues ! here is the time to take out our another weapon :-)**
# <h3>KNN Model</h3>

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


error_rate = []

# Will take some time
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[ ]:


plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=12)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)
print('\n')
print(classification_report(y_test,pred))


# Ooops ! this weapon resulted in even worse accuracy.

# So finally we will stick with our favourite Logistic Regression.

# **Now lets apply our model to the test set **

# In[ ]:


df_test = pd.read_csv('../input/test.csv')


# In[ ]:


df_test.head()


# In[ ]:


df_test = df_test.drop('Cabin',axis=1)


# In[ ]:


df_test=df_test.drop(['PassengerId','Name','Ticket'],axis=1)


# In[ ]:


df_test.info()


# In[ ]:


df_test[df_test['Fare'].isnull()]


# In[ ]:


plt.figure(figsize=(14,6))
sns.boxplot(x='Pclass',y='Fare',hue='Embarked',data=df_test)


# In[ ]:


df_test.groupby(['Pclass','Embarked'])['Fare'].mean()


# In[ ]:


df_test.loc[152,'Fare']=14


# In[ ]:


df_test[df_test['Fare'].isnull()]


# In[ ]:


df_test['Age'] = df_test[['Age','Pclass']].apply(compute_age,axis=1)


# In[ ]:


df_test.info()


# In[ ]:


se = pd.get_dummies(df_test['Sex'],drop_first=True)
emb = pd.get_dummies(df_test['Embarked'],drop_first=True)


# In[ ]:


dft=df_test.drop(['Sex','Embarked'],axis=1)


# In[ ]:


dft = pd.concat([dft,se,emb],axis=1)


# In[ ]:


dft.head()


# In[ ]:


predictions=lgr.predict(dft)


# In[ ]:


sub_df = pd.read_csv('../input/test.csv')


# In[ ]:


sub_id=sub_df['PassengerId'].values


# In[ ]:


sbdf = pd.DataFrame({"PassengerId":sub_id,"Survived":predictions})


# In[ ]:


sbdf.head(10)


# <h1>Congratulations  Folks ! We have reached the end of this kernel.</h1>

# As for now lets be satisfied with the accuracy of **81%**. However this is not the end.Stay tuned with this kernel as we will try to push our accuracy factor to the top level.
# Till then **Take Care** and **Good Bye**.
