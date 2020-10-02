#!/usr/bin/env python
# coding: utf-8

# # Hello Community!
# Lets find out who are more likely to survive...

# Importing Libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Great! Now lets read the data.

# In[ ]:


trainData = pd.read_csv('../input/titanic/train.csv')


# In[ ]:


testData = pd.read_csv('../input/titanic/test.csv')


# Lets have a look at our dataset

# In[ ]:


trainData.head()


# Survived - dependent variable

# In[ ]:


trainData.Survived


# # Data Exploration
# Lets do some exploration

# In[ ]:


plt.hist(trainData.Survived, bins = 2)


# In[ ]:


trainData.describe()


# In[ ]:


pie_data = trainData.drop(columns=['PassengerId', 'Name', 'Ticket', 'Age','Fare','Cabin'])


# In[ ]:


fig = plt.figure(figsize=(15,15))
for i in range(1, pie_data.shape[1] +1):
    plt.subplot(2, 3, i)
    
    fig = plt.gca()
    fig.axes.get_yaxis().set_visible(False)
    
    fig.set_title(pie_data.columns.values[i-1])
    values =  pie_data.iloc[:, i-1].value_counts(normalize = True).values
    index =  pie_data.iloc[:, i-1].value_counts(normalize = True).index
    fig = plt.pie(values, labels = index, autopct = '%1.1f%%')
    
    plt.axis('equal')
plt.tight_layout(rect = [0, 0.03, 1, 0.95])


# In[ ]:


sns.set_style('darkgrid')
plt.figure(figsize=(12,12))
trainData.drop(columns=['Survived','PassengerId']).corrwith(trainData.Survived).plot.bar()
plt.title('Correlation with the dependent vairable')
plt.show()


# In[ ]:


plt.figure(figsize=(12,12))
sns.heatmap(trainData.corr(),annot = True, fmt='g')


# Let's clean the Data next

# # Data Cleaning

# In[ ]:


trainData.isnull().sum()


# I'll drop the cabin column and replace the null values in age column with the mean. Also I'll remove the 2 rows which have null values in Embarked column

# In[ ]:


trainData2 = trainData.copy(deep = True)


# In[ ]:


testData2 = testData.copy(deep=True)


# In[ ]:


trainData2.drop(columns = ['Cabin'], inplace = True)


# In[ ]:


testData2.drop(columns = ['Cabin'], inplace = True)


# In[ ]:


trainData2.Age.fillna(trainData2.Age.mean(),inplace = True)


# In[ ]:


testData2.Age.fillna(testData2.Age.mean(),inplace = True)
testData2.Fare.fillna(testData2.Fare.mean(), inplace = True)


# In[ ]:


trainData2.isnull().sum()


# In[ ]:


testData2.isnull().sum()


# In[ ]:


trainData2[trainData2.Embarked.isnull()].index


# In[ ]:


trainData2.drop(index=[61,829],inplace=True)


# In[ ]:


trainData2.isnull().sum()


# In[ ]:


trainData2.shape


# In[ ]:


testData2.shape


# Let's do some Feature engineering next

# # Feature Engineering
# I'll be extracting the titles from the Name column and drop the name column as it is of no use to us

# In[ ]:


name = trainData.Name


# In[ ]:


title = np.asarray([])
for i in name:
    if i.find('Mrs') != -1:
        r = i.find('Mrs')
        re = i[r: r+3]
    elif i.find('Mr') != -1:
        r = i.find('Mr')
        re = i[r:r+2]
    elif i.find('Master') != -1:
        r = i.find('Master')
        re = i[r:r+6]
    else:
        r = i.find('Miss')
        re = i[r:r+4]
    title = np.append(title, re)
    


# In[ ]:


title.head()


# In[ ]:


title = pd.DataFrame(title)


# In[ ]:


title[0].unique()


# The missing value can be filled as

# In[ ]:


indexes = title[title[0] ==''].index


# In[ ]:


for i in indexes:
    if trainData.iloc[i, 4] == 'male':
        if trainData.iloc[i, 5] >= 18:
            title.iloc[i,0] = 'Mr'
        else:
            title.iloc[i,0] = 'Master'
    else:
        title.iloc[i,0] = 'Miss'


# In[ ]:


title[0].unique()


# In[ ]:


name_test = testData2.Name
title_test = np.asarray([])
for i in name_test:
    if i.find('Mrs') != -1:
        r = i.find('Mrs')
        re = i[r: r+3]
    elif i.find('Mr') != -1:
        r = i.find('Mr')
        re = i[r:r+2]
    elif i.find('Master') != -1:
        r = i.find('Master')
        re = i[r:r+6]
    else:
        r = i.find('Miss')
        re = i[r:r+4]
    title_test = np.append(title_test, re)


# In[ ]:


title_test = pd.DataFrame(title_test)
title_test[0].unique()


# In[ ]:


indexes_test = title_test[title_test[0] ==''].index
for i in indexes_test:
    if testData2.iloc[i, 3] == 'male':
        if testData2.iloc[i, 4] >= 18:
            title_test.iloc[i,0] = 'Mr'
        else:
            title_test.iloc[i,0] = 'Master'
    else:
        title_test.iloc[i,0] = 'Miss'


# In[ ]:


title_test[0].unique()


# In[ ]:


trainData2['Title'] = title[0]


# In[ ]:


trainData2.head()


# In[ ]:


testData2['Title'] = title_test[0]
testData2.head()


# ----

# We can now drop the Name variable

# In[ ]:


y=trainData2.Survived
Id = trainData2.PassengerId
x = trainData2.drop(columns = ['PassengerId', 'Name', 'Ticket','Survived'])


# In[ ]:


Id_test = testData2.PassengerId
testData2.drop(columns = ['PassengerId','Name','Ticket'], inplace = True)


# In[ ]:


x_dummies = pd.get_dummies(x, drop_first=True)
testData_dummies = pd.get_dummies(testData2, drop_first = True)


# In[ ]:


x_dummies.head()


# In[ ]:


testData_dummies.head()


# In[ ]:


x = x_dummies


# Now that the data is ready, I'll jump to building the model.
# # Standardization

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)


# In[ ]:


sc = StandardScaler()


# In[ ]:


x_train2 = pd.DataFrame(sc.fit_transform(x_train), columns = x_train.columns.values)


# In[ ]:


x_test2 = pd.DataFrame(sc.transform(x_test), columns = x_test.columns.values)


# In[ ]:


testData_dummies2 = pd.DataFrame(sc.transform(testData_dummies), columns = testData_dummies.columns.values)


# In[ ]:


x_train2


# In[ ]:


x_test2


# In[ ]:


testData_dummies2


# # Training the model
# I'll be using Random Forest 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


classifier = RandomForestClassifier()


# In[ ]:


classifier.fit(x_train2,y_train)


# In[ ]:


y_pred = classifier.predict(x_test2)


# In[ ]:


from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score


# In[ ]:


confusion_matrix(y_test,y_pred)


# In[ ]:


precision_score(y_test,y_pred)


# In[ ]:


accuracy_score(y_test, y_pred)


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


param = {'n_estimators':[100,300,500],'criterion':['gini','entropy'],'max_depth':[3,5,7],'max_features':['sqrt','log2','auto']}


# In[ ]:


classifier = RandomForestClassifier()
random = RandomizedSearchCV(estimator=classifier,param_distributions=param,n_iter=10,scoring='accuracy',n_jobs = -1,cv = 5)


# In[ ]:


random.fit(x_train2,y_train)


# In[ ]:


random.best_params_


# In[ ]:


classifier2 = RandomForestClassifier(n_estimators=100,
                                     max_features='sqrt',
                                     max_depth=5,
                                     criterion='entropy')


# In[ ]:


classifier2.fit(x_train2,y_train)


# In[ ]:


y_pred2 = classifier2.predict(x_test2)


# In[ ]:


confusion_matrix(y_test,y_pred2)


# In[ ]:


precision_score(y_test,y_pred2)


# In[ ]:


accuracy_score(y_test,y_pred2)


# I dont know why I didnt get a better accuracy here . If someone can help me out with it...it would be great.

# In[ ]:


classifier.fit(x_train2,y_train)


# In[ ]:


prediction = classifier.predict(testData_dummies2)


# In[ ]:


Id = pd.DataFrame(Id_test, columns = ['Id'])

predi = pd.DataFrame(prediction, columns = ['Survived'])


# In[ ]:


result = pd.concat([Id,predi],axis = 1)


# So this is it 
# I would really appreciate if someone can help me out with improving my model
# Happy Learning!

# In[ ]:




