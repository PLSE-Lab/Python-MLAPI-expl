#!/usr/bin/env python
# coding: utf-8

# # Titanic Prediction 
# ## I just do my super simple example as start of this competition and will improve myself based on this
# ## I share this notebook as a example Please comment if you have a question or there is somethin wrong
# 
# ## I will very grateful about your feedback
# 

# ## What am I gonna do?
# 1. Import packages which we need 
# 2. See what kind of data we have
#     1. Type
#     2. Feature
#     3. leakage
#     4. etc...
# 3. Preprocessing
# 4. Classification
# 5. Submission
#    

# ## What did I most concerned?
# 
# we could make so many mistakes in data handling. And that gives us not even only wrong insight, but also that could be a manipulation
# 
# So, I tried not to hack the test data. 
# 
# It sounds very easy not to hack but it really is.
# 
# Let me show you what kind of process I consider and let's talk about that
# 

# ## Import
# I did use the package to process and analasys this data set

# In[ ]:


import pandas as pd # processing data
import matplotlib.pyplot as plt # Visualization
import numpy as np # Linear Algebra
import re # Regular Expression
import seaborn as sns # Visaulization
from sklearn.preprocessing import LabelEncoder # Encoding object to number
from sklearn.ensemble import RandomForestClassifier #Classification model
from sklearn.model_selection import train_test_split #split train and validation data


# In[ ]:


train_data=pd.read_csv("../input/titanic/train.csv")
test_data=pd.read_csv("../input/titanic/test.csv")


# In[ ]:


train_data.head(10)


# In[ ]:


test_data.head(10)


# As you can see and you can even see many notebooks that plenty of people handled the 'Name' feature 
# 
# Because there is a 'title' (Mr, Mrs, Miss)
# 
# And to extract that feature I used regular expression package called 're'
# 
# you can see a reference about this package on https://docs.python.org/3/library/re.html

# In[ ]:


p = re.compile('[^\,.$\.]+')


# Sample extraction using re package

# In[ ]:


print(p.findall(train_data.Name[0]))
print(p.findall(test_data.Name[0]))


# extract all tilts from all of row at a object 'xxx_title'
# 
# I will add this to raw data

# In[ ]:


train_title = pd.Series(train_data.Name.map(lambda x : p.findall(x)[1]))
test_title = pd.Series(test_data.Name.map(lambda x : p.findall(x)[1]))


# You can see a space like ' mr'
# 
# I wanted to delete this space

# In[ ]:


train_title = pd.Series(list(map(lambda x : x.strip() ,train_title)))
test_title = pd.Series(list(map( lambda x : x.strip(),test_title)))


# ## Visualization

# In[ ]:


plt.figure(figsize=(20,13))
sns.barplot(x = train_title.value_counts().index,y=train_title.value_counts().values)
plt.title("The title of aboarded")
plt.xlabel("title")
plt.ylabel("number of aboarded")
plt.show()


# In[ ]:


plt.figure(figsize=(20,13))
sns.barplot(x = test_title.value_counts().index,y=test_title.value_counts().values)
plt.title("The title of aboarded")
plt.xlabel("title")
plt.ylabel("number of aboarded")
plt.show()


# In[ ]:


train_data['title'] = train_title
test_data['title'] = test_title


# In[ ]:


train_data.head(10)


# In[ ]:


test_data.head()


# In[ ]:


list_for_check_train = []
for i in range(len(train_data)):
    if train_data.title[i] in train_data.Name[i]:
        list_for_check_train.append(True)
    else:
        list_for_check_train.append(False)
np.sum(list_for_check_train)


# ## Visualize correalation between features
# 
# In this heatmap you can see 'Sibsp' and 'Parch' related very well and this is pretty straitfoward
# 
# and look at the corr between survied and Fare. This is what we want to know
# 
# Fare and Survived related pretty much 
# 
# did you expect it?

# In[ ]:


relation = train_data.corr()
plt.figure(figsize=(16,9))
sns.heatmap(data=relation,annot=True,cmap='YlGnBu')
plt.show()


# ## replace Nan
# 
# As you can see there are some missing vlaue (Age, Embarked, Cabin ...)
# 
# I replaced missing value as a mean value
# 
# But not replaced Cabin column, which I droped

# In[ ]:


train_data.count()


# I will replace missing value refer to title.
# 
# But there's something suspicious.
# 
# The result titles at the train data, which has missing value is different from the titles at the test data

# In[ ]:


train_data[train_data.Age.isna()].title.value_counts()


# In[ ]:


test_data[test_data.Age.isna()].title.value_counts()


# In[ ]:


train_data.groupby('title').mean().T['Mr'].Age.mean()


# So, I replaced it seperatly
# 
# But what I concerned is, is it okay to replace each other?
# 
# Let's think aout this
# 
# We make a model refer to train dataset not test data, which we don't know what kind of value there are
# 
# And if we let the model learn from train dataset, which doesn't has a categorical value from test dataset, will the model predict and work well?
# 
# please leave your idea.

# In[ ]:


for title in train_data[train_data.Age.isna()].title.value_counts().index:
    mean_age = train_data.groupby('title').mean().T[title].Age
    mean_age_list = train_data[train_data.title == title].Age.fillna(mean_age)
    train_data.update(mean_age_list)
train_data.Age.isna().sum()


# In[ ]:


for title in test_data[test_data.Age.isna()].title.value_counts().index:
    mean_age = train_data.groupby('title').mean().T[title].Age
    mean_age_list = test_data[test_data.title == title].Age.fillna(mean_age)
    test_data.update(mean_age_list)
test_data.Age.isna().sum()


# In[ ]:


train_data.count()


# In[ ]:


test_data.count()


# In[ ]:


test_data.Fare = test_data.Fare.fillna(train_data.Fare.mean())


# In[ ]:


most_frequnt_embarked_value = train_data.Embarked.value_counts()
train_data.Embarked = train_data.Embarked.fillna(most_frequnt_embarked_value.index[0])


# In[ ]:


train_data = train_data.drop(['Cabin','Ticket','PassengerId','Name'], axis= 1)
test_data = test_data.drop(['Cabin','Ticket','PassengerId','Name'], axis= 1)


# In[ ]:


train_data.select_dtypes(include='object').T.index


# In[ ]:


test_data.select_dtypes(include='object').T.index


# In[ ]:


train_data.count()


# In[ ]:


test_data.count()


# In[ ]:


label_encoder = LabelEncoder()


# In[ ]:


for col in list(train_data.select_dtypes(include='object').T.index):
    print(col)


# In[ ]:


train_data.head()


# In[ ]:


test_data.head()


# Here is another(?) topic!
# 
# I fitted the Labelencoder by using test dataset.
# 
# It's pretty similar what I mentioned about.

# In[ ]:


for col in train_data.select_dtypes(include='object').T.index:
    train_data[col] = label_encoder.fit_transform(train_data[col])
    test_data[col] = label_encoder.fit_transform(test_data[col])


# In[ ]:


X_train = train_data.drop('Survived',axis=1)
y_train = train_data.Survived


# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,test_size =0.2, random_state =2045)


# In[ ]:


modeler = RandomForestClassifier(random_state=2045)


# In[ ]:


modeler.fit(X_train,y_train)
pred = modeler.predict(X_valid)


# In[ ]:


np.sum(pred == y_valid)/len(pred == y_valid)


# In[ ]:


final_pred = modeler.predict(test_data)


# In[ ]:


final_pred


# In[ ]:


PassengerId = pd.read_csv("../input/titanic/test.csv").PassengerId


# In[ ]:


final = pd.DataFrame({'PassengerId':PassengerId,'Survived':final_pred})


# In[ ]:


final.to_csv('submission_MJ.csv', index=False)


# With this process, I got 76% score. 
# 
# You and I could improve a model based on this or some other faboulaus method!
# 
# Please leave a comment, if you have a good idea and interesting topic!
