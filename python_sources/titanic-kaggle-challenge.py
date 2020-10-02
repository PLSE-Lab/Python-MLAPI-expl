#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Kaggle Titanic 
# ### Logistic Regression with Python

# ## Step - 1: Frame The Problem 
# 
# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.

# 

# ## Step - 2: Obtain the Data

# ### Import Libraries

# In[ ]:


get_ipython().system('pwd')


# In[ ]:


get_ipython().system('pip install -q  missingno')


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as ms
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


get_ipython().system('ls -l')


# Pandas provides two important data types with in built functions to be able to provide extensive capability to handle the data.The datatypes include Series and DataFrames.

# Pandas provides ways to read or get the data from various sources like read_csv,read_excel,read_html etc.The data is read and stored in the form of DataFrames.

# In[ ]:


get_ipython().system('wget -q https://www.dropbox.com/s/8grgwn4b6y25frw/titanic.csv')


# In[ ]:


get_ipython().system('ls -l')


# In[ ]:


data = pd.read_csv("../input/train.csv")


# In[ ]:


data.head(3)


# In[ ]:


#to get the last 5 entries of the data
data.tail(5)


# In[ ]:


type(data)


# In[ ]:


data.shape


# we have 891 rows and 12 columns

# In[ ]:


data.info()


# In[ ]:


data.isnull().sum()


# In[ ]:


data.info()


# In[ ]:


data.describe()


# The statistics shows the variable Age has 177 missing values
# 

# ## Step - 3: Analyse the Data

# In[ ]:


ms.matrix(data)


# In[ ]:


data.info()


# We can observe that there are missing values in 'Age', 'Cabin' and 'Embarked'. Lets continue

# #### Visualization of data with Seaborn

# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',data=data,palette='RdBu_r')


# The target variable is the Survived column such that if survival = 1, the passenger is alive, otherwise dead.
# 

# The graph shows the number of dead passengers are higher than the survivals by more than 1/3.

# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=data,palette='RdBu_r')


# The above graph also illustrates that female passangers survived by more than 50% of their counterpart male passangers. Thus very few females are dead as compared to males (about 20% of the males who succumbed during the accident) 

# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data = data,palette='rainbow')


# Passangers who are onboarded in the lower class had lower chance of survivial and thus around 375 people died from third class while only less than 175 people are dead from both second and third class. This is presumably correct as passangers with higher status are likely to be rescued or are given first priority of safety.  

# In[ ]:


sns.distplot(data['Fare'])
#KDE?


# The graph is highly skewed to the left. Most of the passengers pay cheaper tickets and a very few people bought expensive tickets.  

# In[ ]:


data['Fare'].hist(color = 'green', bins = 40, figsize = (8,3))


# In[ ]:


data.corr()


# In[ ]:


sns.heatmap(data.corr(),cmap='coolwarm')
plt.title('data.corr()')


# In[ ]:


sns.swarmplot
sns.swarmplot(x='Pclass',y='Age',data=data,palette='Set1')


# The balched feature at the middle shows us more people are concentrated within the rangeof that age

# In[ ]:


data['Age'].hist(bins = 40, color = 'darkred', alpha = 0.8)


# Here the histogram shows bimodal distribution of age, though age of the passangers seems to concentrate between 15 and 35 years of age. 

# In[ ]:





# ## Step - 4: Feature Engineering
# 
# We want to fill the missing values of the age in the dataset with the average age value for each of the classes. This is called data imputation.

# In[ ]:


data.info()


# In[ ]:


plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=data,palette='winter')


# In[ ]:


data['Age'].fillna(28, inplace=True)


# In[ ]:


data['Age'].median()


# In[ ]:


ms.matrix(data)


# In[ ]:


data['Cabin'].value_counts()


# In[ ]:





# Applying the function.

# In[ ]:


data.info()


# In[ ]:


data.drop('Cabin',axis=1, inplace=True)


# In[ ]:


data.head()


# In[ ]:


data['Embarked'].value_counts()


# In[ ]:


data.dropna(inplace = True) # dropping missing embarked.


# In[ ]:


ms.matrix(data)


# In[ ]:


data.info()


# ## Converting Catagorical Features
# 
# We'll need to convert categorical features to dummy variables using pandas! Otherwise our machine learning algorithm won't be able to directly take in those features as inputs.

# In[ ]:


data['Sex'].value_counts()


# In[ ]:


sex = pd.get_dummies(data['Sex'],drop_first=True)
sex.head()


# In[ ]:


data['Embarked'].value_counts()


# In[ ]:


embark = pd.get_dummies(data['Embarked'],drop_first=True)
embark.head(10)


# In[ ]:


sex.head()


# In[ ]:


old_data = data.copy()
data.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
data.head()


# In[ ]:


data = pd.concat([data,sex,embark],axis=1)


# In[ ]:


data.dropna(inplace = True) # dropping missing embarked.data.info()
data.info()


# In[ ]:


data.describe()


# ## Step - 5:Model Selection

# ### Building a Logistic Regression model

# In[ ]:


X = data.drop('Survived',axis=1)
y = data['Survived']


# In[ ]:


y.shape


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y 
                                                    , test_size=0.20, 
                                                    random_state=42)


# In[ ]:


X_test.shape


# In[ ]:


len(y_test)


# In[ ]:


178/889


# In[ ]:


X.describe()


# In[ ]:


X_train.describe()


# In[ ]:


y_train.describe()


# In[ ]:


from sklearn.linear_model import LogisticRegression

# Build the Model.
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train) # this is where training happens


# In[ ]:


logmodel.coef_


# In[ ]:


logmodel.intercept_


# In[ ]:


predict =  logmodel.predict(X_test)
predict[:5]


# In[ ]:


y_test[:5]


# Let's move on to evaluate our model.

# ## Step - 6 : Evaluation

# ### Evaluation
# We can check precision, recall, f1 - score using classification report!

# #### Confusion Matrix

# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report


# #### Confusion 
# 
# True positive | False positive,
# ____|____
# |
# False negative | True negative

# In[ ]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predict))


# In[ ]:


print(confusion_matrix(y_test, predict))


# #### Precision Score

# The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.
# 
# The best value is 1 and the worst value is 0.

# In[ ]:


from sklearn.metrics import precision_score


# In[ ]:


print(precision_score(y_test,predict))


# #### Recall score

# The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.
# 
# The best value is 1 and the worst value is 0.

# In[ ]:


from sklearn.metrics import recall_score


# In[ ]:


print(recall_score(y_test,predict))


# #### f_score

# The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal. The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)

# In[ ]:


from sklearn.metrics import f1_score


# In[ ]:


print(f1_score(y_test,predict))


# To get all the above metrics at one go, use the following function:

# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(y_test,predict))


# ## Step - 7 : Predict on New Cases

# ### Prediction on Test Data From Kaggle

# In[ ]:


prod_data=pd.read_csv('../input/test.csv')


# In[ ]:


prod_data.info()


# In[ ]:


ms.matrix(prod_data)


# ### Data Cleaning

# There are inconsistencies in test data.We can use the same graph functions that are used to visualize the train data for test data as well.We use the same data cleaning techniques like removing the cabin column and applying impute_age function on age column on test data.
# But we cannot remove any rows because kaggle wants same number of rows in submission csv also. So we fill the missing values in fare with mean.

# In[ ]:


prod_data['Age'].fillna(28, inplace=True)


# In[ ]:


ms.matrix(prod_data)


# In[ ]:


prod_data.drop('Cabin', axis = 1, inplace= True)


# In[ ]:


ms.matrix(prod_data)


# In[ ]:


prod_data.fillna(prod_data['Fare'].mean(),inplace=True)


# In[ ]:


prod_data.info()


# In[ ]:


ms.matrix(prod_data)


# In[ ]:


sex = pd.get_dummies(prod_data['Sex'], drop_first=False)
embark = pd.get_dummies(prod_data['Embarked'], drop_first=False)


# In[ ]:


sex = pd.get_dummies(prod_data['Sex'], drop_first=False)
embark = pd.get_dummies(prod_data['Embarked'], drop_first=False)




# In[ ]:


prod_data.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[ ]:


prod_data = pd.concat([prod_data,sex,embark],axis=1)


# In[ ]:


prod_data.head()


# In[ ]:


prod_data.drop(["female", 'C'], axis = 1, inplace = True)


# In[ ]:


prod_data.head()


# In[ ]:


prod_data.info()


# In[ ]:


prod_data['Fare'].fillna(prod_data['Fare'].median(), inplace = True)


# In[ ]:


prod_data.info()


# In[ ]:


predict1=logmodel.predict(prod_data)
predict1


# In[ ]:


df1=pd.DataFrame(predict1,columns=['Survived'])


# In[ ]:


df2=pd.DataFrame(prod_data['PassengerId'],columns=['PassengerId'])


# In[ ]:


df2.head()


# In[ ]:


result = pd.concat([df2,df1],axis=1)
result.head()


# In[ ]:


result.to_csv('result.csv',index=False)


# In[ ]:





# In[ ]:




