#!/usr/bin/env python
# coding: utf-8

# **EXPLORATORY DATA ANALYSIS FOR TITANIC SURVIVAL STATUS**

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


import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd


titanic_data = pd.read_csv("../input/titanic/train.csv")
titanic_data.shape


# The training dataset for Titanic has 12 features and 891 rows of data

# In[ ]:


titanic_data.head()


# In[ ]:


titanic_data['Survived'].value_counts()


# From the datset we know that 342 survived but 549 died in the Titanic disaster.

# In[ ]:


titanic_data["Survived"] = titanic_data["Survived"].apply(lambda x: "Alive" if x == 1  else "Dead")
titanic_data.head()


# In[ ]:


titanic_data.columns


# In[ ]:


titanic_data.describe()


# # OBJECTIVE
# Determine the best features which help in the classifying the people in to either "Survived" or "Dead" Category

# In[ ]:


gender_count = titanic_data.Sex.groupby(titanic_data.Sex).count()
print("Total count of men and Women")
print(gender_count)

print("\nPercentage of female on ship: {}".format(gender_count['female']*100/(gender_count['female']+gender_count['male'])))
print("\nPercentage of male on ship: {}".format(gender_count['male']*100/(gender_count['female']+gender_count['male'])))

titanic_data_survived = titanic_data[titanic_data.Survived == "Alive"]
print("\n")
survival_gender_count = titanic_data_survived['Sex'].groupby(titanic_data_survived.Sex).count()
print("Survival Count of Men and Women")
print(survival_gender_count)
print("\nPercentage of female survived: {}".format(survival_gender_count['female']*100/(survival_gender_count['female']+survival_gender_count['male'])))
print("\nPercentage of male survived: {}".format(survival_gender_count['male']*100/(survival_gender_count['female']+survival_gender_count['male'])))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")
sns.countplot(x="Sex", hue='Survived',data=titanic_data)
plt.show()


# From the above statistics and the plot it is evident that women had higher chances of survival than men, and hence it is an important feature which can help in the classification.

# In[ ]:


plt.figure(figsize=(16, 6))
sns.violinplot(x="Survived", y="Age", data=titanic_data, size=8,hue='Survived')   .set(title='Violin plot for age',yticks=np.arange(0,100,5))
plt.show()


# From the above violin plot it is evident that if the age is in the range 0-10, there is more chance of survival.
# Conclusion is that children have higher chances of survival than others.

# In[ ]:


sns.FacetGrid(titanic_data, hue="Survived",palette = ['Red','Blue'],size=8)    .map(sns.distplot, "Fare")    .set(title='Histogram/PDF for Fare')   .add_legend();
plt.show();


# Peak for 'Dead' PDF is much higher than 'Alive' PDF, so people who paid lesser fare have lesser chances of survival than people who paid higher fare. Also the PDF for 'Alive' decays slower than 'Dead', hence higher fare means higher chances of survival. 

# In[ ]:



sns.set_style("whitegrid")
sns.countplot(x="Pclass", hue='Survived',data=titanic_data)
plt.show()


# From the above CountPlot it can be concluded that people in Pclass 1 has higher chance of survival. For Pclass 2 there is equal chance of survival and death while Pclass 3 has very high chance of death. Hence Pclass is also an important feature for classification. 

# In[ ]:


sns.countplot(x="Embarked", hue='Survived',data=titanic_data)
plt.show()


# In[ ]:


embarked_count = titanic_data.Embarked.groupby(titanic_data.Embarked).count()
print(embarked_count)

print("\nEmbarked survived")
embarked_survival_count = titanic_data_survived.Embarked.groupby(titanic_data_survived.Embarked).count()
print(embarked_survival_count)

print("Percentage of people survived who embarked from Cherbourg is {}".format(embarked_survival_count['C']*100/embarked_count['C']))
print("Percentage of people survived who embarked from Queenstown is {}".format(embarked_survival_count['Q']*100/embarked_count['Q']))
print("Percentage of people survived who embarked from Southampton is {}".format(embarked_survival_count['S']*100/embarked_count['S']))


# From above statistics it is evident people who embarked from Cherbourg had highest chance of survival followed by Queenstown and Southampton.

# In[ ]:


sns.countplot(x="SibSp", hue='Survived',data=titanic_data)
plt.show()


# In[ ]:


sns.countplot(x="Parch", hue='Survived',data=titanic_data)
plt.show()


# Number of siblings / spouses and number of parents / children does not seem to have any significant impact on the survivability of a person.

# From the above analysis, we can conclude that Age,Sex,Fare,Pclass and Embarked are the most important features that can be used for classification.
# 
# * Children and women  had higher chances of survival.
# * Plcass 1 had higher chances of survival than 2 and 3.
# * Higher fare also indicated higher chances of survival.
# * People who embarked from Cherbourg had higher chances of survival than Queenstown and Southampton.

# In[ ]:


titanic_data["Survived"] = titanic_data["Survived"].apply(lambda x: 1 if x == 'Alive'  else 0)
titanic_data.head()


# In[ ]:


#Removing columns not required for creating the model
titanic_model_data = titanic_data.drop(columns=['PassengerId','SibSp','Parch','Ticket','Cabin','Name'])


# In[ ]:


titanic_model_data.head()


# In[ ]:


print(titanic_model_data[['Age']].mean())
print(titanic_model_data[['Survived','Embarked']].groupby(titanic_model_data.Survived).count())


# In[ ]:


titanic_model_data['Age'] = titanic_model_data['Age'].replace(np.nan, 29.699118)


# In[ ]:


#Checking for NaN values
titanic_model_data.isnull().sum().sum()


# In[ ]:


titanic_model_data['Embarked'] = titanic_model_data['Embarked'].replace(np.nan, 'C')


# In[ ]:


titanic_model_data.head()


# In[ ]:



titanic_model_data_Y = titanic_model_data.drop(columns=['Pclass','Sex','Age','Fare','Embarked'])
titanic_model_data_X = titanic_model_data.drop(columns=['Survived'])
print(titanic_model_data_Y.head(2))
print(titanic_model_data_X.head(2))


# In[ ]:


titanic_model_data_X.shape


# In[ ]:



from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder() 
titanic_model_data_X['Pclass']= le.fit_transform(titanic_model_data['Pclass']) 
titanic_model_data_X['Sex']= le.fit_transform(titanic_model_data['Sex']) 
titanic_model_data_X['Embarked']= le.fit_transform(titanic_model_data['Embarked'])
titanic_model_data_X.head()


# In[ ]:


titanic_model_data_X.shape


# In[ ]:


onehotencoder = OneHotEncoder(categorical_features=[0,4]) 
titanic_encoded = onehotencoder.fit_transform(titanic_model_data_X)


# In[ ]:


print(titanic_encoded.shape)


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(with_mean=False)
titanic_scaled = scaler.fit_transform(titanic_encoded)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

knnmodel = KNeighborsClassifier(n_neighbors=3)

knnmodel.fit(titanic_scaled,titanic_model_data_Y)


# In[ ]:


titanic_test_data = pd.read_csv("../input/titanic/test.csv")


# In[ ]:


titanic_test_data.head()


# In[ ]:


titanic_test_data = titanic_test_data.drop(columns=['PassengerId','SibSp','Parch','Ticket','Cabin','Name'])


# In[ ]:


titanic_test_data.head()


# In[ ]:


lenew = LabelEncoder()
titanic_test_data['Pclass']= lenew.fit_transform(titanic_test_data['Pclass']) 
titanic_test_data['Sex']= lenew.fit_transform(titanic_test_data['Sex']) 
titanic_test_data['Embarked']= lenew.fit_transform(titanic_test_data['Embarked'])
titanic_test_data.head()


# In[ ]:


titanic_test_data.shape


# In[ ]:


print(titanic_test_data.isnull().sum().sum())


# In[ ]:


print(titanic_test_data['Age'].mean())


# In[ ]:


titanic_test_data['Age'] = titanic_test_data['Age'].replace(np.nan, 30.272590361445783)


# In[ ]:


titanic_test_data.head()


# In[ ]:


print(titanic_test_data['Fare'].isnull().sum().sum())


# In[ ]:


print(titanic_test_data['Fare'].mean())


# In[ ]:


titanic_test_data['Fare'] = titanic_test_data['Fare'].replace(np.nan, 35.6271884892086)


# In[ ]:


print(titanic_test_data.isnull().sum().sum())


# In[ ]:


onehotencodern = OneHotEncoder(categorical_features=[0,4]) 
titanic_test_encoded = onehotencodern.fit_transform(titanic_test_data)


# In[ ]:


titanic_test_encoded.shape


# In[ ]:


scalern = StandardScaler(with_mean=False)
titanic_scaled = scalern.fit_transform(titanic_test_encoded)


# In[ ]:


predicted_y = knnmodel.predict(titanic_scaled)


# In[ ]:


passengerId = 892
output = []
for i in range(len(predicted_y)):
    l = []
    l.append(passengerId)
    l.append(predicted_y[i])
    output.append(l)
    passengerId+=1


# In[ ]:


output


# In[ ]:


outdf = pd.DataFrame(output, columns = ['PassengerId', 'Survived'])


# In[ ]:


outdf


# In[ ]:


outdf.to_csv('titanic_submission.csv',index=False)
from IPython.display import FileLink
FileLink('titanicsubmission.csv')


# <a href="./titanicsubmission.csv"> Download File </a>

# In[ ]:




