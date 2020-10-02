#!/usr/bin/env python
# coding: utf-8

# # <center>Titanic Dataset for Beginners</center>

# I'm a beginner in the field of data science. I'm very much interested in data analytics. This is my first kernel on kaggle. I am using the titanic dataset, which is very popular among the beginners that are using kaggle. I will predict the survival rate of passengers. Since I'm a beginner, I'm using simple approaches to reach at a solution.

# ### 1. Import the required packages

# In[ ]:


# To analyse
import pandas as pd
import numpy as np

#To visualise
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
get_ipython().run_line_magic('matplotlib', 'inline')
#This will display the plots below the code and store it in the notebook itself

#To ignore warnings
import warnings
warnings.filterwarnings('ignore')


# ### 2. Read the input files

# We are inputting two files
# 1. Test.csv
# 2. Train.csv
# 
# We will train the model using 'Train.csv' and we are going to test it using 'Test.csv'

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# ### 3. Analyse the data

# First we will analyse the 'train' data. The following code shows a brief description about it.

# In[ ]:


print(train.columns)
train.sample(5)


# In[ ]:


train.describe(include='all')


# Check for any null values

# In[ ]:


print(pd.isnull(train).sum())
print(pd.isnull(train).mean())


# There are a total of 891 rows (passengers) in the dataset.
# 
# The cabin column has 687 (77.1%) null values. As the null values are higher, we can drop this column when we are training the model.
# 
# The age column has 177 (19.8%) null values. It is better to modify this column rather than dropping it because age is an important aspect in determining the survival rate.
# 
# The embarked column has 2 (0.2%) null values. We can ignore this column as it will not make that much impact when we are training our model.
# 
# Rest of the columns has no null values. 

# ### Visualise the data

# We can visualise the data on the basis of different columns. Age, Sex, Pclass and Sibsp are some of the important columns to be considered while visualising the data.

# #### Based on Sex

# In[ ]:


sns.barplot(x="Sex",y="Survived",data = train)
plt.show()


# In[ ]:


train[['Sex','Survived']].groupby('Sex').mean()*100


# We can see that 74.2% of females survived compared to 18.89% of males. So females have much higher chance of survival.

# #### Based on PClass

# In[ ]:


sns.barplot(x="Pclass",y="Survived",data = train)
plt.show()


# In[ ]:


train[['Pclass','Survived']].groupby('Pclass').mean()*100


# 62.96% of the people in Pclass 1 survived.
# 
# 47.28% of the people in Pclass 2 survived.
# 
# 24.23% of the people in Pclass 3 survived.
# 
# 

# So people in Pclass 1 had a higher chance of survival and as the Pclass level increased, the rate of survival decreased.

# #### Combining PClass and Sex

# In[ ]:


sns.catplot(x='Sex', y='Survived',  kind='bar', data=train, hue='Pclass')
plt.show()


# From the above graph, we can see that almost all of the females in Pclass 1 survived.
# 
# So the females in PClass 1 and 2 had the highest chance to survive.

# #### Based on SibSp

# SibSp denotes the number of siblings and spouses

# In[ ]:


sns.barplot(x="SibSp",y="Survived",data = train)
plt.show()


# People with 2 or less sibilings and spouses had more chance to survive as they looked after only less number of people.

# In[ ]:


train[['SibSp','Survived']].groupby("SibSp").mean()*100


# 34.53% of people with 0 sibilings or spouses survived.
# 
# 53.58% of people with 1 sibiling or spouse survived. (More chance to survive as they helped each other)
# 
# 46.42% of people with 2 sibilings or spouses survived 
# 
# 

# #### Based on age

# Now we can compare the rate of survival based on age.
# 
# But as several values of the age column is null, we need to modify age column. Instead of finding the age of each and every person, we can group each person on the basis of their age group(Like child,adult,elder etc.)

# In[ ]:


train["Age"] = train["Age"].fillna(-0.5)
test["Age"] = test["Age"].fillna(-0.5)
value = [-1, 0, 5, 12, 18, 30, 65, 100]
names = ['Missing', 'Baby', 'Child', 'Teen', 'Youth', 'Adult', 'Elder']
train['AgeGroup'] = pd.cut(train["Age"], value, labels = names)
test['AgeGroup'] = pd.cut(test["Age"], value, labels = names)



# In[ ]:


sns.barplot(x="AgeGroup", y="Survived", data=train)
plt.show()


# Babies had the highest chance to survive as they were looked after by the adults. A baby would always have an adult accompanying him and at the time of rescue, these babies will be given more priority.
# 
# Elders had the least chance to survive as they became extremely fatigued.

# ### 5. Process the data 

# Now we should remove the Null values from the dataset and get them ready for training

# We have Null values in age column. To properly give a value to the missing places, we should compare it with the title(Mr, Mrs etc.) which is obtained from the name of the passenger.Through the title, we can estimate the approximate age of the passenger and add that value to the age column.

# First we should extract title from all the names and add another column ("Title) to the dataset

# In[ ]:


train['Name'].head()


# By looking at the above output,we can see that the title is the second word. So we can extract the title from the second word of the name. Be careful to remove the punctuation marks.

# In[ ]:


for item in [train,test]:
    item['Title'] = item['Name'].str.split(',').str[1].str.split('.').str[0].str.strip()
    # Here we split the second word from the name and stripped the excess whitespaces.


# In[ ]:


pd.crosstab(train['Title'],train['Sex'])


# Earlier we have replaced Nan values in the age column with -0.5. We should replace that with Nan as -0.5 would interfere when calculating mode age.

# In[ ]:


train["Age"] = train["Age"].replace({-0.5:np.nan})
test["Age"] = test["Age"].replace({-0.5:np.nan})


# From the above table we can see the relation between title and sex. We can compare it with the title of the passengers with the missing ages. We can replace the Nan value with the mode age of the correspoding title.

# In[ ]:


Null_List = train[train['Age'].isna()].groupby('Title').count()['Survived']
Null_Title = Null_List.index.values
Null_List


# The above output corresponds to the number of passengers with the age column as Nan. We can replace Nan with the mode age of the respective title

# In[ ]:


for item in Null_Title:
        val = train[train.Title == item]['Age'].median()
        train_list = train[(train.Title==item)& (train.AgeGroup == 'Missing')].index
        for elem in train_list:
            train.iloc[elem,train.columns.get_loc('Age')]=val
    


# Now the missing Age values are filled (At least somewhat accurately). Repeat the same for test dataset.
# 

# In[ ]:


Null_List = test[test['Age'].isna()].groupby('Title').count()['PassengerId']
Null_Title = Null_List.index.values
for item in Null_Title:
        val = train[train.Title == item]['Age'].median()
        test_list = test[(test.Title==item) & (test.AgeGroup == 'Missing')].index
        for elem in test_list:
            test.iloc[elem,test.columns.get_loc('Age')]=val
    


# For the rest of the columns we can either drop them or map them with integer values(For better modelling)

# #### Fare Column

# We can map fare column into four groups from 1 to 4 based on its value. We can also fill the missing fare value in the test dataset.

# In[ ]:


# First we can fill the missing value

test[pd.isnull(test)['Fare']]



# So the fare of only one passenger is missing. To fill it, we can find the mean fare of the passengers in Pclass column.

# In[ ]:


val = int(test[pd.isnull(test)['Fare']].Pclass)
amount = round(test[test.Pclass ==val].Fare.mean())
id = test[pd.isnull(test)['Fare']].index
test.iloc[id,test.columns.get_loc('Fare')]=amount


# Next we can map fare values into four groups based on its values

# In[ ]:


train['Fare'] = pd.cut(train['Fare'], 4, labels = [1, 2, 3, 4])
test['Fare'] = pd.cut(test['Fare'], 4, labels = [1, 2, 3, 4])


# Check to see if there are any null values in the train dataset.

# In[ ]:


pd.isnull(train).sum()


# There are 687 values missing in the cabin column which is too much to predict. So we can drop the cabin column.

# In[ ]:


train = train.drop(['Cabin'], axis = 1)
test = test.drop(['Cabin'], axis = 1)


# There are 2 missing values in Embarked column, we can fill that by taking the mode value in the Embarked column

# In[ ]:


print(train.groupby("Embarked").count()["Survived"])


# The most repeating value in the Embarked column is S (Southampton) which is repeated 644 times.
# 
# So we can fill the missing values with "S"

# In[ ]:


train = train.fillna({"Embarked": "S"})


# #### Name and Ticket column

# We can drop both name and ticket column as they are no longer useful for us

# In[ ]:


train = train.drop(['Name'], axis = 1)
test = test.drop(['Name'], axis = 1)
train = train.drop(['Ticket'], axis = 1)
test = test.drop(['Ticket'], axis = 1)


#  #### Sex and Embarked column

# We can map values in the sex column with numerical values (0 for male and 1 for female)
# 

# In[ ]:


sex_num = {"male":0,"female":1}
train["Sex"] = train["Sex"].map(sex_num)
test["Sex"] = test["Sex"].map(sex_num)


# We can map values in the Embarked column with numerical values (0 for male and 1 for female)

# In[ ]:


embarked_num = {"S": 1, "C": 2, "Q": 3}
train['Embarked'] = train['Embarked'].map(embarked_num)
test['Embarked'] = test['Embarked'].map(embarked_num)


# Check for any null values

# In[ ]:


print(pd.isnull(test).sum())
print(pd.isnull(train).sum())


# No null values means we are almost ready for creating a model from the data

# Check a sample of the train dataset

# In[ ]:


train.sample(5)


# Since AgeGroup and Title contains string values, we need to replace them with integer values before creating the model

# We can simply map the values in the AgeGroup with numerical values

# In[ ]:


group = list(map(str,train.AgeGroup.unique().sort_values()))
val = pd.Series(group)
print(val)


# Since we have already filled the missing ages, we don't require the missing column anymore.

# In[ ]:


value = [0, 5, 12, 18, 30, 65, 100]
names = ['Baby', 'Child', 'Teen', 'Youth', 'Adult', 'Elder']
train['AgeGroup'] = pd.cut(train["Age"], value, labels = names)
test['AgeGroup'] = pd.cut(test["Age"], value, labels = names)


# In[ ]:


group = list(map(str,train.AgeGroup.unique().sort_values()))
val = pd.Series(group)
print(val)


# In[ ]:


item = val.to_dict()
item


# We need to invert this dict to properly map the AgeGroup column

# In[ ]:


item = {v: k for k, v in item.items()}
item


# Now we can map AgeGroup with its respective numerical values

# In[ ]:


train['AgeGroup'] = train['AgeGroup'].map(item)
test['AgeGroup'] = test['AgeGroup'].map(item)


# For the title column, we can replace various titles with their column names

# In[ ]:


train['Title'] = train['Title'].replace('Ms', 'Miss').replace('Mme', 'Mrs').replace('Mlle', 'Miss').replace(['Countess', 'Lady', 'Sir'], 'Royal').replace(['Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Special')
test['Title'] = test['Title'].replace('Ms', 'Miss').replace('Mme', 'Mrs').replace('Mlle', 'Miss').replace(['Countess', 'Lady', 'Sir'], 'Royal').replace(['Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Special')

title_num = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Special": 6}
train['Title'] = train['Title'].map(title_num)
train['Title'] = train['Title'].fillna(0)
test['Title'] = test['Title'].map(title_num)
test['Title'] = test['Title'].fillna(0)


# That's it we're good to go. Let's get to modelling.

# ### Selecting the best model

# I will be using 5 different models to test the data and will select the best one out of it.
# 
# The models used are:
# 
# 1. Logistic Regression
# 2. Support Vector Machines (SVM)
# 3. K-Nearest Neighbours (KNN)
# 4. Gradient Boosting Classifier
# 5. Random Forest Classifier

# First we need to split the train dataset to compare the prediction of models.

# In[ ]:


from sklearn.model_selection import train_test_split

predictors = train.drop(['Survived', 'PassengerId'], axis=1)
target = train["Survived"]
x_train, x_val, y_train, y_val = train_test_split(predictors, target, random_state = 0)


# #### 1. Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_predict = logreg.predict(x_val)
result1 = round(accuracy_score(y_predict, y_val) * 100, 2)
print(result1)


# #### 2. Support Vector Matrices

# In[ ]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

svc = SVC()
svc.fit(x_train, y_train)
y_predict = svc.predict(x_val)
result2 = round(accuracy_score(y_predict, y_val) * 100, 2)
print(result2)


# #### 3. K-Nearest Neighbours

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_predict = knn.predict(x_val)
result3 = round(accuracy_score(y_predict, y_val) * 100, 2)
print(result3)


# #### 4. Gradient Boosting Classifier

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_predict = gbk.predict(x_val)
result4 = round(accuracy_score(y_predict, y_val) * 100, 2)
print(result4)


# #### 5.Random Forest Classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_predict = randomforest.predict(x_val)
result5 = round(accuracy_score(y_predict, y_val) * 100, 2)
print(result5)


# Compare all the models

# In[ ]:


model = ["Logistic Regression","Support Vector Machines (SVM)","K-Nearest Neighbours (KNN)","Gradient Boosting Classifier","Random Forest Classifier"]
value = [result1,result2,result3,result4,result5]
result = pd.DataFrame({"Model":model,"Value":value}).sort_values(by="Value", ascending = False)
result


# So the best model is Gradient Boosting Classifier. We can use it to predict the survival of test dataset.

# ### Creating the submission file

# We require two columns: PassengerId and Survived of the test dataset.

# In[ ]:


index = test['PassengerId']
prediction = gbk.predict(test.drop('PassengerId', axis=1))

output = pd.DataFrame({ 'PassengerId' : index, 'Survived': prediction })
output.to_csv('submission.csv', index=False)

