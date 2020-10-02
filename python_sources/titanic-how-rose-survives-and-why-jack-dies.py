#!/usr/bin/env python
# coding: utf-8

# # Titanic - How Rose survives and why Jack dies ?

# In this Titanic dataset we will analyse the factors which have contributed to the person's survival.
# We will use the Logistic Regression to predict if the person is survived or not.

# Let us import the necessary librariers

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
get_ipython().run_line_magic('matplotlib', 'inline')


# Load the Titanic dataset

# In[ ]:


data = pd.read_csv("../input/train.csv")


# Print top few rows to understand about the data

# In[ ]:


data.head()


# How many passengers are there ?

# In[ ]:


print("# of passengers in dataset:"+str(len(data)))


# Let us analyse the data by creating the different plots

# How many passengers survived ?

# In[ ]:


sns.countplot(x = "Survived", data = data);


# passengers survived based on the gender wise

# In[ ]:


sns.countplot(x = "Survived", hue = "Sex", data = data);


# Passengers survived based on the passenger class

# In[ ]:


sns.countplot(x = "Survived", hue ="Pclass", data = data);


# Let us explore the age of the passengers

# In[ ]:


data["Age"].plot.hist();


# Let us check the fare details also

# In[ ]:


data["Fare"].plot.hist(bins = 20, figsize=(10,5));


# Let us analyse the columns

# In[ ]:


data.info()


# check the siblings status on the ship

# In[ ]:


sns.countplot(x="SibSp", data = data);


# Data Cleaning / Data Wrangling ( we will remove  unnecessary columns wherever possible )

# In[ ]:


data.isnull()


# In[ ]:


data.isnull().sum()


# From above data we can notice that there are missing values in Age, Cabin and Embarked

# Let us see the null values using the heatmap

# In[ ]:


sns.heatmap(data.isnull(), yticklabels = False, cmap="viridis")


# Let us analyse the age column

# In[ ]:


sns.boxplot(x = "Pclass", y = "Age", data = data);


# From the above we can notice that the passengers travelling in the first class and second class are older than the 3rd class

# Imputation : we can drop the missing values or fill in some other values

# In the data set we have the column Survived which categorical.
# So we apply Logistic regression on the columns ( i.e we need to predict the y value )

# The cabin column has lot of null values, so we drop them

# In[ ]:


data.drop("Cabin", axis = 1, inplace = True)


# drop all NA values

# In[ ]:


data.dropna(inplace = True)


# Let use the heatmap again to check null values removed or not

# In[ ]:


sns.heatmap(data.isnull(), yticklabels = False, cbar = False);


# Let us check table again

# In[ ]:


data.isnull().sum()


# Now our data set is clean

# We see a lot of string values in our dataset, we need to convert it to categorical variables inorder to implement logistic regression. So the process is we will convert this to categorical variable into dummy variable as logistic regression takes only two values. So we will be creating dummy variables.

# In[ ]:


pd.get_dummies(data['Sex'])


# So '0' basically tells its not afemale and 1 tells it's a female column. Similar is the case with male column.

# We don't require both these columns. One column is enough to enough to tell whether it is male or female. So we will keep only one column, male column in this case. 

# In[ ]:


sex = pd.get_dummies(data['Sex'], drop_first = True)


# So the female column is dropped and let us print few columns

# In[ ]:


sex.head()


# Similarly we apply the dummy function on Embarked column also:

# In[ ]:


embark = pd.get_dummies(data['Embarked'])
embark.head()


# we have C, Q and S columns. Here also we can drop the first column as the other two columns are enough where it tells if the passenger is travelling for Q(Queen's town), S(Southampton) or if the both these values are 0 then we can assume that he/she is travelling for C(Cherbourg).
# 
# So let us drop the first value

# In[ ]:


embark = pd.get_dummies(data['Embarked'], drop_first=True)
embark.head(5)


# Applying dummy function on Pclass

# In[ ]:


pcl = pd.get_dummies(data['Pclass'])
pcl.head()


# In[ ]:


pcl = pd.get_dummies(data['Pclass'], drop_first = True)
pcl.head()


# We have 2 and 3, meaning if both these values are 0 then the passenger is travelling in the 1st class.
# 
# So the next step is we will concatenate all the above categorical values to the dataset

# In[ ]:


data = pd.concat([data, sex, embark, pcl], axis = 1)
data.head()


# It's time to drop the Pclass, Sex, and Embarked categorical data columns

# In[ ]:


data.drop(['Sex', 'Embarked', 'Pclass', 'PassengerId', 'Name', 'Ticket'], axis = 1, inplace = True)
data.head()


# ## Train and Test data

# we will build the model on the train data and predict output on test data

# In[ ]:


X = data.drop("Survived", axis = 1)


# in the above, except Survived all other columns will become the independent variables ( i.e features )

# In[ ]:


Y = data["Survived"]  # this is our target variable


# We will split data in training and test data sets

# In[ ]:


from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)


# We will train and predict by creating a model.

# In[ ]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train, Y_train)
predictions = logmodel.predict(X_test)


# It is time to evaluate how the model is performaing.
# 
# We can find the accuracy or classification report 

# In[ ]:


from sklearn.metrics import classification_report
classification_report(Y_test, predictions)


# Now let find the accuracy by creating a confusion matrix

# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test, predictions)


# calculate accuracy

# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(Y_test, predictions)


# So we got 78% accuracy
