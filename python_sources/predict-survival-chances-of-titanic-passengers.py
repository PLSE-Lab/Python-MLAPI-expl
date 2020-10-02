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


# Importing required Libraries
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


import pandas as pd
dataset = pd.read_csv("../input/titanic/train.csv",header=0)
dataset.head()


# *A few attributes were unnecessary, such as Name, passengerId and the ticket number, so we remove them*

# In[ ]:


dataset = dataset.drop(columns=['Name', 'PassengerId','Ticket'], axis=0)
dataset.head()


# **Analysis of data**
# 
# *Next, I performed some exploratory data analysis, to look for some trends in data, like relation of Sex and Survival chance, age and survival chance and the class of passengers and their survival chance. Here are the plots*
# 

# In[ ]:


ax = sns.countplot(x="Pclass", hue = "Survived", data=dataset)


# **As we can see, more people belonging to class 1 survived and more people belong to class 3 died. This shows that there is *some* relation between class of passenger and their survival rates. It could be that, passengers of class 1 were kept on the upper decks and could be rescued before the other classes, hence the result**

# In[ ]:


ax1 = sns.countplot(x="Survived", hue = "Sex", data=dataset)


# **In this plot, we can see the relationship between the gender of a passenger and their chance of survival. The barplot clearly shows that, among the passengers who did not survive, a higher number of them were males. On the other hand, among those who survived, a majority of them were females. The reason for this could be that women were evacuated from the ship before men, and hence the result **

# Next we try to find the relation between the age of passengers and their survival chance:
# For that, we need to do some more operations on the data, to make it useful for plotting a chart

# In[ ]:


ageclass = ["0-20","21-40","41-60","61-80","81-100","100+"]
#We make groups of age, in the range of 20 years

agesurvival = pd.DataFrame(index = ageclass, columns = ['0','1'])
agesurvival.loc[:,:] = 0
#Initialising the Dataframe with 0

for index,rows in dataset.iterrows():
    if rows["Age"] <= 20:
        if rows["Survived"] == 0:
            agesurvival.loc["0-20"]['0'] += 1
        else:
            agesurvival.loc["0-20"]['1'] += 1
    elif rows["Age"] > 20 and rows["Age"] <= 40:
        if rows["Survived"] == 0:
            agesurvival.loc["21-40"]['0'] += 1
        else:
            agesurvival.loc["21-40"]['1'] += 1
    elif rows["Age"] > 40 and rows["Age"] <= 60:
        if rows["Survived"] == 0:
            agesurvival.loc["41-60"]['0'] += 1
        else:
            agesurvival.loc["41-60"]['1'] += 1
    elif rows["Age"] > 60 and rows["Age"] <= 80:
        if rows["Survived"] == 0:
            agesurvival.loc["61-80"]['0'] += 1
        else:
            agesurvival.loc["61-80"]['1'] += 1
    elif rows["Age"] > 80 and rows["Age"] <= 100:
        if rows["Survived"] == 0:
            agesurvival.loc["81-100"]['0'] += 1
        else:
            agesurvival.loc["81-100"]['1'] += 1
    elif rows["Age"] > 100:
        if rows["Survived"] == 0:
            agesurvival.loc["100+"]['0'] += 1
        else:
            agesurvival.loc["100+"]['1'] += 1
        
agesurvival.plot.bar(rot=0)

#We get the following graph:


# **The number of people who died in a certain age group, is maximum for the group "21 to 40 years". This might be because, children and senior citizens were given priority to evacuate from the ship**

# *Like these, more graphs can be plotted, that show how many men and women of age group 21 to 40 years died / lived, how many people of class 3 that survived were men/women and so on. For now, we plot these 3 graphs and move further.*

# **Preparing the data for predictions**

# In[ ]:


#Separating dependent and independent variables
x = dataset.iloc[:,1:]
y = dataset.iloc[:,0]


# In[ ]:


x.isna().sum()


# *As we can see 687 rows in the Cabin column are nan. which is about 77% of the records. We can safely discard this column as well*

# In[ ]:


del dataset["Cabin"]
#dataset = dataset.drop(columns=['Name', 'PassengerId','Ticket'], axis=0)
dataset.head()


# In[ ]:


#Separating dependent and independent variables. [Repeating without the 'Cabin' column]
x = dataset.iloc[:,1:]
y = dataset.iloc[:,0]
x


# **Taking care of missing values**

# *Here we replace the nan values in 'Age' column with the mean of ages in other column*

# In[ ]:


from sklearn.impute import SimpleImputer 
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(x.iloc[:, 2:3])
x.iloc[:, 2:3] = imputer.transform(x.iloc[:, 2:3])
x


# *Here we replace the nan values in 'Embarked' column with the most frequently occuring value in that column*

# In[ ]:


imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer = imputer.fit(x.iloc[:, 6:7])
x.iloc[:, 6:7] = imputer.transform(x.iloc[:, 6:7])
x


# **Taking care of categorical variables**

# In[ ]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X = LabelEncoder()

x.iloc[:, 0] = labelencoder_X.fit_transform(x.iloc[:,0])
x.iloc[:, 1] = labelencoder_X.fit_transform(x.iloc[:,1])
x.iloc[:, 6] = labelencoder_X.fit_transform(x.iloc[:,6])

columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0,1,6])], remainder='passthrough')
x = columnTransformer.fit_transform(x)


# In[ ]:


x


# **Feature Scaling**

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)


# Splitting data into test set and training set

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# **Importing the Keras libraries and packages**

# In[ ]:


import keras
#sequential - initialize NN
from keras.models import Sequential

#dense - build NN
from keras.layers import Dense,Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
#init - 
#activation - 
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 12))
classifier.add(Dropout(0.3))
# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
classifier.add(Dropout(0.3))
# Adding the third hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
classifier.add(Dropout(0.3))
# Adding the fourth hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
classifier.add(Dropout(0.3))
# Adding the output layer
#sigmoid - will give probabilities of classes
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6), loss = 'mean_squared_error', metrics = ['accuracy'])


# Fitting the ANN to the Training set
history = classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 500,validation_split=0.25)


# **Once our model is ready, we perform similar prepossessing steps on our testing data:**

# In[ ]:


#Preparing the test set
datasetest = pd.read_csv("../input/titanic/test.csv")
datasetest = datasetest.drop(columns=['Name', 'PassengerId','Ticket','Cabin'])

#Separating dependent and independent variables
x_test = datasetest.iloc[:,:]
#y_test = no independent variable in test set

#Taking care of missing values
from sklearn.impute import SimpleImputer 
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(x_test.iloc[:, 2:3])
x_test.iloc[:, 2:3] = imputer.transform(x_test.iloc[:, 2:3])

imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer = imputer.fit(x_test.iloc[:, 6:7])
x_test.iloc[:, 6:7] = imputer.transform(x_test.iloc[:, 6:7])


#Taking care of categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X = LabelEncoder()

x_test.iloc[:, 0] = labelencoder_X.fit_transform(x_test.iloc[:,0])
x_test.iloc[:, 1] = labelencoder_X.fit_transform(x_test.iloc[:,1])
x_test.iloc[:, 6] = labelencoder_X.fit_transform(x_test.iloc[:,6])

columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0,1,6])], remainder='passthrough')
x_test = columnTransformer.fit_transform(x_test)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_test = sc.fit_transform(x_test)


# In[ ]:


y_pred = classifier.predict(x_test)
y_pred1 = []
for i in range(len(y_pred)):
    if y_pred[i] > 0.5:
        y_pred1.append(1)
    else:
        y_pred1.append(0)


# In[ ]:


result = pd.DataFrame(columns=['PassengerId','Survived'])
for i in range(len(y_pred1)):
    result = result.append([{'PassengerId':i+892, 'Survived':y_pred1[i]}], ignore_index = True)


# In[ ]:


result.head()

