#!/usr/bin/env python
# coding: utf-8

# Objective of this Kernel is to cover the basics of Python and its application in Data Science.This kernel is work in process and I will be updating this is coming days.If you like my work please fo vote.

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


# ## 1. Strings 
# **1.1)Center Method **: Here we will be putting symbols like * in front and after the String **titanic**

# In[ ]:


# we will add * in front and after the name philip.
name='Titanic'   #Name is defined as string  
#there are 6 letters in the name philip.We can use string center method to put star in front and behind the word philip 
name.center(8,'*')


# In[ ]:


name.center(10,'*')


# In[ ]:


name.center(8,'!')


# We will put four * in front and end of the string by taking name as input from user 

# In[ ]:


name='Ship'
#The number of letter in name is calculated by using length function and 8 is added to add 4 stars before and end of the string
print(name.center(len(name)+8,'*'))


# **1.2) String Concatenation** :It is used to combine two strings 

# In[ ]:


# We will be combaining the First and the Last Name 
first_name='Leonardo'
last_name='DiCaprio'
full_name=first_name+last_name
print(full_name)


# In[ ]:


# We will add space between the two names 
full_name=first_name+' ' +last_name
print(full_name)


# In[ ]:


print(first_name +'3')
print(first_name+str(3))
print(first_name*3)


# ## 2.Printing the version of the Python Modules used 

# In[ ]:


# Importing the modules for which we want to find out the Version
import matplotlib
import sklearn
import scipy 
import seaborn as sns
import pandas as pd 
import numpy as np
import sys
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')  
import warnings
warnings.filterwarnings('ignore')  #this will ignore the warnings.it wont display warnings in notebook


# In[ ]:


print('matplotlib: {}'.format(matplotlib.__version__))
print('matplotlib: {}'.format(matplotlib.__version__))
print('sklearn: {}'.format(sklearn.__version__))
print('scipy: {}'.format(scipy.__version__))
print('seaborn: {}'.format(sns.__version__))
print('pandas: {}'.format(pd.__version__))
print('numpy: {}'.format(np.__version__))
print('Python: {}'.format(sys.version))


# ## 3.Printing emojis with Python

# In[ ]:


# Link for getting the list of the Unicode emoji characters and sequences, with images from different vendors, CLDR name, date, source, and keywords - https://unicode.org/emoji/charts/full-emoji-list.html
# For the codes obtained from this website replace + with 000 and put \ in front of the code as dhown below 
#U+1F600 has to be represented as \U000F600 in the code 
print('\U0001F600')  #Grimming face emoji
print('\U0001F600','\U0001F600') # Printing the Emoji twice 
print('\U0001F600','\U0001F602','\U0001F643') # Printing three different emojis


# ## 4.Solving linear equation with Numpy 
# X1+X2=2200
# 
# 1.5X1+4X2=5050
# 
# Find the values of X1 and X2

# In[ ]:


A=np.array([[1,1],[1.5,4]])
b=np.array([2200,5050])
np.linalg.solve(A,b)


# X1=1500 & X2=700

# ## 5.Matplot Lib
#   a) Line Plot 

# In[ ]:


import matplotlib.pyplot as plt 
x=np.linspace(0,10,10)  # Equally spaced data with 10 points 
y=np.sin(x) # Generating the sine function 
plt.plot(x,y)
plt.xlabel('Time')     #Specifying the X axis label 
plt.ylabel('Speed')    #Specifying the Y axis label 
plt.title('My Cool Chart') #Specifying the title 
plt.show()


# b)Scatter plot

# In[ ]:


x=np.linspace(0,10,10)
y=np.array([1,5,6,3,7,9,13,50,23,56])
plt.scatter(x,y,color='r') # We can specify the color to the points
plt.xlabel('X-Value')     #Specifying the X axis label 
plt.ylabel('Y-Value')    #Specifying the Y axis label 
plt.title('My Cool Chart') #Specifying the title 
plt.show()


# ## 6.Pandas

# 
# **6.1 Importing the data with Pandas **

# In[ ]:


import pandas as pd
data=pd.read_csv('../input/train.csv')


# **6.2 Displaying the data**

# **Viewing head of the data **

# In[ ]:


data.head()


# **Viewing the tail of the data**

# In[ ]:


data.tail() 


# **6.3 Finding missing values**

# In[ ]:


Total=data.isnull().sum().sort_values(ascending=False)
Percent=round(Total/len(data)*100,2)
pd.concat([Total,Percent],axis=1,keys=['Total','Percent'])


# Cabin and Age Column have missing data.Age can be replaced with mean age.

# **6.4 Crosstab**

# In[ ]:


pd.crosstab(data.Survived,data.Pclass,margins=True).style.background_gradient(cmap='gist_rainbow') 
#Margins=True gives us the All column values that is sum of values colums


# We we can see that more people survived in the First class cabin and least survived in third class cabin.It clearly shows the rescue efferts were baised based on the class of the travellers.

# **6.5 Concanat ** to get  P Class details 

# In[ ]:


percent = pd.DataFrame(round(data.Pclass.value_counts(dropna=False, normalize=True)*100,2))
## creating a df with the #
total = pd.DataFrame(data.Pclass.value_counts(dropna=False))
## concating percent and total dataframe

total.columns = ["Total"]
percent.columns = ['Percent']
pd.concat([total, percent], axis = 1)


# We can see that 55 % of the people on Titanic travelled in Pclass 3

# **6.6 Statistics characteristics of the data **

# In[ ]:


data.describe()


# Statistic charasteristic of only one feature Age

# In[ ]:


data['Age'].describe()


# **6.7 Finding out people with age greater than 70 in the dataset**

# In[ ]:


data[data['Age']>70]


# We can see than five people over age of 70 travelled on the Fatal Ship.

# **6.8 Unique values of a Feature **

# In[ ]:


data['Pclass'].unique()


# There were three classes on the Ship 1,2,3.Elite Class is represented by 1.

# In[ ]:


data[data['Pclass']==1].head()


# This way we can get the information on all the people you travelled 1st Class on the Titanic.

# In[ ]:


data[data['Pclass']==1].count()


# **6.9 Finding out the Family Size **

# In[ ]:


# Create a family size variable including the passenger themselve
data["FamilySize"] = data["SibSp"] + data["Parch"]+1
#titanic_test["FamilySize"] = titanic_test["SibSp"] + titanic_test["Parch"]+1
print(data["FamilySize"].value_counts())


# Most Passengers on Board Titanic were travelling Alone.

# **6.10 Group by **

# In[ ]:


data[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived',ascending=False)


# 62 % of first class,47 % of second class and 24 % of the thrid class of passangers survived.

# **6.11 Data types **

# In[ ]:


data.dtypes


# **6.12 Viewing the data randomly **

# In[ ]:


data.sample(5)


# In[ ]:





# **7.Data Vizualization **
# 

# In[ ]:


import seaborn as sns #importing seaborn module 
import warnings
warnings.filterwarnings('ignore')  #this will ignore the warnings.it wont display warnings in notebook
plt.style.use('fivethirtyeight')


# **7.1.Pie Plot and Bar Plot ** 

# **Survival Details**

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
data['Survived'].value_counts().plot.pie(ax=ax[0],explode=[0,0.1],shadow=True,autopct='%1.1f%%')
ax[0].set_title('Survived',fontsize=30)
ax[0].set_ylabel('Count')
sns.set(font="Verdana")
sns.set_style("ticks")
sns.countplot('Survived',hue='Sex',linewidth=2.5,edgecolor=".2",data=data,ax=ax[1])
plt.ioff() # This removes the matplotlib notifications


# From pie chare we can see that only 38% people survived on the Titanic.From bar chart we can see that more women survived than men.

# **Family Size Details **

# In[ ]:


data["FamilySize"] = data["SibSp"] + data["Parch"]+1
#titanic_test["FamilySize"] = titanic_test["SibSp"] + titanic_test["Parch"]+1
#data["FamilySize"].value_counts())
sns.countplot('FamilySize',data=data)


# We can see that most people on Titanic travelled alone.

# **Embarkment Details**

# In[ ]:


sns.barplot(x="Embarked", y="Survived",data=data);


# We can see that the people who embarked from port C survived most.

# In[ ]:


sns.barplot(x="Embarked", y="Survived", hue="Sex", data=data);


# More women survived compared to Men.

# **Getting details of Mr , Mrs etc **

# In[ ]:


import re
#GettingLooking the prefix of all Passengers
data['Title'] = data.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))

#defining the figure size of our graphic
plt.figure(figsize=(12,5))

#Plotting the result
sns.countplot(x='Title', data=data, palette="hls")
plt.xlabel("Title", fontsize=16) #seting the xtitle and size
plt.ylabel("Count", fontsize=16) # Seting the ytitle and size
plt.title("Title Name Count", fontsize=20) 
plt.xticks(rotation=45)
plt.show()


# People were addresses as Mr,Mrs,Miss,Master,Rev,Dr,Major,Mlle and Col on the titanic

# In[ ]:


Title_Dictionary = {
        "Capt":       "Officer",
        "Col":        "Officer",
        "Major":      "Officer",
        "Dr":         "Officer",
        "Rev":        "Officer",
        "Jonkheer":   "Royalty",
        "Don":        "Royalty",
        "Sir" :       "Royalty",
        "the Countess":"Royalty",
        "Dona":       "Royalty",
        "Lady" :      "Royalty",
        "Mme":        "Mrs",
        "Ms":         "Mrs",
        "Mrs" :       "Mrs",
        "Mlle":       "Miss",
        "Miss" :      "Miss",
        "Mr" :        "Mr",
        "Master" :    "Master"
                   }
data['Title']=data.Title.map(Title_Dictionary)


# In[ ]:


print('Chance of Survival based on Titles:')
print(data.groupby("Title")["Survived"].mean())
#plt.figure(figsize(12,5))
sns.countplot(x='Title',data=data,palette='hls',hue='Survived')
plt.xlabel('Title',fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.title('Count by Title',fontsize=20)
plt.xticks(rotation=30)
plt.show()


# In[ ]:





# **7.2 Factor Plot **

# In[ ]:


sns.factorplot('Pclass','Survived',hue='Sex',data=data)
plt.ioff() 


# Again the factor plot clearly shows us that more women have survived compared to men,This shows women were given priority over men while performing the rescue.Also we can clearly see moreof  P class one people survived compared to other two classes.

# In[ ]:


sns.factorplot('Embarked','Survived',hue='Sex',data=data)
plt.ioff() 


# **7.3 Histogram**

# In[ ]:


plt.figure(figsize=(12,6))
data[data['Age']<200000].Age.hist(bins=80,color='red')
plt.axvline(data[data['Age']<=100].Age.mean(),color='black',linestyle='dashed',linewidth=3)
plt.xlabel('Age of Passengers',fontsize=20)
plt.ylabel('Number of People',fontsize=20)
plt.title('Age Distribution of Passengers on Titanic',fontsize=25)


# In[ ]:


#print('Minumum salary for H1b1 Visa holder is:',int(data[data['PREVAILING_WAGE']<=200000].PREVAILING_WAGE.min()),'$')
print('Mean age of Passenger on Titanic:',int(data[data['Age']<=100].Age.mean()),'Years')
print('Median age of Passenger on Titanic:',int(data[data['Age']<=100].Age.median()),'Years')
#print('Maximum salary for H1b1 Visa holder is:',int(data[data['PREVAILING_WAGE']<=9000000].PREVAILING_WAGE.max()),'$')


# **7.4 Strip Plot **

# In[ ]:


sns.stripplot(x="Survived", y="Age", data=data,jitter=True)


# **7.5 Box Plot **

# In[ ]:


sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=data);


# As expected the first class ticket prices are very high and it highest when Q was port of embarkment.

# In[ ]:




