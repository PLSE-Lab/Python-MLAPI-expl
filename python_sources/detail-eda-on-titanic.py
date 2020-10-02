#!/usr/bin/env python
# coding: utf-8

# 
# <font size="3"><b>Check the notebook to perform Exploratory Data Analysis(EDA) Step by Step

# <h1 style='color:green'>Please upvote if you find it helpful.
# 

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# <font size = "3"><b>Loading both the data sets and finding out the rows and columns of each data set using shape method.

# In[ ]:


df1 = pd.read_csv('../input/titanic/test.csv')
df1.head()


# In[ ]:


df2 = pd.read_csv('../input/titanic/train.csv')
df2.head()


# In[ ]:


df1.shape


# In[ ]:


df2.shape


# <font size="3"><b>Let us look at the columns of both the data sets. After analyzing the columns, we can see that there are same columns in both the data sets except survived column in train data set which shows that the particular passenger is survived or not. But the data in both the data sets is different.

# In[ ]:


df1.columns


# In[ ]:


df2.columns


# <font size="3"><b>Let us count the values of train data set of columns such as Survived, Sex, Parch, SibSp, Pclass.

# <font size = "3"><b>Let me explain two or three columns for better understanding. If we look at Survived column then it shows that number of passengers survived is less than the number of passengers not survived. If we look at Sex column then it shows that male passengers are more than female passengers.Similarly, we can understand other columns.

# In[ ]:


df2['Survived'].value_counts()


# In[ ]:


df2['Sex'].value_counts()


# In[ ]:


df2['Parch'].value_counts()


# In[ ]:


df2['SibSp'].value_counts()


# In[ ]:


df2['Pclass'].value_counts()


# <font size="3"><b>Now, we will look at the unique values of each column of both the data sets.

# In[ ]:


df1.nunique() 


# In[ ]:


df2.nunique() 


# <font size="3"><b>Finding out the rows and columns

# In[ ]:


df2.apply(len) 


# In[ ]:


df2.apply(len, axis = 'columns') 


# <font size="3"><b>Looking at the largest and smallest values of the columns.

# In[ ]:


df2['Fare'].nlargest(10) 


# In[ ]:


df2['Fare'].nsmallest(10) 


# In[ ]:


df2['Age'].nlargest(10) 


# In[ ]:


df2['Age'].nsmallest(10) 


# In[ ]:


df2['SibSp'].nlargest(10) 


# In[ ]:


df2['SibSp'].nsmallest(10)


# In[ ]:


df2['Parch'].nlargest(10) 


# In[ ]:


df2['Parch'].nsmallest(10) 


# <font size="3"><b>Describe method shows count,mean,standard deviation, minimum value, 25%, median(50%), 75%, maximum value of each column.

# In[ ]:


df2.describe()


# <font size ="3"><b>Let us look at the total null values of each column.

# In[ ]:


df2.isnull().sum()


# <h1>Let us start Plotting the Columns.

# In[ ]:


plt.style.available


# In[ ]:


plt.style.use('fivethirtyeight')


# <font size="3"><b>Starting with the relationship among the dataset.

# In[ ]:


sns.heatmap(df2.corr(), annot=True);


# <font size="3"><b>Now, let us take SibSp column and will see total count of number of siblings or spouses.

# In[ ]:


df2['SibSp'].value_counts().plot(kind = 'bar', title = 'Count of Siblings or Spouses', color = ['#ff2e63','#fe9881']);
plt.xlabel('Number of Sib/Sp');
plt.ylabel('Count');


# <font size="3"><b>Below graph indicates number of siblings or spouses at particular index.

# In[ ]:


plt.bar(df2.index, df2['SibSp'], align = 'center', color = 'green', width=5);
plt.title('Count of the Siblings or Spouses')
plt.xlabel('Index')
plt.ylabel('Number of Siblings or Spouses');


# <font size="3"><b>Looking at the total count of number of parent or children.

# In[ ]:


df2['Parch'].value_counts().plot(kind='bar', title = 'Count of Parent or Chilredn', color = '#B33B24');
plt.xlabel('Number of Parch')
plt.ylabel('Count');


# <font size ="3"><b>Looking at the total count of survived column. It shows that the passengers survived is less than the passengers not survived.

# In[ ]:


df2['Survived'].value_counts().plot(kind = 'bar', title = 'Count for the Survival', color = ['#FE4C40', '#FFCC33'] );
plt.xlabel('Survived or not')
plt.ylabel('Count');


# <font size="3"><b>Looking at the count of Sex column.

# In[ ]:


sns.countplot(x = 'Sex', data = df2, palette = 'Blues')
plt.xlabel('Male of Female')
plt.ylabel('Count')
plt.title('Count of Male and Female');


# <font size="3"><b>Looking at the count of Embarked column. It shows the count of Type of Embarkation.

# In[ ]:


df2['Embarked'].value_counts().plot(kind = 'bar', title = 'Count for the Port of Embarkation', color = ['#B33B24', '#CC553D','#E6735C']);
plt.xlabel('Type of Embarkation')
plt.ylabel('Count');


# <font size="3"><b>Looking at the count of Pclass column. It shows the count of Type of Pclass.

# In[ ]:


df2['Pclass'].value_counts().plot(kind = 'bar', title = 'Count for the Pclass', color = ['#FFCC33','#FF6037','#FE4C40']);
plt.xlabel('Type of Pclass')
plt.ylabel('Count');


# <font size="3"><b>Now, we will plot the largest and smallest values of Fare, Age, SibSp and Parch columns.

# In[ ]:


df2['Fare'].nlargest(10).plot(kind='bar', title = '10 largest Fare', color = ['#C62D42', '#FE6F5E']);
plt.xlabel('Index')
plt.ylabel('Fare');


# In[ ]:


df2['Age'].nlargest(10).plot(kind='bar', color = ['#5946B2','#9C51B6']);
plt.title('10 largest Ages')
plt.xlabel('Index')
plt.ylabel('Ages');


# In[ ]:


df2['Age'].nsmallest(10).plot(kind='bar', color = ['#A83731','#AF6E4D'])
plt.title('10 smallest Ages')
plt.xlabel('Index')
plt.ylabel('Ages');


# In[ ]:


df2['SibSp'].nlargest(10).plot(kind='bar', color = ['#33CC99','#00755E'])
plt.title('Index having largest number of SibSp')
plt.xlabel('Index')
plt.ylabel('Number of SibSp');


# In[ ]:


df2['Parch'].nlargest(10).plot(kind='bar', color = ['#319177','#0A7E8C'])
plt.title('Index having largest no. of Parch')
plt.xlabel('Index')
plt.ylabel('Number of Parch');


# <font size="3"><b>Let us now plot the count for the range of Ages and Fare.It shows count of the values that are included in the particular range.

# In[ ]:


bins = [10,20,30,40,50,60,70,80,90,100]
plt.hist(df2['Age'], bins = bins, edgecolor = 'black', color = '#008080');
plt.title('Count for the range of ages')
plt.xlabel('Ages')
plt.ylabel('Number of Counts');


# In[ ]:


bins = [10,20,30,40,50,60,70,80,90]
plt.hist(df2['Fare'], bins = bins, edgecolor = 'black', color = '#CD607E');
plt.title('Count for the range of Fare')
plt.xlabel('Fare')
plt.ylabel('Number of Counts');


# <font size="3"><b>Let us plot Survival of passengers according to Pclass.It clearly shows that passengers of class 1st(Lower) and 2nd(Middle) have survived more than the passengers of class 3rd(Upper).

# In[ ]:


sns.countplot(x = 'Survived', data = df2, hue = 'Pclass', palette = 'Greens');
plt.title('Survival of people according to Pclass');


# <font size ="3"><b>Plotting the Survival of passengers according to Sex.We can get from the graph that female have survived more than male.

# In[ ]:


sns.countplot(x = 'Survived', data = df2, hue = 'Sex', palette = 'Greys')
plt.title('Survival of people according to sex');


# <font size="3"><b>Plotting survival of passengers according to Port of Embarkation.

# In[ ]:


sns.countplot(x = 'Survived', data = df2, hue = 'Embarked', palette = 'Accent');
plt.title('Survival of people according to Embarked');


# <font size="3"><b>Let us plot the box plot of Fare and Age column. It shows that how the values in the data are spread out. We can also get the median and outliers of the column.

# In[ ]:


sns.boxplot(data = df2['Fare'],orient = 'h', palette = 'Blues');


# In[ ]:


sns.boxplot(data = df2['Age'], orient = 'h', palette = 'Greens');


# <font size="3"><b>Plotting the scatter plot of Fare and Pclass column. It shows the relationship of both the columns. It tells us that the Fare is decided according to Pclass.

# In[ ]:


plt.scatter(df2['Pclass'], df2['Fare'], color = '#676767')
plt.title('Fare according to Pclass')
plt.xlabel('Pclass')
plt.ylabel('Fare');


# <font size = "3"><b>Let us get into comparision of plots.

# <font size = "3"><b>Below plot tells us Count of Pclass on one side and Survival of passengers according to Pclass on other side.Similarly, the next two plots shows the same thing.

# In[ ]:


plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
sns.countplot(x='Pclass', data = df2, palette = 'rainbow');
plt.title('Count of Pclass');


plt.subplot(1,2,2)
sns.countplot(x='Survived', data = df2, hue = 'Pclass', palette = 'rainbow');
plt.title('Survival according to Pclass');


# In[ ]:


plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
sns.countplot(x='Sex', data = df2, palette = 'Blues')
plt.title('Count of Sex');

plt.subplot(1,2,2)
sns.countplot(x='Survived', data = df2, hue = 'Sex', palette = 'Blues')
plt.title('Survival according to Sex');


# In[ ]:


plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
sns.countplot(x = 'Embarked', data = df2, palette = 'Greens')
plt.title('Count of Embarkation');

plt.subplot(1,2,2)
sns.countplot(x = 'Survived', data = df2, hue='Embarked', palette = 'Greens')
plt.title('Survival according to point of embarkation');


# <font size = "3"><b>Each graph present in comparision plots have been perform individually. But the main goal of doing these is that the user can easily compare between two graph.

# <font color = "green"><font size = "5"><b>So these were the steps of performing EDA. Throw your doubts and feedback in the comment section.
# Thank You and Be Safe.
