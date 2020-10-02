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


# 
# **Hello everyone**, 
# * *This is my first kernel on the platform and also my first competition submission. *
# * *Being a beginner myself, I've tried to make this kernel simple and easy to understand for other fellow beginners who have just started.*
# * *Feel free to write down your opinions & comments.*

# **LOADING LIBRARIES**
# 
# The first step is to load all the necessary packages and libraries that will be used in this kernel. This includes :
# * Numpy & Pandas for handling our Data & DataFrames.
# * Matplotlib, Seaborn for visualitsations.
# * Machine learning packages (from SKlearn) for training and prediction.
# 

# In[ ]:


#for analysis of data, dataframe
import numpy as np
import pandas as pd

#for plotting and stuffs
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
#the above line of code is known as a magic function, helps to display our plots just below our code in the notebook.

#for model training & prediction
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# **LOAD THE DATA !**
# 
# We are given 3 files, one for training(train.csv) one for testing(test.csv) and a sample submission file(gender_submission.csv).
# We shall read the first 2 files using pandas.

# In[ ]:


#read training data into 'train_df' dataframe
train_df=pd.read_csv('../input/train.csv')

#read testing data into 'test_df' dataframe
test_df=pd.read_csv('../input/test.csv')

#combined dataset, will be handy in wrangling steps.
combined_df=[train_df,test_df]


# **Get to know the data we've loaded.** 
# 
# * This shall be the first step after we read data into dataframes.
# * We should know how many rows, columns the dataset has, their names, the values stored and their datatypes.
# * All this forms the first part of any analysis.

# In[ ]:


train_df.columns


# In[ ]:


test_df.columns


# Shown above - name of columns of both training & testing data, and it is clear from it that the test data is missing "Survived" column, and it is our **aim to predict that column**.

# In[ ]:


#to know what type of data columns hold ; 'object' type means they hold string values
train_df.dtypes


# In[ ]:


test_df.dtypes


# From above it is clear which columns hold 'int', 'float' and 'strings'. This is important as we will convert them accordingly ahead, to make it understandable for the machine(models) to do training, prediction etc.
# 
# Another method to have a detailed info about our dataset is to use ".info()". As shown below.

# In[ ]:


train_df.info()


# In[ ]:


test_df.info()


# From above it is clear that, our training data has 891 entries, with "Age", "Cabin" & "Embarked" columns having missing values.
# Our test data has 418 entries, and for these given 418 entries, we have to predict their survival. Even the test data is missing some values in "Age", "Fare", "Cabin" columns.

# In[ ]:


#train_df.info(verbose=False) will give a compact version of the above output, it set to True by default(in above case).
train_df.info(verbose=False)


# **First look of our data !**

# In[ ]:


train_df.head() #by default it prints first 5 rows, any other integer can also be given inside parenthesis.


# In[ ]:


test_df.head()


# As seen above- this is how data looks in the dataframe -train_df & test_df we created. Sweet. We can infer that:
# * There are few categorical variables.
# * Pclass has 3 categories namely 1,2 & 3
# * Sex - male, female category
# * Embarked has 3 catefories - S,Q,C
# * Survived has 2 - 1(survived) or 0(not survived)
# * Other columns such as Age, Ticket, Fare have continuous numerical values.

# In[ ]:


train_df.describe()
#this gives metric/stats of various columns.


# From above it is clear that, 
# * mean age of traveller onboard the Titanic is about 30.
# * The max age is 80 and min. is 0.42 (few months old baby !)
# * the avg. survival rate is 0.38, meaning the survival rate was less than 50% for passengers ! [Note: 1 means survived & 0 means not survived]
# * The mean fare is around 32 and the max fare is 512. Also looking at the std.deviation of Fare, it seems Fare varied greatly.
# 
# 
# 

# **Further Analysis**
# 
# Now, we shall do detailed analysis of the data. Our goal is to predict survival for test data, and to do that we should find out what all factors(features) led to the survival(Survived=1) of a person.
# 
# For such detailed analysis, plots are the best !

# In[ ]:


ax=train_df['Sex'].value_counts().plot.bar(title='Sex Distribution aboard Titanic',figsize=(8,4))

#below loop is to print numeric value above the bars
for p in ax.patches:
    ax.annotate(str(p.get_height()),(p.get_x(),p.get_height()*1.005))

sns.despine()  #to remove borders (by default : from top & right side)


# It is clear from above bar graph that the male population on the titanic exceeded the female population. But what was their survival rate ? We'll find out.

# In[ ]:


sns.set(style='whitegrid')
ax=sns.kdeplot(train_df['Age'])
ax.set_title('Age Distribution aboard the Titanic')
ax.set_xlabel('<---AGE--->')


# The above graph shows that,
# * most population was in the age group 20-40 years
# * toddlers+children exceeded old folks(>60 yrs) in number

# In[ ]:


print(train_df['Survived'].value_counts())
l=['Not Survived','Survived']
ax=train_df['Survived'].value_counts().plot.pie(autopct='%.2f%%',figsize=(6,6),labels=l)
#autopct='%.2f%%' is to show the percentage text on the plot
ax.set_ylabel('')


# * We see from above, among 891 entries in training data, just 342 (38% as seen in pie chart) survived !
# * 549 passengers or ~61% did not make it !

# In[ ]:


sns.countplot(train_df['Pclass'])
sns.despine()


# We can see from above graph, majority(more than 400) travelled in Class3, followed by Class1(~200). We will soon find out chances of survival based on PClass !

# In[ ]:


sns.countplot(train_df['Embarked'])


# More than 600 passengers embarked from 'S', followed by 'C' and then 'Q'.

# Now, we'll find how these columns relate to Survival (if at all).

# In[ ]:


train_df[['Sex','Survived']].groupby('Sex').mean()


# The above table makes it clear that, the survival of a female passenger was much higher than male passenger.

# In[ ]:


train_df[['Pclass','Survived']].groupby('Pclass').mean()


# In[ ]:


train_df.groupby(['Pclass','Survived'])['Pclass'].count()


# We see above that, higher the Passenger class, higher the survival rate.

# In[ ]:


sns.countplot(x='Pclass',hue='Survived',data=train_df)


# from the above table, we infer that, 
# * Survival chances of passengers from Class 1 was the highest. Interms of numbers also, Class 1 passengers survived most.
# * Despite Class3 having the most passengers their **Survival rate** was the lowest, though the numbers were high compared to Class2 !
# * Above can be corroborated from the bar graph above.
# 
# * **Hence we can say, the PClass in which passengers travelled had a role to play in their Survival**

# In[ ]:


train_df[['Embarked','Survived']].groupby('Embarked').mean()


# Embarked vs Survived table shows that :
# * People embarked from 'C' had high Survival.
# * Followed by from 'Q' and 'S'

# In[ ]:


train_df[['Parch','Survived']].groupby('Parch').mean()


# In[ ]:


train_df[['SibSp','Survived']].groupby('SibSp').mean()


# We can see from above 2 tables that:
# SibSp i.e number of siblings ans spouse aboard the Titanic, Parch- number of parent/children don't show any pattern/trend with survival, meaning :
# Looking at the table of Parch & Survival, first Survival rate increases first, then decreases, then again increases.
# The graph below makes it clear.

# In[ ]:


ax=train_df[['Parch','Survived']].groupby('Parch').mean().plot.line(figsize=(8,4))
ax.set_ylabel('Survival')
sns.despine()


# In[ ]:


ax=train_df[['SibSp','Survived']].groupby('SibSp').mean().plot.line(figsize=(8,4))
ax.set_ylabel('Survival')
sns.despine()


# The above SibSp vs Survival graph first increases (for SibSp 0 to 1) then comes down (for SibSp 1 to 5) eventually to become zero.

# **Now, lets map certain features on survival and see how they relate to it and understand their PLOTS**

# In[ ]:


a=sns.FacetGrid(train_df,col='Survived')
a.map(sns.distplot, 'Age')


# Looking at the peaks in above plots, we observe :
# * Many toddlers+children(upto 10 yrs age) survived
# * Many teenagers 20-25 years old, didn't survive
# * Many middle aged passengers, 30-45 survived.

# In[ ]:


a=sns.FacetGrid(train_df,col='Pclass',row='Survived')
a.map(plt.hist,'Age')


# The above plot further strengthens our observations made earlier:
# * Among the the Classes, Class 3 had most deaths, that too in 20-30 age group.
# * Class 1 had the lowest passenger deaths.
# * Survival percentage of Class 1 passenger was highest, that too in age group 30-40.

# In[ ]:


train_df['Embarked'].value_counts()


# In[ ]:


a=sns.FacetGrid(train_df,col='Embarked')
a.map(sns.distplot,'Survived')


# In[ ]:


train_df.groupby(['Embarked','Survived'])['Embarked'].count()


# The above plots and result analyse 'Embarked' and whether it impacts Survival. We find that :
# * In terms of numbers, max survival and also death, comes from passengers who embarked from 'S'.
# * Survival of members from 'S' is high compared to 'Q' and 'C'.
# * Passengers embarked from 'Q' had the least survival.

# In[ ]:


a=sns.FacetGrid(train_df,col='Embarked')
a.map(sns.pointplot, 'Pclass','Survived','Sex') #colum order is x='Pclass', y='Survived', hue='Sex'
a.add_legend()


# In[ ]:


train_df.groupby(['Embarked','Sex'])['Embarked'].count()


# The above pointplot may seem complex, but is simple and insightful one.
# 
# * Among passengers embarked from S, females had high survival compared to males. Also, as seen in first plot, the Survival decresed as we go from Pclass 1 to 3, for both sexes. (NOTE : Strong dip in Survival of females from Pclass 2 to 3)
# 
# * Third plot, Embarked='Q', also has similar pattern, with females of high Pclass having high survival. BUT, males of Pclass 3 had slightly high survival chance compared to Pclass 2 & 3 (strange).
# 
# * Now from the second plot, Embarked='C', this plot is unusual, as we can see** males survived more than female**. 

# In[ ]:


a=sns.FacetGrid(train_df,col='Survived')
a.map(sns.barplot,'Sex', 'Fare')


# The above plot confirms that :
# * Those who paid a higher ticket fare had more chances of Survivng.
# * Also,females paid higher ticket price compared to males.

# **Data Wrangling**
# 
# * Data Wrangling involves cleaning, organizing the data, making it more suitable for machine learning.
# * it involves steps, such as removing nulls, mapping, encoding etc.

# Remember, we had combined both training and testing data earlier, below is a view of it.

# In[ ]:


combined_df[0].head(3) #[0] is train_df


# In[ ]:


combined_df[1].head(3)  #[1] is test_df


# In[ ]:


print('training data dimensions :',train_df.shape)
print('testing data dimensions :', test_df.shape)
print('combined data\'s dimension are :\n',combined_df[0].shape,'\n',combined_df[1].shape)


# In[ ]:


train_df[['PassengerId','Name','Ticket','Cabin']].head()


# We can see that, 'Name' 'Cabin' and 'Ticket' columns are random, and have no impact on Survival of passenger as other features had. Seriously - "Whats in a name?!"
# 
# * Hence, we shall remove these columns (done below), as they don't contribute to our analysis.
# * Also, we saw earlier, SibSp, Parch didn't have any effect on Survival of a passenger, so remove those too.
# * Note : We will remove Passenger Id from the training data set also.

# In[ ]:


#removing mentioned columns from dataset
train_df=train_df.drop(['Name','Ticket','Cabin','SibSp','Parch','PassengerId'],axis=1)
test_df=test_df.drop(['Name','Ticket','Cabin','SibSp','Parch'],axis=1)


# In[ ]:


# the combined data
combined_df=[train_df, test_df]


# In[ ]:


#lets check the new dimensions
print('new training data dimensions :',train_df.shape)
print('new testing data dimensions :', test_df.shape)
print('new combined data\'s dimension are :\n',combined_df[0].shape,'\n',combined_df[1].shape)


# In[ ]:


train_df.head(3)


# In[ ]:


#checking for any null values
train_df.isnull().any() #True means null present


# In[ ]:


test_df.isnull().any()


# Null values interfere with our training and prediciton. So they have to be removed or be filled with relevant, suitable data.
# * Above results show, which columns have null values.. we'll correct them one by one.

# In[ ]:


# age columns
print('mean age in train data :',train_df['Age'].mean())
print('mean age in test data :',test_df['Age'].mean())


# Since mean age in both datasets is near 30, we'll replace null values with 30.

# In[ ]:


#replacing null values with 30 in age column
for df in combined_df:
    df['Age']=df['Age'].replace(np.nan,30).astype(int)


# In training data, 'Embarked' also has missing values..

# In[ ]:


train_df['Embarked'].value_counts()


# In[ ]:


#most people embarked from 'S'. So, we'll replace the missing missing Embarked value by 'S'.
train_df['Embarked']=train_df['Embarked'].replace(np.nan,'S')


# The test data is missing values in Fare column. Lets deal with that now.

# In[ ]:


#finding mean fare in test data
test_df['Fare'].mean()


# In[ ]:


#replace missing fare values in test data by mean
test_df['Fare']=test_df['Fare'].replace(np.nan,36).astype(int)


# In[ ]:


combined_df=[train_df,test_df]
for df in combined_df:
    print(df.isnull().any()) #bool value = False means that there are no nulls in the column.


# We have successfully dealt with NULL values. 

# As we know, some column have ***categorical values*** such as Sex, Pclass, Embarked.
# 
# The values in these columns can be "categorised" or can be put into certain categories. For eg: Sex in our dataset can be categorised as either male or female, similarly Embarked into S,Q,C.
# 
# We will now convert these into numeric data, or codify them. As done below. This is known as **ENCODING**
# 

# In[ ]:


#will code female as 1 and male as 0
for df in combined_df:
    df['Sex']=df['Sex'].map({'female':1,'male':0}).astype(int)


# In[ ]:


train_df.head(3)


# As seen above, 'Sex' column has been changed, males have been coded as 0 and females as 1.

# In[ ]:


#coding Embarked column as: S=2, C=1, Q=0
for df in combined_df:
    df['Embarked']=df['Embarked'].map({'S':2,'C':1,'Q':0}).astype(int)


# In[ ]:


train_df.head(3)


# As seen above, 'Embarked' column has been changed, S, C and Q have been coded or changed to numeric values.

# We know from earlier analysis that age was a factor in the survival of a passenger.
# 
# Also the range of values age takes is very high(from 0.42 to 80).
# 
# * So we shall divide age in to age groups/bands for easier training and prediction. This is called **BINNING**.

# In[ ]:


#binning or making bands of age into intervals and then assigning labels to them(encoding the bands as 0,1,2,3,4)
for df in combined_df:
    df['Age']=pd.cut(df['Age'],5,labels=[0,1,2,3,4]).astype(int) #pandas cut will help us divide age in bins


# In[ ]:


train_df.head(3)


# Similarly, we found out earlier that Fare played imp. role in survival, so we bin fare in to groups, just as we did for age.

# In[ ]:


#binning fares and assigning label 0,1,2,3 to their respective bins
for df in combined_df:
    df['Fare']=pd.qcut(df['Fare'],4,labels=[0,1,2,3]).astype(int)


# In[ ]:


train_df.head(3)


# In[ ]:


test_df.head(3)


# As seen above, we have modified our train and test datas, making it suitable for our models, to do training & prediciton properly.

# So, our final dataframe looks like the one shown above,
# * Pclass, Sex, Age(binned), Embarked and Fare(binned) will be our feature set, i.e these play a factor in the survival of passenger and will be used in PREDICTION of test data. 

# **MODELLING and PREDICTION**
# 
# The following models have been used:
# * Logistic Regression
# 
# * Random Forest Classifier
# 
# * Decision Tree Classifier

# In[ ]:


X_train=train_df.drop('Survived',axis=1)
Y_train=train_df['Survived']

#X_train is the entire training data except the Survived column, which is separately stored in Y_train. We will use these to train our MODEL !

X_test=test_df.drop('PassengerId',axis=1).copy()
#X_test is the test data, for on which we will apply model and predict the "SURVIVED" column for its entries.


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


#first applying Logistic Regression

lg = LogisticRegression()
lg.fit(X_train, Y_train)
Y_pred1 = lg.predict(X_test)
accu_lg = (lg.score(X_train, Y_train))
round(accu_lg*100,2)


# In[ ]:


#applying decision tree

dtree = DecisionTreeClassifier()
dtree.fit(X_train, Y_train)
Y_pred2 = dtree.predict(X_test)
accu_dtree = (dtree.score(X_train, Y_train))
round(accu_dtree*100,2)


# In[ ]:


#applying random forest

rafo = RandomForestClassifier(n_estimators=100)
rafo.fit(X_train, Y_train)
Y_pred3 = rafo.predict(X_test)
accu_rafo = rafo.score(X_train, Y_train)
round(accu_rafo*100,2)


# We note that :
# 
# * Score of Logistic Regression is lowest
# 
# * The score from Decision Tree and Random forest is similar.

# In[ ]:


#our goal was to predict survived column for test data, and were asked to submit a dataframe with 'PassengerId' and 'Survived' columns

submission=pd.DataFrame({'PassengerId':test_df['PassengerId'],'Survived':Y_pred3})


# In[ ]:


submission.shape


# In[ ]:


submission.head(10)


# From above it is clear, the Submission file is as per the requirement. Now writing it to csv file format.

# In[ ]:


submission.to_csv('submission.csv', index=False)


# *Thank You*
