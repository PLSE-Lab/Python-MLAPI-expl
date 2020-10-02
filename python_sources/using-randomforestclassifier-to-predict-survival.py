#!/usr/bin/env python
# coding: utf-8

# # Machine Learning from Start to Finish with Scikit-Learn
# 
# This notebook covers the basic Machine Learning process in Python step-by-step. Go from raw data to at least 78% accuracy on the Titanic Survivors dataset. 
# 
# ### Steps Covered
# 
# 
# 1. Importing  a DataFrame
# 2. Visualize the Data
# 3. Cleanup and Transform the Data
# 4. Encode the Data
# 5. Split Training and Test Sets
# 6. Fine Tune Algorithms
# 7. Cross Validate with KFold
# 8. Upload to Kaggle

# # Change CSV into DataFrame
# 
# CSV files can be loaded into a dataframe by calling `pd.read_csv` . After loading the training and test files, print a `sample` to see what you're working with.

# # **Import required libraries and packages. Convert CSV files into dataframes named data_train and data_test**

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

data_train = pd.read_csv('../input/train.csv')
data_test = pd.read_csv('../input/test.csv')


# ## **Initial exploration of data**

# In[ ]:


display(data_train.sample(3))
display(data_test.sample(3))


# In[ ]:


display(data_train.head())
display(data_test.head())
display(data_test.tail())


# In[ ]:


data_train.describe() 


# In[ ]:


data_test.describe()


# **As we can see from above, data_test is similar to data_train, except that data_test does not contain the 'Survived' column. The reason is that the target or dependent variable is 'Survived'. That is, we are going to predict 'Survived' values. Therefore, it is not going to be given to us in data_test. **

# ### Analyzing shapes of the dataframes

# In[ ]:


print(str(data_train.shape)+ ' -> data_train')
print(str(data_test.shape)+ ' -> data_test')


# In[ ]:





# ### Filling in null values with means of their respective columns

# In[ ]:


data_train.info()


# In[ ]:


data_train = data_train.drop(columns=['Name', 'Ticket', 'Fare', 'Cabin']) # dropping columns which are unnecessary for analysis
data_test = data_test.drop(columns=['Name', 'Ticket', 'Fare', 'Cabin']) # doing same for data_test to maintain similar structure of dataframes for both train and test sets


# NOTE : drop passengerid for both above ****

# In[ ]:


display(data_train.Age.value_counts(dropna=False).sort_index())
display(data_test.Age.value_counts(dropna=False).sort_index())


# 177 missing ages in train set
# 
# 86 missing ages in test set 
# 
# I will replace the missing ages with average age

# In[ ]:


data_train.Age = data_train.Age.fillna(data_train.Age.mean()) #filling all nulls in 'Age' column with the mean age
data_test.Age = data_test.Age.fillna(data_test.Age.mean()) #filling all nulls in 'Age' column with the mean age


# In[ ]:


display(data_train.Embarked.value_counts(dropna=False))
display(data_test.Embarked.value_counts(dropna=False))


# so there are 2 NaNs to take care of in data_train. I will replace those NaNs with the mode of the Embarked column, which we can see is 'S'.

# In[ ]:


data_train.Embarked = data_train.Embarked.fillna('S') #filling all nulls in 'Embarked' column with 'S'


# Now, will be shifting the 'Survived' column to the end of the dataframe for easier analysis

# Using 'dropna = False' inside value_counts() method enables us to include counts of null values when performing value_counts() 

# In[ ]:


b = data_train.pop('Survived') # from data_train, pop the 'Survived' column 
data_train = pd.concat([data_train, b], axis=1) # and add it to the end of data_train
display(data_train.head())


# ### CHECKING FOR NULL VALUES IN ALL COLUMNS IN BOTH DATA_TRAIN AND DATA_TEST

# In[ ]:


display ( data_train.Age.value_counts(dropna=False).sort_index() )
display ( data_test.Age.value_counts(dropna=False).sort_index() )


# In[ ]:


display ( data_train.Sex.value_counts(dropna=False).sort_index() )
display( data_test.Sex.value_counts(dropna=False).sort_index() )


# In[ ]:


display ( (data_train.Pclass.value_counts(dropna=False).sort_index()) )
display ( (data_test.Pclass.value_counts(dropna=False).sort_index()) )


# In[ ]:


display ( data_train.Survived.value_counts(dropna=False).sort_index() )


# In[ ]:


display ( data_train.SibSp.value_counts(dropna=False).sort_index() )
display ( data_test.SibSp.value_counts(dropna=False).sort_index() )


# Having more than 5 siblings / spouses was very rare.

# In[ ]:


display ( data_train.Parch.value_counts(dropna=False).sort_index() )
display ( data_test.Parch.value_counts(dropna=False).sort_index() )


# In[ ]:


display ( data_train.Embarked.value_counts(dropna=False).sort_index() )
display ( data_test.Embarked.value_counts(dropna=False).sort_index() )


# It is confirmed that there are no more nulls in any of the columns of data_train and data_test.

# -----

# ## DATA VISUALIZATION 

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

# notes : 
# sns.set()
# f, ax = plt.subplots(figsize=(19, 19))
# sns.heatmap(data_train, annot=True, linewidths=.5, ax=ax)
# -- above not working -- 



# In[ ]:


print( ' data_train ')

sns.set(style="dark")
# Compute the correlation matrix
corr = data_train.corr()
display(corr)
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(2, 900, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, cmap=cmap, vmax=1, vmin=-1, center=0,
            square=True, linewidths=.05, linecolor='grey') 





print( ' data_test ')

sns.set(style="dark")
# Compute the correlation matrix
corr = data_test.corr()
display(corr)
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(2, 900, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, cmap=cmap, vmax=1, vmin=-1, center=0,
            square=True, linewidths=.05, linecolor='grey') 




# Important to note : Embarked and Sex are not included in the correlation tables or matrices, since they do not have numerical values.
# 
# DATA_TRAIN 
# 
# From the correlation matrix of data_train, we can see that:
# 1. As Pclass increases (going from 1st to 3rd class), the survival rate decreases, as denoted by the light red shade between Pclass and Survived.
# 2. As Age increases, the Pclass tends to be lower (that is, older passengers tended to be in the first class more than second class , and more than third class.)
# 3. As Age increased, the survival rate tended to be lower. That is, older passengers were less likely to survive, even though they had greater probability of being in upper classes. This is very ironic.
# 
# DATA_TEST
# 
# This also seems to be the case for data_test, since the colors are approximately the same in both heatmaps.

# ## Boxplot Analysis of passengers' ages in each class

# In[ ]:


plt.figure(figsize=(15,5))

plt.subplot(1,2,1)
plt.title('TRAIN')
my_palette = {1:'g', 2:'c', 3:'y'}
sns.boxplot(data_train['Pclass'], data_train['Age'], palette=my_palette, saturation= 40) 
plt.xticks = [1,2,3]

plt.subplot(1,2,2)
plt.title('TEST')
my_palette = {1:'g', 2:'c', 3:'y'}
sns.boxplot(data_test['Pclass'], data_test['Age'], palette=my_palette, saturation= 40) 
plt.xticks = [1,2,3]


# the distribution of people in 3rd class in data_train and data_test are different from each other when we consider the outliers. Otherwise, all classes have similar distributions in both data_train and data_test.

# ## SCATTERPLOT ANALYSIS OF PASSENGERS' AGES IN EACH CLASS

# In[ ]:


plt.figure(figsize=(30,20))
plt.subplot(1,2,1)
plt.title('TRAIN')
plt.scatter( data_train['Pclass'], data_train['Age'], c='red', marker='d', s= 6.0)
plt.xticks = ([1,2,3])

plt.subplot(1,2,2)
plt.title('TEST')
plt.scatter( data_test['Pclass'], data_test['Age'], c='red', marker='d', s= 6.0)
plt.xticks = ([1,2,3])


# There seems to be similar distributions of ages in Classes 1, 2, and 3 in data_train and data_test with the exception of 1st class in data_train which has more younger people than data_test does. Another exception is 3rd class in data_train has more elder people than data_test does.

# ## ANALYSIS OF GENDER AND SURVIVAL STATUS

# In[ ]:


display ( sns.countplot(x=data_train['Survived'], hue=data_train['Sex'], data=data_train, palette='Spectral', saturation=10) )


# **DATA_TRAIN**
# 
# more males lost their lives than females did
# 
# more females survived than the males did

# In[ ]:


plt.figure(figsize=(30,20))
plt.subplot(2,6,1)
plt.title('EMBARKED VS. SURVIVAL')
sns.countplot(x=data_train['Embarked'], hue=data_train['Survived'], palette='winter')
plt.ylim(0,700)
plt.legend()
plt.subplot(2,6,2)
plt.title('EMBARKED VS. SEX')
sns.countplot(x=data_train['Embarked'], hue=data_train['Sex'], palette='spring')
plt.ylim(0,700)

# plt.figure(figsize=(5,5))
# sns.countplot(x=data_train['Survived'], hue=data_train['Embarked'], palette= 'summer', alpha=0.3)
# plt.ylim(0,600) -- this code will put two graphs in one -- 

plt.subplot(2,6,3)
plt.title('SURVIVAL VS. SEX')
sns.countplot(x=data_train['Survived'], hue=data_train['Sex'], palette= 'autumn')
plt.ylim(0,700)

plt.subplot(2,6,4)
plt.title('Distribution of Gender')
sns.countplot(x=data_train['Sex'], hue=data_train['Sex'], palette= 'summer')
plt.ylim(0,700)

plt.subplot(2,6,5)
plt.title('Distribution of Embarked Location')
sns.countplot(x=data_train['Embarked'], hue=data_train['Embarked'], palette= 'summer')
plt.ylim(0,700)


plt.subplot(2,6,6)
plt.title('Distribution of Survival')
sns.countplot(x=data_train['Survived'], hue=data_train['Survived'], palette= 'summer')
plt.ylim(0,700)


# In[ ]:


plt.figure(figsize=(30,20))

plt.subplot(2,6,2)
plt.title('EMBARKED VS. SEX')
sns.countplot(x=data_test['Embarked'], hue=data_test['Sex'], palette='spring')
plt.ylim(0,700)

plt.subplot(2,6,4)
plt.title('Distribution of Gender')
sns.countplot(x=data_test['Sex'], hue=data_test['Sex'], palette= 'summer')
plt.ylim(0,700)

plt.subplot(2,6,5)
plt.title('Distribution of Embarked Location')
sns.countplot(x=data_test['Embarked'], hue=data_test['Embarked'], palette= 'summer')
plt.ylim(0,700)


# 

# ## **DATA_TRAIN **
# 
# * We can see from 1, of those who embarked on area 'S', more people lost their lives than survived compared to 'C' and 'Q'.
# * Also, of those who embarked on 'S', majority of them were males. Therefore, in 3, we can see that males tended to lose their lives more than females, possibly due to the huge imbalance in the number of men on the Titanic in general. As a result, the number of females who survived is greater than the number of males who survived (see 6).
# * Looking at 4, we see that huge imbalance in the distribution of males vs females. Therefore, it is not surprising that more men died than women. 
# * Looking at 5, we see a huge imbalance in how many embarked at 'S' vs 'C' and 'Q'. Therefore, it is not surprising that those who embarked at 'S' had the highest proportion of those who died compared to 'C' or 'Q'.
#   
# * However, in the cases of 'C' and 'Q' embarkments, the survived vs not survived levels are approximately balanced. It seems like those who embarked at 'S' and were male had the highest chances of dying.
# * The graph 6 suggests that more people died than survived. It is evident that the reasons for this are that :   1. More people boarded at 'S' than 'C' or 'Q'. 2. More men boarded at 'S', 'C', and 'Q' than women. 3. Hence, more men were on the Titanic than women. 4. More men died than women. 5. Therefore, more passengers died than survived.
# 
# 
# ## **DATA_TEST **
# 
# * Embarked vs. Sex graphs for data_train and data_test are similar in proportions (number of males (or females) who embarked in any area / total number of passengers)
# * 'Distribution of Gender' graphs for data_train and data_test are similar in proportions (number of males / total number of passengers, number of females / total number of passengers)
# * 'Distribution of Embarked Location' graphs for data_train and data_test are similar in proportions ( number of people who embarked in any area / total number of passengers) with slight variations. The variations are that the proportions can be little higher or lower than expected, but it's not significant enough to skew the results.

# In[ ]:


plt.figure(figsize=(20,6))
plt.subplot(1,3,1)
plt.title('data_train : CLASS VS. SURVIVAL')
sns.countplot(x=data_train['Pclass'], hue=data_train['Survived'], palette='cool')
plt.ylim(0,700)
plt.legend(['Died','Survived'])

plt.subplot(1,3,2)
plt.title('data_train : CLASS VS. GENDER')
sns.countplot(x=data_train['Pclass'], hue=data_train['Sex'], palette='magma')
plt.ylim(0,700)
plt.legend(['Male', 'Female'])

plt.subplot(1,3,3)
plt.title('data_train : GENDER VS. SURVIVAL')
sns.countplot(x=data_train['Sex'], hue=data_train['Survived'], palette='prism')
plt.ylim(0,700)
plt.legend(['Died', 'Survived'])


plt.figure(figsize=(10,10))
plt.subplot(1,1,1)
plt.title('data_test : CLASS VS. GENDER')
sns.countplot(x=data_test['Pclass'], hue=data_test['Sex'], palette='magma')
plt.ylim(0,300)
plt.legend(['Male', 'Female'])


# The distribution of males and females in 2nd class are slightly different in proportions and slightly more different in 3rd class for data_train and data_test dataframes. However, this should not alter the results significantly.

# ---

# **Exploring the data further : **

# In[ ]:





# In[ ]:


data_train[(data_train.Embarked=='S')].groupby(['Pclass', 'Sex']).size() #.plot(kind='bar', cmap='summer')


# In[ ]:


data_test[(data_test.Embarked=='S') ].groupby(['Pclass', 'Sex']).size()#.plot(kind='bar', cmap='summer')


# try out other variations of this to see if you get similar results as before !

# In[ ]:


data_train.shape


# In[ ]:


data_train.head()


# In[ ]:


data_test.shape


# In[ ]:


data_test.head()


# **The columns 'SibSp', 'Parch', 'Sex', 'Embarked', and 'Pclass' contain categorical data. Machine Learning Algorithms cannot process categorical data. So, one-hot encoding is applied to these columns in order to convert the data into numbers. One-hot encoding is done below using pandas get_dummies() . Also, it cannot process **

# In[ ]:


data_train = pd.get_dummies(data_train, columns=['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked'])
data_train.info()


# Successfully completed one-hot encoding on the dataframe's columns

# In[ ]:


data_test = pd.get_dummies(data_test, columns=['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked'])
data_test.info()


# In[ ]:


data_train['Age'] = (data_train.Age//10*10)


# In[ ]:


data_test['Age'] = (data_test.Age//10*10)


# In[ ]:


data_train = pd.get_dummies(data_train, columns=['Age'])


# In[ ]:


data_test = pd.get_dummies(data_test, columns=['Age'])


# In[ ]:


b = data_train.pop('Survived')
data_train = pd.concat([data_train, b], axis=1)
data_train.head()


# In[ ]:


X = data_train.drop(columns = ['Survived', 'PassengerId'], axis=1)


# In[ ]:


X


# Even after using .drop() method, data_train hasn't been permanently altered. From this, the 'Survived' column is selected as 'y'.

# In[ ]:


y = data_train.Survived


# In[ ]:


y


# In[ ]:


# X   #418 rows
# y   #891 rows

data_train.info()


# In[ ]:


y.head()


# In[ ]:


plt.figure(figsize=(8,8))
plt.title('data_train : CLASS 1 VS. SURVIVAL')
sns.countplot(x=data_train['Pclass_1'], hue=data_train['Survived'], palette='magma')
plt.ylim(0,700)
plt.legend(['Did not survive', 'Survived'])


# In[ ]:


plt.figure(figsize=(8,8))
plt.title('data_train : CLASS 2 VS. SURVIVAL')
sns.countplot(x=data_train['Pclass_2'], hue=data_train['Survived'], palette='magma')
plt.ylim(0,700)
plt.legend(['Did not survive', 'Survived'])


# In[ ]:


plt.figure(figsize=(8,8))
plt.title('data_train : CLASS 3 VS. SURVIVAL')
sns.countplot(x=data_train['Pclass_3'], hue=data_train['Survived'], palette='magma')
plt.ylim(0,700)
plt.legend(['Did not survive', 'Survived'])


# In[ ]:





# Of the people who were in first class, more people died than survived. 
# 
# Of the people who were in second class, approximately equal numbers of passengers survived or died. So the ratio of survived : not survived would be almost 1:1 .
# 
# Considering the people who were in third class , more passengers died than those who survived.

# ## Importing train_test_split

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)


# In[ ]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# In[ ]:


num_trees = 1000
max_features = 3
kfold = KFold(n_splits=10, random_state=7)
rfc = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)


# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# In[ ]:


y_train.shape


# In[ ]:


y_test.shape


# In[ ]:


type(y_train)


# In[ ]:


rfc.fit(X_train, y_train)


# **The model has been fitted to the training data, X_train and y_train.**

# In[ ]:


rfc.score(X_train, y_train) 


# Interesting question : why is the training accuracy score only 88.77% and not 100% ? 

# ---

# In[ ]:


rfc.score(X_test, y_test)


# Interesting question : The test accuracy score is 79.82%. What can be done to increase the score? 
# 

# In[ ]:


y_pred = rfc.predict(X_test)


# In[ ]:


y_pred # this is an array of predictions


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


acc = accuracy_score(y_test, y_pred)


# In[ ]:


acc


# **As earlier, the model predicted survival with an accuracy of 79.82%.**

# ----

# In[ ]:


type(y_test)


# In[ ]:


X_test.shape


# In[ ]:


y_test.shape


# In[ ]:


data_train.shape


# In[ ]:


data_test.shape


# In[ ]:


data_test.isnull().sum()


# In[ ]:


data_test.head()


# In[ ]:





# In[ ]:


np.array(y_test).reshape(-1,1).shape


# In[ ]:


data_train.shape


# In[ ]:


data_train.head()


# In[ ]:


X_train.head()


# In[ ]:


X_train.shape


# In[ ]:


y_train.head()


# In[ ]:


y_train.shape


# In[ ]:


X_test.head()


# In[ ]:


X_test.shape


# In[ ]:


y_test.head()


# In[ ]:


y_test.shape


# In[ ]:





# In[ ]:


data_test = data_test.drop(columns=['PassengerId'])


# In[ ]:


data_test.shape


# In[ ]:





# In[ ]:


df = pd.DataFrame({'PassengerId': range(892, 1310), 'Survived': (rfc.predict(data_test))})


# In[ ]:


type(df)


# In[ ]:


df.to_csv('TitanicDataSetKaggleVersion2.csv', index=False)


# 
# 
# 
# 
# 

# ---

# Using RandomForestClassifier on the Titanic dataset helped predict survival of passengers with an accuracy of 79.37% .
