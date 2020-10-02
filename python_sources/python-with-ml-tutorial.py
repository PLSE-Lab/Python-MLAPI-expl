#!/usr/bin/env python
# coding: utf-8

# # Python ML Book

# # Introduction to ML
# 
# What is Machine Learning??
# 
# ML is used for many things - Image/Voice Recognition, Fraud detection, Self driving cars etc. So what is ML? 
# 
# Humans vs Computers - Learn from Experience - Heuristics! 
# 
# Well the person who coined the term "Machine Learning" defined it as :
# 
# >Field of study that gives computers the ability to learn without being explicitly programmed.  ~ Arthur Samuel, circa 1959
# 
# Another person came along and said..hmm, that's too vague, let me make it a defintion worthy for engineering students - and he said :
# 
# > A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E.  ~Tom Mitchell, 1988

# ![MLBox.png](attachment:MLBox.png)

# You initially train the system by showing a different set of images/documents or inputs and adjust the weights so that the output it shows is accurate. Later, when you just give the input, the machine automatically shows the correct output, since its weights have been tuned.
# 
# That's machine learning :)
# 
# But, hold on -- that's only one type of Machine Learning. And this type of machine learning is called Supervised learning.

# ## Types of Machine Learning
# 
# So broadly speaking there are multiple type of Machine learning (algorithms) and  they are :
# 
# 1. Supervised Learning  - Any type of ML algorithm which takes in a labeled data set (called training data) to learn from
#         a. Classification
#         b. Regression
# 2. Unsupervised Learning - In this type of ML algorithm, the data provided is not labeled and the ML algorithm needs to learn on its own and identify the patterns and associations within the data.
# 
# 3. Reinforcement Learning - In this, there typically is an agent which is provided positive and negative feedback of its previous action along with the input. If its previous action was positive, it will react with more of those action and its negative, it will react with less of those actions. There is not predefined data provided here.
# 
# We will concentrate more on the Supervised Learning algorithms in this session
# 

# ## Machine Learning workflow
# 
# A typical ML workflow is as follows :
# 1. Problem Defintion
# 2. Acquire data.
# 3. Data tidying, cleansing/preprocessing.
# 4. Exploratory data analysis & Feature engineering
# 5. Modelling, prediction &  optimization
# 6. Visualization & reporting
# 
# These steps are indicative only and they can be combined (visualization & EDA) or some steps can be ignored or done more than once (feature engg.) etc. 
# 
# In this tutorial, I will introduce you to section 3 using a library called Pandas. We will looking into Data tidying a bit and then pick up a problem definition and create a model for it..
# 
# But before I do, there is a small introduction to Numpy - which is the core of Pandas.

# # Tidying Dataset using Pandas.
# https://github.com/chendaniely/pydatadc_2018-tidy -- 
# 
# Pandas is used to work with tabular data. Most of the time, we work with ML with structured data like CSV or TSV.
# 
# So how do we get into pandas to work for us? Every library needs to be first imported.
# 

# In[ ]:


import pandas as pd


# So lets read some data using Pandas
# 
# 

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


print(os.listdir("../input"))
#gapminder_tsv_df = pd.read_csv('../input/tidyData2/billboard.csv')


# In[ ]:


gapminder_df = pd.read_csv('../input/tidydata2/gapminder.tsv',sep='\t')


# Pandas has 2 datastructures - Series and Dataframes.
# 
# Series are like numpy array - contains a list of similar things.
# 
# Dataframes on other hands are a set of Series...so you can consider a group of series attached together..to provide you a tabular structure and thats Dataframes.
# 
# Lets print some of the values of the data frame we just read
# 

# In[ ]:


gapminder_df.head()


# In[ ]:


gapminder_df.columns


# In[ ]:


gapminder_df.index


# In[ ]:


##You see the numpy array 
gapminder_df.values


# Maybe you want to look at some random values in the dataframe

# In[ ]:


gapminder_df.sample(10)


# In[ ]:


gapminder_df.describe()


# In[ ]:


gapminder_df.info()


# In[ ]:


#Subsetting
country_df = gapminder_df[['country']]
country_df


# In[ ]:


type(country_df) ## Magic! Its a series..not a df


# In[ ]:


cc_df = gapminder_df[['country','continent']]
type(cc_df)


# In[ ]:


##Subsetting by rows - 3 ways. 

gapminder_df.head()


# In[ ]:


gapminder_df.loc[2]  ##This is not the index. This is the row label 2.


# In[ ]:


gapminder_df.iloc[2] ## This is the index.. this counts 0,1,2 -- takes the data at count 2 and prints!


# In[ ]:


gapminder_df.loc[[2,0]]


# In[ ]:


gapminder_df.iloc[[2,10]]


# In[ ]:


#Boolean subsetting. loc can be used for subseting both rows and columns.
subset = gapminder_df.loc[(gapminder_df['year']==1967) & (gapminder_df['pop']>10000000),['country']]
subset.head()


# In[ ]:


subset.shape


# ## Tidying
# http://vita.had.co.nz/papers/tidy-data.pdf
# 
# What is the formal defintion of Tidy data?
# 
# > 1. Each variable forms a column.
# > 2. Each observation forms a row.
# > 3. Each type of observational unit forms a table.
# 
# We mostly would be looking at point 1 & 2.
# 
# Data is messy in the real world...and the typical problems the paper mentions are :
# 
# > 1. Column headers are values, not variable names.
# > 2. Multiple variables are stored in one column.
# > 3. Variables are stored in both rows and columns.
# > 4. Multiple types of observational units are stored in the same table.
# > 5. A single observational unit is stored in multiple tables
# 
# Most of the data we might get would be having the first 3 issues..and we will look at how to solve them now.
# 
# 

# In[ ]:


pew_df = pd.read_csv('../input/tidydata2/pew.csv')
pew_df.head()


# In[ ]:


#How to fix this? Long data - lot more rows, Wide data - lot more columns. Data scientists hate wide!
pew_df.melt(id_vars=['religion'],var_name='income',value_name='counts')


# #### Scenario 2 :: Multi-variable columns

# In[ ]:


ebola_df = pd.read_csv('../input/tidydata2/country_timeseries.csv')
ebola_df.head()


# In[ ]:


ebola_df = ebola_df.melt(id_vars=['Date','Day'],value_name='counts')
ebola_df.sample(20)


# In[ ]:


#Assigning the split back to the same series
ebola_df['variable']=ebola_df['variable'].str.split('_')


# In[ ]:


#Now this series is a list - of which we try to get the 1st index
ebola_df['variable'].str.get(0)


# In[ ]:


#assigning the 1st and 2nd index to 2 new columns or series in the data frame
ebola_df['scenario'] = ebola_df['variable'].str.get(0)
ebola_df['country'] = ebola_df['variable'].str.get(1)


# In[ ]:


ebola_df.sample(10)


# #### Scenario 3 :: Variables are stored in both rows and columns.
# One of symptoms of this happening is a lot of repeated data b/w 2 rows except of a single column or so

# In[ ]:


weather_df = pd.read_csv('../input/tidydata2/weather.csv')
weather_df.head()


# In[ ]:


weather_df = weather_df.melt(id_vars=['id','year','month','element'],var_name='day',value_name='temp')
weather_df.head(10)


# In[ ]:


weather_df = weather_df.pivot_table(index=['id','year','month','day'],columns='element',values='temp')
weather_df.sample(10)


# In[ ]:


weather_df.reset_index().head()


# # Data Cleansing
# 
# 1. Outliers in target variable
# 
# 1.5 times 25th percentile to 1.5 times 75th percentile - Any data beyond this, can be considered an outlier and thereby removed.
# 
# 2. Missing values
# 
# Most of the times, real world data would contain null/missing values. In Python they are represented by the symbol 'NaN'. There are multiple ways in which missing data can be handled. They are :
# 
#     a. Ignore the column which contains the missing data
# 
#     b. Impute the values withe mean, median or mode of the other observations available.
# 
#     c. Impute  0 (sometimes 0 is better than a null)
# 
#     d. Use a simple model to predict the value based on other features available and then impute 
#     values based on predictions.
# 
# 3. Removing un-necessary data / multi-collinear features
# 
# 4. Feature scaling or Normalization - Bring all numerical values to the same scale
# 
# 5. Handling of categorical data :
# 
#         a. Label Encoding
#    
#         b. One-hot Encoding

# In[ ]:


train = pd.read_csv('../input/sessiondata/train.csv')
test = pd.read_csv('../input/sessiondata/test.csv')
# Concatenate train & test
train_objs_num = len(train)
y = train['Survived']
dataset = pd.concat(objs=[train.drop(columns=['Survived']), test], axis=0)
dataset.info()


# In[ ]:


total = dataset.isnull().sum().sort_values(ascending=False)
percent = (dataset.isnull().sum()/dataset.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head()


# In[ ]:


df = dataset
# Will drop all features with missing values 
df.dropna(inplace = True)
df.isnull().sum()


# In[ ]:


df1 = dataset
# Will drop the rows only if all of the values in the row are missing
df1.dropna(how = 'all',inplace = True)
df1.head()


# In[ ]:


df2 = train
df2['Age'].fillna(df2['Age'].median(),inplace=True)
df2.head()


# In[ ]:


data_unique = train
data_unique['Cabin'].head(10)


# In[ ]:


data_unique['Cabin'].fillna('U').head(10)


# You can also use another library called scikit-learn (sklearn) for doing these data preprocessing steps with much more control.
# 
# Lets take a look at Feature scaling

# In[ ]:


from sklearn import preprocessing

train.head()

fare_df = train[['Fare','Age']]
Fare_scaled = preprocessing.scale(fare_df)  #Returns an np.array
fare_df.head()


# In[ ]:


pd.DataFrame(Fare_scaled).head()


# In[ ]:


category_data = data_unique['Cabin'].fillna('U')
le = preprocessing.LabelEncoder()
le.fit(category_data)
pd.Series(le.transform(category_data)).to_frame(name="cabin_type").sample(10)


# # Modelling 
# 
# ### Problem defn:
# 
# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.

# In[ ]:


##Importing all libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = 10,8
sns.set(style='whitegrid', palette='muted',
        rc={'figure.figsize': (15,10)})


# In[ ]:


# Reading the data

train = pd.read_csv('../input/cleantitanic/train_clean.csv', )
test = pd.read_csv('../input/cleantitanic/test_clean.csv')
df = pd.concat([train, test], axis=0, sort=True)
df.head()


# In[ ]:


## Data Cleansing

#Dropping unnecessary columns
df.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)

# convert to category dtype
df['Sex'] = df['Sex'].astype('category')
# convert to category codes
df['Sex'] = df['Sex'].cat.codes

df.head()


# In[ ]:


#Label encoding remaining categories. You can try to do one hot encoding.
enc = LabelEncoder()
df_cat  = df[['Embarked','Title']].apply(enc.fit_transform)

df_cat.sample(10)


# In[ ]:


df['Embarked'] = df_cat['Embarked']
df['Title'] = df_cat['Title']
df.head()


# In[ ]:


df.shape


# In[ ]:


#Splitting test data out of the training data (remember we had 2 CSV)
train = df[pd.notnull(df['Survived'])]
test = df[pd.isnull(df['Survived'])].drop(['Survived'], axis=1)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(
    train.drop(['Survived'], axis=1),
    train['Survived'],
    test_size=0.2, random_state=42)

for i in [X_train, X_test]:
    print(i.shape)


# # Linear Classifier

# How it fits --  Minimization of errors due to unclassified points. 

# In[ ]:


from sklearn.linear_model import LogisticRegression # Logistic Regression

model = LogisticRegression()
model.fit(X_train,y_train)
prediction_lr=model.predict(X_test)

print('--------------The Accuracy of the model----------------------------')
print('The accuracy of the Logistic Regression is',round(accuracy_score(prediction_lr,y_test)*100,2))


# # Decision Tree & Ensemble

# In[ ]:


rf = RandomForestClassifier(random_state=40)
rf.fit(X_train, y_train)
accuracy_score(y_test, rf.predict(X_test))


# **Explain Overfitting / Underfitting**

# # Cross validation
# 
# 

# We lose data, to train when we split the input into 2 chunks. We can try creating cross validation sets - which 

# In[ ]:


X_train_2 = pd.concat([X_train, X_test])
y_train_2 = pd.concat([y_train, y_test])


# In[ ]:


rf = RandomForestClassifier(n_estimators=10, random_state=88)
cross_val_score(rf, X_train, y_train, cv=5)
cross_val_score(rf, X_train, y_train, cv=5).mean()


# # Hyperparameter Tuning

# In[ ]:


n_estimators = [10, 100, 1000, 2000]
max_depth = [None, 5, 10, 20]
param_grid = dict(n_estimators=n_estimators, max_depth=max_depth)


# In[ ]:


# create the default model
rf = RandomForestClassifier(random_state=42)

# search the grid
grid = GridSearchCV(estimator=rf, 
                    param_grid=param_grid,
                    cv=3,
                    verbose=2,
                    n_jobs=-1)

grid_result = grid.fit(X_train, y_train)


# In[ ]:


grid_result.best_params_


# In[ ]:


grid_result.best_score_

