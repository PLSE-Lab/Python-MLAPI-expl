#!/usr/bin/env python
# coding: utf-8

# # Introduction

# Hello Viewers!
# 
# Thanks for viewing my notebook on classifying the Penguins to different species based on available attributes. The problem statement we have in our hand is to classify the species of the penguin, given the different predictor variables. We have the dataset available in hand which has the following predictors and targets.
# 
# ### Predictors:
# * Sex
# * Culmen Length (mm)
# * Culmen Depth (mm)
# * Flipper Length (mm)
# * Body Mass (g)
# * Island
# 
# ### Target - Species:
# * Adelie
# * Chinstrap
# * Gentoo

# # Problem Type:
# 
# ### Supervised Learning:
# Supervised learning is a type of training the system by providing labelled inputs. While we feed the system the input features, we also say the expected output. In this case, we are training the system with predictors (independant variables) along with the target (dependant variable).
# 
# ### Classification:
# Classification is a subset of supervised learning where the output or dependant variable is discrete. We have the 'Species' feature as the target which is discrete.
# 
# Hence this is a classification problem. However this dataset can also be used to carry out clustering tasks as well. We are not covering clustering in this notebook.

# # Importing Libraries and Datasets

# The first thing to do is to import the required libraries. I've listed down the libraies we are going to use in this notebook.
# 
# The libraries which are used in this Kernel are,
# * Numpy - Matrices and Mathematical Functions
# * Pandas - Data Manipulation and Analysis
# * Matplotlib - Simple Visualization
# * Seaborn - More Sophisticated Visualizations
# * Scikit Learn - Machine Learning Algorithms and Evaluation Metrics

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #simple data visualization
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns #some advanced data visualizations
import warnings
warnings.filterwarnings('ignore') # to get rid of warnings
plt.style.use('seaborn-white') #defining desired style of viz

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Let's load the dataset and store it in a variable. We'll have a copy of the original dataset so that we can rollback to the original version of the dataset whenever required.

# In[ ]:


df = pd.read_csv('../input/palmer-archipelago-antarctica-penguin-data/penguins_size.csv')
original = df.copy()


# # Quick Inspection of the Data

# In[ ]:


print('Dataset has', df.shape[0] , 'rows and', df.shape[1], 'columns')


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df.isnull().sum()


# This data seems to have some missing values. Let's leave this for now, we'll impute missing values later.

# In[ ]:


df.head(10)


# # Exploratory Data Analysis

# ## Univariate Analysis

# Let's try to understand how the categorical variables are distributed. I'll use the value_counts() method with an argument 'normalize' set to True to see the result i terms of percentage.

# In[ ]:


plt.rcParams['figure.figsize'] = (10,7)


# In[ ]:


df['species'].value_counts(normalize = True).plot(kind = 'bar', color = 'seagreen', linewidth = 1, edgecolor = 'k')
plt.title('Penguin Species')
plt.xlabel('Species')
plt.ylabel('% (100s)')
plt.xticks(rotation = 360)
plt.show()


# In[ ]:


df['island'].value_counts(normalize = True).plot(kind = 'bar', color = 'seagreen', linewidth = 1, edgecolor = 'k')
plt.title('Islands where Penguins live')
plt.xlabel('Island')
plt.ylabel('% (100s)')
plt.xticks(rotation = 360)
plt.show()


# In[ ]:


df['sex'].value_counts(normalize = True).plot(kind = 'bar', color = 'seagreen', linewidth = 1, edgecolor = 'k')
plt.title('Penguins - Sex')
plt.xlabel('Sex')
plt.ylabel('% (100s)')
plt.xticks(rotation = 360)
plt.show()


# The third bar in this graph shows the inconsistency in this feature. This would be treated in the upcoming sections.

# Okay! We explored the categorical features. What about the numerical features?
# 
# Shall we use histograms for this?
# 
# We can use histograms, but it suffers from binning bias. I would go with the Probability Density Function which says the probability of a random variable x picked at a time. Since the variable is continuous, we have chosen PDF.
# 
# We also have something called Empirical Cumulative Distribution Function, which says the probability of getting a value less than or equal to a random value picked at a time. Simple! This is a cumulative distribution function basically, except the fact that the CDF works on samples whereas the ECDF works on the real data.
# 
# Let me write a simple function which can plot both ECDF and PDF. 

# In[ ]:


def ecdf(x):
    n = len(x)
    a = np.sort(x)
    b = np.arange(1, 1 + n) / n
    plt.subplot(211)
    plt.plot(a, b, marker = '.', linestyle = 'None', c = 'seagreen')
    mean_x = np.mean(x)
    plt.axvline(mean_x, c = 'k', label = 'Mean')
    plt.title('ECDF')
    plt.legend()
    plt.show()
    plt.subplot(212)
    sns.distplot(x, color = 'r')
    plt.title('Probability Density Function')
    plt.show()


# In[ ]:


ecdf(df['culmen_length_mm'])


# What does the ECDF shows?
# 
# Well, do you notice a black line there? It is the mean value of this feature, which is at 44. Look at the value in y-axis corresponding to the mean. It is somewhere around 0.5. 
# 
# This infers that the probability of getting a value less than the mean culmen length is 0.5!
# 
# Quite interesting right?
# 
# Let's now look for the probability of getting a value less than the culmen length 40mm. It is around 0.2, which means there is only a 20% chance of the culmen length to be less than 40mm if you randomly pick a value.
# 
# You got the logic?

# In[ ]:


ecdf(df['culmen_depth_mm'])


# In[ ]:


ecdf(df['flipper_length_mm'])


# In[ ]:


ecdf(df['body_mass_g'])


# ## Multivariate Analysis

# As we have analyzed the distribution of every features, let's try to analyze the relationship between them. Let me write a simple function which plots the boxplot of features which is classified by the species and their sex.
# 
# This is a great way to check how the features vary for different sex and species.

# In[ ]:


def box(f):
    sns.boxplot(y = f, x = 'species', hue = 'sex',data = df)
    plt.title(f)
    plt.show()


# In[ ]:


box('culmen_length_mm')


# In[ ]:


box('culmen_depth_mm')


# In[ ]:


box('flipper_length_mm')


# In[ ]:


box('body_mass_g')


# A common thing which I noticed from all the above graphs is that the male penguins have more culmen length, depth, flipper length and body mass irrespective of their species. This would help us immensely during our modelling.

# Now let's plot a pairplot to see the multivariate trends all at the same time.

# In[ ]:


sns.pairplot(df, hue = 'species')
plt.show()


# # Missing Values Treatment

# As you have seen earlier, we were having some missing values in the original dataset. Let's treat them.
# 
# Since the missing values are negligible in number, let's use the most common imputation strategies - mean and mode. For numeric variables, I would use the mean technique and for cateogorical variables mode is used.

# In[ ]:


new_df = original.copy()

new_df['culmen_length_mm'].fillna(np.mean(original['culmen_length_mm']), inplace = True)
new_df['culmen_depth_mm'].fillna(np.mean(original['culmen_depth_mm']), inplace = True)
new_df['flipper_length_mm'].fillna(np.mean(original['flipper_length_mm']), inplace = True)
new_df['body_mass_g'].fillna(np.mean(original['body_mass_g']), inplace = True)
new_df['sex'].fillna(original['sex'].mode()[0], inplace = True)


# In[ ]:


new_df.head()


# In[ ]:


new_df.isnull().sum()


# Cool, now we have got rid of all the missing values. Let's move ahead to the feature transformation.

# # Feature Transformation

# Let's check whether the dataset is skewed. As we have noticed from the density plots of the numeric variables, there was not seen any normal distribution. But let's check the skewnesss of the features once. If the skewness is more, we can transform the variables using np.sqrt, np.log etc.

# In[ ]:


print('Skewness of numeric variables')
print('-' * 35)

for i in new_df.select_dtypes(['int64', 'float64']).columns.tolist():
    print(i, ' : ',new_df[i].skew())


# I do not see that the data is highly skewed. Let's quickly move to the normalization section.

# Why do we need to normalize our data?
# 
# The reason being, the scale of every feature in the dataset is different. We noticed this during our inspection of the dataset at an initial stage. This is something to be treated. 
# 
# I've chosen MinMaxScaler for this exercise. This scales the values in the particular feature such that they lie within 0 and 1. This makes the dataset to have the same range.

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()


# In[ ]:


new_df['culmen_length_mm'] = mms.fit_transform(new_df['culmen_length_mm'].values.reshape(-1, 1))
new_df['culmen_depth_mm'] = mms.fit_transform(new_df['culmen_depth_mm'].values.reshape(-1, 1))
new_df['flipper_length_mm'] = mms.fit_transform(new_df['flipper_length_mm'].values.reshape(-1, 1))
new_df['body_mass_g'] = mms.fit_transform(new_df['body_mass_g'].values.reshape(-1, 1))


# In[ ]:


new_df.head()


# Now the dataset seems to have normalized, let's check this by seeing the summary stats of the data.

# In[ ]:


new_df.describe()


# Did you notice the mean is now in the same range?
# Also the min and max of every variable are 0 and 1. So the dataset is now normalized. 

# We have categorical variables in our dataset. What are we going to do for that? Fine, let's use the pd.get_dummies function to create dummy variables, as these variables can't be randomly assigned any values.

# In[ ]:


new_df_dummy = pd.get_dummies(new_df, columns = ['sex', 'island'], drop_first = True)


# In[ ]:


new_df_dummy['species'].unique()


# In[ ]:


new_df_dummy['species'].replace({'Adelie' : 0,
                                'Chinstrap' : 1,
                                'Gentoo': 2}, inplace = True)


# In[ ]:


sns.heatmap(new_df_dummy.corr(), annot = True, cmap = 'Blues')


# As you see in the correlation map, there is a significant correlation seen between the predictors and the target. This would help us during the modelling stage.

# # Model Building

# Since we are all set, let's start the modelling. Let's import the required machine learning libraries and evaluation metrics from sklearn. 
# 
# Then we'll separate the independant and dependant variables before splitting them into train and test sets using train_test_split.

# In[ ]:


from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# In[ ]:


X = new_df_dummy.drop(columns = ['species', 'sex_FEMALE', 'sex_MALE'])
Y = new_df_dummy['species']


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 123)


# Let's first try with a simple Logistic Regression model.

# In[ ]:


LR = LogisticRegression()
LR.fit(X_train, Y_train)

pred = LR.predict(X_test)


# In[ ]:


print('Accuracy : ', accuracy_score(Y_test, pred))
print('F1 Score : ', f1_score(Y_test, pred, average = 'weighted'))


# This turned out to be a cool task! Let's try cross validation with different models and then pick up one.

# In[ ]:


models = []
models.append(('LR', LogisticRegression()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('RF', RandomForestClassifier()))
models.append(('kNN', KNeighborsClassifier()))
models.append(('SVC', SVC()))


# In[ ]:


for name, model in models:
    kfold = KFold(n_splits = 5, random_state = 42)
    cv_res = cross_val_score(model, X_train, Y_train, scoring = 'accuracy', cv = kfold)
    print(name, ' : ', cv_res.mean())


# In[ ]:


svc = SVC()
svc.fit(X_train, Y_train)

pred = LR.predict(X_test)


# # Model Evaluation

# In[ ]:


print('Accuracy : ', accuracy_score(Y_test, pred))
print('F1 Score : ', f1_score(Y_test, pred, average = 'weighted'))
print('Precision : ', precision_score(Y_test, pred , average = 'weighted'))
print('Recall : ', recall_score(Y_test, pred, average = 'weighted'))


# In[ ]:


confusion_matrix(Y_test, pred)


# We tried modelling using the SVC model and it resulted in a good model. In this kernel, we tried out the basic stuff in all aspects. We can improve this by applying feature engineering (where we create more features which could result in a better model) and hyperparamater tuning.
# 
# We can also use this dataset to apply clustering algorithm to cluster the penguins to 3 clusters based on the species.
# 
# That's it from me!

# Hope you enjoyed this. Kindly upvote if you like this kernel and leave a comment. Thanks!

# In[ ]:




