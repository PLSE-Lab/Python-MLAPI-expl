#!/usr/bin/env python
# coding: utf-8

# # **Graduate Admission Analysis for UCLA**

# ### **Data Description**
# The dataset contains information about a student's:
# * GRE Score
# * TOEFL Score
# * University Ratings
# * Statement of Purpose Score
# * Letter of Recomendation Score
# * CGPA
# * Whether the Student Has Done Any Research
# * Chance of Admission (What We're Trying to Predict)

# ## **Importing Libraries and Data**
# Importing libraries and setting the default style in Seaborn.

# In[ ]:


import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
sns.set(style='white')
sns.set(style='whitegrid', color_codes=True)


# Next, let's import our dataset and see what we're working with.

# In[ ]:


df = pd.read_csv("../input/Admission_Predict.csv")
df.head()


# In[ ]:


df.describe()


# ## **Exploratory Analysis**
# 
# From these charts it looks like we have no missing values! 
# 
# It seems as though Serial No. is just an index for students, which we can take out. 
# 
# Two columns also have an added space in the label which we'll take out

# In[ ]:


df.rename(columns = {'Chance of Admit ':'Chance of Admit', 'LOR ':'LOR'}, inplace=True)
df.drop(labels='Serial No.', axis=1, inplace=True)


# Let's plot a heatmap to see the correlation of all the features compared to Chance to Admit:

# In[ ]:


fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(df.corr(), annot=True, cmap='Blues')


# The top three features that affect the Chance to Admit are:
# 1. CGPA
# 2. GRE Score
# 3. TOEFL Score
# 
# Let's explore these three features to get a better understanding.

# ### **CGPA**
# 
# The Cumulative Grade Point Average is a 10 point grading system.
# 
# From the data shown below, it appears the submissions are normally distributed. With a mean of 8.6 and standard deviation of 0.6.
# 
# ### **CGPA vs Chance of Admit**
# 
# It appears as applicant's CGPA has a strong correlation with their chance of admission.

# In[ ]:


plt.figure(figsize=(20,6))
plt.subplot(1,2,1)
sns.distplot(df['CGPA'])
plt.title('CGPA Distribution of Applicants')

plt.subplot(1,2,2)
sns.regplot(df['CGPA'], df['Chance of Admit'])
plt.title('CGPA vs Chance of Admit')


# ### **GRE Score**
# 
# The Graduate Record Examination is a standarized exam, often required for admission to graduate and MBA programs globally. It's made up of three components:
# 1. Analytical Writing (Scored on a 0-6 scale in half-point increments)
# 2. Verbal Reasoning (Scored on a 130-170 scale)
# 3. Quantitative Reasoning (Scored on a 130-170 scale)
# 
# In this dataset, the GRE Score is based on a maximum of 340 points. The mean is 317 with a standard deviation of 11.5.
# 
# ### **GRE Score vs Chance of Admit**
# 
# GRE scores have a strong correlation with the chance of admission however not as strong as one's CGPA.
# 

# In[ ]:


plt.figure(figsize=(20,6))
plt.subplot(1,2,1)
sns.distplot(df['GRE Score'])
plt.title('Distributed GRE Scores of Applicants')

plt.subplot(1,2,2)
sns.regplot(df['GRE Score'], df['Chance of Admit'])
plt.title('GRE Scores vs Chance of Admit')


# ### **TOEFL Score**
# 
# The Test of English as a Foreign Language is a standarized test for non-native English speakers that are choosing to enroll in English-speaking universities.
# 
# The test is split up into 4 sections:
# 1. Reading
# 2. Listening
# 3. Speaking
# 4. Writing
# 
# All sections are scored out of 30, giving the exam a total score of 120 marks. In this dataset, the TOEFL scores have a mean of 107 and a standard deviation of 6.
# 
# ### **TOEFL Score vs Chance of Admit**
# 
# Like GRE scores, the scores received for the TOEFL strongly correlate to an applicants chance of admission.

# In[ ]:


plt.figure(figsize=(20,6))
plt.subplot(1,2,1)
sns.distplot(df['TOEFL Score'])
plt.title('Distributed TOEFL Scores of Applicants')

plt.subplot(1,2,2)
sns.regplot(df['TOEFL Score'], df['Chance of Admit'])
plt.title('TOEFL Scores vs Chance of Admit')


# For my curiosity, I want to explore the data a little bit further regarding research and university rankings. Even though they hold a lower importance in the chance of admission, it would be nice to understand their characteristics in the dataset.

# ### **Research**
# 
# Let's explore how many applicants have research experience.
# 
# It seems the majority of applicants have research experience. However, this is the least important feature, so it doesn't matter all too much if an applicant has the experience or not.

# In[ ]:


fig, ax = plt.subplots(figsize=(8,6))
sns.countplot(df['Research'])
plt.title('Research Experience')
plt.ylabel('Number of Applicants')
ax.set_xticklabels(['No Research Experience', 'Has Research Experience'])


# ### **University Rating**
# 
# Let's see the distribution of applicants coming from each kind of university.
# 
# Most applicants come from a tier 3 and tier 2 university.

# In[ ]:


fig, ax = plt.subplots(figsize=(8,6))
sns.countplot(df['University Rating'])
plt.title('University Rating')
plt.ylabel('Number of Applicants')


# ## **Preparing Data for Machine Learning**
# 
# Now that we understand our dataset, it's time to implement machine learning methods to predict future applicant's chances of admission.
# 
# First we have to prepare our data, by splitting it into training and testing data. We'll also scale our data, from 0 to 1, to receive more accurate predictions.

# In[ ]:


targets = df['Chance of Admit']
features = df.drop(columns = {'Chance of Admit'})

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)


# In[ ]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# ## **Machine Learning **
# 
# The end goal is to determine if an applicant will be admitted to the univeristy or not. Therefore, this is a classification case. We'll use multiple techniques and eventually select the method with the best score. The methods used will be:
# 
# 1. Logistic Regression
# 2. Decision Tree Classification
# 3. Random Forest Classification

# ### **Logistic Regression**

# Our data contains the chance to admit, which is a float value, we should classify the chance of a student's admission as a binary value, accepted and not accepted.
# 
# To transform the chance to admit to binary values, we'll assume that if the student's chance is above the mean, they are accepted. If not, they will not be accepted.
# 
# So, let's find the mean chance of admission and transform the 'Chance to Admit' column to binary values.

# In[ ]:


mean_chance = df['Chance of Admit'].mean()


# In[ ]:


y_train_binary = (y_train > 0.72).astype(int)
y_test_binary = (y_test > 0.72).astype(int)


# In[ ]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train_binary)
y_predict = logreg.predict(X_test)
logreg_score = (logreg.score(X_test, y_test_binary))*100
logreg_score


# ### **Decision Trees**

# In[ ]:


dec_tree = DecisionTreeClassifier(random_state=0, max_depth=6)
dec_tree.fit(X_train, y_train_binary)
y_predict = dec_tree.predict(X_test)
dec_tree_score = (dec_tree.score(X_test, y_test_binary))*100
dec_tree_score


# ### **Random Forests**

# In[ ]:


forest = RandomForestClassifier(n_estimators=110,max_depth=6,random_state=0)
forest.fit(X_train, y_train_binary)
y_predict = forest.predict(X_test)
forest_score = (forest.score(X_test, y_test_binary))*100
forest_score


# ### **Comparing Scores**
# 
# Let's put all the scores in a table and display their scores side-by-side.

# In[ ]:


Methods = ['Logistic Regression', 'Decision Trees', 'Random Forests']
Scores = np.array([logreg_score, dec_tree_score, forest_score])

fig, ax = plt.subplots(figsize=(8,6))
sns.barplot(Methods, Scores)
plt.title('Algorithm Prediction Accuracies')
plt.ylabel('Accuracy')


# ### **Selecting the Best Algorithm**
# 
# 1. Logistic Regression - 96.25%
# 2. Random Forests - 96.25%
# 3. Decision Trees - 92.5%
# 
# It seems that Logistic Regression and Random Forests are the most accurate methods and will be used to predict the future applicant's chances of admission.
# 
# 

# ## **Conclusion**
# 
# This was a great way to get started on Kaggle and for my first project outside of coursework. It gave me some practice some exploratory analysis and simple machine learning techniques. 
# 
# It's great to see what specific variables contribute to the chance of admission and how they are weighted against eachother.
