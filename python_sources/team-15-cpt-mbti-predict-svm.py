#!/usr/bin/env python
# coding: utf-8

# # MBTI personality profile prediction

# # 1. Introduction
# ## 1.1 Problem Brief

# In this challenge, we are tasked with building and training a model(s) capable of predicting a person's MBTI label using only what they post in online forums.
# 
# To convert the post data into machine learning format, we are required to use Natural Language Processing. Ultimately, this converted data will be used to train a classifier capable of assigning MBTI labels to a person's online forum posts.
# 
# Each MBTI personality type consists of four binary variables, they are: 
# - Mind: Introverted (I) or Extraverted (E) 
# - Energy: Sensing (S) or Intuitive (N) 
# - Nature: Feeling (F) or Thinking (T) 
# - Tactics: Perceiving (P) or Judging (J)
# 
# We will need to build and train a model to predict labels for each of the four MBTI variables. For each person, four separate labels are predicted that when combined results in that person's personality type.

# ## 1.2 Importing the Data and necessary Packages

# In[ ]:


# importing the necessary modules
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, accuracy_score, log_loss
import warnings
import re
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 1.3 Looking at the data
# Some exploratory analysis is always needed when processing new data. After getting an idea of some of the data's features, gaps, traits, we can begin our preprocessing.

# In[ ]:


# importing train data
mbti = pd.read_csv('../input/train.csv')


# In[ ]:


mbti.head()


# In[ ]:


mbti.info()


# From the information above, we can see that there are no missing values in the 6506 rows and two columns of the dataset.
# Let's have a look at how many of the different MBTI types we have data for.

# In[ ]:


mbti['type'].value_counts().plot(kind='bar')
plt.show()


# # 2. Train Data Preprocessing
# ## 2.1 Encoding categories
# Converting the categories from a single text item into a numerical format for each of the 4 MBTI categories, in order for our model to process the data.

# In[ ]:


# M = 1 if extrovert M = 0 if introvert
mbti['mind'] = mbti['type'].apply(lambda x: 1 if x[0] == 'E' else 0)


# In[ ]:


# E = 1 if intuitive(N) E = 0 if observant(S)
mbti['energy'] = mbti['type'].apply(lambda x: 1 if x[1] == 'N' else 0)


# In[ ]:


# N = 1 if thinking N = 0 if feeling
mbti['nature'] = mbti['type'].apply(lambda x: 1 if x[2] == 'T' else 0)


# In[ ]:


# T = 1 if judging N = 0 if prospecting
mbti['tactics'] = mbti['type'].apply(lambda x: 1 if x[3] == 'J' else 0)


# In[ ]:


mbti.head()


# By creating a pie chart for each of the 4 functions, we can see that there is quite a disparity in representation.

# In[ ]:


# mind category
labels = ['Extraversion', 'Introversion']
sizes = [mbti['mind'].value_counts()[1], mbti['mind'].value_counts()[0]]

fig, ax = plt.subplots(2, 2, figsize=(8, 8))
ax[0, 0].pie(sizes, labels=labels, autopct='%1.1f%%',
             shadow=False, startangle=90)
ax[0, 0].axis('equal')

# energy category
labels = ['Intuitive', 'Observant']
sizes = [mbti['energy'].value_counts()[1], mbti['energy'].value_counts()[0]]

ax[0, 1].pie(sizes, labels=labels, autopct='%1.1f%%',
             shadow=False, startangle=90)
ax[0, 1].axis('equal')

# nature category
labels = ['Thinking', 'Feeling']
sizes = [mbti['nature'].value_counts()[1], mbti['nature'].value_counts()[0]]

ax[1, 0].pie(sizes, labels=labels, autopct='%1.1f%%',
             shadow=False, startangle=90)
ax[1, 0].axis('equal')

# tactics category
labels = ['Judging', 'Prospecting']
sizes = [mbti['tactics'].value_counts()[1], mbti['tactics'].value_counts()[0]]

ax[1, 1].pie(sizes, labels=labels, autopct='%1.1f%%',
             shadow=False, startangle=90)
ax[1, 1].axis('equal')
plt.tight_layout()
plt.show()


# ## 2.2 Removing noise
# Before the data can be fit and train to the model, the data needed to be preprocessed to remove noise (i.e. URLs, punctuation, capital letters, numerical digits) so as to make our predictions as accurate as possible. 

# #### Remove the web-urls

# In[ ]:


# replacing url links with 'url-web'
first_pattern = '[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|'
second_pattern = '[!*,]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'
pattern_url = r'http'+first_pattern+second_pattern

subs_url = r'url-web'
mbti['posts'] = mbti['posts'].replace(
                to_replace=pattern_url, value=subs_url, regex=True)


# #### Remove numbers

# In[ ]:


# replacing numbers with ''
pattern_numbers = r'\d+'
subs_numbers = r''
mbti['posts'] = mbti['posts'].replace(
                to_replace=pattern_numbers, value=subs_numbers, regex=True)


# #### Making everything lower case

# In[ ]:


mbti['posts'] = mbti['posts'].str.lower()


# #### Remove '|||' that separates posts

# In[ ]:


# replacing '|||' with ' '
pattern_lines = r'\[|]|[|]|[|]+'
subs_lines = r' '
mbti['posts'] = mbti['posts'].replace(to_replace = pattern_lines, value = subs_lines, regex = True)


# #### Remove punctuation

# In[ ]:


def remove_punctuation(post):
    return ''.join([l for l in post if l not in string.punctuation])


# In[ ]:


mbti['posts'] = mbti['posts'].apply(remove_punctuation)


# Let's have a final look at the data after cleaning it up.

# In[ ]:


mbti.head()


# ## 2.3 Vectorizer
# Next we need to build our vectorisor. This will convert our collection of text documents to a matrix of token counts, further allowing us to begin our Natural Language Processing.

# In[ ]:


# building our vectorizer
count_vec = CountVectorizer(stop_words='english',
                            lowercase=True,
                            max_df=0.5,
                            min_df=2,
                            max_features=200)


# In[ ]:


# transforming posts to a matrix of token counts
X_count = count_vec.fit_transform(mbti['posts'])


# In[ ]:


X = X_count


# # 3. MBTI Categories
# We need to split our data into the various train and test sets before we can begin our classification model building and training. Here we are separating the different MBTI categories into separate train and test sets, each going to be used in their own classification model.
# ## 3.1 Mind Classifier
# The _Mind_ category refers to where one focuses one's attention, either Extraversion (E) or Introversion (I).

# In[ ]:


y_mind = mbti['mind']


# In[ ]:


a, b, c, d = train_test_split(X, y_mind, test_size=0.2, random_state=42)
X_train_mind = a
X_test_mind = b
y_train_mind = c
y_test_mind = d


# #### Building SVM model for mind classifier
# Building the classifcation model to predict the _Mind_ category, using a Support Vector Machine model (SVM) as well as the GridSearchCV() model in order to select the best possible parameters.

# In[ ]:


svm = SVC()
params = {'C': [1.0, 0.001, 0.01],
          'gamma': ['auto', 'scale', 1.0, 0.001, 0.01]}
clf_mind = GridSearchCV(estimator=svm, param_grid=params)
clf_mind.fit(X_train_mind, y_train_mind)


# In[ ]:


#  gives parameter setting that returned the best results
clf_mind.best_params_


# In[ ]:


y_pred_mind = clf_mind.predict(X_test_mind)
print("Accuracy score:", accuracy_score(y_test_mind, y_pred_mind))


# ## 3.2 Energy Classifier
# The _Energy_ category refers the way one takes in information, either Sensing (S) or INtuition (N).

# In[ ]:


y_energy = mbti['energy']


# In[ ]:


a, b, c, d = train_test_split(X, y_energy, test_size=0.2, random_state=42)
X_train_energy = a
X_test_energy = b
y_train_energy = c
y_test_energy = d


# #### Building SVM model for energy classifier 
# Building the classifcation model to predict the _Energy_ category, using a Support Vector Machine model (SVM) as well as the GridSearchCV() model in order to select the best possible parameters.

# In[ ]:


svm = SVC()
params = {'C': [1.0, 0.001, 0.01],
          'gamma': ['auto', 'scale', 1.0, 0.001, 0.01]}
clf_energy = GridSearchCV(estimator=svm, param_grid=params)
clf_energy.fit(X_train_energy, y_train_energy)


# In[ ]:


# gives parameter setting that returned the best results
clf_energy.best_params_


# In[ ]:


y_pred_energy = clf_energy.predict(X_test_energy)
print("Accuracy score:", accuracy_score(y_test_energy, y_pred_energy))


# ## 3.3 Nature Classifier
# The _Nature_ category refers to how one makes decisions, either Thinking (T) or Feeling (F).

# In[ ]:


y_nature = mbti['nature']


# In[ ]:


a, b, c, d = train_test_split(X, y_nature, test_size=0.2, random_state=42)
X_train_nature = a
X_test_nature = b
y_train_nature = c
y_test_nature = d


# #### Building SVM model for nature classifier
# Building the classifcation model to predict the _Nature_ category, using a Support Vector Machine model (SVM) as well as the GridSearchCV() model in order to select the best possible parameters.

# In[ ]:


svm = SVC()
params = {'C': [1.0, 0.001, 0.01],
          'gamma': ['auto', 'scale', 1.0, 0.001, 0.01]}
clf_nature = GridSearchCV(estimator=svm, param_grid=params)
clf_nature.fit(X_train_nature, y_train_nature)


# In[ ]:


# gives parameter setting that returned the best results
clf_nature.best_params_


# In[ ]:


y_pred_nature = clf_nature.predict(X_test_nature)
print("Accuracy score:", accuracy_score(y_test_nature, y_pred_nature))


# ## 3.4 Tactics Classifier
# The _Tactics_ category refers to how one deals with the world, either Judging (J) or Perceiving (P).

# In[ ]:


y_tactics = mbti['tactics']


# In[ ]:


a, b, c, d = train_test_split(X, y_tactics, test_size=0.2, random_state=42)
X_train_tactics = a
X_test_tactics = b
y_train_tactics = c
y_test_tactics = d


# #### Building SVM model for tactics classifier
# Building the classifcation model to predict the _Tactics_ category, using a Support Vector Machine model (SVM) as well as the GridSearchCV optimiser in order to select the best possible parameters.

# In[ ]:


svm = SVC()
params = {'C': [1.0, 0.001, 0.01, 0.1],
          'gamma': ['auto', 'scale', 1.0, 0.001, 0.01, 0.1]}
clf_tactics = GridSearchCV(estimator=svm, param_grid=params)
clf_tactics.fit(X_train_tactics, y_train_tactics)


# In[ ]:


# gives parameter setting that returned the best results
clf_tactics.best_params_


# In[ ]:


y_pred_tactics = clf_tactics.predict(X_test_tactics)
print("Accuracy score:", accuracy_score(y_test_tactics, y_pred_tactics))


# # 4. Test Data Preprocessing 
# Now that the training data has been successfully preprocessed, the test data needs to go through the same processes in order to match the training dataset.

# In[ ]:


# importing test data
test_mbti = pd.read_csv('../input/test.csv')


# In[ ]:


test_mbti.head()


# ### Removing noise
# Just like the train data before, the test data needs to be preprocessed to remove noise (i.e. URLs, punctuation, capital letters, numerical digits) so as to make our predictions as accurate as possible.
# #### Removing the web-urls

# In[ ]:


# replacing url links with 'url-web'
x = pattern_url
y = subs_url
z = True
test_mbti['posts'] = test_mbti['posts'].replace(to_replace=x, value=y, regex=z)


# #### Remove numbers

# In[ ]:


# replacing numbers with ''
a = pattern_numbers
b = subs_numbers
c = True
test_mbti['posts'] = test_mbti['posts'].replace(to_replace=a, value=b, regex=c)


# #### Making everything lower case

# In[ ]:


test_mbti['posts'] = test_mbti['posts'].str.lower()


# #### Remove '|||' that separates posts

# In[ ]:


test_mbti['posts'] = test_mbti['posts'].replace(to_replace = pattern_lines, value = subs_lines, regex = True)


# #### Remove punctuation

# In[ ]:


test_mbti['posts'] = test_mbti['posts'].apply(remove_punctuation)


# In[ ]:


test_mbti.head()


# ### Transforming posts in test data to a matrix of token counts
# Once again using a CountVectorisor() to tokenise the data.

# In[ ]:


X_count_test = count_vec.fit_transform(test_mbti['posts'])
X_test = X_count_test


# ## 5. Predicting the test data labels

# In[ ]:


# mind predictions of the test data
mind_predictions = clf_mind.predict(X_test)


# In[ ]:


# energy predictions of the test data
energy_predictions = clf_energy.predict(X_test)


# In[ ]:


# nature predictions of the test data
nature_predictions = clf_nature.predict(X_test)


# In[ ]:


# tactics predictions of the test data
tactics_predictions = clf_tactics.predict(X_test)


# ## 6. Submission
# Bringing all our predicted data into a single dataframe, ready for export to .csv for submission.

# In[ ]:


# creating a dataframe to be exported to a csv file,
# containing all the relevant column names.
listym = ['id', 'mind', 'energy', 'nature', 'tactics']
submission = pd.DataFrame(columns=listym)


# In[ ]:


submission['id'] = test_mbti['id']


# In[ ]:


submission['mind'] = mind_predictions

submission['energy'] = energy_predictions

submission['nature'] = nature_predictions

submission['tactics'] = tactics_predictions


# In[ ]:


# saving submission to csv file
submission.to_csv('submission.csv', index=False)


# ## 7. Conclusion
# We first tried to predict personality types without breaking them up into the 4 Functions they consist of (Mind, Energy, Nature and Tactics) but we found that approach to be significantly more inaccurate, so we instead decided to break them up and predict each of those 4 functions individually.
# 
# Initially we tried using Logistic Regression and Random Forest models for the predictions, but Support Vector Classification models proved to be more accurate.
# 
# The single biggest step we took that managed to significantly improve our Kaggle score was using the GridSearchCV function to find the optimal parameters for our SVC models.
