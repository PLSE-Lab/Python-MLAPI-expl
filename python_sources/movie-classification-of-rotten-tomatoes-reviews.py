#!/usr/bin/env python
# coding: utf-8

# # Project Overview
# - We have a dataset from the moview review website RottenTomatoes.com.
# - The dataset includes written reviews on movies, the names of each critic, review date, and a verdict of whether of not a movie was consider "Fresh" (positive review) or "Rotten" (negative review).
# - The goal of this project is to use the text in each review to create a model to predict whether or not a review is considered "Fresh" or "Rotten" based solely on the information extracted from the text review alone.

# In[ ]:


# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import os
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


print(os.listdir('../input'))


# In[ ]:


# Loading in dataset
df = pd.read_csv('../input/rotten-tomatoes/reviews.tsv', sep='\t', encoding = 'ISO-8859-1')
df.head()


# # Exploratory Data Analysis
# - For our movie review classification, we will stick with the Rotten Tomoatoes benchmark of "Fresh" vs "Rotten".
# - This is a simpler task than predicting a rating as many of the critics appear to use their own metrics (e.g. numeric rating system out of 4, 5 or out of 10. Other critics use a letter grade system.

# In[ ]:


# Way too many types of reviews from critics with each reviewer using their own set of review rating system
print('List of Reviews:')
print(df['rating'].unique())
print('\n')
print('Number of unique reviews:')
print(df['rating'].nunique())


# In[ ]:


# We'll stick with Rotten Tomatoes' final review classification of "Fresh" vs. "Rotten" when training our model
df['fresh'].value_counts()


# In[ ]:


df['fresh'].unique()


# In[ ]:


# Distribution of "Fresh" vs "Rotten" reviews is roughly balanced. 
sns.countplot(df['fresh'])
plt.show()


# In[ ]:


# Checking for missing values
df.isnull().sum()


# In[ ]:


# Since we cannot work with missing data or find a viable way to replace missing text reviews, we will drop these missings rows under reviews.
df = df.dropna(subset=['review'])


# In[ ]:


df.isnull().sum()


# In[ ]:


df.describe(include='all')


# In[ ]:


df_fresh = df[['fresh', 'review']]


# In[ ]:


# Checking for reviews with no text in review
blank_reviews = []

# (index, label, review text)
for i, label, review in df_fresh.itertuples():
    if type(review) == str:
        if review.isspace():
            blank_reviews.append(i)


# In[ ]:


# All remaining reviews contain text
blank_reviews


# In[ ]:


# Addining in a new feature to see if there is any correlation to the length of the review to the fresh rating.
df_fresh['review length'] = df_fresh['review'].apply(lambda review: len(review))


# In[ ]:


df_fresh.head()


# In[ ]:


bins = 20
plt.hist(df_fresh[df_fresh['fresh']=='fresh']['review length'],bins=bins,alpha=0.5)
plt.hist(df_fresh[df_fresh['fresh']=='rotten']['review length'],bins=bins,alpha=0.5)
plt.legend(('fresh','rotten'))
plt.show()


# - Not a clear trend to see if the length of reviews has any relation to the movie review.
# - The distribution of review length looks pretty similar between "Fresh" movies and "Rotten" movies.

# # Model Selection
# - Feature extraction of the text reviews was performed using TfidfVectorizer.
# - The classification models evaluated include:
# - LinearSVC
# - Logistic Regression Model
# - XGBoost
# - Random Forest
# 

# In[ ]:


# Splitting data into training and testing datasets
from sklearn.model_selection import train_test_split

X = df_fresh['review']
y = df_fresh['fresh']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# **Linear SVC**

# In[ ]:


# Building a simple pipeline to preprocess text data
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

text_clf_svc = Pipeline([('tfidf', TfidfVectorizer()),
                    ('clf', LinearSVC())])


# In[ ]:


# Fitting and generating predictions
text_clf_svc.fit(X_train, y_train)
y_pred_svc = text_clf_svc.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(classification_report(y_test, y_pred_svc))
print(confusion_matrix(y_test, y_pred_svc))


# **Logistic Regression**

# In[ ]:


from sklearn.linear_model import LogisticRegression
# Building pipeline
text_clf_lr = Pipeline([('tfidf', TfidfVectorizer()),
                    ('clf', LogisticRegression())])
# Fitting and generating predictions
text_clf_lr.fit(X_train, y_train)
y_pred_lr = text_clf_lr.predict(X_test)


# In[ ]:


print(classification_report(y_test, y_pred_lr))
print(confusion_matrix(y_test, y_pred_lr))


# **XGBoost**

# In[ ]:


from xgboost import XGBClassifier
# Building pipeline
text_clf_xgb = Pipeline([('tfidf', TfidfVectorizer()),
                    ('clf', XGBClassifier())])
# Fitting and generating predictions
text_clf_xgb.fit(X_train, y_train)
y_pred_xgb = text_clf_xgb.predict(X_test)


# In[ ]:


print(classification_report(y_test, y_pred_xgb))
print(confusion_matrix(y_test, y_pred_xgb))


# **Random Forest Classifier**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
# Building pipeline
text_clf_rf = Pipeline([('tfidf', TfidfVectorizer()),
                    ('clf', RandomForestClassifier())])
# Fitting and generating predictions
text_clf_rf.fit(X_train, y_train)
y_pred_rf = text_clf_rf.predict(X_test)


# In[ ]:


print(classification_report(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))


# # Model Comparisons

# In[ ]:


model_performance = [accuracy_score(y_test, y_pred_svc),accuracy_score(y_test, y_pred_lr),accuracy_score(y_test, y_pred_xgb),accuracy_score(y_test, y_pred_rf)]
models = ['Linear SVC', 'Logistic Regression', 'XGBoost', 'Random Forest']
df_model = pd.DataFrame(model_performance, columns=['Accuracy'])
df_model['Model'] = models
df_model


# In[ ]:


plt.figure(figsize=(8,6))
plt.ylim(0.5,0.8)
sns.barplot(x='Model', y='Accuracy', data=df_model)
plt.show()


# - The top two performers were the linear SVC and logistic regression models.
# - Ensemble tree based models don't appear to be that great.
# - Surprisingly the XGBoost model performed worse than the normal random forest model, and the worst out of the four models.

# # Sample Predictions
# - We will use the linear SVC model for the sample predictions.
# - Randomly selected reviews from the testing data will be used for these predictions.

# **Sample Prediction 1**

# In[ ]:


np.random.seed(42)
rand_sample_1 = int(np.random.randint(0, len(X_test), size=1))
list(X_test)[rand_sample_1]


# In[ ]:


y_pred_1 = text_clf_svc.predict([list(X_test)[rand_sample_1]])
y_pred_1


# In[ ]:


df[df['review'] == 'As a work of cinema, The Passion of the Christ possesses a majestic beauty within its horror, one that comes most effectively through a tiny, solitary teardrop.']['fresh']


# - Model Predition: 'fresh'
# - True Result: 'fresh'

# **Sample Prediction 2**

# In[ ]:


np.random.seed(43)
rand_sample_2 = int(np.random.randint(0, len(X_test), size=1))
list(X_test)[rand_sample_2]


# In[ ]:


y_pred_2 = text_clf_svc.predict([list(X_test)[rand_sample_2]])
y_pred_2


# In[ ]:


df[df['review'] == 'A character-driven dramedy with equal parts humor and heart, Safety Not Guaranteed is a magical film about the human spirit whose charm is impossible to ignore.']['fresh']


# - Model Predition: 'fresh'
# - True Result: 'fresh'

# **Sample Prediction 3**

# In[ ]:


np.random.seed(44)
rand_sample_3 = int(np.random.randint(0, len(X_test), size=1))
list(X_test)[rand_sample_3]


# In[ ]:


y_pred_3 = text_clf_svc.predict([list(X_test)[rand_sample_3]])
y_pred_3


# In[ ]:


df[df['review'] == 'My mother is going to love this movie. ']['fresh']


# - Model Predition: 'rotten'
# - True Result: 'rotten'

# **Some notes on these preditions:** <br>
# - After surveying the text reviews for these sample predictions, it seems reasonable to predict on our own that the first two reviews would be given a "Fresh" rating given how the reviews were written.
# - The third review also looked like it had a positive sentiment to it using the word "love". However, the model predicted that this review would be "Rotten".
# - Interestingly, the actual review was indeed "Rotten", perhaps this review was sarcastic and the model appears to have understood this after feature extraction of the text.
