#!/usr/bin/env python
# coding: utf-8

# # Obtain Optimal Probability Threshold Using PR Curve
# 
# The aim of this notebook is to demonstrate how to use the Precision Recall (PR) curve to obtain optimal probability threshold to improve the predictive capability of a machine learning model.
# 
# This notebook is similar to https://www.kaggle.com/nicholasgah/obtain-optimal-probability-threshold-using-roc, just that now, the curve used is that of precision-recall (PR).

# ## Table of Contents
# 
# 1. [Import Packages](#1)
# 2. [Import Data](#2)
# 3. [Extracting X, Y](#3)
# 4. [Preprocess Texts](#4)
# 5. [Train Test Split](#5)
# 6. [Feature Extraction and Train Model](#6)
# 7. [Evaluate Model (Before Thresholding)](#7)
# 8. [Confusion Matrix of Model (Before Thresholding)](#8)
# 9. [PR Curve](#9)
# 10. [Obtain Optimal Probability Thresholds with PR Curve](#10)
# 11. [Evaluate Model (After Thresholding)](#11)
# 12. [Confusion Matrix of Model (After Thresholding)](#12)
# 13. [Conclusion](#13)
# 14. [References](#14)

# ## Import Packages <a class="anchor" id="1"></a>

# In[ ]:


import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, plot_precision_recall_curve
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## Import Data <a class="anchor" id="2"></a>
# 
# With respect to the dataset, we will only focus on the columns named "Phrase" and "Sentiment". The unique values in "Sentiment" are 0, 1, 2, 3, 4, where increasing values would represent more positive sentiment. Hence, this would make this a binary classification problem.
# 
# **For this notebook, we shall just focus on the training set as the main objective is to showcase the retrieval of optimal probability thresholds using the PR Curve.**

# In[ ]:


df = pd.read_csv("/kaggle/input/sentiment-analysis-on-movie-reviews/train.tsv.zip", sep="\t")
df.head()


# In[ ]:


print("Unique Values of Sentiment are: {}".format(", ".join(list(map(str,df["Sentiment"].unique())))))


# ## Extracting X, Y <a class="anchor" id="3"></a>
# 
# For this dataset, we shall let sentiment values above 2 represent positive ones. As a result, positive movie reviews make up less than 50% of the dataset. 

# In[ ]:


X = df["Phrase"].tolist()
Y = df["Sentiment"].apply(lambda i: 0 if i <= 2 else 1)


# In[ ]:


Y.value_counts()


# ## Preprocess Texts <a class="anchor" id="4"></a>
# 
# Some typical text preprocessing steps will be performed:
# 
# 1. Removal of markup, html
# 2. Obtain only words in lower case
# 3. Lemmatization
# 4. Removal of stop words

# In[ ]:


lemmatizer = WordNetLemmatizer()
def proc_text(messy): #input is a single string
    first = BeautifulSoup(messy, "lxml").get_text() #gets text without tags or markup, remove html
    second = re.sub("[^a-zA-Z]"," ",first) #obtain only letters
    third = second.lower().split() #obtains a list of words in lower case
    fourth = set([lemmatizer.lemmatize(str(x)) for x in third]) #lemmatizing
    stops = set(stopwords.words("english")) #faster to search through a set than a list
    almost = [w for w in fourth if not w in stops] #remove stop words
    final = " ".join(almost)
    return final


# In[ ]:


X = [proc_text(i) for i in X]


# ## Train Test Split <a class="anchor" id="5"></a>

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=100, test_size=0.2, stratify=Y)


# In[ ]:


print("Training Set has {} Positive Labels and {} Negative Labels".format(sum(y_train), len(y_train) - sum(y_train)))
print("Test Set has {} Positive Labels and {} Negative Labels".format(sum(y_test), len(y_test) - sum(y_test)))


# ## Feature Extraction and Train Model <a class="anchor" id="6"></a>
# 
# Features will be built using tfidf.
# 
# Model selected here is the RandomForestClassifier, larger weight is given to the positive class since the number of samples with positive labels are significantly smaller. The weights would be calculated as 
# 
# $$ W_p = \frac{N_n}{N_p}, $$
# 
# where $ W_p $ is a float indicating the weight for positive class, $ N_n $ is the number of negative samples and $ N_p $ is the number of positive samples. The output of this computation will be included in the *class_weight* parameter of RandomForestClassifier.
# 
# These steps will be collated by using sklearn's Pipeline.

# In[ ]:


pos_weights = (len(y_train) - sum(y_train)) / (sum(y_train)) 
pipeline_tf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', DecisionTreeClassifier(random_state=100, class_weight={0: 1, 1: pos_weights}))
])


# In[ ]:


pipeline_tf.fit(X_train, y_train)


# ## Evaluate Model (Before Thresholding) <a class="anchor" id="7"></a>

# In[ ]:


predictions = pipeline_tf.predict(X_test)
predicted_proba = pipeline_tf.predict_proba(X_test)


# In[ ]:


print("Accuracy Score Before Thresholding: {}".format(accuracy_score(y_test, predictions)))
print("Precision Score Before Thresholding: {}".format(precision_score(y_test, predictions)))
print("Recall Score Before Thresholding: {}".format(recall_score(y_test, predictions)))
print("F1 Score Before Thresholding: {}".format(f1_score(y_test, predictions)))
print("ROC AUC Score: {}".format(roc_auc_score(y_test, predicted_proba[:, -1])))


# ## Confusion Matrix of Model (Before Thresholding) <a class="anchor" id="8"></a>

# In[ ]:


y_actual = pd.Series(y_test, name='Actual')
y_predict_tf = pd.Series(predictions, name='Predicted')
df_confusion = pd.crosstab(y_actual, y_predict_tf, rownames=['Actual'], colnames=['Predicted'], margins=True)
print(df_confusion)


# ## PR Curve <a class="anchor" id="9"></a>
# 
# The curve is plots values of precision scores (y-axis) against those of recall scores (x-axis) and these values are plotted at various probability thresholds.
# 
# There can be two ways of obtaining a more optimal probability threshold for the positive class:
# 
# 1. Minimize the difference between precision and recall scores
#     - Select the probability threshold of which precision and recall scores are closest to each other
#    
# 2. Euclidean Distance
#     - The most optimal point on the PR curve should be (1,1), i.e. precision and recall scores of 1.
#     - Select the probability threshold as the most optimal one if precision and recall scores are closest fo the ones mentioned in the previous point in terms of Euclidean distance, i.e. $$ d(recall, precision) = \sqrt{({recall_1 - recall_2})^{2} + {precision_1 - precision_2})^{2}}. $$

# In[ ]:


precision_, recall_, proba = precision_recall_curve(y_test, predicted_proba[:, -1])

disp = plot_precision_recall_curve(pipeline_tf, X_test, y_test)
disp.ax_.set_title('Precision-Recall curve')


# ## Obtain Optimal Probability Thresholds with PR Curve <a class="anchor" id="10"></a>
# 
# In this notebook, we will be obtaining the optimal probability threshold based on minimizing the distance between precision and recall scores.

# In[ ]:


optimal_proba_cutoff = sorted(list(zip(np.abs(precision_ - recall_), proba)), key=lambda i: i[0], reverse=False)[0][1]
roc_predictions = [1 if i >= optimal_proba_cutoff else 0 for i in predicted_proba[:, -1]]


# ## Evaluate Model (After Thresholding) <a class="anchor" id="11"></a>

# In[ ]:


print("Accuracy Score Before and After Thresholding: {}, {}".format(accuracy_score(y_test, predictions), accuracy_score(y_test, roc_predictions)))
print("Precision Score Before and After Thresholding: {}, {}".format(precision_score(y_test, predictions), precision_score(y_test, roc_predictions)))
print("Recall Score Before and After Thresholding: {}, {}".format(recall_score(y_test, predictions), recall_score(y_test, roc_predictions)))
print("F1 Score Before and After Thresholding: {}, {}".format(f1_score(y_test, predictions), f1_score(y_test, roc_predictions)))


# ## Confusion Matrix of Model (After Thresholding) <a class="anchor" id="12"></a>

# In[ ]:


y_actual = pd.Series(y_test, name='Actual')
y_predict_tf = pd.Series(roc_predictions, name='Predicted')
df_confusion = pd.crosstab(y_actual, y_predict_tf, rownames=['Actual'], colnames=['Predicted'], margins=True)
print (df_confusion)


# ## Conclusion <a class="anchor" id="13"></a>
# 
# Obtaining optimal probability thresholds using the PR curves is one way of maximizing the predictive capability of your machine learning model. There are a few ways of obtaining these thresholds and they do not necessarily have the same effects on performance. Of course, you can pick probability thresholds manually with the aim of maximizing either precision or recall, which depends on the problem you are trying to solve.
# 
# Example problems which exemplify the need to maximize precision or recall are as follows:
# 
# - **Minimize number of false positives, i.e. maximize precision**: You have a model which identifies spam and non-spam emails. This model should focus on reducing the number of falsely identified spam emails as this would increase the possibility of users missing out on important emails.
# 
# - **Minimize number of false negatives, i.e. maximize recall**: You have a model which identifies cancer and non-cancer cases. This model should focus on reducing the number of false identified non-cancer cases since this would prevent concerned parties from seeking early treatment for cancer.
# 
# Another important point to note is that when obtaining optimal probability thresholds, precision-recall (PR) curves are normally preferred as compared to receiver operating characteristic (ROC) curves when dealing with datasets with **severe class imbalance**. PR curves focus more on the minority class whereas ROC curves attempts to place equal emphasis on both classes.

# ## References <a class="anchor" id="14"></a>
# 
# - https://github.com/nicholaslaw/roc-optimal-cutoff
# - https://www.kaggle.com/nicholasgah/obtain-optimal-probability-threshold-using-roc
# - https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-imbalanced-classification/
# - https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py
