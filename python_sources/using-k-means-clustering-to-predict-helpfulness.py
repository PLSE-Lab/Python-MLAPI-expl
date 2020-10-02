#!/usr/bin/env python
# coding: utf-8

# ## Predicting the Helpfulness of Amazon Fine Food Reviews

# ### Purpose
# Build a model to predict the helpfulness of Amazon Fine Food Reviews. This will improve Amazon's selection of helpful reviews at the top of the review section and improve customer's purchasing decisions. It could also help other reviewers as a guide to writing helpful reviews.
# 
# This dataset comes from over 568,0454 Amazon Fine Food Reviews. 

# Variable: Description | Type of Variable
# 
# HelpfulnessNumerator: number of users who found the review helpful | continuous
# 
# HelpfulnessDenominator: number of users who indicated whether they found the review helpful or not helpful | continuous
# 
# Score: rating between 1 and 5 | categorical
# 
# Text: text of the review | text

# ## Load the Data

# In[ ]:


#imports
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# read data into a DataFrame
data = pd.read_csv("../input/Reviews.csv")

#make a copy of columns I need from raw data
df1 = data.iloc[:, [4,5,6,9]]
df1.head()


# In[ ]:


#change data type of non-Text features from string to integer
df1.iloc[:, 1:3] = df1.iloc[:, 1:3].apply(pd.to_numeric)


# In[ ]:


#include reviews that have more than 10 helpfulness data point only
df1 = df1[(df1.HelpfulnessDenominator > 10)]


# In[ ]:


df1['Score'].shape


# ## Notes
# I have only included reviews that have more than 10 votes from users on whether the review was helpful or not. With this filter, the dataset is significantly reduce from 560,000+ reviews to 21,463 reviews.

# # Clean the Data

# In[ ]:


#check for missing values
df1.isnull().sum()


# In[ ]:


# convert text to lowercase
df1.loc[:, 'Text'] = df1['Text'].str.lower()
df1["Text"].head(10)


# In[ ]:


#remove html tags
#import bleach
#df1["Text"] = df1['Text'].apply(lambda x: bleach.clean(x, tags=[], strip=True))
#df1["Text"].head(4)


# In[ ]:


#remove punctuation
import unicodedata
import sys

tbl = dict.fromkeys(i for i in range(sys.maxunicode)
                      if unicodedata.category(chr(i)).startswith('P'))
def remove_punctuation(text):
    return text.translate(tbl)

df1['Text']=df1['Text'].apply( lambda x: remove_punctuation(x))
df1["Text"].head(4)


# In[ ]:


df1['Score'].shape


# #### Notes
# I chose not to use the Porter Stemmer method after reviewing other kernels on Kaggle where the method generated less accurate predictions.

# ## Exploratory Data Analysis

# In[ ]:


#transform Helpfulness into a binary variable with 0.50 ratio
df1.loc[:, 'Helpful'] = np.where(df1.loc[:, 'HelpfulnessNumerator'] / df1.loc[:, 'HelpfulnessDenominator'] > 0.50, 1, 0)
df1.head(3)


# In[ ]:


df1.groupby('Helpful').count()


# In[ ]:


df1.corr()


# ### (Bag of Words model)

# In[ ]:


#make a copy
df2 = df1.copy(deep = True)


# In[ ]:


#tokenize text with Tfidf
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(min_df = 0.1, max_df=0.9,
                             ngram_range=(1, 4), 
                             stop_words='english')
vectorizer.fit(df2['Text'])


# In[ ]:


X_train = vectorizer.transform(df2['Text'])
vocab = vectorizer.get_feature_names()


# In[ ]:


#find best logistic regression parameters
from sklearn import grid_search, cross_validation
from sklearn.linear_model import LogisticRegression
feature_set = X_train
gs = grid_search.GridSearchCV(
    estimator=LogisticRegression(),
    param_grid={'C': [10**-i for i in range(-5, 5)], 'class_weight': [None, 'balanced']},
    cv=cross_validation.StratifiedKFold(df1.Helpful,n_folds=10),
    scoring='roc_auc'
)


gs.fit(X_train, df2.Helpful)
gs.grid_scores_


# In[ ]:


#plot ROC/AUC curve
from sklearn.metrics import roc_auc_score, roc_curve
actuals = gs.predict(feature_set) 
probas = gs.predict_proba(feature_set)
plt.plot(roc_curve(df2[['Helpful']], probas[:,1])[0], roc_curve(df2[['Helpful']], probas[:,1])[1])


# In[ ]:


# ROC/AUC score
y_score = probas
test2 = np.array(list(df2.Helpful))
test2 = test2.reshape(21463,1)
y_true = test2
roc_auc_score(y_true, y_score[:,1].T)


# #### Notes
# The Bag of Words model performs poorly with only 72% accuracy.

# ## Improving Prediction with K-Means Clustering of Reviews
# Hypothesis: There's a natural clustering to review vocabulary. I can use the most descriptive clusters to simplify the model.

# In[ ]:


#Apply TfidfVectorizer to review text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn import metrics


# In[ ]:


model = KMeans(n_clusters=4, init='k-means++', max_iter=100, n_init=1,random_state=5)

vectorizer = TfidfVectorizer(min_df = 0.05, max_df=0.95,
                             ngram_range=(1, 2), 
                             stop_words='english')
vectorizer.fit(df1['Text'])


# ### Select Top 10 words per cluster

# In[ ]:


X_train = vectorizer.transform(df1['Text'])
vocab = vectorizer.get_feature_names()
sse_err = []
res = model.fit(X_train)
vocab = np.array(vocab)
cluster_centers = np.array(res.cluster_centers_)
sorted_vals = [res.cluster_centers_[i].argsort() for i in range(0,np.shape(res.cluster_centers_)[0])]
words=set()
for i in range(len(res.cluster_centers_)):
    words = words.union(set(vocab[sorted_vals[i][-10:]]))
words=list(words)


# In[ ]:


#top 10 words for each cluster
words


# In[ ]:


#add top words to train set
train_set=X_train[:,[np.argwhere(vocab==i)[0][0] for i in words]]


# In[ ]:


# how many observations are in each cluster
df1['cluster'] = model.labels_
df1.groupby('cluster').count()


# In[ ]:


# what does each cluster look like
df1.groupby('cluster').mean()


# In[ ]:


# correlation matrix
df1.corr()


# #### Notes
# There doesn't seem to be a clear trend to the clusters. I cannot make a silhoute coefficient plot due to computer storage capacity, so I chose 4 clusters. With more clusters, the number of overlapping "top words" from each cluster seems to increase. In total there are only 30 "top words" instead of 40, because some top words overlapped among clusters. There may be some common words that I should consider removing in further analysis, like "food"" or "coffee".

# ## Logistic Regression to Predict Review Helpfulness with Top Cluster Words 

# In[ ]:


print(train_set.shape)


# In[ ]:


#add Score column to top words
import scipy as scipy

score = np.array(list(df1.Score))
score = score.reshape(21463, 1)

features = scipy.sparse.hstack((train_set,scipy.sparse.csr_matrix(score)))

features = scipy.sparse.csr_matrix(features)


# In[ ]:


features.shape


# In[ ]:


#find best logistic regression parameters
from sklearn import grid_search, cross_validation
from sklearn.linear_model import LogisticRegression
feature_set = features
gs = grid_search.GridSearchCV(
    estimator=LogisticRegression(),
    param_grid={'C': [10**-i for i in range(-5, 5)], 'class_weight': [None, 'balanced']},
    cv=cross_validation.StratifiedKFold(df1.Helpful,n_folds=10),
    scoring='roc_auc'
)


gs.fit(features, df1.Helpful)
gs.grid_scores_


# In[ ]:


print(gs.best_estimator_)


# In[ ]:


y_pred = gs.predict(feature_set)


# In[ ]:


# Coefficients represent the log-odds
print(gs.best_estimator_.coef_)
print(gs.best_estimator_.intercept_)


# In[ ]:


#roc curve
from sklearn.metrics import roc_auc_score, roc_curve
actuals = gs.predict(feature_set) 
probas = gs.predict_proba(feature_set)
plt.plot(roc_curve(df1[['Helpful']], probas[:,1])[0], roc_curve(df1[['Helpful']], probas[:,1])[1])


# In[ ]:


#roc auc score
y_score = probas
test2 = np.array(list(df1.Helpful))
test2 = test2.reshape(21463,1)
y_true = test2

roc_auc_score(y_true, y_score[:,1].T)


# In[ ]:


#plot a confusion matrix
from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True Helpfulness')
    plt.xlabel('Predicted Helpfulness')


# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
plt.figure()
plot_confusion_matrix(cm)

# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

plt.show()


# In[ ]:


#view top parameters
words.extend(['score'])
sorted(zip(words,gs.best_estimator_.coef_[0]),key=lambda x:x[1])


# #### Notes
# There seem to be common words that I should remove from the text in further analysis, like food, product, or Amazon.
# 
# My model is 82% accurate, which is 10% increase in accuracy over the Bag of Words model.

# ## Recommendations
# Price, Flavor, and Great are the top indicators of a helpful review. This indicates a possible bias among customers to mark a review as helpful when the review is positive. Eating, Like, Don't, Order, Good, and Eat are all negatively correlated with a helpful review, which is difficult to interpret. These may be more common words to remove.
# 
# Moving forward, I would explore the following methods to improve this analysis:
# 
# 1) I would explore alternative definitions of an "unhelpful" review. For example, reviews that are not market as "helpful" could be classified as unhelpful. This may help counter consumer-bias if consumers are less likely to mark a negative review as helpful, because it did not enable them to buy the product. This problem requires more domain expertise on consumer behavior.
# 
# 2) I would make Score into dummy variables to further explore potential biases related to Score. For example, consumers may find that reviews with a score of 1 or 5 are more helpful than scores of 2, 3, and 4.
# 
# 3) I would explore curating a domain-specific dictionary for this project to avoid common food words and Amazon words in reviews.
# 
# 4) I would explore using these findings as a guide for reviewers. For example, when writing a review, Amazon could show "Tips for writing a helpful review": "Describe the flavor of this product" ("Flavor" is the most highly correlated parameter with "helpfulness"), "Describe the value of this product compared to its price", etc.
