#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis of Amazon fine food reviews

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score


# ### Read the data

# In[ ]:


review = pd.read_csv('../input/Reviews.csv')
review.head()


# In[ ]:


print('The number of entries in the data frame: ', review.shape[0])


# In[ ]:


review['ProductId'].nunique()


# In[ ]:


review['UserId'].nunique()


# ### check for null values

# In[ ]:


review.isnull().sum()


# In[ ]:


# drop the rows with null values
review.dropna(inplace=True)


# In[ ]:


# recheck if null values are dropped
review.isnull().sum()


# # Neutral reviews
# 
# We drop the rows where score = 3 because neutral reviews don't provide value to the prediction

# In[ ]:


review = review[review['Score'] != 3]


# # Target variable
# 
# Next we create a column called positivity where any score above 3 is encoded as 1 otherwise 0

# In[ ]:


review['Positivity'] = np.where(review['Score'] > 3, 1, 0)
review.head()


# In[ ]:


sns.countplot(review['Positivity'])
plt.show()


# # Memory usage

# In[ ]:


review.info(memory_usage='deep')


# # Low memory
# 
# For other applications we could have applied various techniques to reduce memory usage,
# here we are going to just drop columns which we don't require

# In[ ]:


review = review.drop(['ProductId','UserId','ProfileName','Id','HelpfulnessNumerator','HelpfulnessDenominator','Score','Time','Summary'], axis=1)


# In[ ]:


# checking the memory usage again
review.info(memory_usage='deep')


# In[ ]:


# split the data into training and testing data

# text will be used for training
# positivity is what we are predicting
X_train, X_test, y_train, y_test = train_test_split(review['Text'], review['Positivity'], random_state = 0)


# In[ ]:


print('X_train first entry: \n\n', X_train[0])
print('\n\nX_train shape: ', X_train.shape)


# ### Tokenization
# 
# In order to perform machine learning on text documents, we first need to turn these text content 
# into numerical feature vectors that Scikit-Learn can use.
# 
# ### Bag of words 
# The simplest way to do so is to use [bags-of-words](https://machinelearningmastery.com/gentle-introduction-bag-words-model/). First we convert the text document into a matrix of tokens. The default configuration tokenizes the string, 
# by extracting words of at least 2 letters or numbers, separated by word boundaries, converts everything to lowercase 
# and builds a vocabulary using these tokens

# In[ ]:


vect = CountVectorizer().fit(X_train)
vect


# In[ ]:


# checking the features
feat = vect.get_feature_names()


# In[ ]:


cloud = WordCloud(width=1440, height=1080).generate(" ".join(feat))


# In[ ]:


# larger the size of the word, more the times it appears
plt.figure(figsize=(20, 15))
plt.imshow(cloud)
plt.axis('off')
plt.show()


# In[ ]:


# checking the length of features
len(vect.get_feature_names())


# # Sparse matrix
# We now transform the document into a bag-of-words representation i.e matrix form.
# The result is stored in a sparse matrix i.e it has very few non zero elements.
# 
# Rows represent the words in the document while columns represent the words in our training vocabulary.

# In[ ]:


X_train_vectorized = vect.transform(X_train)

# the interpretation of the columns can be retreived as follows
# X_train_vectorized.toarray()


# In[ ]:


model = LogisticRegression()
model.fit(X_train_vectorized, y_train)


# In[ ]:


# accuracy
predictions = model.predict(vect.transform(X_test))


# In[ ]:


accuracy_score(y_test, predictions)


# In[ ]:


# area under the curve
roc_auc = roc_auc_score(y_test, predictions)
print('AUC: ', roc_auc)
fpr, tpr, thresholds = roc_curve(y_test, predictions)


# In[ ]:


plt.title('ROC for logistic regression on bag of words', fontsize=20)
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate', fontsize = 20)
plt.xlabel('False Positive Rate', fontsize = 20)
plt.show()


# In[ ]:


# coefficient determines the weight of a word (positivity or negativity)
# checking the top 10 positive and negative words

# getting the feature names
feature_names = np.array(vect.get_feature_names())

# argsort: Integer indicies that would sort the index if used as an indexer
sorted_coef_index = model.coef_[0].argsort()

print('Smallest Coefs: \n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}\n'.format(feature_names[sorted_coef_index[:-11:-1]]))


# # TF IDF (term-frequency-inverse-document-frequency).
# 
# This means that we weigh the terms by how uncommon they are, meaning that we care more about rare words than common ones.
# 
# ### Why use TF IDF over bag of words?
# 
# In large texts, some words may be repeated often but will carry very little meaningful information about the actual
# contents of the document. If we were to feed the count data directly to a classifier those very frequent terms would 
# shadow the frequencies of rarer yet more interesting terms.
# 
# #### Tf-idf allows us to weight terms based on how important they are to a document.

# In[ ]:


# ignore terms that appear in less than 5 documents
vect = TfidfVectorizer(min_df = 5).fit(X_train)
len(vect.get_feature_names())


# In[ ]:


# check the top 10 features for positive and negative
# reviews again, the AUC has improved
feature_names = np.array(vect.get_feature_names())
sorted_coef_index = model.coef_[0].argsort()

# print('Smallest Coef: \n{}\n'.format(feature_names[sorted_coef_index][:10]))
# print('Largest Coef: \n{}\n'.format(feature_names[sorted_coef_index][:-11:-1]))


# In[ ]:


feat = vect.get_feature_names()


# In[ ]:


cloud = WordCloud(width=1440, height=1080).generate(" ".join(feat))


# In[ ]:


# larger the size of the word, more the times it appears
plt.figure(figsize=(20, 15))
plt.imshow(cloud)
plt.axis('off')
plt.show()


# In[ ]:


X_train_vectorized = vect.transform(X_train)


# In[ ]:


model = LogisticRegression()
model.fit(X_train_vectorized, y_train)


# In[ ]:


predictions = model.predict(vect.transform(X_test))


# In[ ]:


accuracy_score(y_test, predictions)


# In[ ]:


roc_auc = roc_auc_score(y_test, predictions)
print('AUC: ', roc_auc)
fpr, tpr, thresholds = roc_curve(y_test, predictions)


# In[ ]:


plt.title('ROC for logistic regression on TF-IDF', fontsize=20)
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate', fontsize = 20)
plt.xlabel('False Positive Rate', fontsize = 20)
plt.show()


# In[ ]:


# even though we reduced the number of features considerably
# the AUC did not change much

# let us test our model
new_review = ['The food was delicious', 'The food was not good']

print(model.predict(vect.transform(new_review)))


# # Bigrams
# 
# Since our classifier misclassifies things like 'not good', we will use groups of words instead of single words.
# This method is called n grams (bigrams for 2 words and so on). Here we take 1 and 2 words into consideration.

# In[ ]:


vect = CountVectorizer(min_df = 5, ngram_range = (1,2)).fit(X_train)
X_train_vectorized = vect.transform(X_train)
len(vect.get_feature_names())


# In[ ]:


feat = vect.get_feature_names()


# In[ ]:


cloud = WordCloud(width=1440, height=1080).generate(" ".join(feat))


# In[ ]:


# larger the size of the word, more the times it appears
plt.figure(figsize=(20, 15))
plt.imshow(cloud)
plt.axis('off')
plt.show()


# In[ ]:


# the number of features has increased again
# checking for the AUC
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)


# In[ ]:


predictions = model.predict(vect.transform(X_test))


# In[ ]:


accuracy_score(y_test, predictions)


# In[ ]:


roc_auc = roc_auc_score(y_test, predictions)
print('AUC: ', roc_auc)
fpr, tpr, thresholds = roc_curve(y_test, predictions)


# In[ ]:


plt.title('ROC for logistic regression on Bigrams', fontsize=20)
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate', fontsize = 20)
plt.xlabel('False Positive Rate', fontsize = 20)
plt.show()


# In[ ]:


# check the top 10 features for positive and negative
# reviews again, the AUC has improved
feature_names = np.array(vect.get_feature_names())
sorted_coef_index = model.coef_[0].argsort()

# print('Smallest Coef: \n{}\n'.format(feature_names[sorted_coef_index][:10]))
# print('Largest Coef: \n{}\n'.format(feature_names[sorted_coef_index][:-11:-1]))


# In[ ]:


new_review = ['The food is not good, I would never buy them again']
print(model.predict(vect.transform(new_review)))


# In[ ]:


new_review = ['One would be disappointed by the food']
print(model.predict(vect.transform(new_review)))


# In[ ]:


new_review = ['One would not be disappointed by the food']
print(model.predict(vect.transform(new_review)))


# In[ ]:


new_review = ['I would feel sorry for anyone eating here']
print(model.predict(vect.transform(new_review)))


# In[ ]:


new_review = ['They are bad at serving quality food']
print(model.predict(vect.transform(new_review)))


# In[ ]:


# there are still more misclassifications
# lets try with 3 grams
# vect = CountVectorizer(min_df = 5, ngram_range = (1,3)).fit(X_train)
# X_train_vectorized = vect.transform(X_train)
# len(vect.get_feature_names())


# In[ ]:




