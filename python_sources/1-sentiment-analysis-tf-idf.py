#!/usr/bin/env python
# coding: utf-8

# # 1. Bag of Words Meets Bags of Popcorn : TF-IDF

# ## Table of Contents

# 1. [Introduction](#intro)<br>
# 2. [Reading the Data](#reading)<br>
# 3. [Text Preprocessing](#preprocessing)<br>
# 4. [TF-IDF](#tf-idf)<br>
# 5. [Visualization](#viz)<br>- [5.1. WordCloud](#wc)<br> - [5.2. Distribution](#dis)<br>
# 6. [Modeling](#modeling)<br>- [6.1. Support Vector Machine](#svm)<br>- [6.2. Bernoulli Naive Bayes Classifier](#bnb)<br>- [6.3. Perceptron](#perceptron)<br>- [6.4. Logistic Regression](#logi)<br>
# 7. [Investigating Model Coefficients](#imc)<br>
# 8. [Submission](#submission)<br><br>
# 
# First notebook: [Bag of Words Meets Bags of Popcorn: CountVectorizer](https://www.kaggle.com/kyen89/0-sentiment-analysis-countvectorizer/)<br>
# Third notebook: [Bag of Words Meets Bags of Popcorn: Word2Vec](https://www.kaggle.com/kyen89/2-sentiment-analysis-word2vec/)

# 

# ## 1. Introduction <a id='intro'></a>
# 
# 

# This is the second notebook for IMDb sentiment analysis (First notebook: [Bag of Words Meets Bags of Popcorn: CountVectorizer](https://www.kaggle.com/kyen89/0-sentiment-analysis-countvectorizer/)). In this notebook, instead of CountVectorizer, we will be analyzing the movie reviews by using TF-IDF (Term Frequency - Inverse Document Frequency). Also, we could compare how differently these methods work and the performance of the predictions. Above all, the most important takeaway from this notebook is to learn how to use TF-IDF and the usage of important TF-IDF parameters.

# In[ ]:





# ## 2. Reading the Data <a id='reading'></a>

# In[ ]:


# Import libraries

import pandas as pd
import numpy as np


# In[ ]:


# Read the data 

X_train = pd.read_csv("../input/labeledTrainData.tsv",quoting = 3, delimiter = "\t", header= 0)
X_test = pd.read_csv("../input/testData.tsv", quoting = 3, delimiter = "\t", header = 0)


# In[ ]:


# Read only the first 600 sentences of the first review.

X_train['review'][0][:600]


# In[ ]:


print('Training set dimension:',X_train.shape)
print('Test set dimension:',X_test.shape)


# In[ ]:


X_train.head()


# In[ ]:





# ## 3. Text Preprocessing <a id='preprocessing'></a>

# In[ ]:


from bs4 import BeautifulSoup
import re
import nltk


# In[ ]:


def prep(review):
    
    # Remove HTML tags.
    review = BeautifulSoup(review,'html.parser').get_text()
    
    # Remove non-letters
    review = re.sub("[^a-zA-Z]", " ", review)
    
    # Lower case
    review = review.lower()
    
    # Tokenize to each word.
    token = nltk.word_tokenize(review)
    
    # Stemming
    review = [nltk.stem.SnowballStemmer('english').stem(w) for w in token]
    
    # Join the words back into one string separated by space, and return the result.
    return " ".join(review)
    


# In[ ]:


# test whether the function successfully preprocessed.
X_train['review'].iloc[:2].apply(prep).iloc[0]


# In[ ]:





# In[ ]:


# If there is no problem at the previous cell, let's apply to all the rows.
X_train['clean'] = X_train['review'].apply(prep)
X_test['clean'] = X_test['review'].apply(prep)


# In[ ]:


X_train['clean'].iloc[3]


# In[ ]:


print('Training dim:',X_train.shape, 'Test dim:', X_test.shape)


# In[ ]:





# ## 4. TF-IDF <a id='tf-idf'></a>
# 
# TF-IDF (Term Frequency - Inverse Document Frequency) can be represented tf(d,t) X idf(t). TF-IDF uses the method diminishing the weight (importance) of words appeared in many documents in common, considered them incapable of discerning the documents, rather than simply counting the frequency of words as CountVectorizer does. The outcome matrix consists of each document (row) and each word (column) and the importance (weight) computed by tf * idf (values of the matrix).
# 

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import words


# In[ ]:


# analyzer is the parameter that the vectorizer reads the input data in word unit or character unit to create a matrix
# vocabulary is the parameter that the vectorizer creates the matrix by using only input data or some other source 
# Other parameters are self-explanatory and already mentioned in other notebooks.

tv = TfidfVectorizer(
                    ngram_range = (1,3),
                    sublinear_tf = True,
                    max_features = 40000)


# In[ ]:


# Handle with care especially when you transform the test dataset. (Wrong: fit_transform(X_test))

train_tv = tv.fit_transform(X_train['clean'])
test_tv = tv.transform(X_test['clean'])


# In[ ]:


# Create the list of vocabulary used for the vectorizer.

vocab = tv.get_feature_names()
print(vocab[:5])


# In[ ]:


print("Vocabulary length:", len(vocab))


# In[ ]:


dist = np.sum(train_tv, axis=0)
checking = pd.DataFrame(dist,columns = vocab)


# In[ ]:


checking


# As you can see the above, due to the vocabulary option 'set(words.words())', a lot of vocabularies are added to the matrix even more than review's vocabularies.

# In[ ]:


print('Training dim:',train_tv.shape, 'Test dim:', test_tv.shape)


# The number of the feature of the matrix are almost ten times larger than the number of reviews. This can cause the curse of dimensionality but this project is for studying and trying many features of text mining tools so I decide to leave the option. Instead, the regularization term must be tuned with care when optimizing the parameters. 

# In[ ]:





# ## 5. Visualization <a id='viz'></a>

# ### 5.1 WordCloud <a id='wc'></a>
# 
# As alluded in the first notebook, the drawback for WordCloud is that the graphics only reflect the frequency of words, which can cause some uninformative words frequently appeared in the text can be highlighted on the cloud instead of informative words which is less frequently appeared in the text. These kind of uninformative words could be stopwords or just some words frequently appeared in documents that particularly longer than other documents. Although the WordCloud is not the best visualization method to show all the aspect of the data, it is worth plotting them so that we can quickly and intuitively see what the text is about.

# In[ ]:


from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


def cloud(data,backgroundcolor = 'white', width = 800, height = 600):
    wordcloud = WordCloud(stopwords = STOPWORDS, background_color = backgroundcolor,
                         width = width, height = height).generate(data)
    plt.figure(figsize = (15, 10))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
    


# In[ ]:


cloud(' '.join(X_train['clean']))


# In[ ]:


cloud(' '.join(X_test['clean']))


# As expected, most of emphasized words are just normal words like "film", "one", "movie", "show", and "stori" which appear to be not informative to distinguish one document from the others or distinguish between negative and positive movie reviews.

# In[ ]:





# ### 5.2 Distribution <a id='dis'></a>

# In[ ]:


# We need to split each words in cleaned review and then count the number of each rows of data frame.

X_train['freq_word'] = X_train['clean'].apply(lambda x: len(str(x).split()))
X_train['unique_freq_word'] = X_train['clean'].apply(lambda x: len(set(str(x).split())))
                                                 
X_test['freq_word'] = X_test['clean'].apply(lambda x: len(str(x).split()))
X_test['unique_freq_word'] = X_test['clean'].apply(lambda x: len(set(str(x).split())))                                                 


# In[ ]:


fig, axes = plt.subplots(ncols=2)
fig.set_size_inches(10,5)

sns.distplot(X_train['freq_word'], bins = 90, ax=axes[0], fit = stats.norm)
(mu0, sigma0) = stats.norm.fit(X_train['freq_word'])
axes[0].legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu0, sigma0)],loc='best')
axes[0].set_title("Distribution Word Frequency")
axes[0].axvline(X_train['freq_word'].median(), linestyle='dashed')
print("median of word frequency: ", X_train['freq_word'].median())


sns.distplot(X_train['unique_freq_word'], bins = 90, ax=axes[1], color = 'r', fit = stats.norm)
(mu1, sigma1) = stats.norm.fit(X_train['unique_freq_word'])
axes[1].set_title("Distribution Unique Word Frequency")
axes[1].legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu1, sigma1)],loc='best')
axes[1].axvline(X_train['unique_freq_word'].median(), linestyle='dashed')
print("median of uniuqe word frequency: ", X_train['unique_freq_word'].median())


# The black contour of the distribution graphs represent the normal distribution if the data would have been distributed as normal. Compared to the black contour, the actual distribution is pretty skwed; therefore, median would be better to use as a measure of representative of data since mean is very sensitive to outliers and noise especially the distribution is highly skewed. As shown in the legend, the mean of the word frequency is 236.89 and the mean of the unique word is 135.61. It means 236.89 words and 135.61 unique words are used for each review. Also the dashed lines represent the median of the distribution. Another thing to notice is that the median values are very closely located to the normal distribution's mean points. Compared to CountVectorizer methods, there are 117.39 words used more for train set and 41.57 words used more for test set. This is due to the different parameter setting and we used more words for max features for TF-IDF. The distribution of the graphs are somehow similar to that of CountVectorizer.

# In[ ]:





# ## 6. Modeling <a id='modeling'></a>
# 
# As text data usually is very sparse and has a high dimensionality, using linear, and simple models such as Linear Support Vector Machine, Bernoulli Naive Bayes, Logistic Regression or MultiLayer Perceptron would be better choice rather than using Random Forest. 

# In[ ]:


from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import GridSearchCV, StratifiedKFold, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier


# In[ ]:


kfold = StratifiedKFold( n_splits = 5, random_state = 2018 )


# In[ ]:





# ### 6.1 Support Vector Machine <a id='svm'></a>

# In[ ]:


# LinearSVC

sv = LinearSVC(random_state=2018)

param_grid2 = {
    'loss':['squared_hinge'],
    'class_weight':[{1:4}],
    'C': [0.2]
}


gs_sv = GridSearchCV(sv, param_grid = [param_grid2], verbose = 1, cv = kfold, n_jobs = 1, scoring = 'roc_auc')
gs_sv.fit(train_tv, X_train['sentiment'])
gs_sv_best = gs_sv.best_estimator_
print(gs_sv.best_params_)

# {'C': 0.1, 'class_weight': {1: 3}, 'loss': 'squared_hinge'} - 0.87220
# {'C': 0.1, 'class_weight': {1: 4}, 'loss': 'squared_hinge'} - 0.86060
# {'C': 0.2, 'class_weight': {1: 4}, 'loss': 'squared_hinge'} - 0.87952


# In[ ]:


submission1 = gs_sv.predict(test_tv)


# In[ ]:


print(gs_sv.best_score_)


# In[ ]:





# ### 6.2 Bernoulli Naive Bayes Classifier <a id='bnb'></a>

# In[ ]:


bnb = BernoulliNB()
gs_bnb = GridSearchCV(bnb, param_grid = {'alpha': [0.001],
                                         'binarize': [0.001]}, verbose = 1, cv = kfold, n_jobs = 1, scoring = "roc_auc")
gs_bnb.fit(train_tv, X_train['sentiment'])
gs_bnb_best = gs_bnb.best_estimator_
print(gs_bnb.best_params_)

# {'alpha': 0.001, 'binarize': 0.001} - 0.86960


# In[ ]:


submission2 = gs_bnb.predict(test_tv)


# In[ ]:


print(gs_bnb.best_score_)


# In[ ]:





# ### 6.3 Perceptron <a id='perceptron'></a>

# In[ ]:


MLP = MLPClassifier(random_state = 2018)

mlp_param_grid = {
    'hidden_layer_sizes':[(5)],
    'activation':['relu'],
    'solver':['adam'],
    'alpha':[0.3],
    'learning_rate':['constant'],
    'max_iter':[1000]
}


gsMLP = GridSearchCV(MLP, param_grid = mlp_param_grid, cv = kfold, scoring = 'roc_auc', n_jobs= 1, verbose = 1)
gsMLP.fit(train_tv,X_train['sentiment'])
print(gsMLP.best_params_)
mlp_best0 = gsMLP.best_estimator_

# {'activation': 'relu', 'alpha': 0.1, 'hidden_layer_sizes': (1,), 'learning_rate': 'constant', 'max_iter': 1000, 'solver': 'adam'} - 0.89996
# {'activation': 'relu', 'alpha': 0.1, 'hidden_layer_sizes': (5,), 'learning_rate': 'constant', 'max_iter': 1000, 'solver': 'adam'} - 0.89896
# {'activation': 'relu', 'alpha': 0.2, 'hidden_layer_sizes': (1,), 'learning_rate': 'constant', 'max_iter': 1000, 'solver': 'adam'} - 0.90284
# {'activation': 'relu', 'alpha': 0.3, 'hidden_layer_sizes': (5,), 'learning_rate': 'constant', 'max_iter': 1000, 'solver': 'adam'} - 0.90356


# In[ ]:


submission3 = gsMLP.predict(test_tv)


# In[ ]:


print(gsMLP.best_score_)


# In[ ]:





# ### 6.4 Logistic Regression <a id='logi'></a>

# In[ ]:


lr = LogisticRegression(random_state = 2018)

lr2_param = {
    'penalty':['l2'],
    'dual':[True],
    'C':[6],
    'class_weight':[{1:1}]
    }

lr_CV = GridSearchCV(lr, param_grid = [lr2_param], cv = kfold, scoring = 'roc_auc', n_jobs = 1, verbose = 1)
lr_CV.fit(train_tv, X_train['sentiment'])
print(lr_CV.best_params_)
logi_best = lr_CV.best_estimator_

# {'C': 6, 'class_weight': {1: 1}, 'dual': True, 'penalty': 'l2'} - 90.360


# In[ ]:


submission6 = lr_CV.predict(test_tv)


# In[ ]:


print(lr_CV.best_score_)


# Among many models, the best Kaggle score is 90.36% performed by Logistic Regression. Compared to the best model (linear SVM) based on CountVectorizer, it was improved by approximately 2 percents. Not only Logistic Regression but also all the models show the improvement except Linear SVM. This improvement is due to the fact that TF-IDF is more complicated method than simple word counting method (CountVectorizer) and also the max features we set for TF-IDF are way more than that of CountVectorizer (40000 vs 18000).

# In[ ]:





# ## 7. Investigating Model Coefficients <a id='imc'></a>
# 
# Since there are 40000 features, it is impossible to look at all of the coefficients at the same time. Therefore, we can sort them and look at the largest coefficients. The following bar chart shows the 30 largest and 30 smallest coefficients of the Logistic Regression model, with the bars showing the size of each coefficients.

# In[ ]:


# Extract the coefficients from the best model Logistic Regression and sort them by index.
coefficients = logi_best.coef_
index = coefficients.argsort()


# In[ ]:


# Extract the feature names.
feature_names = np.array(tv.get_feature_names())


# In[ ]:


# From the smallest to largest.
feature_names[index][0][:30]


# In[ ]:


# From the smallest to largest.
feature_names[index][0][-31::1]


# In[ ]:


# feature names: Smallest 30 + largest 30.
feature_names_comb = list(feature_names[index][0][:30]) + list(feature_names[index][0][-31::1])


# In[ ]:


# coefficients magnitude: Smallest 30 + largest 30.
index_comb = list(coefficients[0][index[0][:30]]) + list(coefficients[0][index[0][-31::1]])


# In[ ]:


# Make sure the x-axis be the number from 0 to the length of the features selected not the feature names.
# Once the bar is plotted, the features are placed as ticks.
plt.figure(figsize=(25,10))
barlist = plt.bar(list(i for i in range(61)), index_comb)
plt.xticks(list(i for i in range(61)),feature_names_comb,rotation=75,size=15)
plt.ylabel('Coefficient magnitude',size=20)
plt.xlabel('Features',size=20)

# color the first smallest 30 bars red
for i in range(30):
    barlist[i].set_color('red')

plt.show()


# As mentioned in previous notebook about CountVectorizer, the blue bar indicates positive movie reviews. On the other hand the red bar indicates negative move reviews. Interestingly, there are many words in common on both barplot based on CountVectorizer and TF-IDF such as: worst, bad, aw, wast, disappoint, excel, perfect, great, high recommend, etc.

# In[ ]:





# ## 8. Submission <a id='submission'></a>

# In[ ]:


output = pd.DataFrame( data = {'id': X_test['id'], 'sentiment': submission6 })
output.to_csv('submission26.csv', index = False, quoting = 3)


# In[ ]:




