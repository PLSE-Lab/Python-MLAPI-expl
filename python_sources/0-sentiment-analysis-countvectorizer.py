#!/usr/bin/env python
# coding: utf-8

# # 0. Bag of Words Meets Bags of Popcorn : CountVectorizer

# ## Table of Contents

# 1. [Introduction](#intro)<br><br>
# 2. [Reading the Data](#reading)<br><br>
# 3. [Data Cleaning and Text Preprocessing](#preprocess)<br> - [3.1. Removing HTML Markup by using BeautifulSoup Package](#beauti)<br> - [3.2. Removing Non-Letter Characters & Converting Reviews to Lower Case](#non-char)<br> - [3.3. Tokenization](#token)<br> -  [3.4. Removing Stop words](#stop)<br> - [3.5. Stemming / Lemmatization](#stlm)<br> - [3.6. Putting It All Together](#together)<br><br>
# 
# 4. [Visualization](#visu)<br>- [4.1. WordCloud](#wc) <br>- [4.2. Distribution](#dist)<br><br>
# 5. [Bag of Words](#bag)<br><br>
# 6. [Modeling](#modeling)<br>- [6.1. Support Vector Machine](#svm)<br>- [6.2. Bernoulli Naive Bayes Classifier](#bnb)<br>- [6.3. Perceptron ](#perceptron)<br>- [6.4. Logistic Regression](#logi)<br><br>
# 7. [Investigating Model Coefficients](#imc)<br><br>
# 8. [Submission](#submission)<br><br>
# 
# Second notebook: [Bag of Words Meets Bags of Popcorn: TF-IDF](https://www.kaggle.com/kyen89/1-sentiment-analysis-tf-idf/)<br>
# Third notebook: [Bag of Words Meets Bags of Popcorn: Word2Vec](https://www.kaggle.com/kyen89/2-sentiment-analysis-word2vec/)

# In[ ]:





# ## 1. Introduction <a id='intro'></a>

# The goal of our project is to classifiy correctly whether 25,000 movie reviews from IMDB are positive or negative. This is the first part of sentiment analysis which will be used a Bag of Words for creating features. Once we obtain the result of the prediction, we will compare it with the seoncd part of our sentiment analysis and then submit the one performs better to Kaggle. In this project, there are many important concepts that text analysis beginner should know. Acclimatization to jargons (tokenization, stopwords, etc) for machine learning for text analysis is one of them. Also, it is very essential to understand that how you tune parameters influence the result of predictions while preprocessing data. The most important thing to take away from this notebook is to understand how a Bag of Words works. 

# In[ ]:





# ## 2. Reading the Data <a id='reading'></a>

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


# Read the labeled training and test data
# Header = 0 indicates that the first line of the file contains column names, 
# delimiter = \t indicates that the fields are seperated by tabs, and 
# quoting = 3 tells python to ignore doubled quotes

train = pd.read_csv("../input/labeledTrainData.tsv", header = 0, delimiter = "\t", quoting = 3)
test = pd.read_csv("../input/testData.tsv", header = 0, delimiter = "\t", quoting = 3)


# In[ ]:


# Display check the dimensions and the first 2 rows of the file.

print('train dim:', train.shape, 'test dim:', test.shape)
train.iloc[0:2]


# In[ ]:


# Let's check the first review.

train.iloc[0]["review"][:len(train.iloc[0]["review"])//2]


# As you can see the above review, the html tags are disturbing and also in order to make the data machine-learning friendly, we need to clean the data.

# In[ ]:





# ## 3. Data Cleaning and Text Preprocessing <a id='preprocess'></a>

# ### 3.1. Removing HTML Markup by using BeautifulSoup Package <a id='beauti'></a>

# In[ ]:


from bs4 import BeautifulSoup


# In[ ]:


example1 = BeautifulSoup(train["review"][0], "html.parser")

# Without the second argument "html.parser", it will pop out the warning message.


# In[ ]:


print(example1.get_text())


# You can clearly see the effect of removing HTML markup. 

# In[ ]:





# ### 3.2. Removing Non-Letter Characters & Converting Reviews to Lower Case <a id='non-char'></a>

# It may be important to include some punctuations and numbers such as :-). However for this project, for simplicity, we remove both of them.

# In[ ]:


import re

letters = re.sub("[^a-zA-Z]", " ", example1.get_text())
letters = letters.lower()


# The meaning of the above regular expression is that except for (^) the letters from a to z and from A to Z ([a-zA-Z]) substitute all the characters to spaces. lower() means conversion any capital letters to lower case.

# In[ ]:


print(letters)


# In[ ]:





# ### 3.3. Tokenization <a id='token'></a>
# 
# Tokenization is the process splitting a sentence or paragraph into the most basic units.

# In[ ]:


# Import Natural Language Toolkit
import nltk


# In[ ]:


# Instead of using just split() method, used word_tokenize in nltk library.
word = nltk.word_tokenize(letters)


# In[ ]:


word


# In[ ]:





# ### 3.4. Removing Stop words <a id='stop'></a>

# "Stop words" is the frequently occurring words that do not carry much meaning such as "a", "and" , "is", "the". In order to use the data as input for machine learning algorithms, we need to get rid of them. Fortunately, there is a function called stopwords which is already built in NLTK library.

# In[ ]:


from nltk.corpus import stopwords


# In[ ]:





# Below is the list of stopwords.

# In[ ]:


print(stopwords.words("english"))


# In[ ]:


# Exclude the stop words from the original tokens.

word = [w for w in word if not w in set(stopwords.words("english"))]


# In[ ]:


word


# In[ ]:





# ### 3.5. Stemming / Lemmatization <a id='stlm'></a>
# 
# It is important to know the difference between these two.
# 
# - __Stemming:__ Stemming algorithms work by cutting off the end of the word, and in some cases also the beginning while looking for the root. This indiscriminate cutting can be successful in some occasions, but not always, that is why we affirm that this an approach that offers some limitations. ex) studying -> study, studied -> studi <br>
# <br>
# - __Lemmatization:__ Lemmatization is the process of converting the words of a sentence to its dictionary form. For example, given the words amusement, amusing, and amused, the lemma for each and all would be amuse. ex) studying -> study, studied -> study. Lemmatization also discerns the meaning of the word by understanding the context of a passage. For example, if a "meet" is used as a noun then it will print out a "meeting"; however, if it is used as a verb then it will print out "meet".  
# <br>
# 
# Usually, either one of them is chosen for text-analysis not both. As a side note, Lancaster is the most aggressive stemmer among three major stemming algorithms (Porter, Snowball, Lancaster) and Porter is the least aggressive. The "aggressive algorithms" means how much a working set of words are reduced. The more aggressive the algorithms, the faster it is; however, in some certain circumstances, it will hugely trim down your working set. Therefore, in this project I decide to use snowball since it is slightly faster than Porter and does not trim down too much information as Lancaster does.

# In[ ]:


snow = nltk.stem.SnowballStemmer('english')
stems = [snow.stem(w) for w in word]


# In[ ]:


stems


# As you can see the word "started", it is converted to "start" and "listening" and "watching" are converted to "listen" and "watch".

# In[ ]:





# ### 3.6. Putting It All Together <a id='together'></a>

# So far, we have cleaned only one datapoint. Now it's time to apply all the cleaning process to all the data.<br>
# To make the code reusable, we need to create a function that can be called many times.

# In[ ]:


def cleaning(raw_review):
    import nltk
    
    # 1. Remove HTML.
    html_text = BeautifulSoup(raw_review,"html.parser").get_text()
    
    # 2. Remove non-letters.
    letters = re.sub("[^a-zA-Z]", " ", html_text)
    
    # 3. Convert to lower case.
    letters = letters.lower()
    
    # 4. Tokenize.
    tokens = nltk.word_tokenize(letters)
    
    # 5. Convert the stopwords list to "set" data type.
    stops = set(nltk.corpus.stopwords.words("english"))
    
    # 6. Remove stop words. 
    words = [w for w in tokens if not w in stops]
    
    # 7. Stemming
    words = [nltk.stem.SnowballStemmer('english').stem(w) for w in words]
    
    # 8. Join the words back into one string separated by space, and return the result.
    return " ".join(words)

    


# In[ ]:


# Add the processed data to the original data. Perhaps using apply function would be more elegant and concise than using for loop
train['clean'] = train['review'].apply(cleaning)
test['clean'] = test['review'].apply(cleaning)


# In[ ]:


train.head()


# In[ ]:





# ## 4. Visualization <a id='visu'></a>
# 

# ### 4.1 WordCloud <a id='wc'></a>
# 
# As a tool for visualization by using the frequency of words appeared in text, we use WordCloud. Note that it can give more information and insight of texts by analyzing correlations and similarities between words rather than analyzing texts only by the frequency of words appeared; however, it can give you some general shape of what this text is about quickly and intuitively. 

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


cloud(' '.join(train['clean']))


# In[ ]:


cloud(' '.join(test['clean']))


# It is not surprising that the most of large words are just the words frequently appeared in the text.

# In[ ]:





# ### 4.2 Distribution <a id='dist'></a>

# In[ ]:


# We need to split each words in cleaned review and then count the number of each rows of data frame.

train['freq_word'] = train['clean'].apply(lambda x: len(str(x).split()))
train['unique_freq_word'] = train['clean'].apply(lambda x: len(set(str(x).split())))
                                                 
test['freq_word'] = test['clean'].apply(lambda x: len(str(x).split()))
test['unique_freq_word'] = test['clean'].apply(lambda x: len(set(str(x).split())))                                                 


# In[ ]:


fig, axes = plt.subplots(ncols=2)
fig.set_size_inches(10,5)

sns.distplot(train['freq_word'], bins = 90, ax=axes[0], fit = stats.norm)
(mu0, sigma0) = stats.norm.fit(train['freq_word'])
axes[0].legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu0, sigma0)],loc='best')
axes[0].set_title("Distribution Word Frequency")
axes[0].axvline(train['freq_word'].median(), linestyle='dashed')
print("median of word frequency: ", train['freq_word'].median())


sns.distplot(train['unique_freq_word'], bins = 90, ax=axes[1], color = 'r', fit = stats.norm)
(mu1, sigma1) = stats.norm.fit(train['unique_freq_word'])
axes[1].set_title("Distribution Unique Word Frequency")
axes[1].legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu1, sigma1)],loc='best')
axes[1].axvline(train['unique_freq_word'].median(), linestyle='dashed')
print("median of uniuqe word frequency: ", train['unique_freq_word'].median())


# The black contour of the distribution graphs represent the normal distribution if the data would have been distributed as normal. Compared to the black contour, the actual distribution is pretty skwed; therefore, median would be better to use as a measure of representative of data since mean is very sensitive to outliers and noise especially the distribution is highly skewed. As shown in the legend, the mean of the word frequency is 119.50 and the mean of the unique word is 94.04. It means 119.50 words and 94.04 unique words are used for each review. Also the dashed lines represent the median of the distribution. Another thing to notice is that the median values are very closely located to the normal distribution's mean points.

# In[ ]:





# ## 5. Bag of Words <a id='bag'></a>
# 
# Even though we cleaned the data with many steps, we still have one more step to create machine learning-friendly input. One common approach is called a Bag of Words. It is simply the matrix that counts how many each word appears in documents (disregard grammar and word order). In order to do that, we use "CountVectorizer" method in sklearn library. As you know already, the number of vocabulary is very large so it is important to limit the size of the feature vectors. In this project, we use the 18000 most frequent words. Also, the other things to notice is that we set min_df = 2 and ngram_range = (1,3). min_df = 2 means in order to include the vocabulary in the matrix, one word must appear in at least two documents. ngram_range means we cut one sentence by number of ngram. Let's say we have one sentence, I am a boy. If we cut the sentence by digram (ngram=2) then the sentence would be cut like this ["I am","am a", "a boy"]. The result of accuracy can be highly dependent on parameters so feel free to alter them and see if you can improve the score.

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


vectorizer = CountVectorizer(analyzer = "word", 
                             tokenizer = None, 
                             preprocessor = None, 
                             stop_words = None, 
                             max_features = 18000,
                             min_df = 2,
                             ngram_range = (1,3)
                            )


# In[ ]:





# As mentioned many times, the matrix is going to be huge so it would be a good idea to use Pipeline for encapsulating and avoiding a data leakage.

# In[ ]:


from sklearn.pipeline import Pipeline


# In[ ]:


pipe = Pipeline( [('vect', vectorizer)] )


# In[ ]:


# Complete form of bag of word for machine learning input. We will be using this for machine learning algorithms.

train_bw = pipe.fit_transform(train['clean'])

# We only call transform not fit_transform due to the risk of overfitting.

test_bw = pipe.transform(test['clean'])


# In[ ]:


print('train dim:', train_bw.shape, 'test dim:', test_bw.shape)


# In[ ]:


# Get the name fo the features

lexi = vectorizer.get_feature_names()


# In[ ]:


lexi[:5]


# In[ ]:


# Instead of 1 and 0 representation, create the dataframe to see how many times each word appears (just sum of 1 of each row)

train_sum = pd.DataFrame(np.sum(train_bw, axis=0), columns = lexi)


# In[ ]:


train_sum.head()


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
    'loss':['hinge'],
    'class_weight':[{1:1}],
    'C': [0.01]
}

gs_sv = GridSearchCV(sv, param_grid = [param_grid2], verbose = 1, cv = kfold, n_jobs = -1, scoring = 'roc_auc' )
gs_sv.fit(train_bw, train['sentiment'])
gs_sv_best = gs_sv.best_estimator_
print(gs_sv.best_params_)

# {'C': 0.01, 'class_weight': {1: 1}, 'loss': 'hinge'} - 0.88104


# In[ ]:


submission1 = gs_sv.predict(test_bw)


# In[ ]:


print(gs_sv.best_score_)


# In[ ]:





# ### 6.2 Bernoulli Naive Bayes Classifier <a id='bnb'></a>

# In[ ]:


bnb = BernoulliNB()
gs_bnb = GridSearchCV(bnb, param_grid = {'alpha': [0.03],
                                         'binarize': [0.001]}, verbose = 1, cv = kfold, n_jobs = -1, scoring = 'roc_auc')
gs_bnb.fit(train_bw, train['sentiment'])
gs_bnb_best = gs_bnb.best_estimator_
print(gs_bnb.best_params_)

# {'alpha': 0.1, 'binarize': 0.001} - 0.85240
# {'alpha': 0.03, 'binarize': 0.001} - 0.85240


# In[ ]:


submission2 = gs_bnb.predict(test_bw)


# In[ ]:


print(gs_bnb.best_score_)


# In[ ]:





# ### 6.3 Perceptron <a id='perceptron'></a>

# In[ ]:


MLP = MLPClassifier(random_state = 2018)

mlp_param_grid = {
    'hidden_layer_sizes':[(1,)],
    'activation':['logistic'],
    'solver':['sgd'],
    'alpha':[0.1],
    'learning_rate':['constant'],
    'max_iter':[1000]
}

gsMLP = GridSearchCV(MLP, param_grid = mlp_param_grid, cv = kfold, scoring = 'roc_auc', n_jobs= -1, verbose = 1)
gsMLP.fit(train_bw,train['sentiment'])
print(gsMLP.best_params_)
mlp_best0 = gsMLP.best_estimator_

# {'activation': 'logistic', 'alpha': 0.1, 'hidden_layer_sizes': (1,), 'learning_rate': 'constant', 'max_iter': 1000, 'solver': 'sgd'} - 0.87732
# {'activation': 'logistic', 'alpha': 0.1, 'hidden_layer_sizes': (5,), 'learning_rate': 'constant', 'max_iter': 1000, 'solver': 'sgd'} - 0.87632


# In[ ]:


submission3 = gsMLP.predict(test_bw)


# In[ ]:


print(gsMLP.best_score_)


# In[ ]:





# ### 6.4 Logistic Regression <a id='logi'></a>

# In[ ]:


lr = LogisticRegression(random_state = 2018)


lr2_param = {
    'penalty':['l2'],
    'dual':[False],
    'C':[0.05],
    'class_weight':['balanced']
    }

lr_CV = GridSearchCV(lr, param_grid = [lr2_param], cv = kfold, scoring = 'roc_auc', n_jobs = -1, verbose = 1)
lr_CV.fit(train_bw, train['sentiment'])
print(lr_CV.best_params_)
logi_best = lr_CV.best_estimator_


# {'C': 0.1, 'class_weight': 'balanced', 'dual': False, 'penalty': 'l2'} - 0.87868
# {'C': 0.05, 'class_weight': 'balanced', 'dual': False, 'penalty': 'l2'} - 0.88028


# In[ ]:


submission4 = lr_CV.predict(test_bw)


# In[ ]:


print(lr_CV.best_score_)


# Among many models, Linear Support Vector Machine,whose Kaggle score was 88.1%, performed the best. Therefore, I decide to use SVM for investigating mdoel coefficients and see which features are important and check the result we obtained makes sense.

# In[ ]:





# ## 7. Investigating Model Coefficients <a id='imc'></a>
# 
# Since there are 18000 features, it is impossible to look at all of the coefficients at the same time. Therefore, we can sort them and look at the largest coefficients. The following bar chart shows the 30 largest and 30 smallest coefficients of the linear SVM model, with the bars showing the size of each coefficients.

# In[ ]:


# Extract the coefficients from the best model Linear SVM and sort them by index.
coefficients = gs_sv_best.coef_
index = coefficients.argsort()


# In[ ]:


# Extract the feature names.
feature_names = np.array(pipe.named_steps['vect'].get_feature_names())


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
    barlist[i].set_color('r')

plt.show()


# As you can see the result, the terms are quite intuitive, like "worst", "bore", "poor", "dull", "disappoint" indicating bad movie reviews, while "perfect","must see","high recommend" indicate positive movie reviews. Some words are slightly less clear such as "wast"(waste), "aw"(awful), "ridicul"(ridiculous), "excel"(excellent) and "amaz"(amazing) since common suffix are dropped because of stemming.

# In[ ]:





# ## 8. Submission <a id='submission'></a>

# In[ ]:


output = pd.DataFrame( data = {'id': test['id'], 'sentiment': submission1 })
output.to_csv('submission12.csv', index = False, quoting = 3)


# In[ ]:




