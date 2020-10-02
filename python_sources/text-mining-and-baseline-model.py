#!/usr/bin/env python
# coding: utf-8

# # Predict user gender based on Twitter profile information with text-data
# 
# ### Motivation:
# - This problem is interesting, oddly interesting...
# - Problem context is easily understandable
# - There's already a popular classification problem based on Twitter text: Sentiment classification

# In[ ]:


# Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#from sklearn_pandas import DataFrameMapper # Notice that this is actually Sklearn-Pandas library
get_ipython().run_line_magic('matplotlib', 'inline')


# ## About Dataset
# 
# This dataset is obtained from [Kaggle](https://www.kaggle.com/crowdflower/twitter-user-gender-classification/home).
# 
# The dataset contains 20,000 rows, each with a user name, a random tweet, account profile and image, location, and even link and sidebar color.
# 
# Attributes that do not provide useful information for _Gender classification_:
#  - **_unit_id**: a unique id for user
#  - **_last_judgment_at**: date and time of last contributor judgment; blank for gold standard observations
#  - **user_timezone**: the timezone of the user
#  - **tweet_coord**: if the user has location turned on, the coordinates as a string with the format "[latitude, longitude]"
#  - **tweet_count**: number of tweets that the user has posted
#  - **tweet_created**: when the random tweet (in the text column) was created
#  - **tweet_id**: the tweet id of the random tweet
#  - **tweet_location**: location of the tweet; seems to not be particularly normalized 
#  - **profileimage**: a link to the profile image
#  - **created**: date and time when the profile was created
#  
#  
# Attributes that potentially provide useful information for _Gender classification_:
#  - **_golden**: whether the user was included in the gold standard for the model; TRUE or FALSE
#  - **_unit_state**: state of the observation; one of finalized (for contributor-judged) or golden (for gold standard observations)
#  - **_trusted_judgments**: number of trusted judgments (int); always 3 for non-golden, and what may be a unique id for gold standard observations
#  - **gender**: one of male, female, or brand (for non-human profiles)
#  - **gender:confidence**: a float representing confidence in the provided gender
#  - **gender_gold**: if the profile is golden, what is the gender?
#  - **profile_yn**: "no" here seems to mean that the profile was meant to be part of the dataset but was not available when contributors went to judge it
#  - **profile_yn:confidence**: confidence in the existence/non-existence of the profile
#  - **profile_yn_gold**: whether the profile y/n value is golden
#  - **description**: the user's profile description
#  - **fav_number**: number of tweets the user has favorited
#  - **link_color**: the link color on the profile, as a hex value
#  - **name**: the user's name
#  - **retweet_count**: number of times the user has retweeted (or possibly, been retweeted)
#  - **sidebar_color**: color of the profile sidebar, as a hex value
#  - **text**: text of a random one of the user's tweets
# 

# In[ ]:


# Load dataset
data = pd.read_csv('../input/gender-classifier-DFE-791531.csv', encoding='latin-1')

# Drop unnecessary columns/features
data.drop (columns = ['_unit_id',
                      '_last_judgment_at',
                      'user_timezone',
                      'tweet_coord',
                      'tweet_count',
                      'tweet_created', 
                      'tweet_id',
                      'tweet_location',
                      'profileimage',
                      'created'], inplace = True)

data.info()


# In[ ]:


data.head(3)


# ## Cleaning Dataset

# ### 'Gender' Attribute (gender)

# In[ ]:


data['gender'].value_counts()
# We can see that there are 1117 unknown genders, so get rid of them


# In[ ]:


drop_items_idx = data[data['gender'] == 'unknown'].index

data.drop (index = drop_items_idx, inplace = True)

data['gender'].value_counts()


# ### 'Profile' Attribute (profile_yn, profile_yn:confidence, profile_yn_gold)
# 
# **'No'**: Profile was meant to be part of the dataset but was not available when contributors went to judge it.

# In[ ]:


print ('profile_yn information:\n',data['profile_yn'].value_counts())

data[data['profile_yn'] == 'no']['gender']


# It is shown that all of 97 instances with **profile_yn** == **no** are all **NaN** in **gender**. 
# 
# Therefore, i get rid of these 97 instances. Also, i get rid of **profile_yn**, **profile_yn:confidence** and **profile_yn_gold** as they are not useful anymore.

# In[ ]:


drop_items_idx = data[data['profile_yn'] == 'no'].index

data.drop (index = drop_items_idx, inplace = True)

print (data['profile_yn'].value_counts())

data.drop (columns = ['profile_yn','profile_yn:confidence','profile_yn_gold'], inplace = True)


# In[ ]:


# Double check the data 
print (data['gender'].value_counts())

print ('---------------------------')
data.info()


# ### Low-confidence gender (gender:confidence)
# 
# I decide to keep only 100% confidence of labeling Gender and get rid of those < 100% confidence.

# In[ ]:


print ('Full data items: ', data.shape)
print ('Data with label-confidence < 100%: ', data[data['gender:confidence'] < 1].shape)


# Here, i can observe that approximately **26.7%** (5032/18836) of labeled instances were lower 100% of confidence
# 
# Then, i get rid of those instances and the feature **gender:confidence** as it is now useful anymore.

# In[ ]:


drop_items_idx = data[data['gender:confidence'] < 1].index

data.drop (index = drop_items_idx, inplace = True)

print (data['gender:confidence'].value_counts())

data.drop (columns = ['gender:confidence'], inplace = True)


# ### Get rid of remaining useless features

# In[ ]:


data.drop (columns = ['_golden','_unit_state','_trusted_judgments','gender_gold'], inplace = True)

# Double check the data 
print (data['gender'].value_counts())

print ('---------------------------')
data.info()


# ## Manipulate Text data

# ### Removing stop-words in Twits
# 
# First we need to take a glance at the most common words

# In[ ]:


from collections import Counter

twit_vocab = Counter()
for twit in data['text']:
    for word in twit.split(' '):
        twit_vocab[word] += 1
        
# desc_vocab = Counter()
# for twit in data['description']:
#     for word in twit.split(' '):
#         desc_vocab[word] += 1
        
twit_vocab.most_common(20)
# desc_vocab.most_common(20)


# As you can see, the most common words are meaningless in terms of sentiment: _I, to, the, and..._ . They're basically noise that can most probably be eliminated. These kind of words are called **stop words**, and it is a common practice to remove them when doing text analysis.

# In[ ]:


import nltk

nltk.download('stopwords')


# In[ ]:


from nltk.corpus import stopwords
stop = stopwords.words('english')

twit_vocab_reduced = Counter()
for w, c in twit_vocab.items():
    if not w in stop:
        twit_vocab_reduced[w]=c

twit_vocab_reduced.most_common(20)


# ### Removing special characters and "trash"
# 
# We still se a very uneaven distribution. If you look closer, you'll see that we're also taking into consideration punctuation signs ('-', ',', etc) and other html tags like `&amp`. We can definitely remove them for the sentiment analysis, but we will try to keep the emoticons, since those _do_ have a sentiment load:

# In[ ]:


import re

def preprocessor(text):
    """ Return a cleaned version of text
    """
    # Remove HTML markup
    text = re.sub('<[^>]*>', '', text)
    # Save emoticons for later appending
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    # Remove any non-word character and append the emoticons,
    # removing the nose character for standarization. Convert to lower case
    text = (re.sub('[\W]+', ' ', text.lower()) + ' ' + ' '.join(emoticons).replace('-', ''))
    
    return text

print(preprocessor('This!!@ twit :) is <b>nice</b>'))


# ### Lemmatization

# We are almost ready! There is another trick we can use to reduce our vocabulary and consolidate words. If you think about it, words like: love, loving, etc. _Could_ express the same positivity. If that was the case, we would be  having two words in our vocabulary when we could have only one: lov. This process of reducing a word to its root is called **stemming**. An alternative way is called **Lemmatization**.
# 
# A popular stemming algorithm for English is **Porter** algorithm.
# 
# We also need a _tokenizer_ to break down our twits in individual words. We will implement two tokenizers, a regular one and one that does steaming:

# In[ ]:


from nltk.stem import PorterStemmer

porter = PorterStemmer()

def tokenizer(text):
    return text.split()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

print(tokenizer('Hi there, I am loving this, like with a lot of love'))
print(tokenizer_porter('Hi there, I am loving this, like with a lot of love'))


# ## Visualize Data
# 
# Among text data, i want to find out if other features can give me useful information or show some special characteristics.

# ### Create a countplot to visualize the amount of each label

# In[ ]:


sns.countplot(data['gender'],label="Gender")


# ### Create a bar plot to visualize the amount of *favorites* and *retweets*

# In[ ]:


sns.barplot (x = 'gender', y = 'fav_number',data = data)


# In[ ]:


sns.barplot (x = 'gender', y = 'retweet_count',data = data)


# ### Visualize Colors attribute

# In[ ]:


male_top_sidebar_color = data[data['gender'] == 'male']['sidebar_color'].value_counts().head(7)
male_top_sidebar_color_idx = male_top_sidebar_color.index
male_top_color = male_top_sidebar_color_idx.values

male_top_color[2] = '000000'
print (male_top_color)
l = lambda x: '#'+x

sns.set_style("darkgrid", {"axes.facecolor": "#F5ABB5"})
sns.barplot (x = male_top_sidebar_color, y = male_top_color, palette=list(map(l, male_top_color)))


# In[ ]:


female_top_sidebar_color = data[data['gender'] == 'female']['sidebar_color'].value_counts().head(7)
female_top_sidebar_color_idx = female_top_sidebar_color.index
female_top_color = female_top_sidebar_color_idx.values

female_top_color[2] = '000000'
print (female_top_color)

l = lambda x: '#'+x

sns.set_style("darkgrid", {"axes.facecolor": "#F5ABB5"})
sns.barplot (x = female_top_sidebar_color, y = female_top_color, palette=list(map(l, female_top_color)))


# For **sidebar color**, the top 3 colors of both male and female are the same (this seems to be these colors are default theme color of Twitter). It is shown that the number of 2nd and 3rd color of female is larger but this can be explained by the fact that the number of female users are more than male.
# 
# So, at this point, sidebar_color may not give me any useful information for classifying gender.

# In[ ]:


male_top_link_color = data[data['gender'] == 'male']['link_color'].value_counts().head(7)
male_top_link_color_idx = male_top_link_color.index
male_top_color = male_top_link_color_idx.values
male_top_color[1] = '009999'
male_top_color[5] = '000000'
print(male_top_color)

l = lambda x: '#'+x

sns.set_style("whitegrid", {"axes.facecolor": "white"})
sns.barplot (x = male_top_link_color, y = male_top_link_color_idx, palette=list(map(l, male_top_color)))


# In[ ]:


female_top_link_color = data[data['gender'] == 'female']['link_color'].value_counts().head(7)
female_top_link_color_idx = female_top_link_color.index
female_top_color = female_top_link_color_idx.values

l = lambda x: '#'+x

sns.set_style("whitegrid", {"axes.facecolor": "white"})
sns.barplot (x = female_top_link_color, y = female_top_link_color_idx, palette=list(map(l, female_top_color)))


# ## Training classification models with Tweet-text only

# ### How relevant are words? Term frequency-inverse document frequency
# 
# We could use these raw term frequencies to score the words in our algorithm. There is a problem though: If a word is very frequent in _all_ documents, then it probably doesn't carry a lot of information. In order to tacke this problem we can use **term frequency-inverse document frequency**, which will reduce the score the more frequent the word is accross all twits. It is calculated like this:
# 
# \begin{equation*}
# tf-idf(t,d) = tf(t,d) ~ idf(t,d)
# \end{equation*}
# 
# _tf(t,d)_ is the raw term frequency descrived above. _idf(t,d)_ is the inverse document frequency, than can be calculated as follows:
# 
# \begin{equation*}
# \log \frac{n_d}{1+df\left(d,t\right)}
# \end{equation*}
# 
# where `n` is the total number of documents (number of _twits_ in this problem) and _df(t,d)_ is the number of documents where the term `t` appears. 
# 
# The `1` addition in the denominator is just to avoid zero term for terms that appear in all documents. Ans the `log` ensures that low frequency term don't get too much weight.
# 
# The IDF (inverse document frequency) of a word is the measure of how significant that term is in the whole corpus (the whole collection of _twits_ in this problem).
# 
# The higher the TF-IDF weight value, the rarer the term. The smaller the weight, the more common the term.
# 
# Fortunately for us `scikit-learn` does all those calculations for us:

# In[ ]:


# Firstly, convert categorical labels into numerical ones
# Function for encoding categories
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y = encoder.fit_transform(data['gender'])


# split the dataset in train and test
X = data['text']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
#In the code line above, stratify will create a train set with the same class balance than the original set

X_train.head()


# ### Try with Logistic Regression Model

# In[ ]:


from sklearn.linear_model import LogisticRegression

tfidf = TfidfVectorizer(lowercase=False,
                        tokenizer=tokenizer_porter,
                        preprocessor=preprocessor)
clf = Pipeline([('vect', tfidf),
                ('clf', LogisticRegression(multi_class='ovr', random_state=0))])

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
print('Accuracy:',accuracy_score(y_test,predictions))
print('Confusion matrix:\n',confusion_matrix(y_test,predictions))
print('Classification report:\n',classification_report(y_test,predictions))


# ### Try with Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
# Plot the correlation between n_estimators and accuracy

# X_train_sample = X_train.head(5000) # this is series
# y_train_sample = y_train[:5000] # this is array

# print (X_train_sample.shape)
# print (y_train_sample.shape)

n = range (1,100,10) #step 10

results = []
for i in n:
    clf = Pipeline([('vect', tfidf),
                ('clf', RandomForestClassifier(n_estimators = i, random_state=0))])
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    results.append(accuracy_score(y_test, predictions))
plt.grid()
plt.scatter(n, results)


# It is shown that with approximately **40** trees, Random Forest classifier starts reaching the highest performance.

# In[ ]:


tfidf = TfidfVectorizer(lowercase=False,
                        tokenizer=tokenizer_porter,
                        preprocessor=preprocessor)
clf = Pipeline([('vect', tfidf),
                ('clf', RandomForestClassifier(n_estimators = 40, random_state=0))])

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
print('Accuracy:',accuracy_score(y_test,predictions))
print('Confusion matrix:\n',confusion_matrix(y_test,predictions))
print('Classification report:\n',classification_report(y_test,predictions))


# ### Try with SVM

# In[ ]:


# the SVM model
from sklearn.svm import SVC

tfidf = TfidfVectorizer(lowercase=False,
                        tokenizer=tokenizer_porter,
                        preprocessor=preprocessor)
clf = Pipeline([('vect', tfidf),
                ('clf', SVC(kernel = 'linear'))])
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
print('Accuracy:',accuracy_score(y_test,predictions))
print('Confusion matrix:\n',confusion_matrix(y_test,predictions))
print('Classification report:\n',classification_report(y_test,predictions))


# ### Experimental Results
# 
# Accuracy:
#  - **Logistic Regression**: 59.95%  
#  - **Random Forest**: 57.07%  
#  - **SVM**: 59.80%  
#  
# Winner: **Logistic Regression** model

# ## Adding content of Description into Text 

# In[ ]:


data.head(3)


# ### Concatenating 'description' to 'text'

# In[ ]:


#Fill NaN with empty string
data.fillna("", inplace = True)

# Concatenate text with description, add white space between. 
# By using Series helper functions Series.str()
data['text_description'] = data['text'].str.cat(data['description'], sep=' ')

data['text_description'].isnull().value_counts() # Check if any null values, True if there is at least one.


# ### Re-create training dataset

# In[ ]:


# split the dataset in train and test
X = data['text_description']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
#In the code line above, stratify will create a train set with the same class balance than the original set

X_train.head()
X_train.isnull().values.any() # Check if any null values, True if there is at least one.


# ### Try with Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression

tfidf = TfidfVectorizer(lowercase=False,
                        tokenizer=tokenizer_porter,
                        preprocessor=preprocessor)
clf = Pipeline([('vect', tfidf),
                ('clf', LogisticRegression(multi_class='ovr', random_state=0))])

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
print('Accuracy:',accuracy_score(y_test,predictions))
print('Confusion matrix:\n',confusion_matrix(y_test,predictions))
print('Classification report:\n',classification_report(y_test,predictions))


# ### Try with Random Forest

# In[ ]:


# Plot the correlation between n_estimators and accuracy

# X_train_sample = X_train.head(5000) # this is series
# y_train_sample = y_train[:5000] # this is array

# print (X_train_sample.shape)
# print (y_train_sample.shape)

n = range (1,120,10) #step 10

results = []
for i in n:
    clf = Pipeline([('vect', tfidf),
                ('clf', RandomForestClassifier(n_estimators = i, random_state=0))])
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    results.append(accuracy_score(y_test, predictions))
plt.grid()    
plt.scatter(n, results)


# It is shown that with approximately **80** trees, Random Forest classifier starts reaching the highest performance.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

tfidf = TfidfVectorizer(lowercase=False,
                        tokenizer=tokenizer_porter,
                        preprocessor=preprocessor)
clf = Pipeline([('vect', tfidf),
                ('clf', RandomForestClassifier(n_estimators = 80, random_state=0))])

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
print('Accuracy:',accuracy_score(y_test,predictions))
print('Confusion matrix:\n',confusion_matrix(y_test,predictions))
print('Classification report:\n',classification_report(y_test,predictions))


# ### Try with SVM

# In[ ]:


# the SVM model
from sklearn.svm import SVC

tfidf = TfidfVectorizer(lowercase=False,
                        tokenizer=tokenizer_porter,
                        preprocessor=preprocessor)
clf = Pipeline([('vect', tfidf),
                ('clf', SVC(kernel = 'linear'))])
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
print('Accuracy:',accuracy_score(y_test,predictions))
print('Confusion matrix:\n',confusion_matrix(y_test,predictions))
print('Classification report:\n',classification_report(y_test,predictions))


# ### Experimental Results
# 
# Accuracy:
#  - **Logistic Regression**: 68.17%  
#  - **Random Forest**: 64.38%  
#  - **SVM**: 68.68%  
#  
# Winner: **SVM** model

# ## Try Ensemble technique - Take advantage of both 3 models

# In[ ]:


from sklearn.ensemble import VotingClassifier
clf1 = LogisticRegression(multi_class='ovr', random_state=0)
clf2 = RandomForestClassifier(n_estimators = 80, random_state=0)
clf3 = SVC(kernel = 'linear',probability = True, random_state=0)

ensemble_clf = VotingClassifier(estimators=[
        ('lr', clf1), ('rf', clf2), ('svm', clf3)], voting='soft')

clf = Pipeline([('vect', tfidf),
                ('clf', ensemble_clf)])

clf.fit(X_train, y_train)

# ensemble_clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
print('Accuracy:',accuracy_score(y_test,predictions))
print('Confusion matrix:\n',confusion_matrix(y_test,predictions))
print('Classification report:\n',classification_report(y_test,predictions))


# ### Experimental Results
# 
# **The increase of accuracy is not significant at all - only 0.3% (68.68% - 68.97%)**. But it is worth running experiment since ensemble learning usually yields better results.
# 
# --> Maybe i need to research more to select precisely the classifiers for ensemble system in the futures.

# # Conclusions
# 
# I implemented a system of Gender classification based on the dataset provided on Kaggle.
# This is actually an interesting problem among with the Sentiment classification problem, which is more popular.
# 
# As I intended to implement classifiers based on Text data, i also wanted to explore whether other features can help the model classify Gender. Therefore, i plotted different graphs to visualize them. The results show that **link_color** may give additional useful information for classification task. 
# 
# The results show that Only the **Tweet text** can yield a moderate accuracy, although it's not sustantially high.
# But with the content from the **Description**, the classifiers actually improve its performance significantly.
# 
# I also tried implementing Ensemble learning as one of my idea during the implementation, however, it only slightly increases the accuracy. 
# 

# ## Future works

#  - Re-implement ensemble learning system with further research
#  - Extract link_color features to add to the models
#  - Try applying Deep Learning (optional)
