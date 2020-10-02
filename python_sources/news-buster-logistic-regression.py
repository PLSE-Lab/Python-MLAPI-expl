#!/usr/bin/env python
# coding: utf-8

# <img src="https://media-assets-02.thedrum.com/cache/images/thedrum-prod/s3-news-tmp-140656-fake_news--default--1280.jpg" />

# <b>What is fake news ?</b><br>
# Fake news, also known as junk news, pseudo-news, alternative facts or hoax news, is a form of news consisting of deliberate disinformation or hoaxes spread via traditional news media (print and broadcast) or online social media. Digital news has brought back and increased the usage of fake news, or yellow journalism. The news is then often reverberated as misinformation in social media but occasionally finds its way to the mainstream media as well. [source](https://en.wikipedia.org/wiki/Fake_news)
# 
# * It creates panic
# * Damages reputation of public and private entities
# * Misleads people, to benefit fradusters
# * Motivated by personal vendatta, some people backs such generators with support
# 
# <br>
# 
# <b>PROBLEM </b>: How to distinguish between a real news and a fake news?
# 
# <br>
# 
# <b>SOLUTION </b>: A human can read the news and cross-check with the original source or internet to find some supporting ancillary news. Depending on the results, the news can be credited as real or bogus.
# 
# OR
# 
# We can show an algorithm huge number of fake and real news articles so that it learns to differenciate between them automatically, and then it will give a probability score or percentage of confidence as an output for a given news article, that it is real or fake.

# <b>DEFINING ML PROBLEM</b>
# * We should be able to correctly classify all the fake news as fake, hence recall is important.
# * We also want to penalize incorrecly classified data points, so we will use log-loss as loss function.
# * We do not have strict latency concerns.

# <hr>

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# <b>Importing Libraries</b>

# In[ ]:


import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report


# <b>Loading Datasets</b>

# In[ ]:


fake = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/Fake.csv')
fake.shape


# In[ ]:


fake.head(3)


# In[ ]:


true = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/True.csv')
true.shape


# In[ ]:


true.head(3)


# <b>Assigning target labels and merging into one dataframe</b>

# In[ ]:


true['label'] = 0
fake['label'] = 1


# In[ ]:


df = true.append(fake.reset_index(drop=True))
print(df.shape)
df = df.reset_index(drop=True)
df


# In[ ]:


df.info()


# From above table it implies that we do not have any null fields, but after some observation it was observed that in text features we have 445 samples that contain just a single space. To apply classical machine learning algorithms we will have to get rid of these samples but for deep learning we can choose to delete them because we will later on merge title feature with text to tokenization and padding will take care of shorter text.
# 
# For now we remove these empty samples.

# In[ ]:


df[df.text == " "]


# In[ ]:


df.drop(df[df.text == " "].index.tolist(), inplace=True)


# <b>Checking for duplicates</b>

# In[ ]:


df.drop('label', axis=1)[df.duplicated(keep=False)]


# We have 405 samples that match with each other in terms of title, text, date and subject, so we remove them.

# In[ ]:


df.drop(df[df.duplicated()].index.tolist(), inplace=True)
df.shape


# Now we check for duplicates by just title

# In[ ]:


df[df.duplicated(['title'])]


# We get 5960 samples that are completely same with other sample titles. We also remove them before prcedding further.

# In[ ]:


df.drop(df[df.duplicated(['title'])].index.tolist(), inplace=True)
df.shape


# Checking for duplicates just by text we observe that there are samples that have same text but different titles, and there are 15 such samples that have their duplicate pair-end in text, subject and date, but not in title. We also remove them.

# In[ ]:


df[df.duplicated(['text'], keep=False)]


# In[ ]:


df.drop(df[df.duplicated(['text'])].index.tolist(),inplace=True)


# Now we are left with 38,269 samples.

# In[ ]:


df.shape


# <b>Checking class distributions</b>

# In[ ]:


df.groupby("label")['title'].count().plot.bar()


# <hr>

# ### Date Feature
# 
# Dateformats we have are DD-MM-YY and MM DD, YYYY. We can clean them and keep a uniform format across the feature but that will not be of much use. Because we do not want our classifier to distinguish a news item on the basis of the date that article was created hence we remove them from our dataset.
# 
# We also see that for fake news, dates are distributed across a three year period and for true news all dates are between ~Jan 2016 to ~Jan 2018.

# <b>Different types of dates</b>

# In[ ]:


df.sort_values('date').date.reset_index(drop=True).iloc[[5,10,20,50,100,200,38264,38268]]


# <b>Dates distribution</b>

# In[ ]:


import plotly.graph_objects as go

fig = go.Figure(layout=go.Layout(title='Day wise article count',yaxis_title='No. of news',xaxis_title='Dates'))

fig.add_trace(go.Scatter(y=df[df.label==0].groupby(['date']).label.count(),
                    mode='lines',
                    name='true'))
fig.add_trace(go.Scatter(x=df[df.label==1].groupby(['date']).label.count().index,y=df[df.label==1].groupby(['date']).label.count(),
                    mode='lines',
                    name='fake'))

fig.show()


# In[ ]:


df.drop('date', axis=1,inplace=True)


# <hr>

# ### Subject Feature
# 
# We have different classes in subject feature for both target classes. We can bin the positive (i.e. fake) class subject unique values to match negative (i.e. true) subject unique values.
# 
# OR
# 
# We can merge the subject into the text.
# 
# We can also bin and merge but then there will not be any information left to distinguish, which also holds true if we bin one class values to match other class unique values. After experimenting with some simple models we chose to remove them.

# In[ ]:


df[df.label==1].subject.value_counts()


# In[ ]:


df[df.label==0].subject.value_counts()


# In[ ]:


df.drop('subject', axis=1,inplace=True)


# <hr>

# <b>We merge title and text features into single feature text</b>

# In[ ]:


df['text'] = df['title'] + ' ' + df['text']
del df['title']

df.head(2)


# <b>We create stopwords for cleaning the text</b>
# 
# We chose to combine nltk and wordcloud stopwords despite more than 90% elements as same, because decontractions became easy with this approach.

# In[ ]:



nltk_stopwords = stopwords.words('english')
wordcloud_stopwords = STOPWORDS

nltk_stopwords.extend(wordcloud_stopwords)

stopwords = set(nltk_stopwords)
print(stopwords)


# <hr>

# <b>Now we clean the title feature</b>
# 
# 1. We remove urls (if any)
# 2. Perform decontractions
# 3. Non-acronize few popular words
# 4. Remove punctuations and all special characters
# 5. Remove stopwords

# In[ ]:



def clean(text):
    
    text = re.sub("http\S+", '', str(text))
    
    # Contractions ref: https://www.kaggle.com/gunesevitan/nlp-with-disaster-tweets-eda-cleaning-and-bert
    
    text = re.sub(r"he's", "he is", str(text))
    text = re.sub(r"there's", "there is", str(text))
    text = re.sub(r"We're", "We are", str(text))
    text = re.sub(r"That's", "That is", str(text))
    text = re.sub(r"won't", "will not", str(text))
    text = re.sub(r"they're", "they are", str(text))
    text = re.sub(r"Can't", "Cannot", str(text))
    text = re.sub(r"wasn't", "was not", str(text))
    text = re.sub(r"aren't", "are not", str(text))
    text = re.sub(r"isn't", "is not", str(text))
    text = re.sub(r"What's", "What is", str(text))
    text = re.sub(r"haven't", "have not", str(text))
    text = re.sub(r"hasn't", "has not", str(text))
    text = re.sub(r"There's", "There is", str(text))
    text = re.sub(r"He's", "He is", str(text))
    text = re.sub(r"It's", "It is", str(text))
    text = re.sub(r"You're", "You are", str(text))
    text = re.sub(r"I'M", "I am", str(text))
    text = re.sub(r"shouldn't", "should not", str(text))
    text = re.sub(r"wouldn't", "would not", str(text))
    text = re.sub(r"i'm", "I am", str(text))
    text = re.sub(r"I'm", "I am", str(text))
    text = re.sub(r"Isn't", "is not", str(text))
    text = re.sub(r"Here's", "Here is", str(text))
    text = re.sub(r"you've", "you have", str(text))
    text = re.sub(r"we're", "we are", str(text))
    text = re.sub(r"what's", "what is", str(text))
    text = re.sub(r"couldn't", "could not", str(text))
    text = re.sub(r"we've", "we have", str(text))
    text = re.sub(r"who's", "who is", str(text))
    text = re.sub(r"y'all", "you all", str(text))
    text = re.sub(r"would've", "would have", str(text))
    text = re.sub(r"it'll", "it will", str(text))
    text = re.sub(r"we'll", "we will", str(text))
    text = re.sub(r"We've", "We have", str(text))
    text = re.sub(r"he'll", "he will", str(text))
    text = re.sub(r"Y'all", "You all", str(text))
    text = re.sub(r"Weren't", "Were not", str(text))
    text = re.sub(r"Didn't", "Did not", str(text))
    text = re.sub(r"they'll", "they will", str(text))
    text = re.sub(r"they'd", "they would", str(text))
    text = re.sub(r"DON'T", "DO NOT", str(text))
    text = re.sub(r"they've", "they have", str(text))
    text = re.sub(r"i'd", "I would", str(text))
    text = re.sub(r"should've", "should have", str(text))
    text = re.sub(r"where's", "where is", str(text))
    text = re.sub(r"we'd", "we would", str(text))
    text = re.sub(r"i'll", "I will", str(text))
    text = re.sub(r"weren't", "were not", str(text))
    text = re.sub(r"They're", "They are", str(text))
    text = re.sub(r"let's", "let us", str(text))
    text = re.sub(r"it's", "it is", str(text))
    text = re.sub(r"can't", "cannot", str(text))
    text = re.sub(r"don't", "do not", str(text))
    text = re.sub(r"you're", "you are", str(text))
    text = re.sub(r"i've", "I have", str(text))
    text = re.sub(r"that's", "that is", str(text))
    text = re.sub(r"i'll", "I will", str(text))
    text = re.sub(r"doesn't", "does not", str(text))
    text = re.sub(r"i'd", "I would", str(text))
    text = re.sub(r"didn't", "did not", str(text))
    text = re.sub(r"ain't", "am not", str(text))
    text = re.sub(r"you'll", "you will", str(text))
    text = re.sub(r"I've", "I have", str(text))
    text = re.sub(r"Don't", "do not", str(text))
    text = re.sub(r"I'll", "I will", str(text))
    text = re.sub(r"I'd", "I would", str(text))
    text = re.sub(r"Let's", "Let us", str(text))
    text = re.sub(r"you'd", "You would", str(text))
    text = re.sub(r"It's", "It is", str(text))
    text = re.sub(r"Ain't", "am not", str(text))
    text = re.sub(r"Haven't", "Have not", str(text))
    text = re.sub(r"Could've", "Could have", str(text))
    text = re.sub(r"youve", "you have", str(text))
    
    # Others
    text = re.sub("U.S.", "United States", str(text))
    text = re.sub("Dec", "December", str(text))
    text = re.sub("Jan.","January", str(text))
    
    # Punctuations & special characters
    text = re.sub("[^A-Za-z0-9]+"," ", str(text))
    
    # Stop word removal
    text = " ".join(str(i).lower() for i in text.split() if i.lower() not in stopwords)

    return text
    


# In[ ]:



df['text'] = df['text'].map(lambda x: clean(x))


# In[ ]:


df.text.iloc[:3]


# <hr>

# ### Splitting into train, test and CV

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df.drop('label',axis=1), df.label, test_size=0.2, random_state=42)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# <hr>

# ### Using sklearn TfidfVectorizer

# <br>

# In[ ]:


vectorizer = TfidfVectorizer(min_df=0.01,ngram_range=(1,3))
vectorizer.fit(X_train.text)

X_tr = vectorizer.transform(X_train.text)
X_te = vectorizer.transform(X_test.text)

print(X_tr.shape, X_te.shape)


# #### Hyperparameter tuning Logistic Regression

# In[ ]:


clf = SGDClassifier(loss='log')

gs = GridSearchCV(
    estimator = clf,
    param_grid = {'alpha':np.logspace(-10,5,16)},
    cv = 5,
    return_train_score = True,
    scoring = 'accuracy'
    )

gs.fit(X_tr,y_train)

results = pd.DataFrame(gs.cv_results_)

results = results.sort_values(['param_alpha'])
train_auc = results['mean_train_score']
cv_auc = results['mean_test_score']
alpha = pd.Series([ math.log(i) for i in np.array(results['param_alpha']) ]) 

plt.plot(alpha, train_auc, label='Train AUC')
plt.plot(alpha, cv_auc, label='CV AUC')
plt.scatter(alpha, train_auc)
plt.scatter(alpha, cv_auc)
plt.legend()
plt.xlabel('log(alpha): hyperparameter')
plt.ylabel('Accuracy')
plt.title('Hyperparameter vs Accuracy Plot')
plt.grid()
plt.show()

print(gs.best_params_)


# #### Training on best parameters

# In[ ]:


clf = SGDClassifier(loss='log',alpha=1e-06, random_state=42).fit(X_tr,y_train)

print('Training score : %f' % clf.score(X_tr,y_train))
print('Test score : %f' % clf.score(X_te,y_test))

train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, clf.predict_proba(X_tr)[:,1])
test_fpr, test_tpr, te_thresholds = roc_curve(y_test, clf.predict_proba(X_te)[:,1])

plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
plt.legend()
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("AUC Plot")
plt.grid()
plt.show()


# In[ ]:


print(classification_report(y_train.values, clf.predict(X_tr)))
confusion_matrix(y_train, clf.predict(X_tr))


# #### Final Test Scores

# In[ ]:


print(classification_report(y_test.values, clf.predict(X_te)))

cm = pd.DataFrame(confusion_matrix(y_test,clf.predict(X_te)) , index = ['Fake','Not Fake'] , columns = ['Fake','Not Fake'])
sns.heatmap(cm,cmap= 'Blues', annot = True, fmt='', xticklabels = ['Fake','Not Fake'], yticklabels = ['Fake','Not Fake'])
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title('Confusion matrix on test data')
plt.show()


# **Top 50 n-grams**

# In[ ]:


coef = [abs(i) for i in clf.coef_.ravel()]
feature_names = vectorizer.get_feature_names()
feature_imp = dict(zip(feature_names,coef))
feature_imp = {k: v for k, v in sorted(feature_imp.items(), key=lambda item: item[1], reverse=True)}

top_50_features = {k: feature_imp[k] for k in list(feature_imp)[0:50]}

fig, ax = plt.subplots(figsize=(6,10))

people = top_50_features.keys()
y_pos = np.arange(len(people))
importance = top_50_features.values()

ax.barh(y_pos, importance,align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(people)
ax.invert_yaxis()
ax.set_xlabel('Importance')
ax.set_ylabel('Features')
ax.set_title('Top 50 Features')

plt.show()


# **Bottom 50 n-grams**

# In[ ]:


feature_imp = dict(zip(feature_names,coef))
feature_imp = {k: v for k, v in sorted(feature_imp.items(), key=lambda item: item[1], reverse=False)}

bottom_50_features = {k: feature_imp[k] for k in list(feature_imp)[0:50]}

fig, ax = plt.subplots(figsize=(6,10))

people = bottom_50_features.keys()
y_pos = np.arange(len(people))
importance = bottom_50_features.values()

ax.barh(y_pos, importance,align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(people)
ax.invert_yaxis()
ax.set_xlabel('Importance')
ax.set_ylabel('Features')
ax.set_title('Least 50 important features')

plt.show()


# <hr>

# ### Other vectorizers that can be tried are:
# 
# 1. Glove vectors
# 2. Avg W2V
# 2. Tfidf weighted W2V

# <hr>

# ### Observations

# We experimented with various hyperparameters with tdidf vectorizer like n-grams, min-df etc and found the chosen parameters as reasonable.
# 
# We also tried few other classifiers like SGD with log loss and SVM, but there was no significant improvement in the incorrectly labelled points.
# 
# While performing text pre-processing we found that there were certain words like 'Reuters' and 'official', that were only present in negative i.e. real news data samples. Having these words in the corpus has significantally contributed in the performance of the model.
# 
# Since much simpler and easy to interpret models are giving good results, therefore we do not apply deep learning to this problem.

# <hr>

# <b>Now we dump the vectorizer and classifier, so that they can used during real-time predictions</b>

# In[ ]:


# import pickle

# with open('vectorizer', 'wb') as f:
#     pickle.dump(vectorizer, f)
    
# with open('model', 'wb') as f:
#     pickle.dump(clf, f)


# <br>
