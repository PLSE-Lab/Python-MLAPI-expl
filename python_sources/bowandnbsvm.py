#!/usr/bin/env python
# coding: utf-8

# I'll start with creating some simple frequency-based baseline models using BOW, Naive Bayes classifier and logistic regression.  In later notebooks, I'll improve on these models

# <h2>Load and split the data</h2>
# 
# Let's load our cleaned data first

# In[ ]:


import pandas as pd

train = pd.read_csv('/kaggle/input/cleaned/train_cleaned.csv')
test = pd.read_csv('/kaggle/input/cleaned/test_cleaned.csv')

print('Train shape:', train.shape)
print('Test shape:', test.shape)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    train['clean_text'], train['target'], shuffle=True, test_size=0.2, random_state=0)


# <h2>Bag on N-grams</h2>
# 
# I'll implement BOW using `CountVectorizer` from `sklearn`. This is pretty straight-forward -- just count the occurrence of each ngram in the corpus. I'll use unigrams and bigrams here.  We don't have that much data and trigrams are likely to just add noise

# <b>Train the Bag of N-grams model</b>

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer

bv = CountVectorizer(max_features=50000, ngram_range=(1, 2))
train_data_features = bv.fit_transform(X_train)
val_data_features = bv.transform(X_val)


# In[ ]:


vocab = bv.get_feature_names()
print(vocab[:10])


# <h2>Naive Bayes</h2>
# 
# Refer to Jeremy Howard's lecture: https://www.youtube.com/watch?v=37sFIak42Sc&feature=youtu.be&t=3745
# 
# Naive Bayes theorem is an extension of <b>Bayes' Theorem</b>, which can be defined as <i>the probability of an event occurring given the probability of another event that has already occurred</i>. It can be mathematically stated as: 
# 
# $P(A|B) = \frac{\text{P(B|A)P(A)}}{\text{P(B)}}$
# 
# - The idea is to find P(A) given the probability of event B
# - P(A) is <i>a priori</i> probability of event A, i.e., prior to seeing new evidence B. 
# - P(A|B) is the <i>a posteriori</i> probability of A conditional upon event B occurring. 
# 
# In our case, it can be generalized to: what is the probability of a certain tweet being a disaster tweet, i.e., classification of 1, given a collection of tokens in that tweet?  
# 
# <h4>Naive Assumption</h4>
# 
# This assumption states that all the features must be <i>independent</i> of each other.  This is definitely not realistic in most real-life scenarios, but it is a good starting point for a theoretical model.  For example, we know and can assert definitively, that the words in a sentence are not independent of each other.  That's the whole idea behind CBOW and SkipGram models that words aren't independent of each other and we can predict them based on context. 
# 
# We define the <b>log-count ratio</b>, <i>r</i>, for each word <i>f</i>:
# 
# $r = \log \frac{\text{ratio of feature $f$ in disaster tweets}}{\text{ratio of feature $f$ in non-disaster tweets}}$
# 
# where ratio of feature, <i>f</i>, in disaster tweets is the number of times a disaster tweet has a feature divided by the number of disaster tweets

# In[ ]:


# Calculate the ratio of feature f
def pr(y_i):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)


# In[ ]:


import numpy as np

x = train_data_features # these are simply counts of unigrams and bigrams as a sparse matrix
y = y_train.values # targets

r = np.log(pr(1)/pr(0)) # probability matrix for each feature based on the training set
b = np.log((y==1).mean() / (y==0).mean()) # bias


# In[ ]:


r.shape, r


# Apply the ratio <i>r</i> to the <b>Naive Bayes formula</b> and get predictions on the validation set

# In[ ]:


y_val_pre_preds = val_data_features @ r.T + b # multiply the probability matrix with the features in the validation set
y_val_preds = y_val_pre_preds.T > 0 # get disaster tweet predictions
(y_val_preds == y_val.values).mean() # estimate accuracy


# Now let's try <b>Binarized Naive Bayes</b>, which disregards the frequency of a token and simply assigns 1, if it is present and 0 if it is not. 

# In[ ]:


x = train_data_features.sign() # binarize
r = np.log(pr(1)/pr(0))

y_val_pre_preds = val_data_features.sign() @ r.T + b 
y_val_preds = y_val_pre_preds.T>0
(y_val_preds == y_val.values).mean()


# About the same as non-binarized model. Naive Bayes is simply a theoretical model, which doesn't really apply well to the real word.  We can surely improve on this model by actually using the data that we are given. 

# <h2>Logistic Regression</h2>

# In[ ]:


from sklearn.linear_model import LogisticRegression

logistic = LogisticRegression(C=0.5)
logistic.fit(train_data_features, y_train)
y_val_preds = logistic.predict(val_data_features)
(y_val_preds==y_val).mean()


# Not bad. We get significant improvement over the theoretical model. Let's see if the binarized version works better

# In[ ]:


logistic = LogisticRegression(C=0.5)
logistic.fit(train_data_features.sign(), y_train)
y_val_preds = logistic.predict(val_data_features.sign())
(y_val_preds==y_val).mean()


# Binarized version is slightly better. Let's take a look at the words used by the model for classifying disaster tweets using the amazing `eli5` package

# In[ ]:


import eli5

eli5.show_weights(logistic, vec=bv, top=25)


# <h2>NBSVM</h2>

# A combined version of Naive Bayes and SVM where rather than training the Logistic Regression model on simply the counts of unigrams and bigrams, we train it on the log-count ratios of features.  See the video from Jeremy Howard above for the intuition and the brilliant explanation for why this works even though seemingly both approaches should give similar results. 

# In[ ]:


x = train_data_features
y = y_train.values

r = np.log(pr(1)/pr(0))
x_nb = x.multiply(r)

logistic = LogisticRegression(C=0.5)
logistic.fit(x_nb, y);

val_x_nb = val_data_features.multiply(r)
y_val_preds = logistic.predict(val_x_nb)
(y_val_preds.T==y_val.values).mean()


# Does not perform as well as the simple logistic regression model, which is a bit surprising given its track record historically, but we have only 50,000 features as opposed to million, which is a huuuuge limiting factor. Given this, it's probably going to suffer in the binarized version as well, but let's run it anyway

# In[ ]:


x = train_data_features.sign()
y = y_train.values

r = np.log(pr(1)/pr(0))
x_nb = x.multiply(r)

logistic = LogisticRegression(C=0.5)
logistic.fit(x_nb, y);

val_x_nb = val_data_features.sign().multiply(r)
y_val_preds = logistic.predict(val_x_nb)
(y_val_preds.T==y_val.values).mean()


# In[ ]:


eli5.show_weights(logistic, vec=bv, top=25)


# <h2>Random Forest Classifier</h2>
# 
# Now let's try how logistic regression compares with a random forest classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators = 200)
forest = forest.fit(train_data_features, y_train)


# In[ ]:


# Predict on training set
y_train_pred = forest.predict(train_data_features)

# Predict on validation set
y_val_pred = forest.predict(val_data_features)


# In[ ]:


(y_val_pred == y_val).mean()


# <b>Back to the Best model -- Binarized Logistic Regression</b>

# In[ ]:


logistic = LogisticRegression(C=0.5)
logistic.fit(train_data_features.sign(), y_train)
y_val_pred = logistic.predict(val_data_features.sign())


# In[ ]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(y_true, y_pred, ax, vmax=None,
                          normed=True, title='Confusion matrix'):
    cm = confusion_matrix(y_true, y_pred)
    if normed:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm, vmax=vmax, annot=True, square=True, ax=ax, 
                cmap='Blues', cbar=False, linecolor='k',
               linewidths=1)
    ax.set_title(title, fontsize=16)
    ax.set_ylabel('True labels', fontsize=12)
    ax.set_xlabel('Predicted labels', y=1.10, fontsize=12)


# In[ ]:


fig, (axis1, axis2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
plot_confusion_matrix(y_train, y_train_pred, ax=axis1, 
                      title='Confusion matrix (train data)')
plot_confusion_matrix(y_val, y_val_pred, ax=axis2, 
                      title='Confusion matrix (validation data)')


# In[ ]:


from sklearn.metrics import classification_report, f1_score

print('Classification report on Test set: \n', classification_report(y_val, y_val_pred))


# In[ ]:


y_val_probs = logistic.predict_proba(val_data_features.sign())[:, 1]

for thresh in np.arange(0.3, 0.5, 0.01):
    thresh = np.round(thresh, 2)
    print("F1 score at threshold {0} is {1}".format(thresh, f1_score(y_val, (y_val_probs>thresh).astype(int))))


# The highest f1 score was achieved was at the threshold of approx. 0.45

# <h2>TFIDF</h2>
# 
# Term-frequency inverse document frequency.  I have described this in detail in many places before. Logic will tell you that the binariezed version will give exactly the same results as BOW so let's just quickly run through regular logistic regression and random forest even though we don't expect either to perform as well as the binarized logistic regression model above

# <b>Train the TFIDF model</b>
# 
# Once again we'll use unigrams and bigrams

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
train_data_features = tfidf.fit_transform(X_train)
val_data_features = tfidf.transform(X_val)


# In[ ]:


vocab = tfidf.get_feature_names()
print(vocab[:10])


# <h2>Logistic Regression</h2>

# In[ ]:


from sklearn.linear_model import LogisticRegression

logistic = LogisticRegression(C=0.5)
logistic.fit(train_data_features, y_train)
y_val_preds = logistic.predict(val_data_features)
(y_val_preds==y_val).mean()


# In[ ]:


import eli5

eli5.show_weights(logistic, vec=tfidf, top=25)


# <h2>NBSVM</h2>

# In[ ]:


x = train_data_features
y = y_train.values

r = np.log(pr(1)/pr(0))
x_nb = x.multiply(r)

logistic = LogisticRegression(C=0.5)
logistic.fit(x_nb, y);

val_x_nb = val_data_features.multiply(r)
y_val_preds = logistic.predict(val_x_nb)
(y_val_preds.T==y_val.values).mean()


# In[ ]:


eli5.show_weights(logistic, vec=tfidf, top=25)


# <h2>Random Forest Classifier</h2>

# In[ ]:


forest = RandomForestClassifier(n_estimators = 200)
forest = forest.fit(train_data_features, y_train)


# In[ ]:


# Predict on training set
y_train_pred = forest.predict(train_data_features)

# Predict on validation set
y_val_pred = forest.predict(val_data_features)


# In[ ]:


(y_val_pred == y_val).mean()


# <h2>Final thoughts on the baseline model</h2>

# There we go. We got the best performance out of binarized logistic regression with an accuracy of ~82.2% on the validation set. This is definitely not a bad start, but we should be able to improve on this. In the next few notebooks I'll use word embeddings using GloVe and FastText and eventually will move on to deep learning models using LSTM!

# <h2>Submission set</h2>
# 
# Make predictions on the submission set using binarized logistic regression

# In[ ]:


test = pd.read_csv('/kaggle/input/cleaned/test_cleaned.csv')

test.head()


# In[ ]:


test['clean_text']


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer

train_data_features = bv.fit_transform(X_train)
test['clean_text'] = test['clean_text'].apply(lambda x: str(x))
test_data_features = bv.transform(test['clean_text'])


# In[ ]:


logistic = LogisticRegression(C=0.5)
logistic.fit(train_data_features.sign(), y_train)
y_test_pred = logistic.predict(test_data_features.sign())


# In[ ]:


y_test_probs = logistic.predict_proba(test_data_features.sign())[:, 1]


# In[ ]:


(y_test_probs>0.45).sum()/3262


# According to the Logistic Regression model, 36% of tweets in the test set are classified as disaster tweets

# In[ ]:


test['target'] = y_test_pred
test.head()


# In[ ]:




