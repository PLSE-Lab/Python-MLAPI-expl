#!/usr/bin/env python
# coding: utf-8

# # SMS Spam Classification

# ## Imports

# In[ ]:


import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, confusion_matrix


# ## Read and clean data

# In[ ]:


data = pd.read_csv('../input/spam.csv', encoding='latin-1')
data.head()


# We see that there are 3 useless columns, we're going to delete them.

# In[ ]:


data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
data.head()


# We're going to rename 'v1' and 'v2' as 'label' and 'message' respectively, because that's more meaningful.

# In[ ]:


data.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)
data.head()


# We check if there are any missing data.

# In[ ]:


data.count()  # check if the two columns have the same number of values


# In[ ]:


data['label'].apply(len).min()  # check for empty labels


# In[ ]:


data['message'].apply(len).min()  # check for empty messages


# In[ ]:


set(data['label'])  # check if there are labels other than 'ham' and 'spam'


# Apparently, there are no missing data. Before finishing this part, we'll transform the values of the 'label' column from 'ham' and 'spam' to 0 and 1 respectively.

# In[ ]:


data['label'] = data['label'].apply(lambda label: 0 if label == 'ham' else 1)
data.head()


# ## Exploratory Data Analysis

# In[ ]:


data.describe()


# In[ ]:


# Empirical distribution of the labels
print('Percentage of spams: {0}%'.format(round(100 * data['label'].sum() / len(data['label']), 2)))
plt.hist(data['label'], bins=3, weights=np.ones(len(data['label'])) / len(data['label']))
plt.xlabel('label')
plt.ylabel('Empirical PDF')
plt.title('Empirical distribution of the labels')


# We see that the labels are unbalanced. 0 (ham) is highly represented wheras 1 (spam) represents only 13.41% of the data.
# 
# Since the classes (labels) are unbalanced, we cannot use accuracy to access the performance of the classifier that we'll build. A classifier which always returns 0 will have an accuracy around 87% (assuming the given data is well sampled). Instead, we'll use the F1-score because it takes into account the precision and recall of the system, which should be the case here.

# Let's have a look at the messages now.

# In[ ]:


# extract spams and hams
spams = data['message'].iloc[(data['label'] == 1).values]
hams = data['message'].iloc[(data['label'] == 0).values]
print(spams[:10])
print(hams[:10])


# We start seeing some caracteristics of spams, most notably the extensive use of the word 'Free'.

# In[ ]:


# message length
data['message_length'] = data['message'].apply(lambda message: len(message))
data.head()


# In[ ]:


plt.hist(data['message_length'], bins=50, weights=np.ones(len(data))/len(data))
plt.xlabel('Message length')
plt.ylabel('Empirical PDF')
plt.title('Messages lengths')


# In[ ]:


plt.hist(spams.apply(lambda x: len(x)),
         bins=50,
         weights=np.ones(len(spams)) / len(spams),
         facecolor='r',
         label='Spams')
plt.hist(hams.apply(lambda x: len(x)),
         bins=50,
         weights=np.ones(len(hams)) / len(hams),
         facecolor='g',
         alpha=0.8,
         label='Hams')
plt.xlabel('Message lenght')
plt.ylabel('Empirical PDF')
plt.title('Spam/ham messages lengths')
plt.legend()


# We see that spams tend to be longer than hams. This information might be useful.
# 
# Now we'll look at which words are most common in spams and hams.

# In[ ]:


# most common words in spam and ham
spam_tokens = []
for spam in spams:
    spam_tokens += nltk.tokenize.word_tokenize(spam)
ham_tokens = []
for ham in hams:
    ham_tokens += nltk.tokenize.word_tokenize(ham)
print(spam_tokens[:10])
print(ham_tokens[:10])


# In[ ]:


# remove stop words and puncuation from tokens
stop_words = ['.', 'to', '!', ',', 'a', '&', 
              'or', 'the', '?', ':', 'is', 'for',
              'and', 'from', 'on', '...', 'in', ';',
              'that', 'of']
for tokens in [spam_tokens, ham_tokens]:
    for stop_word in stop_words:
        try:
            while True:
                tokens.remove(stop_word)
        except ValueError:  # all occurrences of the stop word have been removed
            pass


# In[ ]:


most_common_tokens_in_spams = Counter(spam_tokens).most_common(20)
most_common_tokens_in_hams = Counter(ham_tokens).most_common(20)
print(most_common_tokens_in_spams)
print(most_common_tokens_in_hams)


# Certain words like 'call' and 'free' seem to occurre often in the spams.

# ## Classification

# Since we're maily dealing with text data (the messages), it makes sense to change the representation of the text, using either bag of words representations (tfidfs or word counts or binary BoW) or word embeddings. we'll use BoW representations, starting with tfidfs.
# 
# But before doing anything else, we'll split the data into train data and test data in order to have an accurate estimation of the final classifier's performance.

# In[ ]:


data, test_data = train_test_split(data, test_size=0.3)
print('Train-valid data length: {0}'.format(len(data)))
print('Test data length: {0}'.format(len(test_data)))


# In[ ]:


binary_vectorizer = CountVectorizer(binary=True)
count_vectorizer = CountVectorizer()
tfidf_vectorizer = TfidfVectorizer()


# In[ ]:


def feature_extraction(df, test=False):
    if not test:
        tfidf_vectorizer.fit(df['message'])
    
    X = np.array(tfidf_vectorizer.transform(df['message']).todense())
    return X


# In[ ]:


train_df, valid_df = train_test_split(data, test_size=0.3)

X_train = feature_extraction(train_df)
y_train = train_df['label'].values

X_valid = feature_extraction(valid_df, test=True)
y_valid = valid_df['label'].values


# In[ ]:


clfs = {
    'mnb': MultinomialNB(),
    'gnb': GaussianNB(),
    'svm1': SVC(kernel='linear'),
    'svm2': SVC(kernel='rbf'),
    'svm3': SVC(kernel='sigmoid'),
    'mlp1': MLPClassifier(),
    'mlp2': MLPClassifier(hidden_layer_sizes=[100, 100]),
    'ada': AdaBoostClassifier(),
    'dtc': DecisionTreeClassifier(),
    'rfc': RandomForestClassifier(),
    'gbc': GradientBoostingClassifier(),
    'lr': LogisticRegression()
}


# In[ ]:


f1_scores = dict()
for clf_name in clfs:
    print(clf_name)
    clf = clfs[clf_name]
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_valid)
    f1_scores[clf_name] = f1_score(y_pred, y_valid)


# In[ ]:


f1_scores


# Multinomial naive bayes seem to give the best results. Before doing hyperparameter optimization, we'll see how other BoW representations perform. We'll reduce the number of classifier to same some computation time.

# In[ ]:


clfs = {
    'mnb': MultinomialNB(),
    'gnb': GaussianNB(),
    'svm': SVC(kernel='linear'),
    'mlp': MLPClassifier(),
    'ada': AdaBoostClassifier(),
    'dtc': DecisionTreeClassifier(),
    'rfc': RandomForestClassifier(),
    'gbc': GradientBoostingClassifier(),
    'lr': LogisticRegression()
}


# In[ ]:


def feature_extraction(df, test=False):
    if not test:
        count_vectorizer.fit(df['message'])
    
    X = np.array(count_vectorizer.transform(df['message']).todense())
    return X


# In[ ]:


X_train = feature_extraction(train_df)
X_valid = feature_extraction(valid_df, test=True)


# In[ ]:


f1_scores = dict()
for clf_name in clfs:
    print(clf_name)
    clf = clfs[clf_name]
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_valid)
    f1_scores[clf_name] = f1_score(y_pred, y_valid)


# In[ ]:


f1_scores


# Multinomial naive bayes is still the best. The f1 score for it is the same.
# 
# Now, let's see with BoW.

# In[ ]:


def feature_extraction(df, test=False):
    if not test:
        binary_vectorizer.fit(df['message'])
    
    X = np.array(binary_vectorizer.transform(df['message']).todense())
    return X


# In[ ]:


X_train = feature_extraction(train_df)
X_valid = feature_extraction(valid_df, test=True)


# In[ ]:


f1_scores = dict()
for clf_name in clfs:
    print(clf_name)
    clf = clfs[clf_name]
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_valid)
    f1_scores[clf_name] = f1_score(y_pred, y_valid)


# In[ ]:


f1_scores


# Same results. I want to see how the binomial naive bayes performes.

# In[ ]:


clf = BernoulliNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_valid)
f1_score(y_pred, y_valid)


# Not that good.
# 
# Conclusion: Multinomial naive bayes is the way to go.

# Now, we'll see if we can do better by using message length.

# In[ ]:


def feature_extraction(df, test=False):
    if not test:
        count_vectorizer.fit(df['message'])
    
    X = np.array(count_vectorizer.transform(df['message']).todense())
    X = np.concatenate((X, df['message_length'].values.reshape(-1, 1)), axis=1)
    return X


# In[ ]:


X_train = feature_extraction(train_df)
X_valid = feature_extraction(valid_df, test=True)


# In[ ]:


clf = MultinomialNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_valid)
f1_score(y_pred, y_valid)


# It went down, we'll forget about it and work only with the word counts.

# ### Hyperparameter optimization

# We'll only optimize the hyperparemeter alpha.

# In[ ]:


def feature_extraction(df, test=False):
    if not test:
        tfidf_vectorizer.fit(df['message'])
    
    X = np.array(tfidf_vectorizer.transform(df['message']).todense())
    return X


# In[ ]:


X_train = feature_extraction(train_df)
X_valid = feature_extraction(valid_df, test=True)


# In[ ]:


alpha_values = [i * 0.1 for i in range(11)]
max_f1_score = float('-inf')
best_alpha = None
for alpha in alpha_values:
    clf = MultinomialNB(alpha=alpha)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_valid)
    current_f1_score = f1_score(y_pred, y_valid)
    if current_f1_score > max_f1_score:
        max_f1_score = current_f1_score
        best_alpha = alpha


# In[ ]:


print('Best alpha: {0}'.format(best_alpha))
print('Best f1-score: {0}'.format(max_f1_score))


# ## Final results

# In[ ]:


clf = MultinomialNB(alpha=0.1)


# In[ ]:


X_train = feature_extraction(data)
y_train = data['label'].values

X_test = feature_extraction(test_data, test=True)
y_test = test_data['label'].values


# In[ ]:


clf.fit(X_train, y_train)


# In[ ]:


y_pred = clf.predict(X_test)
print('F1-score: {0}'.format(f1_score(y_pred, y_test)))
print('Confusion matrix:')
confusion_matrix(y_pred, y_test)


# We obtained an f1-score of 0.92, which is pretty good for the task.
