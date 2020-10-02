#!/usr/bin/env python
# coding: utf-8

# # News Articles Classification (TF-IDF + Voting Ensemble)

# In this notebook, we will present the winning solution of the **BigData2020 Classification**, a Kaggle contest created for the **"Big Data Analytics"** postgraduate course of the Department of Informatics at the University of Athens. The contest was a very close call, which can be seen from the fact that while our solution was 2nd on the public leaderboard, it eventually came 1st on the private leaderboard (Dimitrios Roussis and Thanasis Polydoros).
# 
# I decided to describe our solution in this kernel exactly because I am sure that most teams went to great lengths to win this competition and would probably find it useful. I certainly do not want to spoil the fun for the future students, but I do hope that they will enjoy the challenge as much as we did and actually come up with new and better solutions! Consequently, I will focus only on the parts which are relevant for our approach.
# 
# Without further ado, I will go on to explain what this contest was about, how we tackled the problem and which measure was used. We are given a dataset which contains 111,795 news articles which are labelled as one of the 4 following categories: **Business, Technology, Entertainment** and **Health**. Below, we will first do some minimal preprocessing, extract features with a TF-IDF vectorizer and train a voting classifier.
# 
# The metrics that we measure are the usual ones: Accuracy, Precision, Recall and the F-Measure (also F-Score and also F1 Score). The competition however used only the **F-Measure (Macro)** for the evaluation and the score of our approach was **97.748%** (public leaderborad). So, let us import the necessary libraries and go through the aforementioned steps.

# In[ ]:


import numpy as np
import pandas as pd
import warnings
import os
from matplotlib import pyplot as plt
from timeit import default_timer as timer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


np.random.seed(42)


# In[ ]:


warnings.simplefilter(action='ignore', category=FutureWarning)


# In[ ]:


df_train = pd.read_csv('/kaggle/input/bigdata2020classification/train.csv/train.csv')
df_test = pd.read_csv('/kaggle/input/bigdata2020classification/test_without_labels.csv//test_without_labels.csv')


# In[ ]:


df_train.head()


# We can see that the dataset contains a column named "Id" which is not useful for our analysis and will be removed.

# In[ ]:


df_train = df_train.drop(['Id'], axis=1)


# We will also go ahead and remove the duplicate articles, i.e. those articles that have the exact same content. We also want to know how many of them were actually removed.

# In[ ]:


len_start = len(df_train)
df_train.info()


# In[ ]:


df_train.drop_duplicates(subset='Content', keep='last', inplace=True)


# In[ ]:


len_end = len(df_train)


# In[ ]:


# Display number of duplicates in the dataset

print('{} duplicate articles were removed from the dataset'
      .format(len_start - len_end))


# It is a always a good idea to check the distribution of the classes in a dataset and as we will see, the current dataset is somewhat imbalanced.

# In[ ]:


# Relative frequency of labels

rel_freq = df_train.groupby('Label').Title.count() / len(df_train)
print(rel_freq)


# In[ ]:


fig, ax = plt.subplots(figsize=(6,5))
rel_freq.plot.bar(ax=ax)
ax.set_title('Relative Frequencies of the Class Labels')
ax.set_xlabel('')
fig.tight_layout()
plt.show()


# In order to use **both the titles and the contents** of each news article, we will concatenate them in a single column from which the features will be extracted.

# In[ ]:


# Concatenate the titles and the contents

df_train['Text'] = df_train['Title'] + ' ' + df_train['Content']
X = df_train['Text'].to_dense().values
y = df_train['Label'].to_dense().values


# Let's see a sample (title and content).

# In[ ]:


df_train['Text'][259]


# Finally, we can move on to the classification of the news articles and for this reason, we will define two functions which will help us evaluate each classifier that we use.

# In[ ]:


def get_metrics(y_true, y_pred, metrics):
    metrics[0] += accuracy_score(y_true, y_pred)
    metrics[1] += precision_score(y_true, y_pred, average='macro')
    metrics[2] += recall_score(y_true, y_pred, average='macro')
    metrics[3] += f1_score(y_true, y_pred, average='macro')
    return metrics


# In[ ]:


def evaluate_classifier(clf, kfold, X, y, vectorizer):
    metrics = np.zeros(4)
    start = timer()
    for train, cv in kfold.split(X, y):
        X_train, X_cv = X[train], X[cv]
        y_train, y_cv = y[train], y[cv]
        X_train_gen = [x for x in X_train]
        vectorizer.fit(X_train_gen)
        X_train_vec = vectorizer.transform(X_train_gen)
        clf.fit(X_train_vec, y_train)
        X_cv_gen = [x for x in X_cv]
        X_cv_vec = vectorizer.transform(X_cv_gen)
        y_pred = clf.predict(X_cv_vec)
        metrics = get_metrics(y_cv, y_pred, metrics)
    dt = timer() - start
    metrics = metrics * 100 / 5
    print('Evaluation of classifier finished in {:.2f} s \n'
          'Average accuracy: {:.2f} % \n'
          'Average precision: {:.2f} % \n'
          'Average recall: {:.2f} % \n'
          'Average F-Measure: {:.2f} % \n'
          .format(dt, metrics[0], metrics[1],
                  metrics[2], metrics[3]))


# Below, we define all the necessary tools that will be used to train our final classifier. Note that we do not show the actual process of how we ended up using the hyper-parameters of the TF-IDF Vectorizer or those of the classification algorithms, as that would be very time-consuming (not to mention that it would spoil the fun). 
# 
# Nevertheless, we will make some important remarks:
# - The **TF-IDF Vectorizer** is a feature extraction technique for texts which treats each word and each sequence of two words as tokens for which their term-frequency multiplied by the inverse document-frequency is returned. This essentially means that the TF-IDF vectorizer takes into account not only how many times each word or **bigram** (sequence of two words) appears in a given text, but also how many times it appears overall. We also set upper (max_df=0.5) and lower (min_df=3) thresholds for the document-frequency; this is because a given term (word or bigram) will probably not improve the accuracy of our classifier if it only occurs in a couple samples (note that they are now combinations of the title and the content of each article and it is possible that a given term which is contained only in a single article seems to appear 2 times) nor if it occurs in more than half the articles. This is an easy way to remove reoccuring words and small phrases and has a similar effect with the removal of the stop words.
# - Each classification algorithm is evaluated using **5-fold cross validation** on just 80% of the training data ('X_train_cv' and 'y_train_cv'), while the rest 20% is meant for the final testing ('X_test' and 'y_test'). The evaluation with 5-fold cross validation can give us a very good sense of which algorithms perform better on the specific task, as well as which hyper-parameters of both the classifiers and the TF-IDF vectorizer we should use.
# - The final classification, however, is done by a **voting ensemble classifier** which combines 3 classifiers and gives as an output the predicted label which has the majority vote among the predictions of those classifiers. It is tested on the final test set in order to get a better estimate of its actual performance. This is partially because the hyper-parameters of the classifiers -as well as the choice of which ones to use- are "fitted" on the training and cross-validation set and thus, we could not be very confident of how well does our voting ensemble classifier actually generalize on uknown data.

# In[ ]:


# 5-Fold Cross-Validation

kf = KFold(n_splits=5, shuffle=True, random_state=56)

# Stop Words and TF-IDF Vectorizer

stop_words = ENGLISH_STOP_WORDS
tfidf = TfidfVectorizer(stop_words=stop_words, min_df=3,
                        max_df=0.5, ngram_range=(1, 2))

# Classifiers 

svm = LinearSVC(tol=1e-05, max_iter=1500)
ridge = RidgeClassifier(alpha=0.8, tol=1e-05)
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)


# In[ ]:


# Divide the dataset into a train/cross-validation set and a test set

X_train_cv, X_test, y_train_cv, y_test = train_test_split(X, y, test_size=0.2, 
                                                          random_state=56)


# In[ ]:


# SVM classifier

evaluate_classifier(svm, kf, X_train_cv, y_train_cv, tfidf)


# In[ ]:


# Ridge classifier

evaluate_classifier(ridge, kf, X_train_cv, y_train_cv, tfidf)


# In[ ]:


# kNN classifier

evaluate_classifier(knn, kf, X_train_cv, y_train_cv, tfidf)


# In[ ]:


# Voting ensemble of 3 classifiers

boost = VotingClassifier(estimators=[
                        ('svc', svm), ('ridge', ridge), ('knn', knn)],
                        voting='hard', n_jobs=-1)


# In[ ]:


# Evaluate the voting classifier on the test set

start = timer()
metrics = np.zeros(4)
X_train_gen = [x for x in X_train_cv]
tfidf.fit(X_train_gen)
X_train_vec = tfidf.transform(X_train_gen)
boost.fit(X_train_vec, y_train_cv)
X_test_gen = [x for x in X_test]
X_test_vec = tfidf.transform(X_test_gen)
y_pred = boost.predict(X_test_vec)
metrics = get_metrics(y_test, y_pred, metrics)
dt = timer() - start
metrics = metrics * 100
print('Evaluation of voting classifier on the test set finished in {:.2f} s . \n'
      'Average accuracy: {:.2f} % \n'
      'Average precision: {:.2f} % \n'
      'Average recall: {:.2f} % \n'
      'Average F-Measure: {:.2f} % \n'
      .format(dt, metrics[0], metrics[1],
              metrics[2], metrics[3]))


# As we can see, the performance of the voting ensemble classifier is better than that of each one of the individual classifiers that it uses. Moreover, the F-Measure seems very close to -but is actually lower than- the one that our solution scored on the public leaderboard (0.97748).
# 
# Since we know have a more trustworthy estimation of the final performance of the voting ensemble, we can train it with the whole dataset and make predictions on the unlabelled test data for the deliverable of the competition.

# In[ ]:


# Concatenate the titles and the contents

df_test['Text'] = df_test['Title'] + ' ' + df_test['Content']
X_final = df_test['Text'].to_dense().values


# In[ ]:


# Make predictions on the unlabeled data

X_train_gen = [x for x in X]
tfidf.fit(X_train_gen)
X_train_vec = tfidf.transform(X_train_gen)
boost.fit(X_train_vec, y)
X_test_gen = [x for x in X_final]
X_test_vec = tfidf.transform(X_test_gen)
y_pred = boost.predict(X_test_vec)


# In[ ]:


df_results = pd.DataFrame({'Id':df_test['Id'], 'Predicted':y_pred})


# In[ ]:


df_results.to_csv('testSet_categories.csv',index=False, header=True)


# **PLEASE DO NOT FORGET TO UPVOTE IF YOU LIKED THIS KERNEL!**
