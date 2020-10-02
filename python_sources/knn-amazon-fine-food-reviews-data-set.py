#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#KNN" data-toc-modified-id="KNN-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>KNN</a></span><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Objective:" data-toc-modified-id="Objective:-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Objective:</a></span></li></ul></li><li><span><a href="#BoW" data-toc-modified-id="BoW-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>BoW</a></span><ul class="toc-item"><li><span><a href="#Brute-force-KNN" data-toc-modified-id="Brute-force-KNN-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Brute force KNN</a></span><ul class="toc-item"><li><span><a href="#Missclassification-Error" data-toc-modified-id="Missclassification-Error-2.1.1"><span class="toc-item-num">2.1.1&nbsp;&nbsp;</span>Missclassification Error</a></span></li><li><span><a href="#Accuracy-Score" data-toc-modified-id="Accuracy-Score-2.1.2"><span class="toc-item-num">2.1.2&nbsp;&nbsp;</span>Accuracy Score</a></span></li><li><span><a href="#Classificatoin-Report" data-toc-modified-id="Classificatoin-Report-2.1.3"><span class="toc-item-num">2.1.3&nbsp;&nbsp;</span>Classificatoin Report</a></span></li><li><span><a href="#Confusion-Matrix" data-toc-modified-id="Confusion-Matrix-2.1.4"><span class="toc-item-num">2.1.4&nbsp;&nbsp;</span>Confusion Matrix</a></span></li></ul></li><li><span><a href="#KD-tree-KNN" data-toc-modified-id="KD-tree-KNN-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>KD tree KNN</a></span><ul class="toc-item"><li><span><a href="#Misclassification-Error" data-toc-modified-id="Misclassification-Error-2.2.1"><span class="toc-item-num">2.2.1&nbsp;&nbsp;</span>Misclassification Error</a></span></li><li><span><a href="#Accuracy-Score" data-toc-modified-id="Accuracy-Score-2.2.2"><span class="toc-item-num">2.2.2&nbsp;&nbsp;</span>Accuracy Score</a></span></li><li><span><a href="#Classification-Report" data-toc-modified-id="Classification-Report-2.2.3"><span class="toc-item-num">2.2.3&nbsp;&nbsp;</span>Classification Report</a></span></li><li><span><a href="#Confusion-Matrix" data-toc-modified-id="Confusion-Matrix-2.2.4"><span class="toc-item-num">2.2.4&nbsp;&nbsp;</span>Confusion Matrix</a></span></li></ul></li></ul></li><li><span><a href="#TF-IDF" data-toc-modified-id="TF-IDF-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>TF-IDF</a></span><ul class="toc-item"><li><span><a href="#Brute-force-KNN" data-toc-modified-id="Brute-force-KNN-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Brute force KNN</a></span><ul class="toc-item"><li><span><a href="#Missclassification-Error" data-toc-modified-id="Missclassification-Error-3.1.1"><span class="toc-item-num">3.1.1&nbsp;&nbsp;</span>Missclassification Error</a></span></li><li><span><a href="#Accuracy-Score" data-toc-modified-id="Accuracy-Score-3.1.2"><span class="toc-item-num">3.1.2&nbsp;&nbsp;</span>Accuracy Score</a></span></li><li><span><a href="#Classificatoin-Report" data-toc-modified-id="Classificatoin-Report-3.1.3"><span class="toc-item-num">3.1.3&nbsp;&nbsp;</span>Classificatoin Report</a></span></li><li><span><a href="#Confusion-Matrix" data-toc-modified-id="Confusion-Matrix-3.1.4"><span class="toc-item-num">3.1.4&nbsp;&nbsp;</span>Confusion Matrix</a></span></li></ul></li><li><span><a href="#KD-tree-KNN" data-toc-modified-id="KD-tree-KNN-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>KD tree KNN</a></span><ul class="toc-item"><li><span><a href="#Misclassification-Error" data-toc-modified-id="Misclassification-Error-3.2.1"><span class="toc-item-num">3.2.1&nbsp;&nbsp;</span>Misclassification Error</a></span></li><li><span><a href="#Accuracy-Score" data-toc-modified-id="Accuracy-Score-3.2.2"><span class="toc-item-num">3.2.2&nbsp;&nbsp;</span>Accuracy Score</a></span></li><li><span><a href="#Classificatoin-Report" data-toc-modified-id="Classificatoin-Report-3.2.3"><span class="toc-item-num">3.2.3&nbsp;&nbsp;</span>Classificatoin Report</a></span></li><li><span><a href="#Confusion-Matrix" data-toc-modified-id="Confusion-Matrix-3.2.4"><span class="toc-item-num">3.2.4&nbsp;&nbsp;</span>Confusion Matrix</a></span></li></ul></li></ul></li><li><span><a href="#Word2Vec" data-toc-modified-id="Word2Vec-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Word2Vec</a></span></li><li><span><a href="#Avg-Word2Vec" data-toc-modified-id="Avg-Word2Vec-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Avg Word2Vec</a></span><ul class="toc-item"><li><span><a href="#Brute-force-KNN" data-toc-modified-id="Brute-force-KNN-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Brute force KNN</a></span><ul class="toc-item"><li><span><a href="#Misclassification-Error" data-toc-modified-id="Misclassification-Error-5.1.1"><span class="toc-item-num">5.1.1&nbsp;&nbsp;</span>Misclassification Error</a></span></li><li><span><a href="#Accuracy-Score" data-toc-modified-id="Accuracy-Score-5.1.2"><span class="toc-item-num">5.1.2&nbsp;&nbsp;</span>Accuracy Score</a></span></li><li><span><a href="#Classification-Report" data-toc-modified-id="Classification-Report-5.1.3"><span class="toc-item-num">5.1.3&nbsp;&nbsp;</span>Classification Report</a></span></li><li><span><a href="#Confusion-Matrix" data-toc-modified-id="Confusion-Matrix-5.1.4"><span class="toc-item-num">5.1.4&nbsp;&nbsp;</span>Confusion Matrix</a></span></li></ul></li><li><span><a href="#KD-tree-KNN" data-toc-modified-id="KD-tree-KNN-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>KD tree KNN</a></span><ul class="toc-item"><li><span><a href="#Misclassification-Error" data-toc-modified-id="Misclassification-Error-5.2.1"><span class="toc-item-num">5.2.1&nbsp;&nbsp;</span>Misclassification Error</a></span></li><li><span><a href="#Accuracy-Score" data-toc-modified-id="Accuracy-Score-5.2.2"><span class="toc-item-num">5.2.2&nbsp;&nbsp;</span>Accuracy Score</a></span></li><li><span><a href="#Classification-Report" data-toc-modified-id="Classification-Report-5.2.3"><span class="toc-item-num">5.2.3&nbsp;&nbsp;</span>Classification Report</a></span></li><li><span><a href="#Confusion-Matrix" data-toc-modified-id="Confusion-Matrix-5.2.4"><span class="toc-item-num">5.2.4&nbsp;&nbsp;</span>Confusion Matrix</a></span></li></ul></li></ul></li><li><span><a href="#TFIDF-Word2Vec" data-toc-modified-id="TFIDF-Word2Vec-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>TFIDF-Word2Vec</a></span><ul class="toc-item"><li><span><a href="#Brute-force-KNN" data-toc-modified-id="Brute-force-KNN-6.1"><span class="toc-item-num">6.1&nbsp;&nbsp;</span>Brute force KNN</a></span><ul class="toc-item"><li><span><a href="#Misclassification-Error" data-toc-modified-id="Misclassification-Error-6.1.1"><span class="toc-item-num">6.1.1&nbsp;&nbsp;</span>Misclassification Error</a></span></li><li><span><a href="#Accuracy-Score" data-toc-modified-id="Accuracy-Score-6.1.2"><span class="toc-item-num">6.1.2&nbsp;&nbsp;</span>Accuracy Score</a></span></li><li><span><a href="#Classification-Report" data-toc-modified-id="Classification-Report-6.1.3"><span class="toc-item-num">6.1.3&nbsp;&nbsp;</span>Classification Report</a></span></li><li><span><a href="#Confusion-Matrix" data-toc-modified-id="Confusion-Matrix-6.1.4"><span class="toc-item-num">6.1.4&nbsp;&nbsp;</span>Confusion Matrix</a></span></li></ul></li><li><span><a href="#KD-tree-KNN" data-toc-modified-id="KD-tree-KNN-6.2"><span class="toc-item-num">6.2&nbsp;&nbsp;</span>KD tree KNN</a></span><ul class="toc-item"><li><span><a href="#Misclassification-Error" data-toc-modified-id="Misclassification-Error-6.2.1"><span class="toc-item-num">6.2.1&nbsp;&nbsp;</span>Misclassification Error</a></span></li><li><span><a href="#Accuracy-Score" data-toc-modified-id="Accuracy-Score-6.2.2"><span class="toc-item-num">6.2.2&nbsp;&nbsp;</span>Accuracy Score</a></span></li><li><span><a href="#Classification-Report" data-toc-modified-id="Classification-Report-6.2.3"><span class="toc-item-num">6.2.3&nbsp;&nbsp;</span>Classification Report</a></span></li><li><span><a href="#Confusion-Matrix" data-toc-modified-id="Confusion-Matrix-6.2.4"><span class="toc-item-num">6.2.4&nbsp;&nbsp;</span>Confusion Matrix</a></span></li></ul></li></ul></li><li><span><a href="#Final-Report" data-toc-modified-id="Final-Report-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Final Report</a></span></li><li><span><a href="#Conclusion" data-toc-modified-id="Conclusion-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Conclusion</a></span></li></ul></div>

# # KNN
# 
# ## Introduction
# 
# ## Objective:
# - Get Amazon Fine Foor Review dataset and prepare 4 categories of datasets: 
#       (i) BoW, 
#       (ii) TF-IDF, 
#       (iii) Word2Vec, 
#       (iv) TFIDF-W2V
#       
# - Apply following on all of the above:
#         1. Split the data into train data(70%) and test data(30%) using timebased slicing.
#         2. Perform 10 fold cross validation to find optimal K in KNN.
#         3. Apply KNN, both: bruteforce and kd-tree.
#         4. Report: Accuracy score with best K, f1 measure, Confusion Matrix

# In[ ]:


import warnings
warnings.filterwarnings("ignore")
import sqlite3
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import numpy as np
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
import seaborn as sb
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import gensim
from gensim.models import Word2Vec, KeyedVectors


# In[ ]:


# Connection to the dataset
con = sqlite3.connect('../input/database.sqlite')

# It is given that the table name is 'Reviews'
# Creating pandas dataframe and storing into variable 'dataset' by help of sql query
dataset = pd.read_sql_query("""
SELECT *
FROM Reviews
""", con)

# Getting the shape of actual data: row, column
display(dataset.shape)


# In[ ]:


# Displaying first 5 data points
display(dataset.head())


# In[ ]:


# Considering only those reviews which score is either 1,2 or 4,5
# Since, 3 is kind of neutral review, so, we are eliminating it
filtered_data = pd.read_sql_query("""
SELECT *
FROM Reviews
WHERE Score != 3
""", con)


# In[ ]:


# Getting shape of new dataset
display(filtered_data.shape)


# In[ ]:


# Changing the scores into 'positive' or 'negative'
# Score greater that 3 is considered as 'positive' and less than 3 is 'negative'
def partition(x):
    if x>3:
        return 'positive'
    return 'negative'

actual_score = filtered_data['Score']
positiveNegative = actual_score.map(partition)
filtered_data['Score'] = positiveNegative


# In[ ]:


# Sorting data points according to the 'ProductId'
sorted_data = filtered_data.sort_values('ProductId', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')

# Eliminating the duplicate data points based on: 'UserId', 'ProfileName', 'Time', 'Summary'
final = sorted_data.drop_duplicates(subset={'UserId', 'ProfileName', 'Time', 'Summary'}, keep='first', inplace=False)

# Eliminating the row where 'HelpfulnessDenominator' is greater than 'HelpfulnessNumerator' as these are the wrong entry
final = final[final['HelpfulnessDenominator'] >= final['HelpfulnessNumerator']]

# Getting shape of final data frame
display(final.shape)


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Creating the set of stopwords\nstop = set(stopwords.words(\'english\'))\n\n# For stemming purpose\nsnow = nltk.stem.SnowballStemmer(\'english\')\n\n# Defining function to clean html tags\ndef cleanhtml(sentence):\n    cleaner = re.compile(\'<.*>\')\n    cleantext = re.sub(cleaner, \' \', sentence)\n    return cleantext\n\n# Defining function to remove special symbols\ndef cleanpunc(sentence):\n    cleaned = re.sub(r\'[?|.|!|*|@|#|\\\'|"|,|)|(|\\|/]\', r\'\', sentence)\n    return cleaned\n\n\n# Important steps to clean the text data. Please trace it out carefully\ni = 0\nstr1 = \'\'\nall_positive_words = []\nall_negative_words = []\nfinal_string = []\ns=\'\'\nfor sent in final[\'Text\'].values:\n    filtered_sentence = []\n    sent = cleanhtml(sent)\n    for w in sent.split():\n        for cleaned_words in cleanpunc(w).split():\n            if ((cleaned_words.isalpha()) & (len(cleaned_words)>2)):\n                if (cleaned_words.lower() not in stop):\n                    s = (snow.stem(cleaned_words.lower())).encode(\'utf-8\')\n                    filtered_sentence.append(s)\n                    if (final[\'Score\'].values)[i] == \'positive\':\n                        all_positive_words.append(s)\n                    if (final[\'Score\'].values)[i] == \'negative\':\n                        all_negative_words.append(s)\n                else:\n                    continue\n            else:\n                continue\n    str1 = b" ".join(filtered_sentence)\n    final_string.append(str1)\n    i += 1\n    \n# Adding new column into dataframe to store cleaned text\nfinal[\'CleanedText\'] = final_string\nfinal[\'CleanedText\'] = final[\'CleanedText\'].str.decode(\'utf-8\')\n\n# Creating new dataset with cleaned text for future use\nconn = sqlite3.connect(\'final.sqlite\')\nc = conn.cursor()\nconn.text_factory = str\nfinal.to_sql(\'Reviews\', conn, schema=None, if_exists=\'replace\', index=True, index_label=None, chunksize=None, dtype=None)\n\n# Getting shape of new datset\nprint(final.shape)')


# In[ ]:


# Creating connection to read from database
conn = sqlite3.connect('./final.sqlite')

# Creating data frame for visualization using sql query
final = pd.read_sql_query("""
SELECT *
FROM Reviews
""", conn)


# In[ ]:


# Displaying first 3 indices
display(final.head(3))


# # BoW

# In[ ]:


# Sampling positive and negative reviews
positive_points = final[final['Score'] == 'positive'].sample(
    n=10000, random_state=0)
negative_points = final[final['Score'] == 'negative'].sample(
    n=10000, random_state=0)
total_points = pd.concat([positive_points, negative_points])

# Sorting based on time
total_points['Time'] = pd.to_datetime(
    total_points['Time'], origin='unix', unit='s')
total_points = total_points.sort_values('Time')
sample_points = total_points['CleanedText']
labels = total_points['Score']#.map(lambda x: 1 if x == 'positive' else 0).values


# In[ ]:


# Splitting into train and test
X_train, X_test, Y_train, Y_test = train_test_split(
    sample_points, labels, test_size=0.30, random_state=0)


# In[ ]:


count_vect = CountVectorizer(ngram_range=(1, 1))
X_train = count_vect.fit_transform(X_train)
X_test = count_vect.transform(X_test)


# ## Brute force KNN

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nneighbors = list(range(20, 80, 4))\ncv_score = []\nfor k in neighbors:\n    knn = KNeighborsClassifier(n_neighbors=k, algorithm='brute')\n    scores = cross_val_score(knn, X_train, Y_train, cv=10, scoring='accuracy')\n    cv_score.append(scores.mean())")


# ### Missclassification Error

# In[ ]:


MSE = [1 - x for x in cv_score]
optimal_k = neighbors[MSE.index(min(MSE))]
print("_" * 101)
print("Optimal number of neighbors: ", optimal_k)
print("_" * 101)
print("Missclassification error for each k values: ", np.round(MSE, 3))
print("_" * 101)

plt.plot(neighbors, MSE)
plt.title("Number of neighbors and error")
plt.xlabel("Number of neighbors")
plt.ylabel("Missclassification error")
plt.show()


# ### Accuracy Score

# In[ ]:


get_ipython().run_cell_magic('time', '', 'optimal_model = KNeighborsClassifier(n_neighbors=optimal_k)\noptimal_model.fit(X_train, Y_train)\nprediction = optimal_model.predict(X_test)\n\ntraining_accuracy = optimal_model.score(X_train, Y_train)\ntraining_error = 1 - training_accuracy\ntest_accuracy = accuracy_score(Y_test, prediction)\ntest_error = 1 - test_accuracy\n\nprint("_" * 101)\nprint("Training Accuracy: ", training_accuracy)\nprint("Train Error: ", training_error)\nprint("Test Accuracy: ", test_accuracy)\nprint("Test Error: ", test_error)\nprint("_" * 101)')


# ### Classificatoin Report

# In[ ]:


print("_" * 101)
print("Classification Report: \n")
print(classification_report(Y_test, prediction))
print("_" * 101)


# ###  Confusion Matrix

# In[ ]:


conf_matrix = confusion_matrix(Y_test, prediction)
class_label = ['negative', 'positive']
df_conf_matrix = pd.DataFrame(
    conf_matrix, index=class_label, columns=class_label)
sb.heatmap(df_conf_matrix, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
print("_" * 101)


# ## KD tree KNN

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nneighbors = list(range(20, 80, 4))\ncv_score = []\nfor k in neighbors:\n    knn = KNeighborsClassifier(n_neighbors=k, algorithm='kd_tree')\n    scores = cross_val_score(knn, X_train, Y_train, cv=10, scoring='accuracy')\n    cv_score.append(scores.mean())")


# ### Misclassification Error

# In[ ]:


MSE = [1 - x for x in cv_score]
optimal_k = neighbors[MSE.index(min(MSE))]
print("_" * 101)
print("Optimal number of neighbors: ", optimal_k)
print("_" * 101)
print("Missclassification error for each k values: ", np.round(MSE, 3))
print("_" * 101)

plt.plot(neighbors, MSE)
#for xy in zip(neighbors, np.round(MSE, 3)):
#    plt.annotate("%s %s" %xy, xy=xy, textcoords='data')
plt.title("Number of neighbors and error")
plt.xlabel("Number of neighbors")
plt.ylabel("Missclassification error")
plt.show()


# ### Accuracy Score

# In[ ]:


get_ipython().run_cell_magic('time', '', '\noptimal_model = KNeighborsClassifier(\n    n_neighbors=optimal_k, algorithm=\'kd_tree\')\noptimal_model.fit(X_train, Y_train)\nprediction = optimal_model.predict(X_test)\n\ntraining_accuracy = optimal_model.score(X_train, Y_train)\ntraining_error = 1 - training_accuracy\ntest_accuracy = accuracy_score(Y_test, prediction)\ntest_error = 1 - test_accuracy\n\nprint("_" * 101)\nprint("Training Accuracy: ", training_accuracy)\nprint("Train Error: ", training_error)\nprint("Test Accuracy: ", test_accuracy)\nprint("Test Error: ", test_error)\nprint("_" * 101)')


# ### Classification Report

# In[ ]:


print("Classification Report: \n")
print(classification_report(Y_test, prediction))


# ### Confusion Matrix

# In[ ]:


conf_matrix = confusion_matrix(Y_test, prediction)
class_label = ['negative', 'positive']
df_conf_matrix = pd.DataFrame(
    conf_matrix, index=class_label, columns=class_label)
sb.heatmap(df_conf_matrix, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
print("_" * 101)


# # TF-IDF

# In[ ]:


# Splitting into train and test
X_train, X_test, Y_train, Y_test = train_test_split(
    sample_points, labels, test_size=0.30)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


# In[ ]:


# Initializing tfidf vectorizer
tfidf_vect = TfidfVectorizer(ngram_range=(1, 1))

# Fitting for tfidf vectorization
X_train = tfidf_vect.fit_transform(X_train)
X_test = tfidf_vect.transform(X_test)

print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


# ## Brute force KNN

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nneighbors = list(range(20, 80, 4))\ncv_score = []\nfor k in neighbors:\n    knn = KNeighborsClassifier(n_neighbors=k, algorithm='brute')\n    scores = cross_val_score(knn, X_train, Y_train, cv=10, scoring='accuracy')\n    cv_score.append(scores.mean())")


# ### Missclassification Error

# In[ ]:


MSE = [1 - x for x in cv_score]
optimal_k = neighbors[MSE.index(min(MSE))]
print("_" * 101)
print("Optimal number of neighbors: ", optimal_k)
print("_" * 101)
print("Missclassification error for each k values: ", np.round(MSE, 3))
print("_" * 101)

plt.plot(neighbors, MSE)
#for xy in zip(neighbors, np.round(MSE, 3)):
#    plt.annotate("%s %s" %xy, xy=xy, textcoords='data')
plt.title("Number of neighbors and error")
plt.xlabel("Number of neighbors")
plt.ylabel("Missclassification error")
plt.show()


# ### Accuracy Score

# In[ ]:


get_ipython().run_cell_magic('time', '', '\noptimal_model = KNeighborsClassifier(\n    n_neighbors=optimal_k, algorithm=\'kd_tree\')\noptimal_model.fit(X_train, Y_train)\nprediction = optimal_model.predict(X_test)\n\ntraining_accuracy = optimal_model.score(X_train, Y_train)\ntraining_error = 1 - training_accuracy\ntest_accuracy = accuracy_score(Y_test, prediction)\ntest_error = 1 - test_accuracy\n\nprint("_" * 101)\nprint("Training Accuracy: ", training_accuracy)\nprint("Train Error: ", training_error)\nprint("Test Accuracy: ", test_accuracy)\nprint("Test Error: ", test_error)\nprint("_" * 101)')


# ### Classificatoin Report

# In[ ]:


print("_" * 101)
print("Classification Report: \n")
print(classification_report(Y_test, prediction))
print("_" * 101)


# ### Confusion Matrix

# In[ ]:


conf_matrix = confusion_matrix(Y_test, prediction)
class_label = ['negative', 'positive']
df_conf_matrix = pd.DataFrame(
    conf_matrix, index=class_label, columns=class_label)
sb.heatmap(df_conf_matrix, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# ## KD tree KNN

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nneighbors = list(range(20, 80, 4))\ncv_score = []\nfor k in neighbors:\n    knn = KNeighborsClassifier(n_neighbors=k, algorithm='kd_tree')\n    scores = cross_val_score(knn, X_train, Y_train, cv=10, scoring='accuracy')\n    cv_score.append(scores.mean())")


# ### Misclassification Error

# In[ ]:


MSE = [1 - x for x in cv_score]
optimal_k = neighbors[MSE.index(min(MSE))]
print("_" * 101)
print("Optimal number of neighbors: ", optimal_k)
print("_" * 101)
print("Missclassification error for each k values: ", np.round(MSE, 3))
print("_" * 101)

plt.plot(neighbors, MSE)
#for xy in zip(neighbors, np.round(MSE, 3)):
#    plt.annotate("%s %s" %xy, xy=xy, textcoords='data')
plt.title("Number of neighbors and error")
plt.xlabel("Number of neighbors")
plt.ylabel("Missclassification error")
plt.show()


# ### Accuracy Score

# In[ ]:


get_ipython().run_cell_magic('time', '', '\noptimal_model = KNeighborsClassifier(\n    n_neighbors=optimal_k, algorithm=\'kd_tree\')\noptimal_model.fit(X_train, Y_train)\nprediction = optimal_model.predict(X_test)\n\ntraining_accuracy = optimal_model.score(X_train, Y_train)\ntraining_error = 1 - training_accuracy\ntest_accuracy = accuracy_score(Y_test, prediction)\ntest_error = 1 - test_accuracy\n\nprint("_" * 101)\nprint("Training Accuracy: ", training_accuracy)\nprint("Train Error: ", training_error)\nprint("Test Accuracy: ", test_accuracy)\nprint("Test Error: ", test_error)\nprint("_" * 101)')


# ### Classificatoin Report

# In[ ]:


print("_" * 101)
print("Classification Report: \n")
print(classification_report(Y_test, prediction))
print("_" * 101)


# ### Confusion Matrix

# In[ ]:


conf_matrix = confusion_matrix(Y_test, prediction)
class_label = ['negative', 'positive']
df_conf_matrix = pd.DataFrame(
    conf_matrix, index=class_label, columns=class_label)
sb.heatmap(df_conf_matrix, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
print("_" * 101)


# # Word2Vec

# In[ ]:


sample_points = total_points['Text']
#labels = total_points['Score']
X_train, X_test, Y_train, Y_test = train_test_split(
    sample_points, labels, test_size=0.3, random_state=0)


# In[ ]:


import re

def cleanhtml(sentence):
    cleantext = re.sub('<.*>', '', sentence)
    return cleantext

def cleanpunc(sentence):
    cleaned = re.sub(r'[?|!|\'|#|@|.|,|)|(|\|/]', r'', sentence)
    return cleaned


# In[ ]:


train_sent_list = []
for sent in X_train:
    train_sentence = []
    sent = cleanhtml(sent)
    for w in sent.split():
        for cleaned_words in cleanpunc(w).split():
            if (cleaned_words.isalpha()):
                train_sentence.append(cleaned_words.lower())
            else:
                continue
    train_sent_list.append(train_sentence)


# In[ ]:


test_sent_list = []
for sent in X_test:
    train_sentence = []
    sent = cleanhtml(sent)
    for w in sent.split():
        for cleaned_words in cleanpunc(w).split():
            if (cleaned_words.isalpha()):
                train_sentence.append(cleaned_words.lower())
            else:
                continue
    test_sent_list.append(train_sentence)


# In[ ]:


train_w2v_model = gensim.models.Word2Vec(
    train_sent_list, min_count=5, size=50, workers=4)
train_w2v_words = train_w2v_model[train_w2v_model.wv.vocab]


# In[ ]:


test_w2v_model = gensim.models.Word2Vec(
    test_sent_list, min_count=5, size=50, workers=4)
test_w2v_words = test_w2v_model[test_w2v_model.wv.vocab]


# In[ ]:


print(train_w2v_words.shape, test_w2v_words.shape)


# # Avg Word2Vec

# In[ ]:


import numpy as np
train_vectors = []
for sent in train_sent_list:
    sent_vec = np.zeros(50)
    cnt_words = 0
    for word in sent:
        try:
            vec = train_w2v_model.wv[word]
            sent_vec += vec
            cnt_words += 1
        except:
            pass
    sent_vec /= cnt_words
    train_vectors.append(sent_vec)
train_vectors = np.nan_to_num(train_vectors)


# In[ ]:


test_vectors = []
for sent in test_sent_list:
    sent_vec = np.zeros(50)
    cnt_words = 0
    for word in sent:
        try:
            vec = test_w2v_model.wv[word]
            sent_vec += vec
            cnt_words += 1
        except:
            pass
    sent_vec /= cnt_words
    test_vectors.append(sent_vec)
test_vectors = np.nan_to_num(test_vectors)


# In[ ]:


X_train = train_vectors
X_test = test_vectors


# ## Brute force KNN

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nneighbors = list(range(20, 50, 2))\ncv_score = []\nfor k in neighbors:\n    knn = KNeighborsClassifier(n_neighbors=k, algorithm='brute')\n    scores = cross_val_score(knn, X_train, Y_train, cv=10, scoring='accuracy')\n    cv_score.append(scores.mean())")


# ### Misclassification Error

# In[ ]:


MSE = [1 - x for x in cv_score]
optimal_k = neighbors[MSE.index(min(MSE))]
print("_" * 101)
print("Optimal number of neighbors: ", optimal_k)
print("_" * 101)
print("Missclassification error for each k values: ", np.round(MSE, 3))
print("_" * 101)

plt.plot(neighbors, MSE)
#for xy in zip(neighbors, np.round(MSE, 3)):
#    plt.annotate("%s %s" %xy, xy=xy, textcoords='data')
plt.title("Number of neighbors and error")
plt.xlabel("Number of neighbors")
plt.ylabel("Missclassification error")
plt.show()


# ### Accuracy Score

# In[ ]:


get_ipython().run_cell_magic('time', '', '\noptimal_model = KNeighborsClassifier(\n    n_neighbors=optimal_k, algorithm=\'kd_tree\')\noptimal_model.fit(X_train, Y_train)\nprediction = optimal_model.predict(X_test)\n\ntraining_accuracy = optimal_model.score(X_train, Y_train)\ntraining_error = 1 - training_accuracy\ntest_accuracy = accuracy_score(Y_test, prediction)\ntest_error = 1 - test_accuracy\n\nprint("_" * 101)\nprint("Training Accuracy: ", training_accuracy)\nprint("Train Error: ", training_error)\nprint("Test Accuracy: ", test_accuracy)\nprint("Test Error: ", test_error)\nprint("_" * 101)')


# ### Classification Report

# In[ ]:


print("_" * 101)
print("Classification Report: \n")
print(classification_report(Y_test, prediction))
print("_" * 101)


# ### Confusion Matrix

# In[ ]:


conf_matrix = confusion_matrix(Y_test, prediction)
class_label = ['negative', 'positive']
df_conf_matrix = pd.DataFrame(
    conf_matrix, index=class_label, columns=class_label)
sb.heatmap(df_conf_matrix, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
print("_" * 101)


# - After trying out lot of hyper parameter tuning, model is always overfitting. It clearly indicates that, the model Brute Force KNN is bias towards Average Word2Vec in our Amazon Fine Food Reviews data set.

# ## KD tree KNN

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nneighbors = list(range(20, 50, 4))\ncv_score = []\nfor k in neighbors:\n    knn = KNeighborsClassifier(n_neighbors=k, algorithm='kd_tree')\n    scores = cross_val_score(knn, X_train, Y_train, cv=10, scoring='accuracy')\n    cv_score.append(scores.mean())")


# ### Misclassification Error

# In[ ]:


MSE = [1 - x for x in cv_score]
optimal_k = neighbors[MSE.index(min(MSE))]
print("_" * 101)
print("Optimal number of neighbors: ", optimal_k)
print("_" * 101)
print("Missclassification error for each k values: ", np.round(MSE, 3))
print("_" * 101)

plt.plot(neighbors, MSE)
#for xy in zip(neighbors, np.round(MSE, 3)):
#    plt.annotate("%s %s" %xy, xy=xy, textcoords='data')
plt.title("Number of neighbors and error")
plt.xlabel("Number of neighbors")
plt.ylabel("Missclassification error")
plt.show()


# ### Accuracy Score

# In[ ]:


get_ipython().run_cell_magic('time', '', '\noptimal_model = KNeighborsClassifier(\n    n_neighbors=optimal_k, algorithm=\'kd_tree\')\noptimal_model.fit(X_train, Y_train)\nprediction = optimal_model.predict(X_test)\n\ntraining_accuracy = optimal_model.score(X_train, Y_train)\ntraining_error = 1 - training_accuracy\ntest_accuracy = accuracy_score(Y_test, prediction)\ntest_error = 1 - test_accuracy\n\nprint("_" * 101)\nprint("Training Accuracy: ", training_accuracy)\nprint("Train Error: ", training_error)\nprint("Test Accuracy: ", test_accuracy)\nprint("Test Error: ", test_error)\nprint("_" * 101)')


# ### Classification Report

# In[ ]:


print("_" * 101)
print("Classification Report: \n")
print(classification_report(Y_test, prediction))
print("_" * 101)


# ### Confusion Matrix

# In[ ]:


conf_matrix = confusion_matrix(Y_test, prediction)
class_label = ['negative', 'positive']
df_conf_matrix = pd.DataFrame(
    conf_matrix, index=class_label, columns=class_label)
sb.heatmap(df_conf_matrix, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
print("_" * 101)


# - After trying out lot of hyper parameter tuning, model is always overfitting. It clearly indicates that, the model  KD Tree KNN is bias towards Average Word2Vec in our Amazon Fine Food Reviews data set.

# # TFIDF-Word2Vec

# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(
    sample_points, labels, test_size=0.3)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


# In[ ]:


tfidf_vect = TfidfVectorizer(ngram_range=(1, 1))
train_tfidf_w2v = tfidf_vect.fit_transform(X_train)
test_tfidf_w2v = tfidf_vect.transform(X_test)
print(train_tfidf_w2v.shape, test_tfidf_w2v.shape)


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ntfidf_feat = tfidf_vect.get_feature_names()\ntrain_tfidf_w2v_vectors = []\nrow = 0\nfor sent in train_sent_list:\n    sent_vec = np.zeros(50)\n    weight_sum = 0\n    for word in sent:\n        if word in train_w2v_words:\n            vec = train_w2v_model.wv[word]\n            tf_idf = train_tfidf_w2v[row, tfidf_feat.index(word)]\n            sent_vec += (vec * tf_idf)\n            weight_sum += tf_idf\n    if weight_sum != 0:\n        sent_vec /= weight_sum\n    train_tfidf_w2v_vectors.append(sent_vec)\n    row += 1')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ntfidf_feat = tfidf_vect.get_feature_names()\ntest_tfidf_w2v_vectors = []\nrow = 0\nfor sent in test_sent_list:\n    sent_vec = np.zeros(50)\n    weighted_sum = 0\n    for word in sent:\n        if word in test_w2v_words:\n            vec = test_w2v_model[word]\n            tf_idf = test_tfidf_w2v[row, tfidf_feat.index(word)]\n            sent_vec += (vec * tf_idf)\n            weight_sum += tf_idf\n    if weight_sum != 0:\n        sent_vec /= weight_sum\n    test_tfidf_w2v_vectors.append(sent_vec)\n    row += 1')


# In[ ]:


X_train = train_tfidf_w2v_vectors
X_test = test_tfidf_w2v_vectors


# ## Brute force KNN

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nneighbors = list(range(1, 50, 2))\ncv_score = []\nfor k in neighbors:\n    knn = KNeighborsClassifier(n_neighbors=k, algorithm='brute')\n    scores = cross_val_score(knn, X_train, Y_train, cv=10, scoring='accuracy')\n    cv_score.append(scores.mean())")


# ### Misclassification Error

# In[ ]:


MSE = [1 - x for x in cv_score]
optimal_k = neighbors[MSE.index(min(MSE))]
print("_" * 101)
print("Optimal number of neighbors: ", optimal_k)
print("_" * 101)
print("Missclassification error for each k values: ", np.round(MSE, 3))
print("_" * 101)

plt.plot(neighbors, MSE)
#for xy in zip(neighbors, np.round(MSE, 3)):
#    plt.annotate("%s %s" %xy, xy=xy, textcoords='data')
plt.title("Number of neighbors and error")
plt.xlabel("Number of neighbors")
plt.ylabel("Missclassification error")
plt.show()


# ### Accuracy Score

# In[ ]:


get_ipython().run_cell_magic('time', '', '\noptimal_model = KNeighborsClassifier(\n    n_neighbors=optimal_k, algorithm=\'kd_tree\')\noptimal_model.fit(X_train, Y_train)\nprediction = optimal_model.predict(X_test)\n\ntraining_accuracy = optimal_model.score(X_train, Y_train)\ntraining_error = 1 - training_accuracy\ntest_accuracy = accuracy_score(Y_test, prediction)\ntest_error = 1 - test_accuracy\n\nprint("_" * 101)\nprint("Training Accuracy: ", training_accuracy)\nprint("Train Error: ", training_error)\nprint("Test Accuracy: ", test_accuracy)\nprint("Test Error: ", test_error)\nprint("_" * 101)')


# ### Classification Report

# In[ ]:


print("_" * 101)
print("Classification Report: \n")
print(classification_report(Y_test, prediction))
print("_" * 101)


# ### Confusion Matrix

# In[ ]:


conf_matrix = confusion_matrix(Y_test, prediction)
class_label = ['negative', 'positive']
df_conf_matrix = pd.DataFrame(
    conf_matrix, index=class_label, columns=class_label)
sb.heatmap(df_conf_matrix, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
print("_" * 101)


# - After trying out lot of hyper parameter tuning, model is always bias towards only one class. It clearly indicates that, the model  Brute Ford KNN is not suitable for TFIDF-Word2Vec in our Amazon Fine Food Reviews data set.

# ## KD tree KNN

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nneighbors = list(range(1, 50, 2))\ncv_score = []\nfor k in neighbors:\n    knn = KNeighborsClassifier(n_neighbors=k, algorithm='kd_tree')\n    scores = cross_val_score(knn, X_train, Y_train, cv=10, scoring='accuracy')\n    cv_score.append(scores.mean())")


# ### Misclassification Error

# In[ ]:


MSE = [1 - x for x in cv_score]
optimal_k = neighbors[MSE.index(min(MSE))]
print("_" * 101)
print("Optimal number of neighbors: ", optimal_k)
print("_" * 101)
print("Missclassification error for each k values: ", np.round(MSE, 3))
print("_" * 101)

plt.plot(neighbors, MSE)
#for xy in zip(neighbors, np.round(MSE, 3)):
#    plt.annotate("%s %s" %xy, xy=xy, textcoords='data')
plt.title("Number of neighbors and error")
plt.xlabel("Number of neighbors")
plt.ylabel("Missclassification error")
plt.show()


# ### Accuracy Score

# In[ ]:


get_ipython().run_cell_magic('time', '', '\noptimal_model = KNeighborsClassifier(\n    n_neighbors=optimal_k, algorithm=\'kd_tree\')\noptimal_model.fit(X_train, Y_train)\nprediction = optimal_model.predict(X_test)\n\ntraining_accuracy = optimal_model.score(X_train, Y_train)\ntraining_error = 1 - training_accuracy\ntest_accuracy = accuracy_score(Y_test, prediction)\ntest_error = 1 - test_accuracy\n\nprint("_" * 101)\nprint("Training Accuracy: ", training_accuracy)\nprint("Train Error: ", training_error)\nprint("Test Accuracy: ", test_accuracy)\nprint("Test Error: ", test_error)\nprint("_" * 101)')


# ### Classification Report

# In[ ]:


print("_" * 101)
print("Classification Report: \n")
print(classification_report(Y_test, prediction))
print("_" * 101)


# ### Confusion Matrix

# In[ ]:


conf_matrix = confusion_matrix(Y_test, prediction)
class_label = ['negative', 'positive']
df_conf_matrix = pd.DataFrame(
    conf_matrix, index=class_label, columns=class_label)
sb.heatmap(df_conf_matrix, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
print("_" * 101)


# - After trying out lot of hyper parameter tuning, model is always bias towards only one class. It clearly indicates that, the model KD Tree KNN is not suitable for TFIDF-Word2Vec in our Amazon Fine Food Reviews data set.

# # Final Report

# - Accuracy Report in %:
# <table>
#    <tbody>
#     <tr>      
#       <th><center> </center></th>
#       <th colspan='4'><center> Brute Force KNN </center></th>
#       <th colspan='4'><center> KD Tree KNN </center></th>      
#     </tr>
#   
#   <tr>
#     <td> </td>
#     <td rowspan="1"><b><center> Train Accuracy </center></b></td>    
#     <td rowspan="1"><b><center> Train Error </center></b></td>
#     <td rowspan="1"><b><center> Test Accuracy </center></b></td>
#     <td rowspan="1"><b><center> Test Error </center></b></td>
#     <td rowspan="1"><b><center> Train Accuracy </center></b></td>    
#     <td rowspan="1"><b><center>Train  Error </center></b></td>
#     <td rowspan="1"><b><center> Test Accuracy </center></b></td>
#     <td rowspan="1"><b><center> Test Error </center></b></td>
#   </tr>
#   <tr>
#     <td><b><center> Bow </center></b></td>
#     <td><center> 72.65 </center></td>
#     <td><center> 27.34</center></td>
#     <td><center> 70.65 </center></td>
#     <td><center> 29.35 </center></td>
# <td><center> 72.65 </center></td>
#     <td><center> 27.34</center></td>
#     <td><center> 70.65 </center></td>
#     <td><center> 29.35 </center></td>
#   </tr>
# <tr>
#     <td><b><center> TF-IDF </center></b></td>
#     <td><center> 73.37 </center></td>
#     <td><center> 26.62 </center></td>
#     <td><center> 71.80 </center></td>
#     <td><center> 28.20 </center></td>
#    <td><center> 73.37 </center></td>
#     <td><center> 26.62 </center></td>
#     <td><center> 71.80 </center></td>
#     <td><center> 28.20 </center></td>
#   </tr>
# <tr>
#     <td><b><center> Avg-W2V </center></b></td>
#     <td><center> 75.00 </center></td>
#     <td><center> 25.00 </center></td>
#     <td><center> 57.91 </center></td>
#     <td><center> 42.08 </center></td>
#      <td><center> 75.00 </center></td>
#     <td><center> 25.00 </center></td>
#     <td><center> 57.91 </center></td>
#     <td><center> 42.08 </center></td>
#   </tr>
# <tr>
#     <td><b><center> TFIDF-W2V </center></b></td>
#     <td><center> 50.20 </center></td>
#     <td><center> 49.79 </center></td>
#     <td><center> 49.51 </center></td>
#     <td><center> 50.48 </center></td>
#     <td><center> 50.20 </center></td>
#     <td><center> 49.79 </center></td>
#     <td><center> 49.51 </center></td>
#     <td><center> 50.48 </center></td>
#   </tr>
#   </tbody>
# </table>
# 

# # Conclusion

# - After executing models and hyper parameter tuning, it is found out that, KNN is giving train and test accuracy around 70% in for both W2V and TF-IDF.
# - As K increased, training accuracy always increases but test accuracy is limited around 70% so it's obvious to adjust the K properly so that model doesn't over fit.
# - Model is always over fitting for Average W2V dataset independent of k value so is not suitable in this case.
# - Model is always lenient towards one class for TFIDF-W2V so clearly it's not suitable in this case too.
# - It is very time consuming to have data with very high dimension. Very small subset were taken but still we are getting high delay due to large dimension and time complexity of KNN.
