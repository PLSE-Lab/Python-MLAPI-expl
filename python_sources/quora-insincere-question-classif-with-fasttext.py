#!/usr/bin/env python
# coding: utf-8

# <center><h1><b>Quora - Detecting Insincere Questions</b></h1></center>

# <center><h4><b>Dejan Porjazovski and Donayorov Maksad</b></h4></center>

# # Introduction
# 
# This notebook aims to show an approach on how to solve an interesting problem, which resolves around classifying insincere questions from the famous platform Quora. The task is a binary classification problem. Such tasks are implemented in many applications that we use in our everyday lives, for example email spam detection,  comments with profane language and etc. In the report, we start with a focus on exploring the data set and after obtaining information about it, we  preprocess it in order to feed it to a classifier. We use the FastText classifier developed by Facebook. At the end we evaluate the performance of our classifier by submitting it's predictions results to Kaggle's "Quora Insincere Questions Classification" competition. The best score that we got using the techniques stated in this report is 0.573 and the leader of the competition has a score of 0.729 (at the time when this report was written).
# 
# The aim of this project is to provide an introductory step by step clarification on how a real-world text classification problem could be resolved by analyzing the data, using common tools for preprocessing and getting a reasonable accuracy score. 
# 
# Some the questions that we will try to address, are:
# 
# <ul>
#    <li> How to explore the data set?</li>
#     <li>How to deal with imbalanced data sets?</li>
#     <li>How to choose the best hyperparameters for the classifier?</li>
#     <li>How to evaluate the performance of the classifier?</li>
# </ul>

# # Data description
# 
# The data that is provided consist of train data, test data and word embedding. Due to some limitations we are not going to use the provided embedding.
# 
# The train data consists of 1,306,122 samples and has three features: *qid*, *question_text* and *target*. 
# <br>
# The number of positive questions is 1,225,312, while the number of insincere is 80,810. This is an indication of  a highly imbalanced data set.
# <br>
# The test data consists of 56,370 samples and has two features: *qid* and *question_text*.

# # Imports

# Let's start by importing the main Python packages that will be used throughout this notebook:

# In[ ]:


# Libraries
import numpy as np 
import pandas as pd
import os
import random
import re
from pathlib import Path
import fastText as ft
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from collections import Counter
import matplotlib.pyplot as plt


# Next, by using Python package *pandas* we can import the testing and training data. 

# In[ ]:


# import data
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


# The rule of thumb is to have a quick look into the data, which gives an insight about we are dealing with.

# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# Once we have seen how the data looks like, it is a good idea to the distribution of labeled data. *0* means all the question that are sincere and *1* are questions tha aren't.

# In[ ]:


train_values = df_train['target'].values
zeros = np.where(train_values == 0)

train_values = df_train['target'].values
ones = np.where(train_values == 1)

y = [len(zeros[0]), len(ones[0])]
x = [0, 1]

plt.bar([0, 1], y)
plt.xticks(x, (0, 1))
plt.title('Class distribution')
plt.show()


# From the above histogram we can see that the ration of *0*s is much higher then the ratio of *1*s. This makes the classification task more complicated, as a trained model could become more bias to the dominant class, which in this case is *0* or sincere questions and predicting the label for *1* becomes harder as the data is very unbalanced.

# # Preprocessing

# Let's start by importing necessary packages for the preprocessing of the questions text.

# In[ ]:


import re, string, unicodedata
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer


# It is very important that cleanup the data, as the questions text might include lots of words that add noise to the data and the classifier might have hard time to predict the true label. For that reason we will normalize the question text by removing all the non ASCII characters, converting the strings into lower case, removing punctuation, converting the words into stem words and lemmatize the words so they all have a unified format.  Once all of these steps are preformed we return a normalized string. For example:
# 
# ```
# How did Quebec nationalists see their province as a nation in the 1960s?
# Do you have an adopted dog, how would you encourage people to adopt and not shop?
# How did Otto von Guericke used the Magdeburg hemispheres?
# Can I convert montra helicon D to a mountain bike by just changing the tyres?
# ```
# 
# becomes:
# ```
# how do quebec nationalists see their province as a nation in the 1960s
# do you have an adopt dog how would you encourage people to adopt and not shop
# how do otto von guericke use the magdeburg hemispheres
# can i convert montra helicon d to a mountain bike by just change the tyres
# ```
# 
# Even though removing stop words is a common practice, we decided to not do it on our data set as we have noticed that it decreased our score.

# In[ ]:


stemmer = LancasterStemmer()
lemmatizer = WordNetLemmatizer()

def remove_non_ascii(word):
    """Remove non-ASCII characters from list of tokenized words"""
    new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return new_word

def to_lowercase(word):
    """Convert all characters to lowercase from list of tokenized words"""
    return word.lower()

def remove_punctuation(word):
    """Remove punctuation from list of tokenized words"""
    new_word = re.sub(r'[^\w\s]', '', word)
    return new_word

def remove_stopwords(word):
    """Remove stop words from list of tokenized words"""
    if word not in stopwords.words('english'):
        return word
    return ''

def stem_words(word):
    """Stem words in list of tokenized words"""
    return stemmer.stem(word)

def lemmatize_verbs(word):
    """Lemmatize verbs in list of tokenized words"""
    return lemmatizer.lemmatize(word, pos='v')

def normalize(word):
    word = remove_non_ascii(word)
    word = to_lowercase(word)
    word = remove_punctuation(word)
    # word = remove_stopwords(word)
    word = lemmatize_verbs(word)
    return word

def get_processed_text(string):
    words = nltk.word_tokenize(string)
    new_words = []
    for word in words:
        new_word = normalize(word)
        if new_word != '':
            new_words.append(new_word)
    return ' '.join(new_words)


# Now that our `get_processed_text` function is ready we can convert all the rows in the `question_text` colum of the `df_train` data frame into "clean" version:

# In[ ]:


df_train.question_text = df_train.question_text.apply(lambda x: get_processed_text(x))
df_train.head()


# Next, we need to create a `label_and_text` column in the `df_train` dataframe. This column will have rows of concatenated label and text that will be used later when we train our classifier using FastText.

# In[ ]:


df_train['label_and_text'] = '__label__' + df_train.target.map(str) + ' '+ df_train.question_text
df_train.head()


# Same preprocessing should be done on the test data, except we don't have to add any extra column.

# In[ ]:


df_test.question_text = df_test.question_text.apply(lambda x: get_processed_text(x))
df_test.head()


# FastText trains a classifier with a training data and it receives as a parameter the path to that training data file. For that reason we have to create a `train.txt` file and export all of the rows in the column `label_and_text`:

# In[ ]:


# Write training data to a file as required by fasttext
training_file = open('train.txt','w')
training_file.writelines(df_train.label_and_text + '\n')
training_file.close()


# # FastText
# 
# FastText is an open-souce, lightweight library, developed by Facebook and it is used for efficient learning of word representations and sentence classification. 
# <br>
# The model allows to create an unsupervised learning or supervised learning algorithm for obtaining vector representations for word. It uses a neural network for creating the word embeddings.

# # Model tuning 
# 
# The next chunk of code tries to find the best hyperparameters for the FastText algorithm. Mainly, we are interested in `lr`,`wordNgrams` ,`epoch` ,`ws` ,`loss`. Here is an explanation for each of them with their default values:
# 
# ```
# wordNgrams:          max length of word ngram [1]
# lr:                  learning rate [0.05]
# epoch:               number of epochs [5]
# ws:                  size of the context window [5]
# loss:                loss function {ns, hs, softmax} [ns]
# ```
# 
# The default values should be good enough for many classification problems, but unfortunately in our case we have to perform a grid search and find the best hyperparameter that classifies our data the best. The reason that it is mandatory in this case is that our data is very unbalanced. We will be looping though these values:
# 
# ```
# lr = [0.05, 0.1, 0.2],
# wordNgrams = [1, 2, 3],
# epoch = [1, 5],
# ws = [5, 10],
# loss = ['ns', 'hs', 'softmax']
# ```
# 
# As you might have noticed there are 108 iterations:
# $3 * 3 * 2 * 2 * 3 = 3^3 * 2^2 =108$ 
# 
# On avarage one itiration can take up to 2 minutes that means that the grid search will take about 3.6 hours: $108*2mins/60mins = 3.6hours$. For that reason we have to stick to these parameters and certainly if we notice some unusual phenomenon, then we will perform another grid search with the best results that these parameters give us.
# 
# We would like to thank Randi Griffin for the post: *"Facebook's fastText algorithm"* as the some of the implementations are based on that post.

# In[ ]:


# Function to do K-fold CV across different fasttext parameter values
def tune(Y, X, YX, k, lr, wordNgrams, epoch, loss, ws):
    # Record results
    results = []
    for lr_val in lr:
        for wordNgrams_val in wordNgrams:
            for epoch_val in epoch:  
                for loss_val in loss:
                    for ws_val in ws:
                        # K-fold CV
                        kf = KFold(n_splits=k, shuffle=True)
                        fold_results = []
                        # For each fold
                        for train_index, test_index in kf.split(X):
                            # Write the training data for this CV fold
                            training_file = open('train_cv.txt','w')
                            training_file.writelines(YX[train_index] + '\n')
                            training_file.close()
                            # Fit model for this set of parameter values
                            model = ft.FastText.train_supervised(
                                'train_cv.txt',
                                lr=lr_val,
                                wordNgrams=wordNgrams_val,
                                epoch=epoch_val,
                                loss=loss_val,
                                ws=ws_val
                            )
                            # Predict the holdout sample
                            pred = model.predict(X[test_index].tolist())
                            pred = pd.Series(pred[0]).apply(lambda x: int(re.sub('__label__', '', x[0])))
                            # Compute accuracy for this CV fold
                            fold_results.append(accuracy_score(Y[test_index], pred.values))
                        # Compute mean accuracy across 10 folds 
                        mean_acc = pd.Series(fold_results).mean()
                        print([lr_val, wordNgrams_val, epoch_val, loss_val, ws_val, mean_acc])
    # Add current parameter values and mean accuracy to results table
    results.append([lr_val, wordNgrams_val, epoch_val, loss_val, ws_val, mean_acc])         
    # Return as a DataFrame 
    results = pd.DataFrame(results)
    results.columns = ['lr','wordNgrams','epoch','loss','ws_val','mean_acc']
    return(results)


# Now that our `tune` function is ready we have to perform an exhaustive search.

# In[ ]:


# results = tune(
#     Y = df_train.target,
#     X = df_train.question_text,
#     YX = df_train.label_and_text,
#     k = 5, 
#     lr = [0.05, 0.1, 0.2],
#     wordNgrams = [1, 2, 3],
#     epoch = [1, 5],
#     ws=[5, 10],
#     loss=['ns', 'hs', 'softmax']
# )


# *We will comment the content of this box, as we don't want it to be executed during the committing.*

# This function call will have 108 iteration and print the best hyperparameters that work on our data set. The output looks like:
# 
# ```
# lr      wordNgrams  epoch  loss       ws_val      mean_acc
# -----------------------------------------------------------
# [0.05,      1,        1,   'ns',        5,        0.951719]
# [0.05,      1,        1,   'ns',        10,       0.951705]
# [0.05,      1,        1,   'hs',        5,        0.951468]
# [0.05,      1,        1,   'hs',        10,       0.951427]
# [0.05,      1,        1,   'softmax',   5,        0.951685]
# ...
# [0.1,       3,        5,   'hs',        10,       0.954775]
# [0.1,       3,        5,   'softmax',   5,        0.954365]
# [0.1,       3,        5,   'softmax',   10,       0.954459]
# [0.2,       1,        1,   'ns',        5,        0.951787]
# [0.2,       1,        1,   'ns',        10,       0.951823]
# [0.2,       1,        1,   'hs',        5,        0.951860]
# ```
# 
# From which we will choose the ones with the highest accuracy and do our predictions.

# From the entire 180 outputs we noticed that our top 10 hyperparameters are:
# ```
#             lr      wordNgrams  epoch  loss       ws_val       mean_acc
#             ----------------------------------------------------------------
# 0.95502     [0.05,       2,      5,   'softmax',   5,    0.9550279373942934]
# 0.95507     [0.05,       3,      5,   'softmax',   5,    0.9550777033236294]
# 0.95510     [0.05,       2,      5,   'hs',        5,    0.9551098597511588]
# 0.95511     [0.05,       2,      5,   'softmax',   10,   0.9551198128567192]
# 0.95512     [0.05,       3,      5,   'softmax',   10,   0.9551282348207829]
# 0.95513     [0.05,       2,      5,   'hs',        10,   0.9551343592226385]
# 0.95515     [0.05,       3,      5,   'hs',        10,   0.9551504378144902]
# 0.95539     [0.05,       3,      5,   'hs',        5,    0.9553962030503383]
# ```
# 
# From this we can immoderately notice that:
# 1. A smaller value of `lr` is better
# 2. `wordNgrams` > 1 work better
# 3. The `epoch` could be increased
# 4. The `softmax` and `hs` wok better as a value of`loss`
# 5. `ws` value can be increased 
# 
# Wih these in mind we can perform our next grid search.

# In[ ]:


# results = tune(
#     Y = df_train.target,
#     X = df_train.question_text,
#     YX = df_train.label_and_text,
#     k = 5, 
#     lr = [0.025, 0.05],
#     wordNgrams = [2, 3],
#     epoch = [5, 10, 15, 20],
#     ws=[5, 10, 30],
#     loss=['hs', 'softmax']
# )


# Here are the top 10:
# ```
#             lr      wordNgrams  epoch   loss      ws_val       mean_acc
#             ----------------------------------------------------------------
# 0.95515     [0.025,     2,       5,    'softmax',   10,  0.9551512035405768]
# 0.95515     [0.025,     3,       5,    'softmax',   10,  0.955158859429776]
# 0.95516     [0.01,      2,       10,   'softmax',   30,  0.9551688125617146]
# 0.95517     [0.025,     2,       5,    'softmax',   5,   0.9551711096696319]
# 0.95521     [0.025,     3,       5,    'softmax',   5,   0.9552124531923372]
# 0.95522     [0.025,     3,       10,   'hs',        10,  0.9552247030042806]
# 0.95523     [0.01,      2,       15,   'hs',        5,   0.9552384845907967]
# 0.95523     [0.05,      2,       5,    'hs',        30,  0.9552392504048104]
# 0.95524     [0.05,      3,       5,    'hs',        5,   0.9552400154040317]
# 0.95529     [0.025,     3,       10,   'hs',        30,  0.9552905475547778]
# ```

# # Training
# 
# Now that we know what are the best parameters for the FastText algorithm, we are ready to move on to the training section. So far we have referred to the trained model as "the classifier", but there are multiple classifiers. Certainly, any classifier could sometimes be wrong and give a wrong predictions, but by increasing the number of classifiers we decrease the chance of predicting wrong, considering the fact that our at least two of our classifiers are good. 
# 
# Will top 3 different outputs that the `tune` function displayed and use them create classifier.
# ```
#             lr      wordNgrams  epoch  loss       ws_val       mean_acc
#             ----------------------------------------------------------------
# 0.95524     [0.05,      3,       5,    'hs',        5,   0.9552400154040317]
# 0.95529     [0.025,     3,       10,   'hs',        30,  0.9552905475547778]
# 0.95539     [0.05,      3,       5,    'hs',        5,   0.9553962030503383]
# ```

# In[ ]:


# train the classifier
classifier1 = ft.FastText.train_supervised(
    'train.txt',  lr=0.05,   wordNgrams=3,  epoch=5,  loss='hs',  ws=5
)
classifier2 = ft.FastText.train_supervised(
    'train.txt',  lr=0.025,  wordNgrams=3,  epoch=1,  loss='hs',  ws=30
)
classifier3 = ft.FastText.train_supervised(
    'train.txt',  lr=0.05,   wordNgrams=3,  epoch=5,  loss='hs',  ws=5
)


# # Predictions
# 
# Intuitively, for every classifier there must be a unique predictions, thus we will create multiple predictions:

# In[ ]:


# make predictions for test data
predictions1 = classifier1.predict(df_test.question_text.tolist())
predictions2 = classifier2.predict(df_test.question_text.tolist())
predictions3 = classifier3.predict(df_test.question_text.tolist())


# Now, that we have our predictions results we have to extract the most common ones. This is done with the methods`Counter` and `.most_common(1)`:

# In[ ]:


# Combine predictions
most_common = np.array([])
for i in range(len(predictions1[0])):
    most_common = np.append(
        most_common, 
        Counter([
            predictions1[0][i][0],
            predictions2[0][i][0],
            predictions3[0][i][0]
        ]).most_common(1)[0][0])


# We are now ready to create a submission file:

# In[ ]:


# Write submission file
submit = pd.DataFrame({
    'qid': df_test.qid,
    'prediction':  pd.Series(most_common)
})
submit.prediction = submit.prediction.apply(lambda x: re.sub('__label__', '', x))
submit.to_csv('submission.csv', index=False)


# At last we can have a quick glance at how the submission file looks like:

# In[ ]:


pd.read_csv('submission.csv')


# # Conclusion

# There many other methods and algorithms that one can use to solve a classification problem like this one, but we intentionally wanted to use FastText. The reason was that we wanted to observe how well does FastText perform on a real world problem and on a serious competition. Certainly, there are some parts of this notebook that could be improved to get a higher score. For instance the preprocessing could be enhanced, by converting all the numbers to their textual representation, converting plural into singular and etc.

# # References

# 1. Randi H Griffin. "Facebook's fastText algorithm", <https://www.kaggle.com/heesoo37/facebook-s-fasttext-algorithm>. Last visited: 2 Feb 2019. 
# 
# 2. KDnuggets. "Text Data Preprocessing: A Walkthrough in Python", <https://www.kdnuggets.com/2018/03/text-data-preprocessing-walkthrough-python.html>. Last visited: 2 Feb 2019. 
