#!/usr/bin/env python
# coding: utf-8

# # Toxic Wikipedia comment analysis
# 
# Some first impressions about the given toxic comments data set that was taken from Wikipedia. 
# 
# ### Contents: 
# 1. Import libraries
# 2. Read the data
# 3. Data analysis
# 5. Fill missing data
# 6. Split comments into words
# 7. Train a word2vec model
# 8. Choose a classifier
# 9. Creating submission file
# 
# Feedback and hints for improvement is greatly appreciated!
# 

# ## 1. Import libraries
# We will need numpy for linear algebra matrix handling and pandas for convenient data import and cleaning purposes. 

# In[ ]:


import time

# import necessary libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#text libraries
import re
import nltk
from nltk.corpus import stopwords
from gensim.models import word2vec

# classifier imports
from sklearn.neural_network import MLPClassifier


# ## 2. Read the data
# Use pandas to read the complete training and test data set. 

# In[ ]:


# Read data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# copy test id column for later submission
result = test[['id']].copy() 
# show first 3 rows of the training set to get a first impression about the data
print(train.head(3))


# ## 3. Data analysis
# Take a closer look at the given data. 
# 
# There are no additional feature columns beside the **comment_text** column. The training data comes with six labels defining the incremental severity of toxity of each comment row. 

# In[ ]:


# count each severity 
print('toxic: %d' % train[train['toxic'] > 0]['toxic'].count())
print('severe_toxic: %d' % train[train['severe_toxic'] > 0]['severe_toxic'].count())
print('obscene: %d' % train[train['obscene'] > 0]['obscene'].count())
print('threat: %d' % train[train['threat'] > 0]['threat'].count())
print('insult: %d' % train[train['insult'] > 0]['insult'].count())
print('identity_hate: %d' % train[train['identity_hate'] > 0]['identity_hate'].count())


# I am curious if the label 'toxic' is the precondition for the label 'severe_toxic'?

# In[ ]:


print('Severe toxic but NOT toxic?: %d' % train[(train['severe_toxic'] > 0) & (train['toxic'] == 0)]['id'].count())
print('Insult but NOT toxic?: %d' % train[(train['insult'] > 0) & (train['toxic'] == 0)]['id'].count())
print('Obscene but NOT toxic?: %d' % train[(train['obscene'] > 0) & (train['toxic'] == 0)]['id'].count())
print('Threat but NOT insult?: %d' % train[(train['threat'] > 0) & (train['insult'] == 0)]['id'].count())


# Check the typical length of a comment.

# In[ ]:


train['len'] = train['comment_text'].str.len()
print('Average comment length: %d' % train['len'].mean())
print('Median comment length: %d' % train['len'].quantile(.5))
print('90th percentile comment length: %d' % train['len'].quantile(.9))


# In[ ]:


print(train[train['comment_text'].isnull()])
print(test[test['comment_text'].isnull()])


# ### Some observations
# 
# 1. Most frequent labels are toxic, obscene and insult. They seem to be the major categories.
# 2. Toxic and severe_toxic label are related while all other labels seem to be independent 

# ## 4. Fill missing data
# We have to make sure that there are no null values within the training and test data sets, otherwise our algorithms might fail.

# In[ ]:


test['comment_text'].fillna(value='none', inplace=True) # there is one 
train['comment_text'].fillna(value='none', inplace=True) 


# ## 5. Split comments into array of words
# We will split the given comments into arrays of single words. We will also remove non letter/number characters.  

# In[ ]:


def text_to_words(raw_text, remove_stopwords=False):
    # 1. Remove non-letters, but including numbers
    letters_only = re.sub("[^0-9a-zA-Z]", " ", raw_text)
    # 2. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english")) # In Python, searching a set is much faster than searching
        meaningful_words = [w for w in words if not w in stops] # Remove stop words
        words = meaningful_words
    return words 

sentences_train = train['comment_text'].apply(text_to_words, remove_stopwords=False)
sentences_test = test['comment_text'].apply(text_to_words, remove_stopwords=False)
# show first three arrays as sample
print(sentences_train[:3])


# ## 6. Train a word2vec model
# Train a word2vec model with the given training sentences. 

# In[ ]:


# Set values for various parameters
num_features = 300    # Word vector dimensionality                      
min_word_count = 40   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words
# Initialize and train the model (this will take some time)
model = word2vec.Word2Vec(sentences_train, workers=num_workers, size=num_features, min_count = min_word_count, window = context, sample = downsampling)
model.init_sims(replace=True) # marks the end of training to speed up the use of the model


# ## 7. Choose and train a classifier
# In this example we will use our trained word2vec model to get avarage word vectors from each sentence and train a neural network.

# In[ ]:


def makeFeatureVec(words, model, num_features):
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,), dtype="float32")
    #
    nwords = 0
    # 
    # Index2word is a list that contains the names of the words in 
    # the model's vocabulary. Convert it to a set, for speed 
    index2word_set = set(model.wv.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1
            featureVec = np.add(featureVec, model[word])
    # Divide the result by the number of words to get the average
    if nwords == 0:
        nwords = 1
    featureVec = np.divide(featureVec, nwords)
    return featureVec

def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
    counter = 0
    # Loop through the reviews
    for review in reviews:
        # Call the function (defined above) that makes average feature vectors
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
        counter = counter + 1
    return reviewFeatureVecs

f_matrix_train = getAvgFeatureVecs(sentences_train, model, num_features)
f_matrix_test = getAvgFeatureVecs(sentences_test, model, num_features)
# we have to train 6 different models with 6 different Y labels
y = [train['toxic'], train['severe_toxic'], train['obscene'], train['threat'], train['insult'], train['identity_hate']]


# We create 6 models, one for each toxic level each.

# In[ ]:


# create 6 MLP models
model = []
for i in range(0, 6):
    m = MLPClassifier(solver='adam', hidden_layer_sizes=(30,30,30), random_state=1)
    model.append(m)
print(model)


# Now train the models with a partial fit approach

# In[ ]:


batch_size = 10000
total_rows = f_matrix_train.shape[0]
duration = 0
start_train = time.time()
pos = 0
classes = [0,1]
# we use a partial fit approach
while duration < 2500 and pos < total_rows:
    for i in range(0, 6):
        if pos+batch_size > total_rows:
            batch_size = total_rows-pos
        X_p = f_matrix_train[pos:pos+batch_size]
        y_p = y[i][pos:pos+batch_size]
        model[i].partial_fit(X_p, y_p, classes)
    pos = pos + batch_size
    duration = time.time() - start_train # how long did we train so far?
    print("Pos %d/%d duration %d" % (pos, total_rows, duration))
    # end test partial fit  


# Now predict the result for each toxic level

# In[ ]:


result['toxic'] = model[0].predict_proba(f_matrix_test)[:,1]
result['severe_toxic'] = model[1].predict_proba(f_matrix_test)[:,1]
result['obscene'] = model[2].predict_proba(f_matrix_test)[:,1]
result['threat'] = model[3].predict_proba(f_matrix_test)[:,1]
result['insult'] = model[4].predict_proba(f_matrix_test)[:,1]
result['identity_hate'] = model[5].predict_proba(f_matrix_test)[:,1]


# ## 8. Write prediction into submission file
# Save the predicted values of all our 6 models into a submission csv file.

# In[ ]:


result.to_csv('submission.csv', encoding='utf-8', index=False)


# Wow, you really read my notebook till the last line, congratulation :)
# 
# My prediction is by far not a top ranking one but if you use any part of this notebook in a published kernel, credit (you can simply link back here) would be greatly appreciated.
# 
# Sources:
# [Word2Vec introduction](https://www.kaggle.com/c/word2vec-nlp-tutorial), Great word2vec tutorial by [Angela Chapman](http://www.linkedin.com/pub/angela-chapman/5/330/b97)
# 
