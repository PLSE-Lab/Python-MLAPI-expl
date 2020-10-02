#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


# ## Load data

# In[ ]:


df_train = pd.read_json('../input/train.json')
df_test = pd.read_json('../input/test.json')


# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# ## Handle ingredients column

# #### Create list of words in each recipe row

# In[ ]:


#Train set
ingredients_train = df_train.ingredients
words_train = [' '.join(x) for x in ingredients_train]
print(len(words_train), words_train[0])

#Test set
ingredients_test = df_test.ingredients
words_test = [' '.join(x) for x in ingredients_test]


# #### Create a word vector based on the training set

# In[ ]:


vectorizer = CountVectorizer(max_features = 1000)
bag_of_words = vectorizer.fit(words_train)
bag_of_words


# #### Transform the word lists into vectors using the vectorizer trained on the training data

# In[ ]:


ing_array_train = bag_of_words.transform(words_train).toarray()
ing_array_test = bag_of_words.transform(words_test).toarray()
ing_array_train


# #### Incorporate the word vectors into the train and test dataframes

# In[ ]:


df_ing_train = pd.DataFrame(ing_array_train, columns=vectorizer.vocabulary_)
df_ing_test = pd.DataFrame(ing_array_test, columns=vectorizer.vocabulary_)
df_ing_train.head()


# In[ ]:


df_train_new = df_train.merge(df_ing_train, 
                          left_index=True, 
                          right_index=True).drop('ingredients', axis=1)
df_train_new.head()


# In[ ]:


df_test_new = df_test.merge(df_ing_test, 
                          left_index=True, 
                          right_index=True).drop('ingredients', axis=1)
df_test_new.head()


# ## Create sets

# In[ ]:


X = df_train_new.drop(['id', 'cuisine'], axis=1)
y = df_train_new.cuisine
X.shape, y.shape


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.85)
X_train.shape, X_val.shape


# ## Create Random Forest

# In[ ]:


m = RandomForestClassifier(oob_score=True)
m.fit(X_train, y_train)


# In[ ]:


m.oob_score_, m.score(X_val, y_val)


# ## Structuring the above into helper functions

# In[ ]:


def create_model(n_words, n_trees, train, test, words=None):
    #create vectorized df's
    df_train, df_test = vect_train_test(train, test, n_words, words)
    
    X = df_train.drop(['id', 'cuisine'], axis=1)
    y = df_train.cuisine
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.85)
    
    m = RandomForestClassifier(n_estimators=n_trees, oob_score=True)
    m.fit(X_train, y_train)
    
    moob_score = m.oob_score_
    score = m.score(X_val, y_val)
    model = m
    
    return moob_score, score, model

def vect_train_test(dftrain, dftest, n_words=1000, words=None):
    vectorizer = CountVectorizer(max_features = n_words)
    ingredients_train = dftrain.ingredients
    words_train = [' '.join(x) for x in ingredients_train]
    ingredients_test = dftest.ingredients
    words_test = [' '.join(x) for x in ingredients_test]
    if isinstance(words, pd.Series):
        bag_of_words = vectorizer.fit(words)
    else:
        bag_of_words = vectorizer.fit(words_train)

    ing_array_train = bag_of_words.transform(words_train).toarray()
    ing_array_test = bag_of_words.transform(words_test).toarray()

    df_ing_train = pd.DataFrame(ing_array_train, columns=vectorizer.vocabulary_)
    df_ing_test = pd.DataFrame(ing_array_test, columns=vectorizer.vocabulary_)

    df_train = dftrain.merge(df_ing_train, 
                          left_index=True, 
                          right_index=True).drop('ingredients', axis=1)
    df_test= dftest.merge(df_ing_test, 
                          left_index=True, 
                          right_index=True).drop('ingredients', axis=1)
    return df_train, df_test

def run_variations(variations, target):
    models = []
    for var in variations:
        moob_score, score, model = create_model(var[0], var[1], df_train, df_test)
        models.append({'n_vectors': var[0],
                       'n_trees': var[1],
                       'moob_score': moob_score,
                      'score': score,
                      'model': model})
        print(var, moob_score, score)
    if target == 'vector':
        plot_vector_score(models)
    elif target == 'trees':
        plot_ntree_score(models)
    return models

def plot_vector_score(models):
    plt.plot([x['n_vectors'] for x in models], [y['moob_score'] for y in models])
    plt.title('Score increase from Vector increase')
    plt.xlabel('Word vector size')
    plt.ylabel('Score')
    plt.show()
    return

def plot_ntree_score(models):
    plt.plot([x['n_trees'] for x in models], [y['moob_score'] for y in models])
    plt.title('Score increase from number of estimators (trees) increase')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Score (350 word vector)')
    plt.show()
    return


# ### Check feature importance

# In[ ]:


top = sorted(list(zip(X_val.columns, 
                      m.feature_importances_)), key=lambda x: x[1], reverse=True)[:200]


# In[ ]:


df_imp = pd.DataFrame(top, columns=['feat', 'imp'])
df_imp.imp = df_imp.imp.astype(float)
df_imp[df_imp.imp > 0.002].plot('feat', 'imp', kind='barh', figsize=(12,12))


# In[ ]:


df_keep = df_imp[df_imp.imp > 0.004]
new_ing = df_keep.feat


# In[ ]:


#Reducing features to the most significants did not improve results...
moob_score, score, m = create_model(350, 30, df_train, df_test, new_ing)
moob_score, score


# ## Create variations

# In[ ]:


variations_1 = [(100, 30),
             (150, 30),
             (200, 30),
             (300, 30),
             (350, 30),
             (500, 30),
             (700, 30),
             (1000, 30),
             (1300, 30)]

models_1 = run_variations(variations_1, 'vector')


# We can see that size 1000 for word vectors is ideal.

# How about number of trees? How does the score improve with changes to that?

# In[ ]:


# variations_2 = [(350, 10),
#                (350, 20),
#                (350, 25),
#                (350, 30),
#                (350, 35),
#                (350, 50),
#                (350, 75),
#                (350, 100),
#                (350, 150),
#                (350, 200),
#                (350, 300)]

# models_2 = run_variations(variations_2, 'trees')


# In[ ]:


# variations_3 = [(1000, 50),
#                 (1000, 100),
#                (1000, 150)]

# models_3 = run_variations(variations_3, 'trees')


# ## Optimal model (current)

# In[ ]:


moob_score, score, m = create_model(1000, 300, df_train, df_test) #after running variations these values seemed the best right now..


# In[ ]:


moob_score, score


# ## Create test set file

# In[ ]:


X_test = df_test_new.drop('id', axis=1)
y_test = m.predict(X_test)
y_test


# In[ ]:


df_sub = pd.DataFrame(np.array([df_test.id, y_test]).T, 
                      columns=['id', 'cuisine']).set_index('id')

df_sub.head()


# In[ ]:


df_sub.to_csv(f'submission_{m.n_estimators}_V4Kernel.csv')


# In[ ]:




