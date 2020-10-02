#!/usr/bin/env python
# coding: utf-8

# <font size =5>**About this Kernel ** </font>
# 
# Writing this Kernel as a sequel to my intial notebook published [here](https://www.kaggle.com/slatawa/tfidf-implementation-to-get-80-accuracy). If you are new to NLP I suggest you check that out first- in this notebook I have demonstrated basic concepts of NLP and achieved around 80% accuracy with basic cleaning and TFIDF.
# 
# Coming to this notebook , I wanted to explore word embeddings using Word2Vec and then apply it for classifcation. In my search over last few days I did not come across any notebook which could demonstrate a simple implementaion of word2vec , so decided to write one which can be used as starting point for any one trying to Dabble with Embeddings. Without further wait let's start with the dataset , another article which I found very usefull if you want to have a quick [read](https://towardsdatascience.com/an-implementation-guide-to-word2vec-using-numpy-and-google-sheets-13445eebd281 )
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import warnings 


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

warnings.filterwarnings("ignore")
# Any results you write to the current directory are saved as output.


# 1. Load datasets

# In[ ]:


df_train= pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
df_test=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')


# 2. Let's take a look through the data

# In[ ]:


print(df_train.head(5))


# In[ ]:


print(df_train.info())


# 3. Clean the data
# 
# As first step in cleaning - let us replace some commonly occuring shorthands 

# In[ ]:


def clean_text(text):
    import re
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"you'll", "you will", text)
    text = re.sub(r"i'll", "i will", text)
    text = re.sub(r"she'll", "she will", text)
    text = re.sub(r"he'll", "he will", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"there's", "there is", text)
    text = re.sub(r"here's", "here is", text)
    text = re.sub(r"who's", "who is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"shouldn't", "should not", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"   ", " ", text) # Remove any extra spaces
    return text


df_train['clean_text'] = df_train['text'].apply(clean_text)
df_test['clean_text'] = df_test['text'].apply(clean_text)


# In the next step we are going to do some further massaging which would make Job of Prediction Algorithm easy
# 
# * Let us remove any characters other then alphabets
# * Convert all dictionary to lower case - for consistency 
# * Lemmatize - More details on Stemming and Lemmatization [here](https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html)
# 

# In[ ]:


def massage_text(text):
    import re
    from nltk.corpus import stopwords
    ## remove anything other then characters and put everything in lowercase
    tweet = re.sub("[^a-zA-Z]", ' ', text)
    tweet = tweet.lower()
    tweet = tweet.split()

    from nltk.stem import WordNetLemmatizer
    lem = WordNetLemmatizer()
    tweet = [lem.lemmatize(word) for word in tweet
             if word not in set(stopwords.words('english'))]
    tweet = ' '.join(tweet)
    return tweet
    print('--here goes nothing')
    print(text)
    print(tweet)

df_train['clean_text'] = df_train['text'].apply(massage_text)
df_test['clean_text'] = df_test['text'].apply(massage_text)


# Let's take a look at the data now 

# In[ ]:


df_train.iloc[0:10][['text','clean_text']]


# Let's tokenize the clean text column now 

# In[ ]:


from nltk.tokenize import word_tokenize

df_train['tokens']=df_train['clean_text'].apply(lambda x: word_tokenize(x))
df_test['tokens'] = df_test['clean_text'].apply(lambda x: word_tokenize(x))


# In[ ]:


df_train


# 4. Apply Word2Vector to get word embeddings (convert your words into vectors) - Till now what we have done is basic standard cleaning now let's start with the actual Embeddings and then training. Word2Vector needs a way to map words to Vectors for this either we can use pretrained models from Glove, for this task  have trained the mappings from the tweets text itself. Wanted to try score while using a pretrained model like Glove but now at the moment , would look to add that in the future versions. If you are interested to explore that more , check [this](http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/) 

# 4.1 Below we create a list corpus which we would be using to train word2vec mappings 

# In[ ]:


import gensim
def fn_pre_process_data(doc):
    for rec in doc:
        yield gensim.utils.simple_preprocess(rec)

corpus = list(fn_pre_process_data(df_train['clean_text']))
corpus += list(fn_pre_process_data(df_test['clean_text']))


# Let's inititate the embedding model , we will come back to the passed arguments later

# In[ ]:


from gensim.models import Word2Vec

print('initiated ...')
wv_model = Word2Vec(corpus,size=150,window=3,min_count=2)
wv_model.train(corpus,total_examples=len(corpus),epochs=10)


# We have the embedding mnodel ready, let's convert the train and text tokens now 
# 
# 
# 

# In[ ]:


def get_word_embeddings(token_list,vector,k=150):
    if len(token_list) < 1:
        return np.zeros(k)
    else:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in token_list] 
    
    sum = np.sum(vectorized,axis=0)
    ## return the average
    return sum/len(vectorized)        
def get_embeddings(tokens,vector):
        embeddings = tokens.apply(lambda x: get_word_embeddings(x, wv_model))
        return list(embeddings)

train_embeddings = get_embeddings(df_train['tokens'],wv_model)
test_embeddings = get_embeddings(df_test['tokens'],wv_model)


# 5 Start applying models - Now that we have word embeddings ready let's start applying Learning Models to it .

# We can start by applying Logisitic Regression to get a baseline , we would be using Gridsearch to get the best combination of parameters for LR

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

lr_model = LogisticRegression()
grid_values ={'penalty':['l1','l2'],'C':[0.0001,0.001,0.01,0.1,1,10]}
grid_search_model = GridSearchCV(lr_model,param_grid=grid_values,cv=3)
grid_search_model.fit(train_embeddings,df_train['target'])
print(grid_search_model.best_estimator_)
print(grid_search_model.best_score_)
print(grid_search_model.best_params_)

predict_lr = grid_search_model.predict(test_embeddings)


# In[ ]:


predict_df = pd.DataFrame()
predict_df['id'] = df_test['id']
predict_df['target'] = predict_lr
predict_df.to_csv('sample_submission_100.csv', index=False)


# 5.2 Let's try SVM 

# In[ ]:


from sklearn.svm import SVC

svc_model = SVC()
grid_values ={'C':[0.0001,0.001,0.01,0.1,1,10]}

grid_search_model = GridSearchCV(svc_model,param_grid=grid_values,cv=3)
grid_search_model.fit(train_embeddings,df_train['target'])
print(grid_search_model.best_estimator_)
print(grid_search_model.best_score_)
print(grid_search_model.best_params_)

predict_svc = grid_search_model.predict(test_embeddings)


# 5.3 Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

rf_model = RandomForestClassifier()
grid_values ={'n_estimators':[10,50,100,150,200,300,400]}

grid_search_model = GridSearchCV(rf_model,param_grid=grid_values,cv=3)
grid_search_model.fit(train_embeddings,df_train['target'])
print(grid_search_model.best_estimator_)
print(grid_search_model.best_score_)
print(grid_search_model.best_params_)

predict_rf = grid_search_model.predict(test_embeddings)


# <font size =5> **Final Thoughts ** </font>
# 
# 
# I had (imagined) word2vec giving better accuracy scores then what I achieved with TFIDF [here](https://www.kaggle.com/slatawa/tfidf-implementation-to-get-80-accuracy) but I scored a mere 71.xx with the LR submissions,a bit of tweaking might get it back on track, but this should give you a good starting point in case you want to play with word-embeddings. 
# 
# **Pleae UPvote if you found this usefull and do leave comments if you think something can be improved here or you have any question regarding the notebook/code.**
