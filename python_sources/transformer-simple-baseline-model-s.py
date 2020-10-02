#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time
start_time = time.time()

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from tqdm import tqdm
import numpy as np
import pandas as pd
import re

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[ ]:


get_ipython().system('pip install tensorflow-text==2.0.0 --user')


# In[ ]:


import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as textb


# In[ ]:


#print full tweet , not a part
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_rows', 100)


# ### Pipeline of ideas in the baseline model(s)

# The Transformer and  The Support Vector Machine are very complex and very powerfool tools. Here the tools are used as 'black boxes', and the code to combine the tools in the pipeline is simple.<br>
# Ideas from other notebooks in the competition help to reach good results.<br> 
# <br>
# NO TEXT CLEANING is used in the model(s), the row text is supplied to the Transformer.<br>
# <br>The model(s) is a 'baseline' for more complex versions.<br>
# <br>
# The execution time for the model(s) is about 700...800 seconds without a GPU.

# ### Model 1
# 
# #### Transformer (Multilingual Universal Sentence Encoder)  + Support Vector Machine
# 
# It gives the public score 0.83742   (Version 1 in this notebook)<br>
# Not a bad result taking into account we just supply the raw text to the Transformer,  which transform every string to 512 dimentional vector, supply the vectors to Support Vector Machine and receive the result. 
# The idea to use Multilingual Encoder is from here: https://www.kaggle.com/gibrano/disaster-universal-sentences-encoder-svm

# ### Model 2
# #### Transformer + Support Vector Machine + Filtering basing on keywords 
# 
# It gives the public score 0.83946 (Version 3 of the notebook)<br>
# The idea for the filtering is from here: https://www.kaggle.com/bandits/using-keywords-for-prediction-improvement

# ### Model 3
# 
# #### Transformer + Support Vector Machine + Filtering basing on keywords + Majority voting  for semantically equivalent but mislabelled tweets
# 
# It gives the score 0.84049 (this version)<br> 
# The idea is from here: https://www.kaggle.com/atpspin/same-tweet-two-target-labels

# Many notebooks in the competition show the Support Vector Machine works quite well for the classificaion. In https://www.kaggle.com/gibrano/disaster-universal-sentences-encoder-svm the Multilingual Universal Sentence Encoder is used for sentence encoding. Here I follow the work and use  the Multilingual Universal Sentence Encoder (from tensorflow_hub).<br>
# <br>
# The approach from https://www.kaggle.com/bandits/using-keywords-for-prediction-improvement is applied for final filtering of the results basing on the 'keywords'.<br>
# <br>
# In the training data there are many 'semantically equivalent' tweets. For example some tweets differ only in the URLs at the tail of string. It is reasonable to expect the URL tails are not very important for prediction of target and the tweets are semantically equal. To find such 'only URL different' tweets some cleaning of the 'text' strings is to be done. After the cleaning the tweets become equal as strings. Such semantically equivalent records in train set generate equivalence classes. What is important, there are classes, where tweets have 'mislabelling'. We can find 1 and 0 labels in the same class. But all tweets in a class are considered as semantically equal and as such must be all 0 xor all 1 labelled.<br>
# In raw 'text' (without a cleaning) in the train set we can find 55 records with 'mislabelling' (the records generate 18 equivalence classes).<br>
# <br>
# In https://www.kaggle.com/atpspin/same-tweet-two-target-labels (there is text cleaning in that model) 79 equivalence classes were detected with mislabelled tweets (here we have only 18). I do here (in model 3) same as in: https://www.kaggle.com/atpspin/same-tweet-two-target-labels. The mean for the 'target' is calculated for each class with mislabelling and the 'target' for the corresponfing records in train set is recalculated depending on the mean value (the majority voting).<br>
# <br>
# 

# ### Data loading

# In[ ]:


train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
length_train = len(train.index)
length_train


# ### Equivalence classes with mislabelling. 

# In[ ]:


# the code in the cell is taken from 
# https://www.kaggle.com/gunesevitan/nlp-with-disaster-tweets-eda-cleaning-and-bert
df_mislabeled = train.groupby(['text']).nunique().sort_values(by='target', ascending=False)
df_mislabeled = df_mislabeled[df_mislabeled['target'] > 1]['target']
index_misl = df_mislabeled.index.tolist()

lenght = len(index_misl)

print(f"There are {lenght} equivalence classes with mislabelling")


# The 18 'mislabelled tweets' (each of them respresent a class with min 2 elements).

# In[ ]:


index_misl


# Let us check how the records with 'mislabelled' tweets looks like in train set. I print the long list here intentionally. Please, check the behaviour of 'location' variable within the classes. 

# In[ ]:


train_nu_target = train[train['text'].isin(index_misl)].sort_values(by = 'text')
train_nu_target.head(60)


# In[ ]:


num_records = train_nu_target.shape[0]
length = len(index_misl)
print(f"There are {num_records} records in train set which generate {lenght} equivalence classes with mislabelling (raw text, no cleaning)") 


# Let us calculate some statistic for each class. Below in table the target mean + number of records in train set for each class are calculated. As we can see there are from 2 to 6 elements in equivalence class.  

# In[ ]:


copy = train_nu_target.copy()
classes = copy.groupby('text').agg({'keyword':np.size, 'target':np.mean}).rename(columns={'keyword':'Number of records in train set', 'target':'Target mean'})

classes.sort_values('Number of records in train set', ascending=False).head(20)


# ### Majority voting
# 
# If Target mean is lower or equal 0.5 , I relabel it to 0, otherwise to 1.

# In[ ]:


majority_df = train_nu_target.groupby(['text'])['target'].mean()
#majority_df.index


# In[ ]:


def relabel(r, majority_index):
    ind = ''
    if r['text'] in majority_index:
        ind = r['text']
#        print(ind)
        if majority_df[ind] <= 0.5:
            return 0
        else:
            return 1
    else: 
        return r['target'] 


# In[ ]:


train['target'] = train.apply( lambda row: relabel(row, majority_df.index), axis = 1)


# In[ ]:


new_df = train[train['text'].isin(majority_df.index)].sort_values(['target', 'text'], ascending = [False, True])
new_df.head(15)


# The 'target' for mislabelled tweets is recalculated. 
# The number of mislabelled tweets is 0 after recalculation. 

# In[ ]:


# the code in the cell is taken from 
# https://www.kaggle.com/gunesevitan/nlp-with-disaster-tweets-eda-cleaning-and-bert
df_mislabeled = train.groupby(['text']).nunique().sort_values(by='target', ascending=False)
df_mislabeled = df_mislabeled[df_mislabeled['target'] > 1]['target']
index_misl = df_mislabeled.index.tolist()
#index_dupl[0:50]
len(index_misl)


# ### Load the Multilingual Encoder module 

# In[ ]:


use = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")


# ### Some words about Universal Sentence Encoders and the Transformer

# A Universal Sentence Encoders encode sentencies to fixed length vectors (The size is 512 in the case of the Multilingual Encoder). The encoders are pre trained on several different tasks: (research article) https://arxiv.org/pdf/1803.11175.pdf. And a use case: https://towardsdatascience.com/use-cases-of-googles-universal-sentence-encoder-in-production-dd5aaab4fc15<br>
# Two architectures are in use in the encoders: Transformer and Deep Averaging Networks. Transformer use "self attention mechanism" that learns contextual relations between words and (depending on model) even subwords in a sentence. Not only a word , but it position in a sentence is also taking into account (like positions of other words). There are different ways to implement the intuitive notion of "contextual relation between words in a sentence" ( so, different ways to construct "representation space" for the contextual words relation). If the several "ways" are implemented in a model in the same time: the term "multi head attention mechanism" is used.<br>
# Transformers have 2 steps. Encoding: read the text and transform it in vector of fixed length, and decoding: decode the vector (produce prediction for the task). For example: take sentence in English, encode, and translate (decode) in sentence in German.<br>
# For our model we need only encoding mechanism: sentencies are encoded in vectors and supplied for classification to Support Vector Machine.<br>
# Good and intuitive explanation of the Transformer: http://jalammar.github.io/illustrated-transformer/ ; The original and quite famous now paper "Attention is all you need": (research article) https://arxiv.org/pdf/1706.03762.pdf. More about multi head attention: (research article) https://arxiv.org/pdf/1810.10183.pdf. How Transformer is used in BERT:<br> https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270.
# 
# The Multilingual Universal Sentence Encoder:(research articles) https://arxiv.org/pdf/1810.12836.pdf; https://arxiv.org/pdf/1810.12836.pdf; Example code: https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3 The Multilingual Encoder uses very interesting Sentence Piece tokenization to make a pretrained vocabulary: (research articles) https://www.aclweb.org/anthology/D18-2012.pdf; https://www.aclweb.org/anthology/P18-1007.pdf.<br>
# 
# About the text preprocessing and importance of its coherence with the text preprocessing that is conducted for pretraining + about the different models of text tokeniation:<br>
# very good article: https://mlexplained.com/2019/11/06/a-deep-dive-into-the-wonderful-world-of-preprocessing-in-nlp/.<br>
# 
# For deep understanding of the Transormer:  http://nlp.seas.harvard.edu/2018/04/03/attention.html ; <br>
# https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec ; <br> ;
# https://towardsdatascience.com/how-to-use-torchtext-for-neural-machine-translation-plus-hack-to-make-it-5x-faster-77f3884d95 ; https://github.com/SamLynnEvans/Transformer/blob/master/Models.py

# Below the encoding is applied to every sentence in train.text and test.text columns and the resulting vectors are saved to lists.<br>

# In[ ]:


X_train = []
for r in tqdm(train.text.values):
  emb = use(r)
  review_emb = tf.reshape(emb, [-1]).numpy()
  X_train.append(review_emb)

X_train = np.array(X_train)
y_train = train.target.values

X_test = []
for r in tqdm(test.text.values):
  emb = use(r)
  review_emb = tf.reshape(emb, [-1]).numpy()
  X_test.append(review_emb)

X_test = np.array(X_test)


# ### Training and Evaluating

# In[ ]:


train_arrays, test_arrays, train_labels, test_labels = train_test_split(X_train,
                                                                        y_train,
                                                                        random_state =42,
                                                                        test_size=0.20)


# In[ ]:


def svc_param_selection(X, y, nfolds):
    Cs = [1.07]
    gammas = [2.075]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=nfolds, n_jobs=8)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search

model = svc_param_selection(train_arrays,train_labels, 5)


# In[ ]:


model.best_params_


# #### Accuracy and confusion matrix

# In[ ]:


pred = model.predict(test_arrays)


# In[ ]:


cm = confusion_matrix(test_labels,pred)
cm


# In[ ]:


accuracy = accuracy_score(test_labels,pred)
accuracy


# ### Support Vector Machine prediction

# In[ ]:


test_pred = model.predict(X_test)
submission['target'] = test_pred.round().astype(int)
#submission.to_csv('submission.csv', index=False)


# ### Using keywords for better prediction

# Here I follow https://www.kaggle.com/bandits/using-keywords-for-prediction-improvement The idea is that some keywords with very high probability (sometimes = 1) signal about disaster (or usual) tweets. It is possible to add the extra 'keyword' feature to the model, but the simple approach also works. I make correction for the disaster tweets prediction to the model basing on the "disaster" keywords.

# In[ ]:


train_df_copy = train
train_df_copy = train_df_copy.fillna('None')
ag = train_df_copy.groupby('keyword').agg({'text':np.size, 'target':np.mean}).rename(columns={'text':'Count', 'target':'Disaster Probability'})

ag.sort_values('Disaster Probability', ascending=False).head(20)


# In[ ]:


count = 2
prob_disaster = 0.9
keyword_list_disaster = list(ag[(ag['Count']>count) & (ag['Disaster Probability']>=prob_disaster)].index)
#we print the list of keywords which will be used for prediction correction 
keyword_list_disaster


# In[ ]:


ids_disaster = test['id'][test.keyword.isin(keyword_list_disaster)].values
submission['target'][submission['id'].isin(ids_disaster)] = 1


# In[ ]:


submission.to_csv("submission.csv", index=False)
submission.head(10)


# Please, upvote if you like.

# In[ ]:


print("--- %s seconds ---" % (time.time() - start_time))

