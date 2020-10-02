#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time
start_time = time.time()


# In[ ]:


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


# In[ ]:


import tensorflow_text as text


# In[ ]:


#it helps to print full tweet , not a part
pd.set_option('display.max_colwidth', -1)


# The results from :
# https://www.kaggle.com/ihelon/starter-nlp-svm-tf-idf ;
# https://www.kaggle.com/dmitri9149/svm-expm-v0/edit/run/27808847 ;
# https://www.kaggle.com/rerere/disaster-tweets-svm ;
# show the Support Vector Machine works quite well for the Real or Not ? (disaster) Tweets classification with with TF-ID for tokenization.<br>
# In https://www.kaggle.com/gibrano/disaster-universal-sentences-encoder-svm the Multilingual Universal Sentence Encoder is used for sentence encoding. Here I follow the work in using the Multilingual Universal Sentence Encoder (from tensorflow_hub).<br>
# The approach from https://www.kaggle.com/bandits/using-keywords-for-prediction-improvement is applied for final filtering of the results basing on the 'keywords'.<br>
# 
# The resulting model is quite simple and relativelly fast (700....900 seconds execution time without GPU). This makes the model suitable for experiments with different parameters and text preprocessing.

# ### Data loading

# In[ ]:


train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")


# ### Data preprosessing

# In[ ]:


def clean(text):
    text = re.sub(r"http\S+", " ", text) # remove urls
    text = re.sub(r"RT ", " ", text) # remove RT
    # remove all characters if not in the list [a-zA-Z#@\d\s]
    text = re.sub(r"[^a-zA-Z#@\d\s]", " ", text)
    text = re.sub(r"[0-9]", " ", text) # remove numbers
    text = re.sub(r"\s+", " ", text) # remove extra spaces
    text = text.strip() # remove spaces at the beginning and at the end of string
    return text


# In[ ]:


train.text = train.text.apply(clean)
test.text = test.text.apply(clean)


# How the text looks like after the cleaning.

# In[ ]:


train['text'][50:70]


# In[ ]:


test['text'][:5]


# Load the multilingual encoder module.

# In[ ]:


use = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")


# ### Some words about Universal Sentence Encoders and Transformer

# A Universal Sentence Encoders encode sentencies to fixed length vectors (The size is 512 in the case of the Multilingual Encoder). The encoders are pre trained on several different tasks: (research article) https://arxiv.org/pdf/1803.11175.pdf. And a use case: https://towardsdatascience.com/use-cases-of-googles-universal-sentence-encoder-in-production-dd5aaab4fc15 <br>
# Two architectures are in use in the encoders: Transformer and Deep Averaging Networks.
# Transformer use "self attention mechanism" that learns contextual relations between words and (depending on model) even subwords in a sentence. Not only a word , but it position in a sentence is also taking into account (like positions of other words). There are different ways to implement the intuitive notion of "contextual relation between words in a sentence" ( so, different ways to construct "representation space" for the contextual words relation). If the several "ways" are implemented in a model in the same time: the term "multi head attention mechanism" is used.<br>
# Transformers have 2 steps. Encoding: read the text and transform it in vector of fixed length, and decoding: decode the vector (produce prediction for the task). For example: take sentence in English, encode, and translate (decode) in sentence in German.<br>
# For our model we need only encoding mechanism: sentencies are encoded in vectors and supplied for classification to Support Vector Machine.<br>
# Good and intuitive explanation of the Transformer: http://jalammar.github.io/illustrated-transformer/ ; The original and quite famous now paper "Attention is all you need": (research article)
# https://arxiv.org/pdf/1706.03762.pdf. More about multi head attention: (research article)
# https://arxiv.org/pdf/1810.10183.pdf. How Transformer is used in BERT: https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270.<br>
# 
# The Multilingual Universal Sentence Encoder:(research articles) https://arxiv.org/pdf/1810.12836.pdf; https://arxiv.org/pdf/1810.12836.pdf;
# Example code: https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3
# The Multilingual Encoder uses very interesting Sentence Piece tokenization to make a pretrained vocabulary: (research articles) https://www.aclweb.org/anthology/D18-2012.pdf; https://www.aclweb.org/anthology/P18-1007.pdf.<br>
# 
# About the text preprocessing and importance of its coherence with the text preprocessing that is conducted for pretraining + about the different models of text tokeniation:
# 
# very good article:
# https://mlexplained.com/2019/11/06/a-deep-dive-into-the-wonderful-world-of-preprocessing-in-nlp/.<br>
# 
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
                                                                        test_size=0.05)


# In[ ]:


import xgboost as xgb
from xgboost import XGBClassifier


# In[ ]:


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
def xgboost_param_selection(X, y, nfolds):
    depth_m=[2,3,5,7,9]
    base_learners=[2,5,50,70,100]
    parameters=dict(n_estimators=base_learners , max_depth=depth_m)
    clf=RandomizedSearchCV(XGBClassifier(n_jobs=-1, class_weight='balanced') ,parameters, scoring='roc_auc', refit=True, cv=3)

    clf.fit(X, y)
#     cv_error=clf.cv_results_['mean_test_score']
#     train_error=clf.cv_results_['mean_train_score']
#     pred=clf.predict(X_train_bow)
#     score=roc_auc_score(y_train, pred)
#     estimator=clf.best_params_['n_estimators']
    clf.best_params_
#     depth=clf.best_params_['max_depth']
    return clf

# model = xgboost_param_selection(train_arrays,train_labels, 5)


# In[ ]:


def svc_param_selection(X, y, nfolds):
    Cs = [1.07]
    gammas = [2.075]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=nfolds, n_jobs=8)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search

# model = svc_param_selection(train_arrays,train_labels, 5)


# In[ ]:


from keras.utils import np_utils 
from keras.datasets import mnist 
import seaborn as sns
from keras.initializers import RandomNormal
import time
# https://gist.github.com/greydanus/f6eee59eaf1d90fcb3b534a25362cea4
# https://stackoverflow.com/a/14434334
# this function is used to update the plots for each epoch and error
def plt_dynamic(x, vy, ty, ax, colors=['b']):
    ax.plot(x, vy, 'b', label="Validation Loss")
    ax.plot(x, ty, 'r', label="Train Loss")
    plt.legend()
    plt.grid()
    fig.canvas.draw()
    
train_labels = np_utils.to_categorical(train_labels, 2) 
# y_test = np_utils.to_categorical(y_test, 10)

from keras.models import Sequential 
from keras.layers import Dense, Activation
output_dim = 2
input_dim = train_arrays.shape[1]

batch_size = 128 


# In[ ]:


model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(input_dim,)))
model.add(Dense(128, activation='relu'))
# The model needs to know what input shape it should expect. 
# For this reason, the first layer in a Sequential model 
# (and only the first, because following layers can do automatic shape inference)
# needs to receive information about its input shape. 
# you can use input_shape and input_dim to pass the shape of input

# output_dim represent the number of nodes need in that layer
# here we have 10 nodes

model.add(Dense(output_dim, input_dim=input_dim, activation='softmax'))
model.summary()


# In[ ]:


train_labels.shape


# In[ ]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
nb_epoch = 10
history = model.fit(train_arrays,train_labels, batch_size=batch_size, epochs=nb_epoch, verbose=1)


# In[ ]:


# model.best_params_


# In[ ]:


pred = model.predict(test_arrays)


# #### Accuracy and confusion matrix

# In[ ]:


# cm = confusion_matrix(train_labels,pred.round())
# cm


# In[ ]:


# accuracy = accuracy_score(test_labels,pred)
# accuracy


# ### Make Support Vector Machine prediction.

# In[ ]:


test_pred = model.predict(X_test)
submission['target'] = test_pred.round().astype(int)
#submission.to_csv('submission.csv', index=False)


# ### Using keywords for better prediction.

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


# The keywords are used for the corretion.

# In[ ]:


ids_disaster = test['id'][test.keyword.isin(keyword_list_disaster)].values
submission['target'][submission['id'].isin(ids_disaster)] = 1


# In[ ]:


submission.to_csv("submission.csv", index=False)
submission.head(10)


# I did experiments with different parameters and text preprocessing within the model configuration. It gives many "clever" variants with a good score. But the very model is attractive by its simplicity.

# Please, upvote, if you like

# In[ ]:


print("--- %s seconds ---" % (time.time() - start_time))


# In[ ]:




