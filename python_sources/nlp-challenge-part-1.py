#!/usr/bin/env python
# coding: utf-8

# # NLP Challange.
#  
#     I have decided a idealistic challange for myself that is learning NLP in 1 week 
#     then participating in new "Jigsaw Multilingual Toxic comment classification challange"
#     
#     I am going to write all the things I will learn in different notebooks.
#  

# # Natural Language Processing 
#     
#     Natural language is the language we speak or write in, Natural Language Processing is a study where
#     we make computer understand how our language works and what are the rules on which our language is based
#     
#     Natural Language is hard to understand for computers because Natural Language has very less rules.
#     and sometimes it requires the context inorder to understand the Language.
#     
#     Natural Language Processing has been around for 50 years, but still there is room for improvement.
#     
#     Natural Language Processing is so hard that Turing should have included it in the test of intellegence.
#     
#     
#         
#     

# ## Importing Imp libraries
# 
# I have just imported every possibly used library of NLP <br/>
# we will see what each imported library and class do eventually.<br/>
# 
# I am following [this](https://www.kaggle.com/abhishek/approaching-almost-any-nlp-problem-on-kaggle) notebook by abhishek thakur. 

# In[ ]:


import pandas as pd
import numpy as np
import xgboost as xgb
from tqdm import tqdm
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers.recurrent import LSTM,GRU
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from sklearn import preprocessing, decomposition, model_selection,metrics,pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from keras.layers import GlobalMaxPooling1D,Conv1D,MaxPooling1D,Flatten,Bidirectional,SpatialDropout1D
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping
from nltk import word_tokenize
from nltk.corpus import stopwords


#  ##  Loading dataset
#  
#  **About Data and compitition**
#     Here the challange is simply to predict which author wrote <br/>
#     given sentence 
#     
#   we are given training data and testing data in separate files
#   training data contains "id" of each sentence "text" the sentence
#   and author(EAP: Edgar Allan Poe, HPL: HP Lovecraft; 
#   MWS: Mary Wollstonecraft Shelley)

# In[ ]:


PATH = '../input/spooky-author-identification'
train = pd.read_csv(f'{PATH}/train.zip')
test = pd.read_csv(f'{PATH}/test.zip')
sample = pd.read_csv(f'{PATH}/sample_submission.zip')


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


sample.head()


# ##  Mertics for calculating loss
# 
# Here kaggle is using multiclass log loss for calculating the error.
# 
# Log Loss is the mathematical function which is used to measure the error rate in the classification problem, means we need to get minimum value of log loss. Log Loss is different because here we need to calculate the probability of each class rather than predicting the class. 
# 
# The twist about the logloss is that it highly penalize the wrong values. that is if in a binary classification if we have probability of 50/ 50 an after that if probability shits 60/40 to the wrong side,
# that is to wrong class it highly penalizes it.
# 
# The graph of the probability vs logloss is shown below.
# 
# ![image](https://datawookie.netlify.app/img/2015/12/log-loss-curve.png)
# 
# ![image2](https://miro.medium.com/max/1192/1*wilGXrItaMAJmZNl6RJq9Q.png)
# 
# as this graph shows as the probability of the correct classification increases logloss decreases gradually but if the probability decreases logloss increase exponentially highly penalizing the output.
# 
# so it is better to have overall good clasification values, rather than some excellent classification values and many worst classification values.
# 
# to learn more about logloss read [this](https://datawookie.netlify.app/blog/2015/12/making-sense-of-logarithmic-loss/) article.
# 
# 
# Here the difference is that instead of using Binary LogLoss Kaggle is usinig Multi logloss which is for Multiclass classification
# 
# and it's formula is
# ![image3](https://miro.medium.com/max/1162/0*i2_eUc_t8A1EJObd.png)
# 

#  ##  Function for Multiclass-logloss

# In[ ]:


def multiclass_logloss(actual,predicted,eps=1e-15):
    
    #converting the 'actual' values to binary values if it's 
    #not binary values
    
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0],predicted.shape[1]))
        
        for i, val in enumerate(actual):
            actual2[i,val] = 1
        actual = actual2
    
    #clip function truncates the number between
    #a max number and min number
    clip = np.clip(predicted,eps,1-eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0/ rows * vsota 


# **we will labelencode the author column using LabelEncoder from scikit-learn**

# In[ ]:


encoder = preprocessing.LabelEncoder()
y = encoder.fit_transform(train["author"].values)


# **we will make train test split of the data using train_test_split function of scikit which has parameter test_size which decides fraction of the values to use as test data**

# In[ ]:


# we will use 10% of data for testing
X_train, X_test, y_train, y_test = train_test_split(train.text.values,y,random_state=42,test_size=0.1,shuffle=True)


# ## Term Frequency(tf) and Inverse document frequency(idf).
# 
# **what is term frequeny(tf)?**: 
# * term frequency if simply a number which indicates how many times a number<br/>
#   occur in the document. So it just count number of occurance of the words. It is used to know what words are<br/>
#   important in the documents.
# * Term Frequency is divided by the total number of words for normaizing, it could also be divided by max fequency<br/>
#   or average frquency.
# 
# **what is inverse documents frequency?**
# * The problem with the term freuency is that in a language there are many connecters common words like "the","is","and"<br/>
#    and this words do not highlight the context of the sentence. one way to solve this problem is to remove stop words<br/>
#    and other is to multiply it with idf score.
# * Idf score of a word is a number determining how unique this word is to given document.
# * It is calculated as log(N/DT) where N is total number of documents and DT is number of doc containing the word.
# 
# so tf*idf will highlight the words which gives us context of the data.
# 

# # countvectorizer
# 
# Basic function of countvectorizer is to make a sparce matrix of word count for each document.<br/>
# for example there are two docs like "say hello to dog", and "say hello to everyone".
# 
# Countvectorizer will create a column for each unique word in all the documents<br/>
# and rows for each document. with values of count of each word<br/>
# so countvectorizer could perform as a feature to machine learning model<br/>
# 
# so we will have 5 columns(unique words) and 2 rows(no of documents)<br/>
# 
# say | hello | to | dog | everyone<br/>
#    1      1     1     1       0    <br/>
#    1      1     1     0       1    <br/>
#    
# Here instead of column names as "words" countvectorizer assigns number to each word<br/>
# like say=1, hello=2, to =3 ....
# 
# Now we will look at various ways we can use countvectorizer    

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
#we are going to use this example as our documents.

cat_in_the_hat_docs=[
       "One Cent, Two Cents, Old Cent, New Cent: All About Money (Cat in the Hat's Learning Library",
       "Inside Your Outside: All About the Human Body (Cat in the Hat's Learning Library)",
       "Oh, The Things You Can Do That Are Good for You: All About Staying Healthy (Cat in the Hat's Learning Library)",
       "On Beyond Bugs: All About Insects (Cat in the Hat's Learning Library)",
       "There's No Place Like Space: All About Our Solar System (Cat in the Hat's Learning Library)" 
      ]

#make object of countvectorizer
cv = CountVectorizer()
count_vector = cv.fit_transform(cat_in_the_hat_docs)


# **Basic steps performed by countvectorizer**
# 1. lowecase words (lowercase=False if not want to lower)
# 2. uses utf-8 encoding
# 3. perform tokenization (converts text to small chunk of text)
# 4. word level tokenization (converts each word to token)
# 5. ignores single character such as "a" and "I"

# In[ ]:


#now let's look at the  unique words countvectorizer was able to find
cv.vocabulary_


# In above output number shown are not counts of words they are<br/>
# their position (column no.) in the matrix

# In[ ]:


count_vector.shape


# Let's remove stopwords from the docs
# 
# There are three ways in which we can remove stop_words in CountVectorizer
# 
# 1. using custom stop_words list
# 2. using sklearn stopword list (not recomended)
# 3. using min_df and max_df for removing stopwords (highly recommended)
# 

# In[ ]:


#using cumstom stopword list
custom_stop_words = ["all","in","the","is","and"]

cv = CountVectorizer(cat_in_the_hat_docs,stop_words=custom_stop_words)
count_vector = cv.fit_transform(cat_in_the_hat_docs)
count_vector.shape


# The shape changed from (5,43) to (5,40) as the stop words are removed

# In[ ]:


#have a look at the stop words
cv.stop_words


# ### using MIN_DF (minimum document frequency)
# Min_df will look for the words which have low occurence in the documents<br/>
# like a name of the person which is occured only once in all the documents coult<br/>
# be removed.
# 
# it can be done by setting min_df argument which can be absolute value like 1,2,3<br/>
# or could be 0.25 means less than 25% of documents
# 

# In[ ]:


cv = CountVectorizer(cat_in_the_hat_docs,min_df=2) #word that has occur in only one document
count_vector = cv.fit_transform(cat_in_the_hat_docs)

#now let's look at stop words
cv.stop_words_
#see the difference of _ at the end it's because we used min_df 
#instead of custom stop_words


# There are too many stop words because our document list is small

# ### Using Max_df (Max document frequency)
# 
# As we have removed words which are less frequent we can remove words<br/>
# which are too frequent among documents using max_df parameter.

# In[ ]:


#using max_df
cv = CountVectorizer(cat_in_the_hat_docs,max_df=0.5) #present in more than 50% of documents
count_vector = cv.fit_transform(cat_in_the_hat_docs)

cv.stop_words_


# Good thing about CountVectorizer is that it allows you to make your own<br/>
# preprocessor and tokenizer and add it to 'tokenizer' and 'preprocessor' parameter
# 
# and other thing is if you just want presence and absence of words instead of counts<br/>
# you can set binary=True in parameters

# ## Tfidftransformer and Tfidfvectorizer
# 
# This are the classes which calculates tfid scores for the documents.
# 
# There is only little difference between both of them<br/>
# Tfidtransformer uses CountVectorizer that we have created manually<br/>
# 
# But Tfidvectorizer performs CountVectorizer function internally and no need to create<br/>
# manual CountVectorizer object.

# In[ ]:


#using Tfidtrasnformer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

#we will use this toy example
docs=["the house had a tiny little mouse",
      "the cat saw the mouse",
      "the mouse ran away from the house",
      "the cat finally ate the mouse",
      "the end of the mouse story"
     ]

cv = CountVectorizer(docs,max_df=0.5)

count_vector = cv.fit_transform(docs)
print(count_vector.shape)

#calculate idf values
tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(count_vector)

df_idf = pd.DataFrame(tfidf_transformer.idf_,index=cv.get_feature_names(),columns=["idf_weights"])
df_idf


# In[ ]:


#using tfid_vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer 

tfidf_vectorizer = TfidfVectorizer(smooth_idf=True,use_idf=True)
tfidf_vectorizer.fit_transform(docs)

#as you can see we don't need CountVectorizer in TfidfVectorizer

df_idf = pd.DataFrame(tfidf_vectorizer.idf_,index=tfidf_vectorizer.get_feature_names(),columns=["idf_weights"])
df_idf


# ## Back to Competition 

# ## LogisticRegression Model

# In[ ]:


# we can also pass countvectorizer parameters in TfidVectorizer
tfv = TfidfVectorizer(min_df=3,max_features=None,strip_accents='unicode',analyzer='word',token_pattern=r'\w{1,}',
                      ngram_range=(1,3),use_idf=1,smooth_idf=1,stop_words='english')

# max_features confines maximum number of words 

tfv.fit(list(X_train) + list(X_test))
X_train_tfv = tfv.transform(X_train)
X_test_tfv = tfv.transform(X_test)



# In[ ]:


# Fitting Logistic Regression on TFIDF
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C=1.0)
clf.fit(X_train_tfv,y_train)
prediction = clf.predict_proba(X_test_tfv)

print("logloss: %0.3f" % multiclass_logloss(y_test,prediction))


# Logistic regression on count vector

# In[ ]:


ctv = CountVectorizer(analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), stop_words = 'english')

ctv.fit(list(X_train)+list(X_test))
X_train_ctv = ctv.transform(X_train)
X_test_ctv = ctv.transform(X_test)


# In[ ]:


clf = LogisticRegression(C=1.0)
clf.fit(X_train_ctv,y_train)
prediction = clf.predict_proba(X_test_ctv)

print("logloss: %0.3f" % multiclass_logloss(y_test,prediction))


# ## Navie Bayes
# 
# Fitting on tfidf

# In[ ]:


clf = MultinomialNB()
clf.fit(X_train_tfv,y_train)

prediction = clf.predict_proba(X_test_tfv)

print("logloss: %0.3f" % multiclass_logloss(y_test,prediction))


# Fitting on counts

# In[ ]:


clf = MultinomialNB()
clf.fit(X_train_ctv,y_train)

prediction = clf.predict_proba(X_test_ctv)

print("logloss: %0.3f" % multiclass_logloss(y_test,prediction))


# ## SVM
# 
# SVM is very slow algorithm so it takes lot of time to fit so we will<br/> 
# use Singular Value Decomposition before applying SVM<br/>
# and we will also standardize the data.
# 

# ## What is Singular Value Decomposition?
# 
# Singular value decomposition  or svd is a matrix decompostion technique<br/>
# svd converts a matrix A  of mxn in dot product of three matrix (U . sigma . V^T) <br/>
# 
# Here U is mxm matrix<br/>
# sigma is diagonal matrix of nxm <br/>
# V^T is transpost of nxn matrix <br/>
# 
# Then reduction to k columns is done by taking sigma with k columns.<br/>
# or V^T with k rows. and preforming one of below function
# 
# T = U dot sigma or T = A dot V^T
# 
# T is final reduced matrix.
# 
# To know about SVD in detail read [this](https://machinelearningmastery.com/singular-value-decomposition-for-machine-learning/) article.
# 

# In[ ]:


svd = decomposition.TruncatedSVD(n_components=120)
svd.fit(X_train_tfv)
X_train_svd = svd.transform(X_train_tfv)
X_test_svd = svd.transform(X_test_tfv)

scl = preprocessing.StandardScaler()
scl.fit(X_train_svd)

X_train_svd_scl = scl.transform(X_train_svd)
X_test_svd_scl = scl.transform(X_test_svd)


# Fitting svm 

# In[ ]:


svm = SVC(C=1.0,probability=True)

svm.fit(X_train_svd_scl,y_train)
prediction = svm.predict_proba(X_test_svd_scl)

print("logloss: %0.3f" % multiclass_logloss(y_test,prediction))


# ## Xgboost
# 
# Fitting on tfidf

# In[ ]:


clf = xgb.XGBClassifier(max_depth=7,n_estimators=200,colsample_bytree=0.8,subsample=0.8,nthread=10,learning_rate=0.1)

clf.fit(X_train_tfv.tocsc(),y_train)
prediction = clf.predict_proba(X_test_tfv.tocsc())

print("logloss: %0.3f" % multiclass_logloss(y_test,prediction))


# Fitting on svd

# In[ ]:


clf = xgb.XGBClassifier(max_depth=7,n_estimators=200,colsample_bytree=0.8,subsample=0.8,nthread=10,learning_rate=0.1)

clf.fit(X_train_svd,y_train)
prediction = clf.predict_proba(X_test_svd)

print("logloss: %0.3f" % multiclass_logloss(y_test,prediction))


# ## Grid Search
# 
# We will do hyperparameter optimization using sklearn GridSearchCV<br/>
# and Data Pipeline is used here to keep code clean

# In[ ]:


# as Multiclass_logloss is user defined we need to define our own scorer for grid search
# greater_is_better is True by default but for our smaller the value of logloss better the result

mll_scorer = metrics.make_scorer(multiclass_logloss, greater_is_better=False,needs_proba=True)


# In[ ]:


svd = decomposition.TruncatedSVD()

scl = preprocessing.StandardScaler()

lr_model = LogisticRegression()

clf = pipeline.Pipeline([('svd',svd),
                         ('scl',scl),
                         ('lr',lr_model)])


# grid of parameters

# In[ ]:


params_grid = {'svd__n_components':[120,180],
               'lr__C':[0.1,1.0,10],
               'lr__penalty':['l1','l2']}


# we are creating parameter search for logistic regression <br/>
# with SVD of 120 and 180 lr of 0.1,1,10 and l1 and l2 penalty.

# In[ ]:


model = GridSearchCV(estimator=clf,param_grid=params_grid,scoring=mll_scorer,verbose=10,n_jobs=-1,iid=True,refit=True,cv=2)

#fitting the model
model.fit(X_train_tfv,y_train)

print('Best score: %0.3f' % model.best_score_)
print("Best parameters set:")
best_parameters = model.best_estimator_.get_params()
for param_name in sorted(params_grid.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))


# Grid Search for Navi Bayes

# In[ ]:


nb = MultinomialNB()

clf = pipeline.Pipeline([('nb',nb)])

params_grid = {'nb__alpha':[0.001,0.01,0.1,1,10,100]}

model  = GridSearchCV(estimator=clf,param_grid=params_grid,scoring=mll_scorer,verbose=10,n_jobs=-1,refit=True,cv=2)

model.fit(X_train_tfv,y_train)

print("Best score: %0.3f" % model.best_score_)
print("Best parameters set:")
best_parameters = model.best_estimator_.get_params()
for param_name in sorted(params_grid.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))


# This is the end of this notebook one, next notebook we will learn about advance stuff<br/>
# like word vectors , word embeddings and Deep Learning Models
