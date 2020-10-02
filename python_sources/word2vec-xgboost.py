#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


train_data=pd.read_csv('../input/train.csv')
test_data=pd.read_csv('../input/test.csv')


# In[ ]:


train_data.head(1)


# In[ ]:


train_data.columns


# In[ ]:


print("Train Shape - "+str(train_data.shape))
print("Test Shape - "+str(test_data.shape))


# # Preprocessing

# In[ ]:


def check_nulls(train,test):
    train_null=pd.Series(pd.isnull(train).sum())
    test_null=pd.Series(pd.isnull(test).sum())
    total_null=pd.concat([train_null,test_null],axis=1)
    total_null.columns=['train','test']
    return total_null.transpose().sum()


# In[ ]:


check_nulls(train_data,test_data)


# In[ ]:


def preprocess(train):
    # Filling value of missing title with first sentence of the corresponding text
    title_null_indices=list(train.loc[pd.isnull(train)['title']].index)
    for i in title_null_indices:
        train.loc[i,'title']=train.loc[i,'text'].split('.')[0]
    return train


# In[ ]:


train=preprocess(train_data)
test=preprocess(test_data)


# In[ ]:


check_nulls(train,test)


# In[ ]:


# Exceptions
print(train.loc[37310,'text'])
print(test.loc[555,'text'])


# In[ ]:


print(test.loc[555])
test.loc[(test['restaurant_rating_food']==4.5)&(test['restaurant_rating_service']==4)&(test['restaurant_rating_value']==4)].head()


# In[ ]:


#Dropping train.loc[37310]
train.drop(37310,inplace=True)

#Replacing text of test.loc[555] with that of 72
test.loc[555,'text']=test.loc[72,'text']


# In[ ]:


#Dropping all non-textual columns
train=train[['text','title','id','ratingValue_target']]


# In[ ]:


def check_correlation(data,n):
    return data.corr()['ratingValue_target'].apply(np.abs).drop('ratingValue_target').sort_values(ascending=False)[:n]


# In[ ]:


def extract_best_features(data,n):
    corr_data=data.corr()['ratingValue_target'].apply(np.abs).drop('ratingValue_target').sort_values(ascending=False)
    best_features=list([x for x in corr_data.items() if x[1]>n])
    return best_features


# # Natural Language Processing

# In[ ]:


from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import re
from gensim.models import Word2Vec
import nltk
tokenizer=nltk.data.load('tokenizers/punkt/english.pickle')


# In[ ]:


def preprocess_to_words(text,remove_stopwords=False):
    text=BeautifulSoup(text).get_text()
    text=re.sub("[^a-zA-Z]"," ",text)
    words=text.lower().split()
    if(remove_stopwords==True):
        stop_words=stopwords.words('english')
        words=[x for x in words if x not in stop_words]
    return words


# In[ ]:


def preprocess_to_sentences(text,tokenizer):
    raw_sentences=tokenizer.tokenize(text.strip())
    sentences=[]
    for i in raw_sentences:
        if(len(i)!=0):
            sentences.append(preprocess_to_words(i,False))
    return sentences


# In[ ]:


def makeFeatureVec(words,model,num_features):
    featureVec=np.zeros((num_features,),dtype="float32")
    nwords=0
    index2word_set=set(model.wv.index2word)
    for word in words:
        if word in index2word_set: 
            nwords=nwords+1
            featureVec=np.add(featureVec,model[word])
    featureVec=np.divide(featureVec,nwords)
    return featureVec


# In[ ]:


def getAvgFeatureVecs(reviews,model,num_features):
    counter=0
    reviewFeatureVecs=np.zeros((len(reviews),num_features),dtype="float32")
    for review in reviews:
        reviewFeatureVecs[counter]=makeFeatureVec(review,model,num_features)
        counter=counter+1
    return reviewFeatureVecs


# In[ ]:


def build_word2vec_model(train,test,num_of_features,model_type):
    print("Extracting sentences from training data")
    total_sentences=[]
    cnt=0
    for i in train[model_type]:
        total_sentences+=preprocess_to_sentences(i,tokenizer)
        if(cnt%2000==0):
            print(str(cnt)+" rows done out of 40,000!")
        cnt+=1

    print("\n\nExtracting sentences from test data")
    cnt=0
    for i in test[model_type]:
        total_sentences+=preprocess_to_sentences(i,tokenizer)
        if(cnt%2000==0):
            print(str(cnt)+" rows done out of 20,000")
        cnt+=1
    word2vec_model = Word2Vec(total_sentences, workers=4, size=num_of_features, min_count = 40, window = 10, sample = 1e-3)
    return word2vec_model


# In[ ]:


def extract_word2vec_features(train,test,word2vec_model,num_of_features,model_type):
    
    train_words=train[model_type].apply(preprocess_to_words,args=(True,))
    avg_feature_vectors_train=getAvgFeatureVecs(train_words,word2vec_model,num_of_features)
    
    test_words=test[model_type].apply(preprocess_to_words,args=(True,))
    avg_feature_vectors_test=getAvgFeatureVecs(test_words,word2vec_model,num_of_features)
    
    train_word2vec_df=pd.DataFrame(avg_feature_vectors_train,columns=[model_type+str(x) for x in range(1,num_of_features+1)])
    test_word2vec_df=pd.DataFrame(avg_feature_vectors_test,columns=[model_type+str(x) for x in range(1,num_of_features+1)])
    
    return train_word2vec_df,test_word2vec_df


# In[ ]:


word2vec_model=build_word2vec_model(train,test,300,'text')
train_word2vec,test_word2vec=extract_word2vec_features(train,test,word2vec_model,300,'text')


# In[ ]:


title_word2vec_model=build_word2vec_model(train,test,100,'title')
train_title_word2vec,test_title_word2vec=extract_word2vec_features(train,test,title_word2vec_model,100,'title')


# In[ ]:


train_x=pd.concat([train_word2vec,train_title_word2vec],axis=1)
train_y=train['ratingValue_target']
test_x=pd.concat([test_word2vec,test_title_word2vec],axis=1)


# In[ ]:


print(train_x.shape,test_x.shape)


# # Model Selection

# In[ ]:


#from sklearn.model_selection import StratifiedKFold,cross_val_score,train_test_split
#from sklearn.metrics import f1_score,confusion_matrix
#from sklearn.model_selection import GridSearchCV


# ## XGBoost Model

# In[ ]:


from xgboost import XGBClassifier


# # Submission

# In[ ]:


xgb_model=XGBClassifier(objective='multi:softmax',learning_rate=0.2,min_child_weight=1,max_depth=6)
xgb_model.fit(train_x,train_y,eval_metric='merror')


# In[ ]:


predictions=xgb_model.predict(test_x)


# In[ ]:


submission_df=pd.DataFrame.from_dict({'id':test['id'],'ratingValue_target':predictions})
submission_df.head()


# In[ ]:


submission_df.to_csv('Submission.csv',index=False)

