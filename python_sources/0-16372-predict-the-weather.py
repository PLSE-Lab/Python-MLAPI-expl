#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


# reading files
train = pd.read_csv("/kaggle/input/crowdflower-weather-twitter/train.csv")
test = pd.read_csv('/kaggle/input/crowdflower-weather-twitter/test.csv')
sub = pd.read_csv('/kaggle/input/crowdflower-weather-twitter/sampleSubmission.csv')
print(train.shape)

print(test.shape)


# In[ ]:


# finding taget cols
target_cols = train.columns.difference(test.columns)
print('done')


# In[ ]:


# saving target columns
target = train[target_cols]
train.drop(target_cols,axis=1,inplace=True)
#df = pd.concat([train,test],axis=0,ignore_index=True)
#print(df.shape)
print('done')


# In[ ]:


# flag where location is missing
for data in (train,test):
    missing_loc_idx = data.index[data.location.isnull()].tolist()
    missing_loc_dict = {}
    for i in range(len(data)):
        if i in missing_loc_idx:
            missing_loc_dict[i] = 1
        else:
            missing_loc_dict[i] = 0
    data['missing_loc']= data.index.map(missing_loc_dict)
    

###########################################################################
# flag where state is missing
for data in (train,test):
    missing_state_idx = data.index[data.state.isnull()].tolist()
    missing_state_dict = {}
    for i in range(len(data)):
        if i in missing_state_idx:
            missing_state_dict[i] = 1
        else:
            missing_state_dict[i] = 0
    data['missing_state']= data.index.map(missing_state_dict)
#################################################################################
print('done')


# In[ ]:


# combining text of columns - location,state,tweet
for data in (train,test):
    data['location'] = data['location'].replace(np.nan, '', regex=True)
    data['state'] = data['state'].replace(np.nan, '', regex=True)
    data['full_text'] = data['tweet']+' '+data['state']+' '+data['location']
#####################################
print('done')


# In[ ]:


### replacing abbreviation of american states with their names

states = {"AL":"Alabama","AK":"Alaska","AZ":"Arizona","AR":"Arkansas","CA":"California","CO":"Colorado","CT":"Connecticut","DE":"Delaware","FL":"Florida","GA":"Georgia","HI":"Hawaii","ID":"Idaho","IL":"Illinois","IN":"Indiana","IA":"Iowa","KS":"Kansas","KY":"Kentucky","LA":"Louisiana","ME":"Maine","MD":"Maryland","MA":"Massachusetts","MI":"Michigan","MN":"Minnesota","MS":"Mississippi","MO":"Missouri","MT":"Montana","NE":"Nebraska","NV":"Nevada","NH":"New Hampshire","NJ":"New Jersey","NM":"New Mexico","NY":"New York","NC":"North Carolina","ND":"North Dakota","OH":"Ohio","OK":"Oklahoma","OR":"Oregon","PA":"Pennsylvania","RI":"Rhode Island","SC":"South Carolina","SD":"South Dakota","TN":"Tennessee","TX":"Texas","UT":"Utah","VT":"Vermont","VA":"Virginia","WA":"Washington","WV":"West Virginia","WI":"Wisconsin","WY":"Wyoming","NYC":"New York"}

import re
def cleanpunc(sentence):
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|)|\|/||:]',r' ',cleaned)      
    return cleaned

def acronym(s, dct):
  return ' '.join([dct.get(i, i) for i in s.split()])

for data in (train,test):
    full_text_dict = {}
    for i,row in data.iterrows():
            full_text_dict[i] = acronym(cleanpunc(row['full_text']),states)
            #print(i)       
    data['full_txt2'] = data.index.map(full_text_dict)
############################################    
print('done')


# In[ ]:


# straight forward text features
for data in (train,test):
    data['num_word'] = data['tweet'].apply(lambda x : len(x.split()))
    data['num_char'] = data['tweet'].apply(lambda x : len(x))
    data['avg_word_length']= data['num_char']+1/data['num_word']
######################################
print('done')


# In[ ]:


# whether text contain link or not
for data in (train,test):
    linkword = "{link}"
    link_dict = {}
    for idx in range(len(data)):
        text = data.loc[idx,'full_text']
        if linkword in text.split():
            link_dict[idx]=1
        else:
            link_dict[idx]=0
    data['link_word'] = data.index.map(link_dict)

# whether '@' is present or not in text
for data in (train,test):
    mention= "@mention"
    mention_dict = {}
    for idx in range(len(data)):
        text = data.loc[idx,'full_text']
        if mention in cleanpunc(text).split():
            mention_dict[idx]=1
        else:
            mention_dict[idx]=0
    data['mention_word'] = data.index.map(mention_dict)

############################################
print('done')


# In[ ]:


### cleaning full_text 
import re
from nltk.stem import PorterStemmer
ps = PorterStemmer()
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

def cleanhtml (sentence):
    cleantext = re.sub(r'http\S+',r'',sentence)
    return cleantext

def cleanpunc(sentence):
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|)|\|/|:]',r' ',cleaned)      
    return cleaned

for data in (train,test):
    str1=' '
    final_string=[]
    s=''
    for sent in data['full_text']:
        filter_sent = []
        rem_html = cleanhtml(sent)
        rem_punc = cleanpunc (rem_html)
        for w in rem_punc.split():
            if ((w.isalpha())):
                if (w.lower() not in stopwords):
                    s=(ps.stem(w.lower())).encode('utf8')
                    s=(w.lower()).encode('utf8')
                    filter_sent.append(s)
                else:
                    continue
            else:
                continue
        str1 = b" ".join(filter_sent)
        final_string.append(str1)
    data['clean_text'] = np.array(final_string)
#################################
print('done')


# In[ ]:


# generating tf-idf vector of clean text
import scipy as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.preprocessing import OneHotEncoder

vectorizer = TfidfVectorizer(encoding='utf-8', ngram_range=(1, 3), max_df=0.75, min_df=5) #max_features=12500

train_vec = vectorizer.fit_transform(train.clean_text)
test_vec = vectorizer.transform(test.clean_text)
############################################
print('done')


# In[ ]:


# TruncatedSVD
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=2, algorithm='randomized', n_iter=5, random_state=None, tol=0.0)
train_svd = svd.fit_transform(train_vec)
test_svd = svd.transform(test_vec)
############################################
print('done')


# In[ ]:


col = ['svd1','svd2']
svd_train =  pd.DataFrame(train_svd, columns=col)
svd_test = pd.DataFrame(test_svd, columns=col)
############################################
print('done')


# In[ ]:


#import time

#import numpy as np
#import matplotlib.pyplot as plt

#from sklearn.cluster import MiniBatchKMeans, KMeans
#from sklearn.metrics.pairwise import pairwise_distances_argmin
#from sklearn.datasets import make_blobs
#batch_size = 45
#mbk = MiniBatchKMeans(init='k-means++', n_clusters=10, batch_size=batch_size,
#                      n_init=25, max_no_improvement=10,verbose=0)

#t0 = time.time()
#grp_x = mbk.fit_predict(train_vec)
#grp_test = mbk.predict(test_vec)
#t_mini_batch = time.time() - t0
#print(t_mini_batch )
#train['grp'] = grp_x
#test['grp']= grp_test 
############################################
#print('done')


# In[ ]:


#train = pd.get_dummies(data=train, columns=['grp'])
#test = pd.get_dummies(data=test, columns=['grp'])
############################################
#print('done')


# In[ ]:


import scipy as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
vectorizer2 = TfidfVectorizer(encoding='utf-8', ngram_range=(1, 3), max_df=0.75, min_df=5,max_features=50000)

X = sp.sparse.hstack((vectorizer2.fit_transform(train.clean_text),normalize(train[['num_word', 'num_char', 'avg_word_length']].values),train[['missing_loc', 'missing_state','link_word', 'mention_word']],svd_train [['svd1','svd2']]),format='csr')

X_columns=vectorizer.get_feature_names()+train[['num_word', 'num_char', 'avg_word_length','missing_loc', 'missing_state','link_word', 'mention_word']].columns.tolist() + svd_train [['svd1','svd2']].columns.tolist()
print(X.shape)
test_sp = sp.sparse.hstack((vectorizer2.transform(test.clean_text),normalize(test[['num_word', 'num_char', 'avg_word_length']].values),test[['missing_loc', 'missing_state','link_word', 'mention_word']],svd_test [['svd1','svd2']]),format='csr')

test_columns=vectorizer.get_feature_names()+test[['num_word', 'num_char', 'avg_word_length','missing_loc', 'missing_state','link_word', 'mention_word']].columns.tolist() + svd_test [['svd1','svd2']].columns.tolist()

print(test_sp.shape)
############################################
print('done')


# In[ ]:


# splitting train data for cross validation
from sklearn.model_selection import train_test_split
x, x_test, y, y_test = train_test_split(X,target,test_size=0.2,train_size=0.8, random_state = 0)
print('done')


# In[ ]:


# running single model for each target columns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn import linear_model
from sklearn.svm import LinearSVR
# Fit regression model
cols = target.columns
params = {'n_estimators': 500, 'max_depth': 3, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
dict_preds = {}
#clf = GradientBoostingRegressor(**params)
clf = Ridge(random_state=25)
#clf = SGDRegressor(max_iter=6000)
#clf = linear_model.Lasso(alpha=1)
#clf = LinearSVR(max_iter=2500)
#clf = LinearRegression()
for col in cols:
    print('start',col)
    clf.fit(x,y[col])
    rmse = sqrt(mean_squared_error(y_test[col], clf.predict(x_test)))
    print("RMSE: %.4f" % rmse)
    dict_preds[col]=clf.predict(test_sp)
    


# In[ ]:


# prediction dataframe and file saved
df_res = pd.DataFrame(dict_preds)
df_res[df_res < 0] = 0
df_res[df_res > 1] = 1
df_res['id']= sub['id']
df_res = df_res[sub.columns]
df_res.to_csv('sub1.csv',index=False)
print('done')


# In[ ]:


# using classifier chains
from sklearn.multioutput import RegressorChain
#from sklearn.naive_bayes import GaussianNB
# initialize classifier chains multi-label classifier
# with a gaussian naive bayes base classifier
regressor = RegressorChain(Ridge(random_state=25))

# train
regressor.fit(x,y)

# predict - cross validation
predictions = regressor.predict(x_test)
sqrt(mean_squared_error(y_test,predictions))


# In[ ]:


# predict - test_sp dataset
pred2 = regressor.predict(test_sp)
print('done')


# In[ ]:


# prediction dataframe and file saved

cols = target.columns
df_res2 = pd.DataFrame(pred2,columns=cols)
df_res2[df_res2 < 0] = 0
df_res2[df_res2 > 1] = 1
df_res2['id']= sub['id']
df_res2 = df_res2[sub.columns]
df_res2.to_csv('sub2.csv',index=False)
print('done')

