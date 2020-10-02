#!/usr/bin/env python
# coding: utf-8

# In[5]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
import re

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[6]:


data=pd.read_csv("../input/poem_data.csv")
data[data.content.isnull()==True].index.tolist()


# In[7]:


data.content[0]
lin=data.content[0].splitlines()
for li in lin:
    print(li)
    
lin=(data.content[0]).replace(","," ").replace("."," ").replace(";"," ").replace(":"," ").replace("!"," ")
line=lin.splitlines()
i=0
Set=[]
Sentence_set=[]
result=""
print ("---last word of the sentence, see the below for the result ---")


# 

# In[8]:


#defining functions
def rhyme(inp, level):
    entries = nltk.corpus.cmudict.entries()
    syllables = [(word, syl) for word, syl in entries if word == inp]
    rhymes = []
    for (word, syllable) in syllables:
        rhymes += [word for word, pron in entries if pron[-level:] == syllable[-level:]]
    return set(rhymes)

def doTheyRhyme ( word1, word2 ):
    if word1.find ( word2 ) == len(word1) - len ( word2 ):
        return False
    if word2.find ( word1 ) == len ( word2 ) - len ( word1 ): 
        return False

    return word1 in rhyme ( word2, 1 )


# In[ ]:


for li in line:
    #print(li)
    Sentence_set.append(li)
    sp=li.split()
    if ("".join(sp[-1:]) is None or  len("".join(sp[-1:])) == 0 or "".join(sp[-1:])==" "):
        continue
    print (str(i)+" "+"".join(sp[-1:]))
    Set.append("".join(sp[-1:]) )
    if(len(Set)%4==0):
        if(  doTheyRhyme(Set[len(Set)-4], Set[len(Set)-3] )==True and  doTheyRhyme(Set[len(Set)-3], Set[len(Set)-2] )==True and doTheyRhyme(Set[len(Set)-2], Set[len(Set)-1] )==True):
            result= ("--AAAA--");
        elif (  doTheyRhyme(Set[len(Set)-4], Set[len(Set)-3] )==True and  doTheyRhyme(Set[len(Set)-3], Set[len(Set)-2] )==False and doTheyRhyme(Set[len(Set)-2], Set[len(Set)-1] )==True):
            result =("--AABB--");
        elif (  doTheyRhyme(Set[len(Set)-4], Set[len(Set)-2] )==True and  doTheyRhyme(Set[len(Set)-3], Set[len(Set)-1] )==False and doTheyRhyme(Set[len(Set)-3], Set[len(Set)-2] )==False):
            result =("--ABAB--");
        elif (  doTheyRhyme(Set[len(Set)-4], Set[len(Set)-1] )==True and  doTheyRhyme(Set[len(Set)-3], Set[len(Set)-1] )==False and doTheyRhyme(Set[len(Set)-3], Set[len(Set)-2] )==True):
            result =("--ABBA--");
    i=i+1
if(result==""):    
    result= ("--No Rhyme--");
print(result)


# In[ ]:


#removing punctuation and line break,converting to lower case and removing word of the list in doc
data.content=data.content.str.replace('\n', " ")
data.content=data.content.str.replace("\t", " ")
data.content=data.content.str.replace("\r", " ")
data.content.str.lower()
data.content=data.content.str.replace(","," ").replace("."," ")


# In[ ]:


print ("Sentences for similes in first poem")
for sen in Sentence_set:
    sen_break = sen.split();
    if (( "like"  in sen_break) or ("as"  in sen_break)):
        print (sen)
  
print ("Sentences for repetitions in first poem")
for sen in Sentence_set:
    sen_break = sen.split();
    if (sen_break is None or  len(sen_break) == 0 or sen_break==" "):
        continue
    for ele_in in sen_break:
        sen_break.remove(ele_in)
        if (ele_in in sen_break):
                print (sen +"      -word of Repetitions:"+ ele_in)


# In[ ]:


data.content = data.content.str.replace("ing( |$)", " ")
data.content = data.content.str.replace("[^a-zA-Z]", " ")
data.content = data.content.str.replace("ies( |$)", "y ")


# 

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True, analyzer= 'word')


# In[ ]:


#splitting the data
data_content_train,data_content_test, data_train_label,data_test_label =train_test_split(data[["content","author","poem name"]],data.type,test_size = 0.1, random_state = 1)
data_test_label_for_age=data.ix[data_test_label.index].age
data_train_label_for_age=data.ix[data_train_label.index].age

train_ = vectorizer.fit_transform(data_content_train.content.as_matrix())
feature_names =vectorizer.get_feature_names()
feature_names
test_ = vectorizer.transform(data_content_test.content.as_matrix())


# In[ ]:


from sklearn import preprocessing
label_au = preprocessing.LabelEncoder()
label_author=label_au.fit_transform(data_content_train.author.as_matrix())
label_authorT=label_au.fit_transform(data_content_test.author.as_matrix())

label_poe_name =TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df = 0, stop_words = 'english')  
label_poena=label_poe_name.fit_transform(data_content_train["poem name"].as_matrix())
label_poenaT  =label_poe_name.fit_transform(data_content_test["poem name"].as_matrix())

label_author=np.reshape(label_author, (label_author.shape[0], 1))
label_authorT=np.reshape(label_authorT, (label_authorT.shape[0], 1))


# In[ ]:


import xgboost as xgb

xgb_params = {'eta': 0.05,'max_depth': 6,'subsample': 0.6,'colsample_bytree': 1,'objective': 'reg:linear',"eval_metric": 'logloss',silent': 1
             }

xgb_params_age = {'eta': 0.05,'max_depth': 6,'subsample': 0.6,'colsample_bytree': 1,'objective': 'reg:linear',"eval_metric": 'error','silent': 1
                 }

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
a=le.fit_transform(data_train_label.as_matrix())

le2 = preprocessing.LabelEncoder()
a_age=le2.fit_transform(data_train_label_for_age.as_matrix())

dtrain = xgb.DMatrix(X_train, a )
dtest = xgb.DMatrix(X_test)
dtrain_age = xgb.DMatrix(X_train, a_age )
dtest_age = xgb.DMatrix(X_test)

num_boost_rounds = 422
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)
result =model.predict(dtest)

num_boost_rounds = 422
model_age = xgb.train(dict(xgb_params_age, silent=0), dtrain_age, num_boost_round=num_boost_rounds)
result_age =model.predict(dtest_age)

result_age

presult=pd.DataFrame(result)
presult_age=pd.DataFrame(result_age)

presult[(presult.values >= 0.5) & (presult.values < 1.5) ]= 1;
presult[(presult.values >= 1.5) & (presult.values < 2.5) ]=2;
presult[(presult.values >= -0.5) & (presult.values < 0.5) ]=0;

presult_age[(presult_age.values >= -0.5) & (presult_age.values < 0.5) ]=0;
presult_age[(presult_age.values >= 0.5) & (presult_age.values < 1.5) ]= 1;

presult=presult.astype(int)
presult_age=presult_age.astype(int)

result_back=le.inverse_transform(presult.values)
result_back_age=le2.inverse_transform(presult_age.values)

result_back_age.ravel()


# In[ ]:


# accuracy for target type 
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(data_test_label, result_back)
accuracy

# accuracy for target age
accuracy_age = accuracy_score(data_test_label_for_age, result_back_age)
accuracy_age

result_back_age


# In[ ]:


pd.DataFrame({  'poem name': data_content_test["poem name"],
                'correct_data' : data_test_label_for_age+ " " +data_test_label,
                'predict result' : result_back_age.ravel()+" " +result_back.ravel()
                    })

