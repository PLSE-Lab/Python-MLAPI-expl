#!/usr/bin/env python
# coding: utf-8

# # DonorsChoose

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")
import gc

import sqlite3
import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from nltk.stem.porter import PorterStemmer
from time import time
from scipy.sparse import hstack
from sklearn.preprocessing import normalize
import re
from prettytable import PrettyTable
# Tutorial about Python regular expressions: https://pymotw.com/2/re/
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.metrics import auc
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD

import math
from sklearn.naive_bayes import ComplementNB
from subprocess import call
from IPython.display import Image

import multiprocessing as mp
from sklearn import tree
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pickle
import seaborn as sns
from sklearn.metrics import confusion_matrix
from nltk.stem import LancasterStemmer
from tqdm import tqdm_notebook as tqdm
import os
from sklearn.linear_model import SGDClassifier

from plotly import plotly
import plotly.offline as offline
import plotly.graph_objs as go
offline.init_notebook_mode()
from collections import Counter


# # Assignment 11: TruncatedSVD

# - <font color='red'>step 1</font> Select the top 2k words from essay text and project_title (concatinate essay text with project title and then find the top 2k words) based on their <a href='https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html'>`idf_`</a> values 
# - <font color='red'>step 2</font> Compute the co-occurance matrix with these 2k words, with window size=5 (<a href='https://www.analyticsvidhya.com/blog/2017/06/word-embeddings-count-word2veec/'>ref</a>)
#     <img src='cooc.JPG' width=300px>
# - <font color='red'>step 3</font> Use <a href='http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html'>TruncatedSVD</a> on calculated co-occurance matrix and reduce its dimensions, choose the number of components (`n_components`) using <a href='https://www.appliedaicourse.com/course/applied-ai-course-online/lessons/pca-code-example-using-non-visualization/'>elbow method</a>
#  >- The shape of the matrix after TruncatedSVD will be 2000\*n, i.e. each row represents a vector form of the corresponding word. <br>
#  >- Vectorize the essay text and project titles using these word vectors. (while vectorizing, do ignore all the words which are not in top 2k words)
# - <font color='red'>step 4</font> Concatenate these truncatedSVD matrix, with the matrix with features
# <ul>
#     <li><strong>school_state</strong> : categorical data</li>
#     <li><strong>clean_categories</strong> : categorical data</li>
#     <li><strong>clean_subcategories</strong> : categorical data</li>
#     <li><strong>project_grade_category</strong> :categorical data</li>
#     <li><strong>teacher_prefix</strong> : categorical data</li>
#     <li><strong>quantity</strong> : numerical data</li>
#     <li><strong>teacher_number_of_previously_posted_projects</strong> : numerical data</li>
#     <li><strong>price</strong> : numerical data</li>
#     <li><strong>sentiment score's of each of the essay</strong> : numerical data</li>
#     <li><strong>number of words in the title</strong> : numerical data</li>
#     <li><strong>number of words in the combine essays</strong> : numerical data</li>
#     <li><strong>word vectors calculated in</strong> <font color='red'>step 3</font> : numerical data</li>
# </ul>
# - <font color='red'>step 5</font>: Apply GBDT on matrix that was formed in <font color='red'>step 4</font> of this assignment, <font color='blue'><strong>DO REFER THIS BLOG: <a href='https://www.kdnuggets.com/2017/03/simple-xgboost-tutorial-iris-dataset.html'>XGBOOST DMATRIX<strong></a></font>
# <li><font color='red'>step 6</font>:Hyper parameter tuning (Consider any two hyper parameters)<ul><li>Find the best hyper parameter which will give the maximum <a href='https://www.appliedaicourse.com/course/applied-ai-course-online/lessons/receiver-operating-characteristic-curve-roc-curve-and-auc-1/'>AUC</a> value</li>
#     <li>Find the best hyper paramter using k-fold cross validation or simple cross validation data</li>
#     <li>Use gridsearch cv or randomsearch cv or you can also write your own for loops to do this task of hyperparameter tuning</li> 
#         </ul>
#     </li>
# 
# 

# ##2. TruncatedSVD

# ##2.1 Splitting data into Train and cross validation(or test): Stratified Sampling</h2>

# In[ ]:


cd  ../input/


# In[ ]:


project_data=pd.read_csv('preprocessed_project_data.csv')


# In[ ]:


upsample=0
if not upsample:
  trn=0.8
  tst=0.2
  drp=[col for col in project_data.columns if col!='project_is_approved']
  y=project_data.drop(labels=drp,axis=1)
  projct_data_strtfy=project_data.drop('project_is_approved',axis=1)
  data,data_test,Y,Y_test=train_test_split(projct_data_strtfy,y,train_size=trn,test_size=tst,stratify=y)
  data_train,data_cv,Y_train,Y_cv=train_test_split(data,Y,test_size=0.2,stratify=Y)
  data_train.reset_index(inplace=True,drop=True)
  data_test.reset_index(drop=True,inplace=True)
  data_cv.reset_index(inplace=True,drop=True)
  Y_train.reset_index(inplace=True,drop=True)
  Y_test.reset_index(inplace=True,drop=True)
  print(data_train.shape,data_cv.shape,data_test.shape)
  
##https://elitedatascience.com/imbalanced-classes
### Doing upsampling of minority class
if upsample:
  from sklearn.utils import resample
  sam=int(project_data.shape[0]*0.8*0.8)
  tst=int(project_data.shape[0]*0.2)
  cv=int(sam*0.2)+int((project_data.shape[0]-sam-tst-sam*0.2)/2)
  tst+=int((project_data.shape[0]-sam-tst-sam*0.2)/2)
  bal1=project_data[:sam][project_data.project_is_approved==1]
  zero=project_data[:sam][project_data.project_is_approved==0]
  bal0=resample(zero,replace=True,n_samples=bal1.shape[0],random_state=20)
  data=pd.concat([bal1,bal0],ignore_index=True)

  drp=[col for col in data.columns if col!='project_is_approved']
  Y_train=data.drop(labels=drp,axis=1)
  data_train=data.drop('project_is_approved',axis=1)

  drp=[col for col in project_data.columns if col!='project_is_approved']
  y_cvtst=project_data.drop(labels=drp,axis=1)
  projct_data_strtfy=project_data.drop('project_is_approved',axis=1)

  data_test,Y_test=projct_data_strtfy[sam+cv:],y_cvtst[sam+cv:]
  data_cv,Y_cv=projct_data_strtfy[sam:sam+cv],y_cvtst[sam:sam+cv]
  
  #data_cv,data_test,Y_cv,Y_test=train_test_split(projct_data_strtfy[sam:],y_cvtst[sam:],train_size=cv,test_size=tst,stratify=y_cvtst[sam:])

  print(data_train.shape,data_cv.shape,data_test.shape)


# ##2.2 Make Data Model Ready: encoding numerical, categorical features</h2>

# In[ ]:


### One hot encoding of Teacher prefix ######
tchr_prfx_ncode=CountVectorizer(lowercase=False).fit(data_train.teacher_prefix)
tchr_prfx_one_hot_train=tchr_prfx_ncode.transform(data_train.teacher_prefix)
print("Shape of matrix after one hot encodig ",tchr_prfx_one_hot_train.shape)
tchr_prfx_one_hot_train.toarray()[:10],tchr_prfx_ncode.get_feature_names()


# In[ ]:


### Transforming Teacher prefix of test data ###
tchr_prfx_one_hot_cv=tchr_prfx_ncode.transform(data_cv.teacher_prefix)
tchr_prfx_one_hot_test=tchr_prfx_ncode.transform(data_test.teacher_prefix)

tchr_prfx_one_hot_test.toarray()[:5],tchr_prfx_one_hot_cv.toarray()[:5]


# In[ ]:


# we use count vectorizer to convert the values into one 
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer( lowercase=False, binary=True).fit(data_train['clean_categories'].values)
categories_one_hot_train = vectorizer.transform(data_train['clean_categories'].values)
print(vectorizer.get_feature_names())
print("Shape of matrix after one hot encodig ",categories_one_hot_train.shape)
print(categories_one_hot_train.toarray()[:5])


# In[ ]:


### Transforming project categories feature of cv and test data ###
categories_one_hot_cv=vectorizer.transform(data_cv.clean_categories)
categories_one_hot_test=vectorizer.transform(data_test.clean_categories)
categories_one_hot_test.toarray()[:5]


# In[ ]:


### Transforming project sub categories feature of train data ###
sub_categories_one_hot = CountVectorizer(lowercase=False, binary=True).fit(data_train.clean_subcategories.values)
sub_categories_one_hot_train = sub_categories_one_hot.transform(data_train.clean_subcategories.values)
print(sub_categories_one_hot.get_feature_names())
print("Shape of matrix after one hot encodig ",sub_categories_one_hot_train.shape)
sub_categories_one_hot_train.toarray()[:5]


# In[ ]:


### Transforming project sub categories feature of cv and test data ###
sub_categories_one_hot_cv=sub_categories_one_hot.transform(data_cv.clean_subcategories)
sub_categories_one_hot_test=sub_categories_one_hot.transform(data_test.clean_subcategories)

sub_categories_one_hot_test.toarray()[:5],sub_categories_one_hot_cv.toarray()[:5],


# In[ ]:


### Transforming project school state feature of train data ###

state_vectorizer=CountVectorizer(lowercase=False, binary=True).fit(data_train.school_state)
state_one_hot_train=state_vectorizer.transform(data_train.school_state)
print(state_vectorizer.get_feature_names())
print("Shape of matrix after one hot encodig ",state_one_hot_train.shape)
print(state_one_hot_train.toarray()[:3])


# In[ ]:


### Transforming project school state feature of cv and test data ###
state_one_hot_cv=state_vectorizer.transform(data_cv.school_state)
state_one_hot_test=state_vectorizer.transform(data_test.school_state)
state_one_hot_test.toarray()[:2]


# In[ ]:


### Transforming project grade category feature of cv and test data ###

project_grade_ncode=CountVectorizer(lowercase=False,binary=True).fit(data_train.project_grade_category)
project_grade_one_hot_train=project_grade_ncode.transform(data_train.project_grade_category)
project_grade_one_hot_train.toarray()[:10],project_grade_ncode.get_feature_names()


# In[ ]:


### Transforming project school state feature of cv and test data ###
project_grade_one_hot_cv=project_grade_ncode.transform(data_cv.project_grade_category)
project_grade_one_hot_test=project_grade_ncode.transform(data_test.project_grade_category)
project_grade_one_hot_test.toarray()[:5]


# In[ ]:


# standardization sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
from sklearn.preprocessing import MinMaxScaler

price_scalar = MinMaxScaler()
price_scalar.fit(data_train['price'].values.reshape(-1,1)) # finding the mean and standard deviation of this data
#print(f"Mean : {price_scalar.mean_[0]}, Standard deviation : {np.sqrt(price_scalar.var_[0])}")

# Now standardize the train data price with above maen and variance.
price_standardized_train = price_scalar.transform(data_train['price'].values.reshape(-1, 1))
price_standardized_train


# In[ ]:


### Transforming project price feature of cv and test data ###
price_standardized_cv=price_scalar.transform(data_cv.price.values.reshape(-1, 1))
price_standardized_test=price_scalar.transform(data_test.price.values.reshape(-1, 1))
price_standardized_test[:5],price_standardized_cv[:5]


# In[ ]:


### Transforming no of previous projects feature of train data ###

previous_projects_stnd=MinMaxScaler().fit(data_train.teacher_number_of_previously_posted_projects.values.reshape(-1,1))
previous_projects_train=previous_projects_stnd.transform(data_train.teacher_number_of_previously_posted_projects.values.reshape(-1,1))
previous_projects_train


# In[ ]:


### Transforming no of previous projects feature of cv and test data ###
previous_projects_cv=previous_projects_stnd.transform(data_cv.teacher_number_of_previously_posted_projects.values.reshape(-1, 1))
previous_projects_test=previous_projects_stnd.transform(data_test.teacher_number_of_previously_posted_projects.values.reshape(-1, 1))
previous_projects_test[:5],previous_projects_cv[:5]


# In[ ]:


print(tchr_prfx_one_hot_train.shape)
print(state_one_hot_train.shape)
print(project_grade_one_hot_train.shape)
print(categories_one_hot_train.shape)
print(sub_categories_one_hot_train.shape)
print(price_standardized_train.shape)
print(previous_projects_train.shape)


# In[ ]:


catg_train=hstack((tchr_prfx_one_hot_train,state_one_hot_train,project_grade_one_hot_train,categories_one_hot_train,sub_categories_one_hot_train))
catg_cv=hstack((tchr_prfx_one_hot_cv,state_one_hot_cv,project_grade_one_hot_cv,categories_one_hot_cv,sub_categories_one_hot_cv))
catg_test=hstack((tchr_prfx_one_hot_test,state_one_hot_test,project_grade_one_hot_test,categories_one_hot_test,sub_categories_one_hot_test))


# In[ ]:


from sklearn.preprocessing import MaxAbsScaler


# In[ ]:


####### Concatinating All Categorical and Neumerical Feauters ###########
catg_train_stnd=MaxAbsScaler()
catg_train=catg_train_stnd.fit_transform(hstack((tchr_prfx_one_hot_train,state_one_hot_train,project_grade_one_hot_train,categories_one_hot_train,sub_categories_one_hot_train)))
catg_cv=catg_train_stnd.transform(hstack((tchr_prfx_one_hot_cv,state_one_hot_cv,project_grade_one_hot_cv,categories_one_hot_cv,sub_categories_one_hot_cv)))
catg_test=catg_train_stnd.transform(hstack((tchr_prfx_one_hot_test,state_one_hot_test,project_grade_one_hot_test,categories_one_hot_test,sub_categories_one_hot_test)))
catg_num_train=hstack((catg_train,price_standardized_train,previous_projects_train))
catg_num_cv=hstack((catg_cv,price_standardized_cv,previous_projects_cv))
catg_num_test=hstack((catg_test,price_standardized_test,previous_projects_test))
catg_num_train.shape,catg_num_test.shape,catg_num_cv.shape


# # 2.3 Make Data Model Ready: encoding eassay, and project_title</h2>

# ## <h2>2.1 Selecting top 2000 words from `essay` and `project_title`</h2>

# In[ ]:


train=[]
for i in range(len(data_train)):
  train.append(data_train.preprocessed_essay.iloc[i]+' '+data_train.preprocessed_title.iloc[i])


# In[ ]:


tfidf_vectorizer = TfidfVectorizer(min_df=10,max_df=100).fit(train)


# In[ ]:


## Getting top 2k words of high idf ##
max_2k_idf=np.argsort(tfidf_vectorizer.idf_)[:2000]
reversed_dict=dict(zip(tfidf_vectorizer.vocabulary_.values(),tfidf_vectorizer.vocabulary_.keys()))
top_wrds_2k_idf=[reversed_dict[i] for i in max_2k_idf]


# In[172]:


top_wrds_2k_idf[:10]


# #<h2>2.2 Computing Co-occurance matrix</h2>

# ## Example Calculation of Co-Occurance Matrix:

# from collections import Counter

# In[258]:


## Co occurance matrix for following corpus of window size 2
vocab=['He','is','not','lazy','smart','intelligent']
corpus='He is not lazy He is intelligent He is smart '

strt=time()
lst=np.array(corpus.split())
lenth=len(lst)
window=2
mn_dct={}
for i,word in enumerate(vocab):
    
  index=np.where(lst==word)[0]
  mn_dct[word]=dict.fromkeys(vocab,0)

  cntr=Counter()  
  for indx in index:
    cntr.update(Counter(lst[max(0,indx-window):min(lenth,indx+window+1)]))
  cntr[word]=0  
  mn_dct[word].update(cntr)
'''
Output:
  array([[0., 4., 2., 1., 1., 2.],
       [4., 0., 1., 2., 1., 2.],
       [2., 1., 0., 1., 0., 0.],
       [1., 2., 1., 0., 0., 0.],
       [1., 1., 0., 0., 0., 0.],
       [2., 2., 0., 0., 0., 0.]])

'''
## Making Co-Variance matrix ##
vocab_mtrx_svd=np.zeros((len(vocab),len(vocab)))
for i in range(len(vocab)):
  vocab_mtrx_svd[i,:]=np.array(list((mn_dct[vocab[i]].values())))

#vocab_mtrx_svd=vocab_mtrx_svd+vocab_mtrx_svd.transpose()

end=time()
print('Time elapsed: ',(end-strt)/60,' Minutes')
pd.DataFrame(vocab_mtrx_svd,index=vocab,columns=vocab)


# In[ ]:


'''## Co occurance matrix for following corpus of window size 2
vocab=['He','is','not','lazy','smart','intelligent']
corpus='He is not lazy He is intelligent He is smart'

strt=time()
lst=np.array(corpus.split())
lenth=len(lst)
window=2
mn_dct={}
for word in vocab:
  dct={}
  index=np.where(lst==word)[0]
  for next in vocab:
    dct[next]=0
    if next in mn_dct.keys():
      dct[next]=mn_dct[next][word]
    else:
      for indx in index:
        end = indx+window+1 if indx+window+1<=lenth else lenth
        start=indx-window  if indx-window>=0  else  0
        check=lst[start:end]
        if next in check and next!=word:
          dct[next]=dct[next]+np.sum(check==next)
  mn_dct[word]=dct
## Making Co-Variance matrix ##
vocab_mtrx_svd=np.zeros((len(vocab),len(vocab)))
for i in range(len(vocab)):
  vocab_mtrx_svd[i,:]=np.array(list((mn_dct[vocab[i]].values())))

pd.DataFrame(vocab_mtrx_svd,index=vocab,columns=vocab)

Output:
  array([[0., 4., 2., 1., 1., 2.],
       [4., 0., 1., 2., 1., 2.],
       [2., 1., 0., 1., 0., 0.],
       [1., 2., 1., 0., 0., 0.],
       [1., 1., 0., 0., 0., 0.],
       [2., 2., 0., 0., 0., 0.]])

'''


# ## Calculation of CO-Occurance Matrix:

# In[218]:


def prllell(wrds,npr):
  mn_dct={}
  chk=len(wrds)
  for i,word in enumerate(wrds):
    dct={}
    index=np.where(lst==word)[0]
    if i==0      :print(' '*(npr*20),'Processer no:',npr)
    if i%50==0   :print(' '*(npr*20),i,'words done')
    if i==chk-1  :print(' '*(npr*20),'Work of Processor %d done'%npr)
    for next in vocab:
      dct[next]=0
      if next in mn_dct.keys():
        dct[next]=mn_dct[next][word]
      else:
        for indx in index:
          end = indx+window+1 if indx+window+1<=lenth else lenth
          start=indx-window  if indx-window>=0  else  0
          check=lst[start:end]
          if next in check and next!=word:
            dct[next]=dct[next]+np.sum(check==next)
    mn_dct[word]=dct
  return mn_dct  


# In[231]:


## with counter
def prllell(wrds,npr):
  if npr==0: print('with counter')
  
  chk=len(wrds)
  mn_dct={}
  for i,word in enumerate(wrds):
    index=np.where(lst==word)[0]
    if i==0      :print(' '*(npr*20),'Processer no:',npr+1)
    if i%50==0   :print(' '*(npr*20),i,'words done')
    if i==chk-1  :print(' '*(npr*20),'Work of Processor %d done'%(npr+1))
        
    mn_dct[word]=dict.fromkeys(vocab,0)

    cntr=Counter()  
    for indx in index:
      cntr.update(Counter(lst[max(0,indx-window):min(lenth,indx+window+1)]))
    cntr[word]=0  
    mn_dct[word].update(cntr)

  return mn_dct  


# In[255]:


vocab=top_wrds_2k_idf
corpus=' '.join(data_train.preprocessed_essay)


# In[232]:


strt=time()
lst=np.array(corpus.split())
lenth=len(lst)
window=5
pool = mp.Pool()
res={}
no_prcss= mp.cpu_count()
nn=int(len(vocab)/no_prcss)
for i in range(no_prcss):
  res[i]=pool.apply_async(prllell,args=(vocab[i*nn:(i+1)*nn],i))
pool.close()
pool.join() 
mn_dct={}
for i in range(no_prcss):
  mn_dct.update(res[i].get())
end=time()
print('Time elapsed: ',(end-strt)/60,' Minutes')


# In[ ]:


##with counter t: 2.5777287324269613 


# In[ ]:


##without counter t: 14.021189844608307


# In[254]:


mn_dct['hotel']['evacu']


# In[251]:


len(lstt)


# In[253]:


lstt[2332]


# In[221]:


## Making Co-Variance matrix ## with out counter
vocab_mtrx_svd=np.zeros((len(vocab),len(vocab)))
for i in range(len(vocab)):
  vocab_mtrx_svd[i,:]=np.array(list((mn_dct[vocab[i]].values())))


# In[233]:


## Making Co-Variance matrix ## with counter
vocab_mtrx_svd=np.zeros((len(vocab),len(vocab)))
for i in range(len(vocab)):
  vocab_mtrx_svd[i,:]=np.array(list((mn_dct[vocab[i]].values()))[:2000])


# In[230]:


max([max(vocab_mtrx_svd[i]) for i in range(2000)])


# In[234]:


max([max(vocab_mtrx_svd[i]) for i in range(2000)])


# ##<h2>2.3 Applying TruncatedSVD and Calculating Vectors for `essay` and `project_title`</h2>

# In[ ]:


n_components=[2,5,10,30,50,100,150,200,250,300,400,500]
variance=[]
for i in tqdm(n_components):
  svd=TruncatedSVD(n_components=i)
  svd.fit(vocab_mtrx_svd)
  variance.append(svd.explained_variance_ratio_.sum())


# In[ ]:


y=np.array(variance)*100
plt.figure(figsize=(15,7))
plt.plot(n_components,y,'r')
plt.scatter(n_components,y,c='b')
plt.xticks(n_components)
plt.yticks(y)
plt.title('Explain total %age of Variance along n_components')
plt.xlabel('n_components')
plt.ylabel('Total %-age of Variance explained')
plt.grid()


# In[ ]:


opt_n_comp=250 ## Explained Total variance is around 94%
svd=TruncatedSVD(n_components=opt_n_comp)
X_transfrmd=svd.fit_transform(vocab_mtrx_svd)


# In[ ]:


word_vec=dict.fromkeys(vocab)
for i,key in enumerate(vocab):
  word_vec[key]=X_transfrmd[i]  


# ## Calculating TFIDF weighted word vectors (calculated above) and All Additional features:

# In[ ]:


### transforming tfidf vectorized word_vec from aboe calculated vectors
def tfidfc(data):
  '''Making dictionary of words and corresponding tfidf value'''
  tfidf_model = TfidfVectorizer()
  tfidf_model.fit(data)
  dictionary = dict(zip(tfidf_model.get_feature_names(), list(tfidf_model.idf_)))
  tfidf_words = set(tfidf_model.get_feature_names())
  return dictionary,tfidf_words


def w2v_tfidfc(data,dictionary,tfidf_words,d=250):
  tfidf_w2v_vectors= []; # the avg-w2v for each sentence/review is stored in this list
  for sentence in tqdm(data): # for each review/sentence
    vector = np.zeros(d) # as word vectors are of zero length
    tf_idf_weight =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if (word in word_vec_words) and (word in tfidf_words):
            vec = word_vec[word] # getting the vector for each word
            # here we are multiplying idf value(dictionary[word]) and the tf value((sentence.count(word)/len(sentence.split())))
            tf_idf = dictionary[word]*(sentence.count(word)/len(sentence.split())) # getting the tfidf value for each word
            vector += (vec * tf_idf) # calculating tfidf weighted w2v
            tf_idf_weight += tf_idf
    if tf_idf_weight != 0:
        vector /= tf_idf_weight
        
    tfidf_w2v_vectors.append(vector)
  return tfidf_w2v_vectors


# In[ ]:


word_vec_words=set(word_vec.keys())


# In[ ]:


essay_tfidf_dict,essay_tfidf_words=tfidfc(data_train.preprocessed_essay.values)


# In[ ]:


tfidf_w2v_essay_train=w2v_tfidfc(data_train.preprocessed_essay.values,essay_tfidf_dict,essay_tfidf_words)


# In[ ]:


tfidf_w2v_essay_cv=w2v_tfidfc(data_cv.preprocessed_essay.values,essay_tfidf_dict,essay_tfidf_words)
tfidf_w2v_essay_test=w2v_tfidfc(data_test.preprocessed_essay.values,essay_tfidf_dict,essay_tfidf_words)


# In[ ]:


from sklearn.preprocessing import MaxAbsScaler
essay_w2v_tfidf_stnd=MaxAbsScaler()
tfidf_w2v_essay_train=essay_w2v_tfidf_stnd.fit_transform(tfidf_w2v_essay_train)
tfidf_w2v_essay_cv=essay_w2v_tfidf_stnd.transform(tfidf_w2v_essay_cv)
tfidf_w2v_essay_test=essay_w2v_tfidf_stnd.transform(tfidf_w2v_essay_test)


# In[ ]:


title_tfidf_dict,title_tfidf_words=tfidfc(data_train.preprocessed_title.values)


# In[ ]:


tfidf_w2v_title_train=w2v_tfidfc(data_train.preprocessed_title.values,title_tfidf_dict,title_tfidf_words)
tfidf_w2v_title_cv=w2v_tfidfc(data_cv.preprocessed_title.values,title_tfidf_dict,title_tfidf_words)
tfidf_w2v_title_test=w2v_tfidfc(data_test.preprocessed_title.values,title_tfidf_dict,title_tfidf_words)


# In[ ]:


title_w2v_tfidf_stnd=MaxAbsScaler()
tfidf_w2v_title_train=title_w2v_tfidf_stnd.fit_transform(tfidf_w2v_title_train)
tfidf_w2v_title_cv=title_w2v_tfidf_stnd.transform(tfidf_w2v_title_cv)
tfidf_w2v_title_test=title_w2v_tfidf_stnd.transform(tfidf_w2v_title_test)


# ### Making Features

# #### Number of words in Title

# In[ ]:


data_train['no of words title']=data_train.preprocessed_title.apply(lambda x: len(x.split()))
data_cv['no of words title']=data_cv.preprocessed_title.apply(lambda x: len(x.split()))
data_test['no of words title']=data_test.preprocessed_title.apply(lambda x: len(x.split()))


# #### Number of words in Essays

# In[ ]:


data_train['no of words essay']=data_train.preprocessed_essay.apply(lambda x: len(x.split()))
data_cv['no of words essay']=data_cv.preprocessed_essay.apply(lambda x: len(x.split()))
data_test['no of words essay']=data_test.preprocessed_essay.apply(lambda x: len(x.split()))


# #### Sentiment scores of each of Essays

# In[ ]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer 
import nltk
nltk.download('vader_lexicon')


# In[ ]:


def sentiment_anlsis(text,data):
  sentiment = SentimentIntensityAnalyzer()
  pos=[]
  neg=[]
  neut=[]
  comp=[]
  for i in tqdm(text):
    ss=sentiment.polarity_scores(i)
    pos.append(ss['pos'])
    neg.append(ss['neg'])
    neut.append(ss['neu'])
    comp.append(ss['compound'])  
  
  data['positive']=pos
  data['negitive']=neg
  data['neutral']=neut
  data['compoud score']=comp
  return data


# In[ ]:


data_train=sentiment_anlsis(data_train.preprocessed_essay.values,data_train)
data_cv=sentiment_anlsis(data_cv.preprocessed_essay,data_cv)
data_test=sentiment_anlsis(data_test.preprocessed_essay,data_test)


# In[ ]:


## The new features we are adding
data_train.iloc[:5,-7:]


# #### Combining All the above Features:

# In[ ]:


## Normalizing the new Features
scaler=MinMaxScaler()
new_feat_train=scaler.fit_transform(data_train.iloc[:,-7:])
new_feat_cv=scaler.transform(data_cv.iloc[:,-7:])
new_feat_test=scaler.transform(data_test.iloc[:,-7:])


# In[ ]:


######### Combining categorical, numerical, tfidf word2vec vectorized title and essay#####
X_tfidf_w2v_train=hstack((catg_num_train,tfidf_w2v_essay_train,tfidf_w2v_title_train,new_feat_train))
X_tfidf_w2v_cv=hstack((catg_num_cv,tfidf_w2v_essay_cv,tfidf_w2v_title_cv,new_feat_cv))
X_tfidf_w2v_test=hstack((catg_num_test,tfidf_w2v_essay_test,tfidf_w2v_title_test,new_feat_test))
X_tfidf_w2v_train.shape,X_tfidf_w2v_cv.shape,X_tfidf_w2v_test.shape


# #<h2>2.5 Apply XGBoost on the Final Features from the above section</h2>

# https://xgboost.readthedocs.io/en/latest/python/python_intro.html

# ### Getting Feature Names

# In[ ]:


feat_dict={0:tchr_prfx_ncode,1:state_vectorizer,2:project_grade_ncode,3:vectorizer,4:sub_categories_one_hot,5:'price',6:'Previous posted projects'}
feat=[]
for i in range(len(feat_dict)):
  if i==5:
    feat.append(feat_dict[5])
  if i==6:
    feat.append(feat_dict[6])
  if i!=5 and i!=6:
    key=list(feat_dict[i].vocabulary_.keys())
    values=list(feat_dict[i].vocabulary_.values())
    sr=np.argsort(values)
    feat.extend([key[sr[i]] for i in range(len(sr))])
feat.extend(['essay_svd_comp_'+str(i) for i in range(250)])
feat.extend(['title_svd_comp_'+str(i) for i in range(250)])
feat.extend(list(data_train.columns[-7:]))

mdfd_ft=[]
for i,ftr in enumerate(feat):
  if ftr in mdfd_ft:
    mdfd_ft.append(ftr+str(i))
  else :
    mdfd_ft.append(ftr)


# ## XGBOOST:

# In[ ]:


import xgboost as xgb


# In[ ]:


w8=Y_test.project_is_approved.value_counts()[1]/Y_test.project_is_approved.value_counts()[0]
w8s=[w8 if i==0 else 1 for i in Y_train.values]


# In[ ]:


parameters = {'num_boost_round': [10, 50, 100, 250],
              'eta': [0.1,0.2, 0.3],
              'max_depth': [2, 3, 5, 10],
}


# In[ ]:


dtrain = xgb.DMatrix(X_tfidf_w2v_train,weight=w8s,label=Y_train.values,feature_names=mdfd_ft)
dcv = xgb.DMatrix(X_tfidf_w2v_cv,label=Y_cv.values,feature_names=mdfd_ft)
dtest = xgb.DMatrix(X_tfidf_w2v_test,feature_names=mdfd_ft)
evallist = [(dcv, 'eval'), (dtrain, 'train')]


# In[ ]:


get_ipython().run_cell_magic('time', '', "eval_dct={}\nprms={}\nxgbst=None\ni=0\nj=0\nfor num_boost in parameters['num_boost_round']:\n  for eta_parm in parameters['eta']:\n    chk_eval=[]\n    chk_trn=[]\n    for depth in parameters['max_depth']:\n      prms.update({i:{'num_boost_round':num_boost,'eta':eta_parm,'max_depth':depth}})        \n      param={'max_depth': depth, 'eta': eta_parm, 'objective': 'binary:logistic','nthread':2,'eval_metric':'auc'}\n      eval_dct[i]={}\n      xgbst = xgb.train(param,num_boost_round=num_boost,evals=evallist,dtrain=dtrain,evals_result=eval_dct[i],verbose_eval=False)\n      chk_eval.append(eval_dct[i]['eval']['auc'])\n      chk_trn.append(eval_dct[i]['train']['auc'])\n      i+=1\n    max_eval=[max(chk_eval[i]) for i in range(len(chk_eval))]\n    loc_mx_cv=np.argmax(max_eval)\n    train_prf=chk_trn[loc_mx_cv][np.argmax(chk_eval[loc_mx_cv])]\n    print('For Parameters :\\n',prms[j+loc_mx_cv])\n    print('Max AUC on CV is %4f and AUC on Train is %4f'%(max_eval[loc_mx_cv],train_prf))\n    print('=='*40)\n    j=i")


# In[ ]:


eval_prfrmnc=[eval_dct[i]['eval']['auc'] for i in range(len(prms))]
train_prfrmnc=[eval_dct[i]['train']['auc'] for i in range(len(prms))]
max_p=[max(eval_prfrmnc[i]) for i in range(len(prms))]
loc_cv=np.argmax(max_p)
loc=np.argmax(eval_prfrmnc[loc_cv])
train_auc_prf=train_prfrmnc[loc_cv][loc]


# In[ ]:


prms[loc_cv],max_p[loc_cv],train_auc_prf


# In[ ]:


eval_test={}
param={'silent': 1, 'objective': 'binary:logistic','nthread':2,'eval_metric':'auc'}
param.update(prms[loc_cv])
xgbst = xgb.train(param,evals=evallist,verbose_eval=0,dtrain=dtrain,evals_result=eval_test,num_boost_round=prms[loc_cv]['num_boost_round'])


# In[ ]:


fig, ax=plt.subplots()
fig.set_size_inches(15,10)
xgb.plot_importance(xgbst,max_num_features=30,ax=ax);


# In[ ]:


ypred_test = xgbst.predict(dtest)
ypred_train = xgbst.predict(dtrain)
test_auc=roc_auc_score(Y_test.values,ypred)
train_auc=roc_auc_score(Y_train.values,ypred_train)
print('Train  and Test AUC are %4f & %4f'%(train_auc,test_auc))


# In[ ]:


cf_mtrx_test=confusion_matrix(Y_test.values,np.array([1 if i>=0.5 else 0 for i in ypred_test]))
cf_mtrx_train=confusion_matrix(Y_train.values,np.array([1 if i>=0.5 else 0 for i in ypred_train]))
fig,ax=plt.subplots(1,2,figsize=(17,6))
fig.subplots_adjust(wspace=0.3)
ax1=sns.heatmap(cf_mtrx_test,annot=True, fmt="d",cmap="YlGnBu",ax=ax[0])
ax1.set_title('Confusion Matrix on Test Data')
ax1.set_xlabel('Predicted Class')
ax1.set_ylabel('Actual Class')
ax2=sns.heatmap(cf_mtrx_train,annot=True, fmt="d",cmap="YlGnBu",ax=ax[1])
ax2.set_title('Confusion Matrix on Train Data')
ax2.set_xlabel('Predicted Class')
ax2.set_ylabel('Actual Class');


# In[ ]:


fpr, tpr, thresholds = metrics.roc_curve(Y_test.values,ypred_test)
fpr_tr, tpr_tr, _ = metrics.roc_curve(Y_train.values,ypred_train)
plt.figure(figsize=(7,6))
plt.plot(fpr,tpr,c='r',label='Test ROC curve (area = %0.3f)' % test_auc);
plt.plot(fpr_tr,tpr_tr,label='Train ROC curve (area = %0.3f)' % train_auc);
plt.plot([0,1],[0,1],'--',c='b')
plt.xlim(0,1)
plt.legend()
plt.title('ROC curves of Train and Test data')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.ylim(0,1);


# ##<h1>3. Conclusion</h1>

# * By XGBOOST we got pretty decent AUC for Test data
# * Selected best hyperparameter by maximizing the AUC on CV data
# * By feature importance we can see, svd features aren't much in top 10.
# * Even though we caluclated co-occurance matrix for only top 2000 words(based on idf values), which is very small in size compared to size of whole vocablulary of total document, the performance might be increased if we able to consider more words.
# * Even if we consider more words, calculating co-occurance matrix is mostly time expensive.
# * If computational power and time permits, then there might be increase in performance, which needed to be experimented
