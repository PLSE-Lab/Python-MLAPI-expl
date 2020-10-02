#!/usr/bin/env python
# coding: utf-8

# # # KNN CLASSFIER on Amazon Reviews by  Sundar Viswanathan
# 
# Data Source: https://www.kaggle.com/snap/amazon-fine-food-reviews
# 
# #### Objective:
# 
# Perform KNN- Classification (both Brute Force and KD-Tree)  for the following vectors
# 
# a) BOW
# b) TF-IDF
# c) Avg Word2Vec
# d) TF-IDF Word2Vec
# 
# 

# In[ ]:


# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")

import sqlite3
import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from nltk.stem.porter import PorterStemmer

import re
# Tutorial about Python regular expressions: https://pymotw.com/2/re/
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pickle



# In[ ]:



# using the SQLite Table to read data.
import sqlite3
show_tables = "select tbl_name from sqlite_master where type = 'table'" 
conn = sqlite3.connect('../input/database.sqlite') 
pd.read_sql(show_tables,conn)


# In[ ]:



#filtering only positive and negative reviews i.e. 
# not taking into consideration those reviews with Score=3
filtered_data = pd.read_sql_query("""SELECT * FROM Reviews WHERE Score != 3""", conn) 


# In[ ]:


# Give reviews with Score>3 a positive rating, and reviews with a score<3 a negative rating.
def partition(x):
    if x < 3:
        return '0'
    return '1'

#changing reviews with score less than 3 to be positive and vice-versa
actualScore = filtered_data['Score']
positiveNegative = actualScore.map(partition) 
filtered_data['Score'] = positiveNegative


# In[ ]:


#print(filtered_data.shape) #looking at the number of attributes and size of the data
filtered_data.head()


# # Data Cleaning: Deduplication
# #It is observed (as shown in the table below) that the reviews data had many duplicate entries. Hence it was necessary to remove duplicates in order to get unbiased results for the analysis of the data. Following is an example:

# In[ ]:


display= pd.read_sql_query("""
SELECT *
FROM Reviews
WHERE Score != 3 AND UserId="AR5J8UI46CURR"
ORDER BY ProductID
""", conn)
display.head()


# In[ ]:


#Sorting data according to ProductId in ascending order
sorted_data=filtered_data.sort_values('ProductId', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')


# In[ ]:


#Deduplication of entries
final=sorted_data.drop_duplicates(subset={"UserId","ProfileName","Time","Text"}, keep='first', inplace=False)
final.shape


# In[ ]:


#Checking to see how much % of data still remains
(final['Id'].size*1.0)/(filtered_data['Id'].size*1.0)*100


# *## Observation:-   It was also seen that in two rows given below the value of HelpfulnessNumerator is greater than HelpfulnessDenominator which is not practically possible hence these two rows too are removed from calcualtions

# In[ ]:


display= pd.read_sql_query("""
SELECT *
FROM Reviews
WHERE Score != 3 AND Id=44737 OR Id=64422
ORDER BY ProductID
""", conn)


# In[ ]:


final=final[final.HelpfulnessNumerator<=final.HelpfulnessDenominator]


# In[ ]:


#Before starting the next phase of preprocessing lets see the number of entries left
print(final.shape)

#How many positive and negative reviews are present in our dataset?
final['Score'].value_counts()


# In[ ]:


# get 40k random reviews from the overall dataset for Brute and 20K for KD-tree

from random import sample
final_dataset_Bruteforce = final.ix[np.random.choice(final.index, 40000)]
final_dataset_KDTREE = final.ix[np.random.choice(final.index, 20000)]


# In[ ]:


#Sorting data according to Time in ascending order
KNN_DATASET_BRUTEFORCE=final_dataset_Bruteforce.sort_values('Time', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')


# In[ ]:


#Sorting data according to Time in ascending order
KNN_DATASET_KDTREE=final_dataset_KDTREE.sort_values('Time', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')


# ## Text Preprocessing: Stemming, stop-word removal and Lemmatization.
# 
# Now that we have fi****nished deduplication our data requires some preprocessing before we go on further with analysis and making the prediction model.
# 
# Hence in the Preprocessing phase we do the following in the order below:-
# 
# 1. Begin by removing the html tags
# 2. Remove any punctuations or limited set of special characters like , or . or # etc.
# 3. Check if the word is made up of english letters and is not alpha-numeric
# 4. Check to see if the length of the word is greater than 2 (as it was researched that there is no adjective in 2-letters)
# 5. Convert the word to lowercase
# 6. Remove Stopwords
# 7. Finally Snowball Stemming the word (it was obsereved to be better than Porter Stemming)<br>
# 
# After which we collect the words used to describe positive and negative reviews

# In[ ]:


# find sentences containing HTML tags (DATASET FOR BRUTEFORCE)
import re
i=0;
for sent in KNN_DATASET_BRUTEFORCE['Text'].values:
    if (len(re.findall('<.*?>', sent))):
        print(i)
        print(sent)
        break;
    i += 1;


# In[ ]:


# find sentences containing HTML tags (DATASET FOR KD_TREE)
import re
i=0;
for sent in KNN_DATASET_KDTREE['Text'].values:
    if (len(re.findall('<.*?>', sent))):
        print(i)
        print(sent)
        break;
    i += 1;


# In[ ]:


import nltk.corpus
from nltk.corpus import stopwords


# In[ ]:



stop = set(stopwords.words('english')) #set of stopwords
sno = nltk.stem.SnowballStemmer('english') #initialising the snowball stemmer

def cleanhtml(sentence): #function to clean the word of any html-tags
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', sentence)
    return cleantext
def cleanpunc(sentence): #function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    return  cleaned
print(stop)
print('************************************')
print(sno.stem('tasty'))


# In[ ]:


#Code for implementing step-by-step the checks mentioned in the pre-processing phase (BRUTE-FORCE_DATASET)
i=0
str1=' '
final_string=[]
all_positive_words=[] # store words from +ve reviews here
all_negative_words=[] # store words from -ve reviews here.
s=''
for sent in KNN_DATASET_BRUTEFORCE['Text'].values:
    filtered_sentence=[]
    #print(sent);
    sent=cleanhtml(sent) # remove HTMl tags
    for w in sent.split():
        for cleaned_words in cleanpunc(w).split():
            if((cleaned_words.isalpha()) & (len(cleaned_words)>2)):    
                if(cleaned_words.lower() not in stop):
                    s=(sno.stem(cleaned_words.lower())).encode('utf8')
                    filtered_sentence.append(s)
                    if (KNN_DATASET_BRUTEFORCE['Score'].values)[i] == '1': 
                        all_positive_words.append(s) #list of all words used to describe positive reviews
                    if(KNN_DATASET_BRUTEFORCE['Score'].values)[i] == '0':
                        all_negative_words.append(s) #list of all words used to describe negative reviews reviews
                else:
                    continue
            else:
                continue 
    #print(filtered_sentence)
    str1 = b" ".join(filtered_sentence) #final string of cleaned words
    #print("***********************************************************************")
    
    final_string.append(str1)
    i+=1


# In[ ]:


#Code for implementing step-by-step the checks mentioned in the pre-processing phase (KD-TREE_DATASET)
i=0
str1=' '
final_string_KDTREE=[]
all_positive_words=[] # store words from +ve reviews here
all_negative_words=[] # store words from -ve reviews here.
s=''
for sent in KNN_DATASET_KDTREE['Text'].values:
    filtered_sentence=[]
    #print(sent);
    sent=cleanhtml(sent) # remove HTMl tags
    for w in sent.split():
        for cleaned_words in cleanpunc(w).split():
            if((cleaned_words.isalpha()) & (len(cleaned_words)>2)):    
                if(cleaned_words.lower() not in stop):
                    s=(sno.stem(cleaned_words.lower())).encode('utf8')
                    filtered_sentence.append(s)
                    if (KNN_DATASET_KDTREE['Score'].values)[i] == '1': 
                        all_positive_words.append(s) #list of all words used to describe positive reviews
                    if(KNN_DATASET_KDTREE['Score'].values)[i] == '0':
                        all_negative_words.append(s) #list of all words used to describe negative reviews reviews
                else:
                    continue
            else:
                continue 
    #print(filtered_sentence)
    str1 = b" ".join(filtered_sentence) #final string of cleaned words
    #print("***********************************************************************")
    
    final_string_KDTREE.append(str1)
    i+=1


# In[ ]:


KNN_DATASET_BRUTEFORCE['CleanedText']=final_string #adding a column of CleanedText which displays the data after pre-processing of the review 
KNN_DATASET_BRUTEFORCE['CleanedText']=KNN_DATASET_BRUTEFORCE['CleanedText'].str.decode("utf-8")


# In[ ]:


KNN_DATASET_KDTREE['CleanedText']=final_string_KDTREE #adding a column of CleanedText which displays the data after pre-processing of the review 
KNN_DATASET_KDTREE['CleanedText']=KNN_DATASET_KDTREE['CleanedText'].str.decode("utf-8")


# In[ ]:


# ============================== loading libraries ===========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn import model_selection


# ![](http://)# #  splitting the 40 k reviews in to training and test data(for BRUTEFORCE) (limiting to 40K reviews** since we run in to Memory Error for reviews more than 40 K)**

# In[ ]:


#  data preprocessing

# define column names
names = ['Id', 'ProductId', 'UserId', 'ProfileName', 'HelpfulnessNumerator',
       'HelpfulnessDenominator', 'Time', 'Summary', 'Text','CleanedText']


# create design matrix X and target vector y
X_bf =  KNN_DATASET_BRUTEFORCE[names]
y_bf = KNN_DATASET_BRUTEFORCE['Score']

X_train_bf, X_test_bf, y_train_bf, y_test_bf = model_selection.train_test_split(X_bf, y_bf, test_size=0.3, random_state=0)


# * # ASSIGNMENT- PART 1: KNN  CLASSIFIER  ON BOW  VECTOR

#    # STEP 1) Computing the Bag of Words (BoW) ( for BRUTEFORCE)

# In[ ]:


# Get the BoW vector for Train and Test data

count_vect = CountVectorizer() 

bow_bf = count_vect.fit(X_train_bf['CleanedText'].values)

bow_train_bf = bow_bf.transform(X_train_bf['CleanedText'].values)

bow_test_bf  = bow_bf.transform(X_test_bf['CleanedText'].values)


# > # Step 2)** KNN Classifier for BOW (using BRUTE force algo and Simple CV)**
# 

# In[ ]:


# ============================== loading libraries ===========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn import model_selection
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
# =============================================================================================


# In[ ]:


### KNN using Cross Validation (K=1) and computing acuracy and CM
knn = KNeighborsClassifier(1)
knn.fit(bow_train_bf, y_train_bf)
pred = knn.predict(bow_test_bf)


# In[ ]:


# Print Accuracy and Confusion Matrix
acc = accuracy_score(y_test_bf, pred, normalize=True) * float(100)
print('\n****Test accuracy for k = 1 is %d%%' % (acc))

labels = ['0','1']
cm = confusion_matrix(y_test_bf, pred)

print("---------------------Plot of Confusion Matrix.....................")

df_cm = pd.DataFrame(cm)

import seaborn as sn
sn.set(font_scale=2) #for label size

sn.heatmap(df_cm, annot=True, cmap='Oranges', annot_kws={"size": 20})# font size


# In[ ]:


# Print Training and Test Error rate (BF AND SIMPLE CV)


    train_error_knn_BF_simple_cv = np.float128(1)- np.float128 (knn.score(bow_train_bf, y_train_bf))
    
    #Compute accuracy on the test set
    test_error_knn_BF_simple_cv = np.float128(1)- np.float128(knn.score(bow_test_bf, y_test_bf))
    
    print('\n****Training Error for k = 1 (simple CV) is %.2f%%' % (train_error_knn_BF_simple_cv))
    
    print('\n****Test Error for k = 1 (10-FOLD CV) is %.2f%%' % (test_error_knn_BF_simple_cv))
    
    


# # STEP 3)  KNN Classifier for BOW (using BRUTE force algo and 10-FOLD CV)

# In[ ]:


#  Finding K in KNN using 10- Fold  Cross-Validation

knn_BF_10FOLD_CV = KNeighborsClassifier(algorithm = 'brute')
parameters = {"n_neighbors": np.arange(1, 5, 2),
	"metric": ["euclidean"]}
clf = GridSearchCV(knn_BF_10FOLD_CV, parameters, cv=10)
clf.fit(bow_train_bf, y_train_bf)


# In[ ]:


clf.best_params_
clf.score(bow_test_bf,y_test_bf)
y_pred = clf.best_estimator_.predict(bow_test_bf)


# In[ ]:


#  plotting the  CONFUSION MATRIX
labels = ['0','1']
cm = confusion_matrix(y_test_bf, y_pred, labels)
print("Computation over... Confusion Matrix is as follows .....................")
df_cm = pd.DataFrame(cm)

import seaborn as sn
sn.set(font_scale=2) #for label size

sn.heatmap(df_cm, annot=True, cmap='Oranges', annot_kws={"size": 20})# font size


# In[ ]:


# Getting the Training and Test Error rate (BF AND 10-fold CV)

    train_error_knn_BF_10FOLD_cv = np.float(1) - np.float(clf.score(bow_train_bf, y_train_bf))
    
    #Compute accuracy on the test set
    test_error_knn_BF_10FOLD_cv = 1- np.float(clf.score(bow_test_bf, y_test_bf))
    
  


# In[ ]:


# Printing  the Training and Test Error rate (BF AND 10-FOLD CV) 
print('\n****Training Error for {0}'.format(clf.best_params_),'with 10-FOLD CV is %.3f%%'%(train_error_knn_BF_10FOLD_cv))
print('\n****Test Error for {0}'.format(clf.best_params_), 'with 10-FOLD CV is %.3f%%'%(test_error_knn_BF_10FOLD_cv))


# In[ ]:


from prettytable import PrettyTable
tablenew = PrettyTable()


# In[ ]:


# Display the Training and Test Errors 

tablenew.field_names = (["Model", "hyper parameter (k)", "train error", "test error"])

tablenew.add_row(["BOW-KNN with BF and simple CV", 1, train_error_knn_BF_simple_cv, test_error_knn_BF_simple_cv])
tablenew.add_row(["BOW -KNN_BF and 10 fold CV", clf.best_params_, train_error_knn_BF_10FOLD_cv, test_error_knn_BF_10FOLD_cv])


# In[ ]:


print (tablenew)


# # Step 4) Computing the Bag of Words (BoW) ( for KD-TREE)

# In[ ]:


# ============================== data preprocessing ===========================================

# define column names
names = ['Id', 'ProductId', 'UserId', 'ProfileName', 'HelpfulnessNumerator',
       'HelpfulnessDenominator', 'Time', 'Summary', 'Text','CleanedText']


# create design matrix X and target vector y
X_kd =  KNN_DATASET_KDTREE[names]
y_kd = KNN_DATASET_KDTREE['Score']
X_train_kd, X_test_kd, y_train_kd, y_test_kd = model_selection.train_test_split(X_kd, y_kd, test_size=0.3, random_state=0)


# In[ ]:


# Get the BoW vector for Train and Test data (kd tree)

count_vect = CountVectorizer() 

bow_kd = count_vect.fit(X_train_kd['CleanedText'].values)


# In[ ]:


bow_train_kd = bow_kd.transform(X_train_kd['CleanedText'].values)

bow_test_kd  = bow_kd.transform(X_test_kd['CleanedText'].values)


# In[ ]:


# Standardizing the data for KD-TREE
from sklearn.preprocessing import StandardScaler

standardized_data_kd_train =StandardScaler(with_mean=False).fit_transform(bow_train_kd) 

standardized_data_kd_test =StandardScaler(with_mean=False).fit(bow_train_kd).transform(bow_test_kd) 


# # Step 5) KNN Classifier for BOW (using KD-TREE algo and Simple CV)**
# 

# In[ ]:


# Reducing Dimensions using Truncated SVD method

from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix


standardized_data_sparse_train = csr_matrix(standardized_data_kd_train)

standardized_data_sparse_test = csr_matrix(standardized_data_kd_test)

tsvd = TruncatedSVD(n_components=10)

standardized_data_sparse_tsvd_train_kd = tsvd.fit(standardized_data_sparse_train).transform(standardized_data_sparse_train)

standardized_data_sparse_tsvd_test_kd = tsvd.fit(standardized_data_sparse_train).transform(standardized_data_sparse_test)


# In[ ]:


#tsvd = TruncatedSVD(n_components=2, random_state=42)
print(tsvd.explained_variance_ratio_)


# Here we will take N_COMPONENTS = 10, **since beyond 10, variance ratio drops by 50%**

# In[ ]:


print('Original number of features:', standardized_data_sparse_train.shape[1])
print('Reduced number of features:', standardized_data_sparse_tsvd_train_kd.shape[1])


# In[ ]:


### KNN using Cross Validation (K=1) and computing acuracy and CM (using KDTREE Algo)
knn_KDTREE = KNeighborsClassifier(1, algorithm ='kd_tree')
knn_KDTREE.fit(standardized_data_sparse_tsvd_train_kd, y_train_kd)
pred_KDTREE = knn_KDTREE.predict(standardized_data_sparse_tsvd_test_kd)


# In[ ]:


# Print Accuracy and Confusion Matrix


acc = accuracy_score(y_test_kd, pred_KDTREE, normalize=True) * float(100)
print('\n****Test accuracy for k = 1 is %d%%' % (acc))

labels = ['0','1']
cm = confusion_matrix(y_test_kd, pred_KDTREE)

print("---------------------Plot of Confusion Matrix.....................")

df_cm = pd.DataFrame(cm)

import seaborn as sn
sn.set(font_scale=2) #for label size

sn.heatmap(df_cm, annot=True, cmap='Oranges', annot_kws={"size": 20})# font size


# In[ ]:


# Getting the Training and Test Error rate (BF AND 10-fold CV)

train_error_knn_KD_simple_cv = 1 - knn_KDTREE.score(standardized_data_sparse_tsvd_train_kd, y_train_kd)
    
#Compute accuracy on the test set
    
test_error_knn_KD_simple_cv = 1- knn_KDTREE.score(standardized_data_sparse_tsvd_test_kd, y_test_kd)
    
  


# In[ ]:


# Printing  the Training and Test Error rate (KD AND simple CV) 

print('\n****Training Error for  KNN with KD and simple CV is %.3f%%'%(train_error_knn_KD_simple_cv))

print('\n****Test Error for  for  KNN with KD and simple CV is %.3f%%'%(test_error_knn_KD_simple_cv))


# In[ ]:


tablenew.add_row(["BOW -KNN_KD and simple CV", 1,train_error_knn_KD_simple_cv,  test_error_knn_KD_simple_cv])


print (tablenew)


# > > # Step 6) KNN Classifier for BOW (using KD-TREE algo and 10-FOLD CV)**
# 

# In[ ]:


#  Finding K in KNN using 10- Fold  Cross-Validation

knn_KDTREE_10FOLD = KNeighborsClassifier(algorithm = 'kd_tree')
parameters = {"n_neighbors": np.arange(1, 21, 2),
	"metric": ["euclidean"]}
clf_BOW_KDTREE_10FOLD = GridSearchCV(knn_KDTREE_10FOLD, parameters, cv=10)
clf_BOW_KDTREE_10FOLD.fit(standardized_data_sparse_tsvd_train_kd, y_train_kd)


# In[ ]:


clf_BOW_KDTREE_10FOLD.best_params_
clf_BOW_KDTREE_10FOLD.score(standardized_data_sparse_tsvd_test_kd,y_test_kd)
y_pred = clf_BOW_KDTREE_10FOLD.best_estimator_.predict(standardized_data_sparse_tsvd_test_kd)


# In[ ]:


# plotting the  CONFUSION MATRIX
labels = ['0','1']
cm = confusion_matrix(y_test_kd, y_pred, labels)
print("Plot of the Confusion Matrix.....................")
df_cm = pd.DataFrame(cm)

sn.set(font_scale=2) #for label size
sn.heatmap(df_cm, annot=True, cmap='Oranges', annot_kws={"size": 20})# font size


# In[ ]:


# Getting the Training and Test Error rate (BF AND 10-fold CV)

train_error_knn_BOW_KD_10FOLD_cv = 1 - clf_BOW_KDTREE_10FOLD.score(standardized_data_sparse_tsvd_train_kd, y_train_kd)
    
#Compute accuracy on the test set

test_error_knn_BOW_KD_10FOLD_cv = 1- clf_BOW_KDTREE_10FOLD.score(standardized_data_sparse_tsvd_test_kd, y_test_kd)
    
# Printing  the Training and Test Error rate (BF AND 10-FOLD CV) 

print('\n****Training Error for {0}'.format(clf_BOW_KDTREE_10FOLD.best_params_),'with 10-FOLD CV is %.3f%%'%(train_error_knn_BOW_KD_10FOLD_cv))

print('\n****Test Error for {0}'.format(clf_BOW_KDTREE_10FOLD.best_params_), 'with 10-FOLD CV is %.3f%%'%(test_error_knn_BOW_KD_10FOLD_cv))


# In[ ]:


tablenew.add_row(["BOW -KNN_KD and 10-fold CV", clf_BOW_KDTREE_10FOLD.best_params_,round(train_error_knn_BOW_KD_10FOLD_cv,3),  round(test_error_knn_BOW_KD_10FOLD_cv,3)])


print (tablenew)


# * 1. # Part 2: KNN classifier on TF-IDF vector  for AMZN reviews

# # Step 1) computing the TF-IDF  vector (for BF algo)

# In[ ]:


tf_idf_vect_bf = TfidfVectorizer(ngram_range=(1,2))

tfidf_bf = tf_idf_vect_bf.fit(X_train_bf['CleanedText'].values)

tfidf_train_bf = tfidf_bf.transform(X_train_bf['CleanedText'].values)

tfidf_test_bf  = tfidf_bf.transform(X_test_bf['CleanedText'].values)

#final_tf_idf = tf_idf_vect.fit_transform(final['CleanedText'].values)


#print("the type of count vectorizer (TRAIN DATA) ",type(tfidf_train_bf))
#print("the shape of out text TFIDF vectorizer (TRAIN DATA) ",tfidf_train_bf.get_shape())
#print("the number of unique words including both unigrams and bigrams(TRAIN DATA) ", tfidf_train_bf.get_shape()[1])


# In[ ]:


features_bf = tf_idf_vect_bf.get_feature_names()
print("some sample features(unique words in the corpus)",features_bf[10000:10010])


# In[ ]:


# source: https://buhrmann.github.io/tfidf-analysis.html
def top_tfidf_feats(row, features, top_n=25):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df

top_tfidf_bf = top_tfidf_feats(tfidf_train_bf[1,:].toarray()[0],features_bf,25)


# In[ ]:


top_tfidf_bf


# # Step 2)** KNN Classifier for TFIDF (using BRUTE force algo and Simple CV)**
# 

# In[ ]:


### KNN using Cross Validation (K=1) and computing acuracy and CM
knn_TFIDF_BF_SIMPLE_CV = KNeighborsClassifier(1, algorithm = 'brute')
knn_TFIDF_BF_SIMPLE_CV.fit(tfidf_train_bf, y_train_bf)
pred_TFIDF_BF_SIMPLE_CV = knn_TFIDF_BF_SIMPLE_CV.predict(tfidf_test_bf)


# In[ ]:


# Print Accuracy and Confusion Matrix

Aacc = accuracy_score(y_test_bf, pred_TFIDF_BF_SIMPLE_CV, normalize=True) * float(100)
print('\n****Test accuracy for k = 1 is %d%%' % (acc))

labels = ['0','1']
cm = confusion_matrix(y_test_bf, pred_TFIDF_BF_SIMPLE_CV)

print("---------------------Plot of Confusion Matrix.....................")

df_cm = pd.DataFrame(cm)

#sn.set(font_scale=2) #for label size
sn.heatmap(df_cm, annot=True, cmap='Oranges', annot_kws={"size": 20})# font size


# In[ ]:


# Print Training and Test Error rate FOR TFIDF(BF AND SIMPLE CV)

train_error_knn_TFIDF_BF_simple_cv = 1- knn_TFIDF_BF_SIMPLE_CV.score(tfidf_train_bf, y_train_bf)
    
#Compute accuracy on the test set

test_error_knn_TFIDF_BF_simple_cv = 1- knn_TFIDF_BF_SIMPLE_CV.score(tfidf_test_bf, y_test_bf)

print('\n****Training Error for k = 1 (simple CV) is %.2f%%' % (train_error_knn_TFIDF_BF_simple_cv))
    
print('\n****Test Error for k = 1 (10-FOLD CV) is %.2f%%' % (test_error_knn_TFIDF_BF_simple_cv))


# In[ ]:


tablenew.add_row(["TFIDF-KNN with BF and simple CV", 1, round(train_error_knn_TFIDF_BF_simple_cv,3), round(test_error_knn_TFIDF_BF_simple_cv,3)])

print (tablenew)


# > # Step 3)** KNN Classifier for TFIDF (using BRUTE force algo and 10-FOLD CV)**
# 

# In[ ]:


#  Finding K in KNN using 10- Fold  Cross-Validation

knn_TFIDF_BF_10FOLD_CV = KNeighborsClassifier(algorithm = 'brute')
parameters = {"n_neighbors": np.arange(1, 5, 2),
	"metric": ["euclidean"]}
clf_TFIDF_BF_10FOLD_CV = GridSearchCV(knn_TFIDF_BF_10FOLD_CV, parameters, cv=10)
clf_TFIDF_BF_10FOLD_CV.fit(tfidf_train_bf, y_train_bf)


# In[ ]:


clf_TFIDF_BF_10FOLD_CV.best_params_
clf_TFIDF_BF_10FOLD_CV.score(tfidf_test_bf,y_test_bf)
y_pred_TFIDF_BF_10FOLD_CV = clf_TFIDF_BF_10FOLD_CV.best_estimator_.predict(tfidf_test_bf)


# In[ ]:


#Arriving at CONFUSION MATRIX
labels = ['0','1']
cm = confusion_matrix(y_test_bf, y_pred_TFIDF_BF_10FOLD_CV, labels)
print("Computation over... time to plot Confusion Matrix.....................")
df_cm = pd.DataFrame(cm)
sn.heatmap(df_cm, annot=True, cmap='Oranges', annot_kws={"size": 20})# font size


# In[ ]:


# Getting the Training and Test Error rate (BF AND 10-fold CV)

train_error_knn_TFIDF_BF_10FOLD_CV = 1 - clf_TFIDF_BF_10FOLD_CV.score(tfidf_train_bf, y_train_bf)
    
#Compute accuracy on the test set

test_error_knn_TFIDF_BF_10FOLD_CV = 1- clf_TFIDF_BF_10FOLD_CV.score(tfidf_test_bf, y_test_bf)
    
# Printing  the Training and Test Error rate (BF AND 10-FOLD CV) 

print('\n****Training Error for {0}'.format(clf_TFIDF_BF_10FOLD_CV.best_params_),'with 10-FOLD CV is %.3f%%'%(train_error_knn_TFIDF_BF_10FOLD_CV))

print('\n****Test Error for {0}'.format(clf_TFIDF_BF_10FOLD_CV.best_params_), 'with 10-FOLD CV is %.3f%%'%(test_error_knn_TFIDF_BF_10FOLD_CV))


# In[ ]:


# Adding the entries to Pretty Table and diplaying the Cumulative list of entries appended so far...

tablenew.add_row(["TFIDF-KNN with BF and 10-fold CV", clf_TFIDF_BF_10FOLD_CV.best_params_, train_error_knn_TFIDF_BF_10FOLD_CV, test_error_knn_TFIDF_BF_10FOLD_CV])
print (tablenew)


# # Step 4) Getting TF-IDF Vector  for KD-Tree Algo
# 

# In[ ]:


# Get the TFIDF  vector for Train and Test data (kd tree)

tf_idf_vect_kd = TfidfVectorizer(ngram_range=(1,2))

tfidf_kd = tf_idf_vect_kd.fit(X_train_kd['CleanedText'].values)

tfidf_train_kd = tfidf_kd.transform(X_train_kd['CleanedText'].values)

tfidf_test_kd  = tfidf_kd.transform(X_test_kd['CleanedText'].values)

#print("the type of count vectorizer (TRAIN DATA) ",type(tfidf_train_kd))
#print("the shape of out text TFIDF vectorizer (TRAIN DATA) ",tfidf_train_kd.get_shape())
#print("the number of unique words including both unigrams and bigrams(TRAIN DATA) ", tfidf_train_kd.get_shape()[1])


# In[ ]:


features_kd = tf_idf_vect_kd.get_feature_names()
#print("some sample features(unique words in the corpus)",features_kd[10000:10010])


# In[ ]:


top_tfidf_kd = top_tfidf_feats(tfidf_train_kd[1,:].toarray()[0],features_kd,25)


# In[ ]:


# Standardizing the data for KD-TREE

standardized_data_tfidf_kd_train =StandardScaler(with_mean=False).fit_transform(tfidf_train_kd) 

standardized_data_tfidf_kd_test =StandardScaler(with_mean=False).fit(tfidf_train_kd).transform(tfidf_test_kd) 


# In[ ]:


# TFIDF- Reducing Dimensions using Truncated SVD method

standardized_data_sparse_train_TFIDF = csr_matrix(standardized_data_tfidf_kd_train)

standardized_data_sparse_test_TFIDF = csr_matrix(standardized_data_tfidf_kd_test)

tsvd = TruncatedSVD(n_components=4)

standardized_data_sparse_tsvd_TFIDF_train_kd = tsvd.fit(standardized_data_sparse_train_TFIDF).transform(standardized_data_sparse_train_TFIDF)

standardized_data_sparse_tsvd_TFIDF_test_kd = tsvd.fit(standardized_data_sparse_train_TFIDF).transform(standardized_data_sparse_test_TFIDF)


# In[ ]:


#tsvd = TruncatedSVD(n_components=2, random_state=42)
print(tsvd.explained_variance_ratio_)


# Here we will take N_COMPONENTS = 4, **since beyond 4, variance ratio drops by 50%**

# # Step 5) KNN Classifier for tfidf (using KDTREE force algo and Simple CV)**
# 

# In[ ]:


### KNN using Cross Validation (K=1) and computing acuracy and CM (using KDTREE Algo)
knn_KDTREE_TFIDF_Simple_CV = KNeighborsClassifier(1, algorithm ='kd_tree')
knn_KDTREE_TFIDF_Simple_CV.fit(standardized_data_sparse_tsvd_TFIDF_train_kd, y_train_kd)
pred_KDTREE_TFIDF_Simple_CV = knn_KDTREE_TFIDF_Simple_CV.predict(standardized_data_sparse_tsvd_TFIDF_test_kd)


# In[ ]:


# Print Accuracy and Confusion Matrix
acc = accuracy_score(y_test_kd, pred_KDTREE_TFIDF_Simple_CV, normalize=True) * float(100)
print('\n****Test accuracy for k = 1 is %d%%' % (acc))

labels = ['0','1']
cm = confusion_matrix(y_test_kd, pred_KDTREE_TFIDF_Simple_CV)


# In[ ]:


# plot of CM
print("---------------------Plot of Confusion Matrix.....................")
df_cm = pd.DataFrame(cm)
sn.heatmap(df_cm, annot=True, cmap='Oranges', annot_kws={"size": 20})# font size


# In[ ]:


# Print Training and Test Error rate (BF AND SIMPLE CV)

train_error_knn_TFIDF_KD_simple_cv = 1-knn_KDTREE_TFIDF_Simple_CV.score(standardized_data_sparse_tsvd_TFIDF_train_kd, y_train_kd)
    
#Compute accuracy on the test set
test_error_knn_TFIDF_KD_simple_cv =1-knn_KDTREE_TFIDF_Simple_CV.score(standardized_data_sparse_tsvd_TFIDF_test_kd, y_test_kd)
    
print('\n****Training Error for k = 1 (simple CV) is %.2f%%' % (train_error_knn_TFIDF_KD_simple_cv))
    
print('\n****Test Error for k = 1 (10-FOLD CV) is %.2f%%' % (test_error_knn_TFIDF_KD_simple_cv))


# In[ ]:



tablenew.add_row(["TFIDF-KNN with KD and simple CV", 1, train_error_knn_TFIDF_KD_simple_cv, test_error_knn_TFIDF_KD_simple_cv])


print (tablenew)


# # Step 6) KNN Classifier for tfidf (using KDTREE force algo and 10-fold CV)**
# 

# In[ ]:


#  Finding K in KNN using 10- Fold  Cross-Validation

knn_KDTREE_10FOLD_TFIDF = KNeighborsClassifier(algorithm = 'kd_tree')
parameters = {"n_neighbors": np.arange(1, 21, 2),
	"metric": ["euclidean"]}
clf_KDTREE_10FOLD_TFIDF = GridSearchCV(knn_KDTREE_10FOLD_TFIDF, parameters, cv=10)
clf_KDTREE_10FOLD_TFIDF.fit(standardized_data_sparse_tsvd_TFIDF_train_kd, y_train_kd)


# In[ ]:


clf_KDTREE_10FOLD_TFIDF.best_params_
clf_KDTREE_10FOLD_TFIDF.score(standardized_data_sparse_tsvd_TFIDF_test_kd,y_test_kd)
y_pred_KDTREE_10FOLD_TFIDF = clf_KDTREE_10FOLD_TFIDF.best_estimator_.predict(standardized_data_sparse_tsvd_TFIDF_test_kd)


# In[ ]:


# PLOT OF CONFUSION MATRIX
labels = ['0','1']
cm = confusion_matrix(y_test_kd, y_pred_KDTREE_10FOLD_TFIDF, labels)
print("Computation over... time to plot Confusion Matrix.....................")
df_cm = pd.DataFrame(cm)
sn.heatmap(df_cm, annot=True, cmap='Oranges', annot_kws={"size": 20})# font size


# In[ ]:


# Getting the Training and Test Error rate (BF AND 10-fold CV)
train_error_TFIDF_KD_10FOLD_cv = 1- clf_KDTREE_10FOLD_TFIDF.score(standardized_data_sparse_tsvd_TFIDF_train_kd,y_train_kd)
    
#Compute accuracy on the test set
test_error_TFIDF_KD_10FOLD_cv = 1- clf_KDTREE_10FOLD_TFIDF.score(standardized_data_sparse_tsvd_TFIDF_test_kd,y_test_kd)


# In[ ]:



tablenew.add_row(["TFIDF-KNN with KD and 10-fold CV", clf_KDTREE_10FOLD_TFIDF.best_params_, train_error_TFIDF_KD_10FOLD_cv, test_error_TFIDF_KD_10FOLD_cv])


print (tablenew)


# # Part 3: KNN classifier on ON AVG-WORD2VEC for AMZN reviews

# # Step 1) **computing** the Word2vec  model  (BF algo)

# In[ ]:


i=0
list_of_sent_bf=[]
for sent in KNN_DATASET_BRUTEFORCE['CleanedText'].values:
    list_of_sent_bf.append(sent.split())


# In[ ]:


print(KNN_DATASET_BRUTEFORCE['CleanedText'].values[0])
print("*****************************************************************")
print(list_of_sent_bf[0])


# In[ ]:


# min_count = 1 considers only words that occured atleast 1 times
w2v_model_bf =Word2Vec(list_of_sent_bf,min_count=1,size=50, workers=4)


# In[ ]:


w2v_words_bf = list(w2v_model_bf.wv.vocab)
print("number of words that occured minimum 1 times ",len(w2v_words_bf))
print("sample words ", w2v_words_bf[0:50])


# In[ ]:


w2v_model_bf.wv.most_similar('tasti')


# In[ ]:


w2v_model_bf.wv.most_similar('like')


# # Step 2) compute Avg W2V (dataset is 40k reviews since I am running in to memory error for data > 40k reviews)  (BRUTEFORCE )****

# In[ ]:


sentence_vectors_bf = []; # the avg-w2v for each sentence/review is stored in this list

for sent in list_of_sent_bf: # for each review/sentence
    sent_vec_bf = np.zeros(50) # initialize numpy vector of size 50  
    cnt_words_bf =0; # num of words with a valid vector in the sentence/review
    for word in sent: # for each word in a review/sentence
        if word in w2v_words_bf:
            vec = w2v_model_bf.wv[word]
            sent_vec_bf += vec
            cnt_words_bf += 1
    if cnt_words_bf != 0:
        sent_vec_bf /= cnt_words_bf
    sentence_vectors_bf.append(sent_vec_bf)
print(len(sentence_vectors_bf))
print(len(sentence_vectors_bf[0]))


# In[ ]:


sentence_vectors_bf[1]


# In[ ]:


w2v_model_bf.wv['tasti']


# # Step3)  Arrive at the  Avg W2V dataframe (Test and Train) (BRUTEFORCE)*

# In[ ]:


w2Vdf_bf= pd.DataFrame(sentence_vectors_bf)


# In[ ]:


w2Vdf_bf.shape


# In[ ]:


sentence_vectors_bf_vec= np.array(sentence_vectors_bf)


# In[ ]:


avgword2vec_train_BF= sentence_vectors_bf_vec[0:28000,]


# In[ ]:


avgword2vec_test_BF= sentence_vectors_bf_vec[28000:40001,]


# In[ ]:


avgword2vec_train_BF.shape


# In[ ]:


avgword2vec_test_BF.shape


# # Step 4) KNN Classifier for WORD2VEC  (using BRUTE force algo and Simple CV)

# In[ ]:


### KNN using Cross Validation (K=1) and computing acuracy and CM
knn_avgw2v_bf_simple_cv = KNeighborsClassifier(1, algorithm = 'brute')
knn_avgw2v_bf_simple_cv.fit(avgword2vec_train_BF, y_train_bf)
pred_avgw2v_bf_simple_cv = knn_avgw2v_bf_simple_cv.predict(avgword2vec_test_BF)


# In[ ]:


# Print Accuracy and Confusion Matrix

acc = accuracy_score(y_test_bf, pred_avgw2v_bf_simple_cv, normalize=True) * float(100)
print('\n****Test accuracy for k = 1 is %d%%' % (acc))

labels = ['0','1']
cm = confusion_matrix(y_test_bf, pred_avgw2v_bf_simple_cv)

print("---------------------Plot of Confusion Matrix.....................")

df_cm = pd.DataFrame(cm)

sn.heatmap(df_cm, annot=True, cmap='Oranges', annot_kws={"size": 20})# font size


# In[ ]:


# Getting the Training and Test Error rate (BF AND simple CV)

train_error_knn_avgword2vec_BF_10FOLD_cv = 1- knn_avgw2v_bf_simple_cv.score(avgword2vec_train_BF, y_train_bf)
    
#Compute accuracy on the test set
test_error_knn_avgword2vec_BF_10FOLD_cv = 1- knn_avgw2v_bf_simple_cv.score(avgword2vec_test_BF, y_test_bf)
    
  
# Printing  the Training and Test Error rate (BF AND 10-FOLD CV) 

print('\n****Training Error for AVGW2VEC with simple CV is %.3f%%'%(train_error_knn_avgword2vec_BF_10FOLD_cv))

print('\n****Test Error for AVGW2VEC with simple CV is %.3f%%'%(test_error_knn_avgword2vec_BF_10FOLD_cv))


# In[ ]:



tablenew.add_row(["AVGWORD2VEC-KNN with BF and simple CV",1, round(train_error_knn_avgword2vec_BF_10FOLD_cv,4), round(test_error_knn_avgword2vec_BF_10FOLD_cv,4)])


print (tablenew)


# In[ ]:


tablenew.del_row(8)


# > > # Step 5) KNN Classifier for WORD2VEC  (using BRUTE force algo and 10-FOLD CV) 

# In[ ]:


# ============================== loading libraries ===========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn import model_selection
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
# =============================================================================================


# In[ ]:


#  Finding K in KNN using 10- Fold  Cross-Validation

knn_avgw2v_bf_10foldcv = KNeighborsClassifier(algorithm = 'brute')
parameters = {"n_neighbors": np.arange(1, 5, 2),
	"metric": ["euclidean"]}
clf_avgw2v_bf_10foldcv = GridSearchCV(knn_avgw2v_bf_10foldcv, parameters, cv=10)
clf_avgw2v_bf_10foldcv.fit(avgword2vec_train_BF, y_train_bf)


# In[ ]:


clf_avgw2v_bf_10foldcv.best_params_
clf_avgw2v_bf_10foldcv.score(avgword2vec_test_BF,y_test_bf)
y_pred_avgw2v_bf_10foldcv = clf_avgw2v_bf_10foldcv.best_estimator_.predict(avgword2vec_test_BF)


# In[ ]:


# Arriving at CONFUSION MATRIX and plotting CM
labels = ['0','1']
cm = confusion_matrix(y_test_bf, y_pred_avgw2v_bf_10foldcv, labels)
print("Computation over... time to plot Confusion Matrix.....................")
# Plot of CM
df_cm = pd.DataFrame(cm)

sn.heatmap(df_cm, annot=True, cmap='Oranges', annot_kws={"size": 20})# font size


# In[ ]:


# Getting the Training and Test Error rate (BF AND 10-fold CV)

train_error_knn_AVGWORD2VEC_BF_10FOLD_cv = 1- clf_avgw2v_bf_10foldcv.score(avgword2vec_train_BF,y_train_bf)
    
#Compute accuracy on the test set
test_error_knn_AVGWORD2VEC_BF_10FOLD_cv = 1-  clf_avgw2v_bf_10foldcv.score(avgword2vec_test_BF,y_test_bf)
    
# Printing  the Training and Test Error rate (BF AND 10-FOLD CV) 

print('\n****Training Error for {0}'.format(clf_avgw2v_bf_10foldcv.best_params_),'with 10-FOLD CV is %.3f%%'%(train_error_knn_AVGWORD2VEC_BF_10FOLD_cv))

print('\n****Test Error for {0}'.format(clf_avgw2v_bf_10foldcv.best_params_), 'with 10-FOLD CV is %.3f%%'%(test_error_knn_AVGWORD2VEC_BF_10FOLD_cv))


# In[ ]:



tablenew.add_row(["AVGWORD2VEC-KNN with BF and 10-fold CV", clf_avgw2v_bf_10foldcv.best_params_, train_error_knn_AVGWORD2VEC_BF_10FOLD_cv, test_error_knn_AVGWORD2VEC_BF_10FOLD_cv])


print (tablenew)


# In[ ]:





# # Step 6) **computing** the Word2vec  model  (KD-TREE)

# In[ ]:


i=0
list_of_sent_kd=[]
for sent in KNN_DATASET_KDTREE['CleanedText'].values:
    list_of_sent_kd.append(sent.split())


# In[ ]:


#print(KNN_DATASET_KDTREE['CleanedText'].values[0])
#print("*****************************************************************")
#print(list_of_sent_kd[0])


# In[ ]:


# min_count = 1 considers only words that occured atleast 1 times
w2v_model_kd =Word2Vec(list_of_sent_kd,min_count=1,size=50, workers=4)


# In[ ]:


w2v_words_kd = list(w2v_model_kd.wv.vocab)
print("number of words that occured minimum 1 times ",len(w2v_words_kd))
print("sample words ", w2v_words_kd[0:50])


# In[ ]:


#w2v_model_kd.wv.most_similar('tasti')


# In[ ]:


#w2v_model_bf.wv.most_similar('like')


# # Step 7) compute Avg W2V    (KD-TREE)     
# 

# In[ ]:


sentence_vectors_kd = []; # the avg-w2v for each sentence/review is stored in this list

for sent in list_of_sent_kd: # for each review/sentence
    sent_vec_kd = np.zeros(50) # initialize numpy vector of size 50  
    cnt_words_kd =0; # num of words with a valid vector in the sentence/review
    for word in sent: # for each word in a review/sentence
        if word in w2v_words_kd:
            vec = w2v_model_kd.wv[word]
            sent_vec_kd += vec
            cnt_words_kd += 1
    if cnt_words_kd != 0:
        sent_vec_kd /= cnt_words_kd
    sentence_vectors_kd.append(sent_vec_kd)
print(len(sentence_vectors_kd))
print(len(sentence_vectors_kd[0]))


# In[ ]:


#sentence_vectors_kd[1]


# In[ ]:


#w2v_model_kd.wv['tasti']


# # Step8)  Arrive at the  Avg W2V dataframe (Test and Train) (KD-TREE)

# In[ ]:



w2Vdf_kd= pd.DataFrame(sentence_vectors_kd)


# In[ ]:


w2Vdf_kd.shape


# In[ ]:


sentence_vectors_kd_vec= np.array(sentence_vectors_kd)


# In[ ]:


avgword2vec_train_KD= sentence_vectors_kd_vec[0:14000,]


# In[ ]:


avgword2vec_test_KD= sentence_vectors_kd_vec[14000:20001,]


# In[ ]:


avgword2vec_train_KD.shape


# In[ ]:


avgword2vec_test_KD.shape


# In[ ]:


# Standardizing the data for KD-TREE
#from sklearn.preprocessing import StandardScaler

standardized_data_avgword2vec_kd_train =StandardScaler(with_mean=False).fit_transform(avgword2vec_train_KD) 

standardized_data_avgword2vec_kd_test =StandardScaler(with_mean=False).fit(avgword2vec_train_KD).transform(avgword2vec_test_KD) 


# In[ ]:


standardized_data_avgword2vec_kd_train.shape


# In[ ]:


##  ABGWORD2VEC- Reducing Dimensions using Truncated SVD method

#from sklearn.decomposition import TruncatedSVD
#from scipy.sparse import csr_matrix


standardized_data_sparse_train_WORD2VEC = csr_matrix(standardized_data_avgword2vec_kd_train)

standardized_data_sparse_test_WORD2VEC = csr_matrix(standardized_data_avgword2vec_kd_test)

tsvd_WORD2VEC = TruncatedSVD(n_components=12)


standardized_data_sparse_tsvd_WORD2VEC_train_kd = tsvd_WORD2VEC.fit(standardized_data_sparse_train_WORD2VEC).transform(standardized_data_sparse_train_WORD2VEC)

standardized_data_sparse_tsvd_WORD2VEC_test_kd = tsvd_WORD2VEC.fit(standardized_data_sparse_train_WORD2VEC).transform(standardized_data_sparse_test_WORD2VEC)


# In[ ]:


print(tsvd_WORD2VEC.explained_variance_ratio_)


# ## ** Here we will take N_COMPONENTS = 12, **since beyond 12, variance explained is very minimal%

# # Step 9) KNN Classifier for WORD2VEC  (using KD-TREE algo and Simple CV)

# In[ ]:



### KNN using Cross Validation (K=1) and computing acuracy and CM
knn_avgw2v_kd_simple_cv = KNeighborsClassifier(1, algorithm = 'kd_tree')
knn_avgw2v_kd_simple_cv.fit(standardized_data_sparse_tsvd_WORD2VEC_train_kd, y_train_kd)
pred_avgw2v_kd_simple_cv = knn_avgw2v_kd_simple_cv.predict(standardized_data_sparse_tsvd_WORD2VEC_test_kd)


# In[ ]:


# Print Confusion Matrix

#acc = accuracy_score(y_test_kd, pred_avgw2v_kd_simple_cv, normalize=True) * float(100)
#print('\n****Test accuracy for k = 1 is %d%%' % (acc))

labels = ['0','1']
cm = confusion_matrix(y_test_kd, pred_avgw2v_kd_simple_cv)

print("---------------------Plot of Confusion Matrix.....................")

df_cm = pd.DataFrame(cm)

sn.heatmap(df_cm, annot=True, cmap='Oranges', annot_kws={"size": 20})# font size


# In[ ]:


# Getting the Training and Test Error rate (KD AND simple CV)

train_error_knn_AVGWORD2VEC_KD_simple_cv = 1- knn_avgw2v_kd_simple_cv.score(standardized_data_sparse_tsvd_WORD2VEC_train_kd, y_train_kd)
    
#Compute accuracy on the test set
test_error_knn_AVGWORD2VEC_KD_simple_cv = 1- knn_avgw2v_kd_simple_cv.score(standardized_data_sparse_tsvd_WORD2VEC_test_kd, y_test_kd)
    
  
# Printing  the Training and Test Error rate (BF AND 10-FOLD CV) 

print('\n****Training Error for AVGWORD2VEC with simple CV is %.3f%%'%(train_error_knn_AVGWORD2VEC_KD_simple_cv))

print('\n****Test Error for AVGWORD2VEC with simple CV is %.3f%%'%(test_error_knn_AVGWORD2VEC_KD_simple_cv))


# In[ ]:



tablenew.add_row(["AVGWORD2VEC-KNN with KD and simple CV", 1, train_error_knn_AVGWORD2VEC_KD_simple_cv, test_error_knn_AVGWORD2VEC_KD_simple_cv])


print (tablenew)


# 
# # Step 10) KNN Classifier for WORD2VEC  (using KD_TREE and 10-FOLD CV)
# 

# In[ ]:


#  Finding K in KNN using 10- Fold  Cross-Validation

knn_avgw2v_kd_10foldcv = KNeighborsClassifier(algorithm = 'brute')
parameters = {"n_neighbors": np.arange(1, 5, 2),
	"metric": ["euclidean"]}
clf_avgw2v_kd_10foldcv = GridSearchCV(knn_avgw2v_kd_10foldcv, parameters, cv=10)
clf_avgw2v_kd_10foldcv.fit(standardized_data_sparse_tsvd_WORD2VEC_train_kd, y_train_kd)


# In[ ]:


clf_avgw2v_kd_10foldcv.best_params_
clf_avgw2v_kd_10foldcv.score(standardized_data_sparse_tsvd_WORD2VEC_test_kd,y_test_kd)
y_pred_avgw2v_kd_10foldcv = clf_avgw2v_kd_10foldcv.best_estimator_.predict(standardized_data_sparse_tsvd_WORD2VEC_test_kd)


# In[ ]:


# Arriving at CONFUSION MATRIX and plotting CM

labels = ['0','1']
cm = confusion_matrix(y_test_kd, y_pred_avgw2v_kd_10foldcv, labels)
print("Computation over... time to plot Confusion Matrix.....................")
# Plot of CM
df_cm = pd.DataFrame(cm)
sn.heatmap(df_cm, annot=True, cmap='Oranges', annot_kws={"size": 20})# font size


# In[ ]:


# Getting the Training and Test Error rate (KD AND 10-fold CV)

train_error_knn_WORD2VEC_KD_10FOLD_cv = 1- clf_avgw2v_kd_10foldcv.score(standardized_data_sparse_tsvd_WORD2VEC_train_kd, y_train_kd)
    
    #Compute accuracy on the test set
test_error_knn_WORD2VEC_KD_10FOLD_cv = 1-clf_avgw2v_kd_10foldcv.score(standardized_data_sparse_tsvd_WORD2VEC_test_kd, y_test_kd)
    
  
# Printing  the Training and Test Error rate (BF AND 10-FOLD CV) 

print('\n****Training Error for {0}'.format(clf_avgw2v_kd_10foldcv.best_params_),'with 10-FOLD CV is %.3f%%'%(train_error_knn_WORD2VEC_KD_10FOLD_cv))

print('\n****Test Error for {0}'.format(clf_avgw2v_kd_10foldcv.best_params_), 'with 10-FOLD CV is %.3f%%'%(test_error_knn_WORD2VEC_KD_10FOLD_cv))


# In[ ]:



tablenew.add_row(["AVGWORD2VEC-KNN with KD and 10-fold CV", clf_avgw2v_kd_10foldcv.best_params_, train_error_knn_WORD2VEC_KD_10FOLD_cv, test_error_knn_WORD2VEC_KD_10FOLD_cv])
print (tablenew)


# # Assignment - part 4: KNN on TFIDF-W2V  (BRUTE-FORCE)

# * # Step1 )  Arriving at TFIDF-W2V  vector  (BRUTEFORE)

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")

import sqlite3
import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from nltk.stem.porter import PorterStemmer

import re
# Tutorial about Python regular expressions: https://pymotw.com/2/re/
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pickle

from tqdm import tqdm
import os


# In[ ]:


tf_idf_vect_bf = TfidfVectorizer()

tfidf_bf = tf_idf_vect_bf.fit(X_bf['CleanedText'].values)

tfidf_train_bf = tfidf_bf.transform(X_train_bf['CleanedText'].values)

tfidf_test_bf  = tfidf_bf.transform(X_test_bf['CleanedText'].values)

#tf_idf_matrix_bf = tf_idf_vect_bf.fit_transform(X_train_bf['CleanedText'].values)

#words_array_bf= tf_idf_matrix_bf.toarray()

# we are converting a dictionary with word as a key, and the idf as a value
dictionary = dict(zip(tf_idf_vect_bf.get_feature_names(), list(tf_idf_vect_bf.idf_)))



# In[ ]:


from tqdm import tqdm


# In[ ]:


# Compute the TFIDF-Word2Vec vector

tfidf_sent_vectors_WEIGHTED_BF = []; # the tfidf-w2v for each sentence/review is stored in this list

row=0;

for sent in tqdm(list_of_sent_bf): # for each review/sentence 
    sent_vec_bf = np.zeros(50) # as word vectors are of zero length
    weight_sum_bf =0; # num of words with a valid vector in the sentence/review
    for word in sent: # for each word in a review/sentence
        if word in w2v_words_bf:
            vec = w2v_model_bf.wv[word]
            # obtain the tf_idfidf of a word in a sentence/review
            
            # --------------->   tf_idf = tfidf_all[row, tfidf_feat.index(word)]
            tf_idf = dictionary[word]*sent.count(word)
            sent_vec_bf += (vec * tf_idf)
            weight_sum_bf += tf_idf
    if weight_sum_bf != 0:
        sent_vec_bf /= weight_sum_bf
    tfidf_sent_vectors_WEIGHTED_BF.append(sent_vec_bf)
    row += 1


# In[ ]:


tfidf_w2vec_BF= pd.DataFrame(tfidf_sent_vectors_WEIGHTED_BF)


# In[ ]:


tfidf_w2vec_BF.shape


# In[ ]:


tfidf_w2vec_BF_vec= np.array(tfidf_w2vec_BF)


# In[ ]:


tfidf_w2vec_train_BF= tfidf_w2vec_BF_vec[0:28000,]


# In[ ]:


tfidf_w2vec_test_BF= tfidf_w2vec_BF_vec[28000:40001,]


# In[ ]:


tfidf_w2vec_train_BF.shape


# In[ ]:


tfidf_w2vec_test_BF.shape


# # Step 2) KNN Classifier for TFIDF-Word2Vec ( BRUTE force and Simple CV)

# In[ ]:


### KNN using Cross Validation (K=1) and computing acuracy and CM
knn_tfidfw2v_bf_simple_cv = KNeighborsClassifier(1, algorithm = 'brute')
knn_tfidfw2v_bf_simple_cv.fit(tfidf_w2vec_train_BF, y_train_bf)
pred_tfidfw2v_bf_simple_cv = knn_tfidfw2v_bf_simple_cv.predict(tfidf_w2vec_test_BF)


# In[ ]:


# Print Accuracy and Confusion Matrix

acc = accuracy_score(y_test_bf, pred_tfidfw2v_bf_simple_cv, normalize=True) * float(100)
print('\n****Test accuracy for k = 1 is %d%%' % (acc))

labels = ['0','1']
cm = confusion_matrix(y_test_bf, pred_tfidfw2v_bf_simple_cv)

print("---------------------Plot of Confusion Matrix.....................")

df_cm = pd.DataFrame(cm)

sn.heatmap(df_cm, annot=True, cmap='Oranges', annot_kws={"size": 20})# font size


# In[ ]:


# Print Training and Test Error rate (BF AND SIMPLE CV)

train_error_knn_tfidf_w2vec_BF_simple_cv = 1- knn_tfidfw2v_bf_simple_cv.score(tfidf_w2vec_train_BF, y_train_bf)
    
    #Compute accuracy on the test set
test_error_knn_tfidf_w2vec_BF_simple_cv = 1 - knn_tfidfw2v_bf_simple_cv.score(tfidf_w2vec_test_BF, y_test_bf)
    
print('\n****Training Error for k = 1 (simple CV) is %.2f%%' % (train_error_knn_tfidf_w2vec_BF_simple_cv))
    
print('\n****Test Error for k = 1 (simple CV) is %.2f%%' % (test_error_knn_tfidf_w2vec_BF_simple_cv))


# In[ ]:



tablenew.add_row(["TFIDF-W2V-KNN with BF and simple CV", 1, train_error_knn_tfidf_w2vec_BF_simple_cv, test_error_knn_tfidf_w2vec_BF_simple_cv])

print (tablenew)


# 

# # Step 3) KNN Classifier for TFIDF-W2V (using Brute-force and 10-FOLD CV)

# In[ ]:


#  Finding K in KNN using 10- Fold  Cross-Validation
knn_tfidfw2v_bf_10foldcv = KNeighborsClassifier(algorithm = 'brute')
parameters = {"n_neighbors": np.arange(1,5, 2),
	"metric": ["euclidean"]}
clf_tfidfw2v_bf_10foldcv = GridSearchCV(knn_tfidfw2v_bf_10foldcv, parameters, cv=10)
clf_tfidfw2v_bf_10foldcv.fit(tfidf_w2vec_train_BF, y_train_bf)


# In[ ]:


clf_tfidfw2v_bf_10foldcv.best_params_
clf_tfidfw2v_bf_10foldcv.score(tfidf_w2vec_test_BF,y_test_bf)
y_pred_tfidfw2v_bf_10foldcv = clf_tfidfw2v_bf_10foldcv.best_estimator_.predict(tfidf_w2vec_test_BF)


# In[ ]:


# Arriving at CONFUSION MATRIX and plotting CM
labels = ['0','1']
cm = confusion_matrix(y_test_bf, y_pred_tfidfw2v_bf_10foldcv, labels)
print("Computation over... time to plot Confusion Matrix.....................")
# Plot of CM
df_cm = pd.DataFrame(cm)

sn.heatmap(df_cm, annot=True, cmap='Oranges', annot_kws={"size": 20})# font size


# In[ ]:


# Getting the Training and Test Error rate (BF AND 10-fold CV)

train_error_knn_tfidf_w2vec_BF_10FOLD_cv = 1 -clf_tfidfw2v_bf_10foldcv.score(tfidf_w2vec_train_BF, y_train_bf)
    
    #Compute accuracy on the test set
test_error_knn_tfidf_w2vec_BF_10FOLD_cv = 1- clf_tfidfw2v_bf_10foldcv.score(tfidf_w2vec_test_BF, y_test_bf)
    
  
# Printing  the Training and Test Error rate (BF AND 10-FOLD CV) 

print('\n****Training Error for {0}'.format(clf_tfidfw2v_bf_10foldcv.best_params_),'with 10-FOLD CV is %.3f%%'%(train_error_knn_tfidf_w2vec_BF_10FOLD_cv))

print('\n****Test Error for {0}'.format(clf_tfidfw2v_bf_10foldcv.best_params_), 'with 10-FOLD CV is %.3f%%'%(test_error_knn_tfidf_w2vec_BF_10FOLD_cv))


# In[ ]:


tablenew.add_row(["TFIDF-W2V-KNN with BF and 10-fold CV", clf_tfidfw2v_bf_10foldcv.best_params_, train_error_knn_tfidf_w2vec_BF_10FOLD_cv, test_error_knn_tfidf_w2vec_BF_10FOLD_cv])
print (tablenew)


# # Step 4)  Arriving at TFIDF-W2V  vector  (KD-TREE)

# In[ ]:


tf_idf_w2v_vect_kd = TfidfVectorizer()

tfidf_w2v_kd = tf_idf_w2v_vect_kd.fit(X_kd['CleanedText'].values)

tfidf_w2v_train_kd = tfidf_w2v_kd.transform(X_train_kd['CleanedText'].values)

tfidf_w2v_test_kd  = tfidf_w2v_kd.transform(X_test_kd['CleanedText'].values)

# we are converting a dictionary with word as a key, and the idf as a value
dictionary_kd = dict(zip(tf_idf_w2v_vect_kd.get_feature_names(), list(tf_idf_w2v_vect_kd.idf_)))



# In[ ]:


# Compute the TFIDF-Word2Vec vector  (KD-TREE)

tfidf_sent_vectors_WEIGHTED_kd = []; # the tfidf-w2v for each sentence/review is stored in this list

row=0;

for sent in tqdm(list_of_sent_kd): # for each review/sentence 
    sent_vec_kd = np.zeros(50) # as word vectors are of zero length
    weight_sum_kd =0; # num of words with a valid vector in the sentence/review
    for word in sent: # for each word in a review/sentence
        if word in w2v_words_kd:
            vec = w2v_model_kd.wv[word]
            # obtain the tf_idfidf of a word in a sentence/review
            
            # --------------->   tf_idf = tfidf_all[row, tfidf_feat.index(word)]
            tf_idf = dictionary_kd[word]*sent.count(word)
            sent_vec_kd += (vec * tf_idf)
            weight_sum_kd += tf_idf
    if weight_sum_kd != 0:
        sent_vec_kd /= weight_sum_kd
    tfidf_sent_vectors_WEIGHTED_kd.append(sent_vec_kd)
    row += 1


# In[ ]:


tfidf_w2vec_kd= pd.DataFrame(tfidf_sent_vectors_WEIGHTED_kd)


# In[ ]:


tfidf_w2vec_vectors_kd = np.array(tfidf_w2vec_kd)


# In[ ]:


tfidf_w2vec_train_KD= tfidf_w2vec_vectors_kd[0:14000,]


# In[ ]:


tfidf_w2vec_test_KD= tfidf_w2vec_vectors_kd[14000:20001,]


# In[ ]:


# Standardizing the data for KD-TREE
#from sklearn.preprocessing import StandardScaler

standardized_data_tfidfw2v_kd_train =StandardScaler(with_mean=False).fit_transform(tfidf_w2vec_train_KD) 

standardized_data_tfidfw2v_kd_test =StandardScaler(with_mean=False).fit(tfidf_w2vec_train_KD).transform(tfidf_w2vec_test_KD) 


# In[ ]:


##  TFIDF-W2V - Reducing Dimensions using Truncated SVD method

#from sklearn.decomposition import TruncatedSVD
#from scipy.sparse import csr_matrix

standardized_data_sparse_train_TFIDFW2V = csr_matrix(standardized_data_tfidfw2v_kd_train)

standardized_data_sparse_test_TFIDFW2V = csr_matrix(standardized_data_tfidfw2v_kd_test)

tsvd_TFIDFW2V = TruncatedSVD(n_components=12)


standardized_data_sparse_tsvd_TFIDFW2V_train_kd = tsvd_TFIDFW2V.fit(standardized_data_sparse_train_TFIDFW2V).transform(standardized_data_sparse_train_TFIDFW2V)

standardized_data_sparse_tsvd_TFIDFW2V_test_kd = tsvd_TFIDFW2V.fit(standardized_data_sparse_train_TFIDFW2V).transform(standardized_data_sparse_test_TFIDFW2V)


# In[ ]:


print(tsvd_TFIDFW2V.explained_variance_ratio_)


# ##  Here we will take N_COMPONENTS = 12, **since beyond 12, variance explained is very minimal%

# # Step 5) KNN Classifier for TFIDF-W2V  (using KD-TREE algo and Simple CV)

# In[ ]:


### KNN using Cross Validation (K=1) and computing acuracy and CM

knn_tfidfw2v_kd_simple_cv = KNeighborsClassifier(1, algorithm = 'kd_tree')
knn_tfidfw2v_kd_simple_cv.fit(standardized_data_sparse_tsvd_TFIDFW2V_train_kd, y_train_kd)
pred_tfidfw2v_kd_simple_cv = knn_tfidfw2v_kd_simple_cv.predict(standardized_data_sparse_tsvd_TFIDFW2V_test_kd)


# In[ ]:


# Print Accuracy and Confusion Matrix

acc = accuracy_score(y_test_kd, pred_tfidfw2v_kd_simple_cv, normalize=True) * float(100)
print('\n****Test accuracy for k = 1 is %d%%' % (acc))

labels = ['0','1']
cm = confusion_matrix(y_test_kd, pred_tfidfw2v_kd_simple_cv)

print("---------------------Plot of Confusion Matrix.....................")

df_cm = pd.DataFrame(cm)

sn.heatmap(df_cm, annot=True, cmap='Oranges', annot_kws={"size": 20})# font size


# In[ ]:


# Getting the Training and Test Error rate (KD AND 10-fold CV)

train_error_knn_TFIDF_W2V_KD_simple_cv = 1 - knn_tfidfw2v_kd_simple_cv.score(standardized_data_sparse_tsvd_TFIDFW2V_train_kd, y_train_kd)
    
#Compute accuracy on the test set
test_error_knn_TFIDF_W2V_KD_simple_cv = 1- knn_tfidfw2v_kd_simple_cv.score(standardized_data_sparse_tsvd_TFIDFW2V_test_kd, y_test_kd)
    
# Printing  the Training and Test Error rate (BF AND 10-FOLD CV) 

print('\n****Training Error for TFDIF-W2V with simple CV is %.3f%%'%(train_error_knn_TFIDF_W2V_KD_simple_cv))

print('\n****Test Error for  for TFDIF-W2V with simple CV is is %.3f%%'%(test_error_knn_TFIDF_W2V_KD_simple_cv))


# In[ ]:


tablenew.add_row(["TFIDF-W2V-KNN with KD and simple CV", 1, train_error_knn_TFIDF_W2V_KD_simple_cv, test_error_knn_TFIDF_W2V_KD_simple_cv])

print (tablenew)


# # Step 6) KNN Classifier for TFIDF-W2V  (using KD-TREE algo and 10-fold  CV)

# In[ ]:


#  Finding K in KNN using 10- Fold  Cross-Validation

knn_TFIDFW2V_kd_10foldcv = KNeighborsClassifier(algorithm = 'kd_tree')
parameters = {"n_neighbors": np.arange(1, 5, 2),
	"metric": ["euclidean"]}
clf_TFIDFW2V_kd_10foldcv = GridSearchCV(knn_TFIDFW2V_kd_10foldcv, parameters, cv=10)


# In[ ]:


clf_TFIDFW2V_kd_10foldcv.fit(standardized_data_sparse_tsvd_TFIDFW2V_train_kd, y_train_kd)


# In[ ]:


clf_TFIDFW2V_kd_10foldcv.best_params_
clf_TFIDFW2V_kd_10foldcv.score(standardized_data_sparse_tsvd_TFIDFW2V_test_kd,y_test_kd)
y_pred_TFIDFW2V_kd_10foldcv = clf_TFIDFW2V_kd_10foldcv.best_estimator_.predict(standardized_data_sparse_tsvd_TFIDFW2V_test_kd)


# In[ ]:


# Arriving at CONFUSION MATRIX and plotting CM

labels = ['0','1']
cm = confusion_matrix(y_test_kd, y_pred_TFIDFW2V_kd_10foldcv, labels)
print("Computation over... time to plot Confusion Matrix.....................")
# Plot of CM
df_cm = pd.DataFrame(cm)

sn.heatmap(df_cm, annot=True, cmap='Oranges', annot_kws={"size": 20})# font size


# In[ ]:


# Getting the Training and Test Error rate (KD AND 10-fold CV)

train_error_knn_TFIDF_W2V_KD_10fold_cv = 1 - clf_TFIDFW2V_kd_10foldcv.score(standardized_data_sparse_tsvd_TFIDFW2V_train_kd, y_train_kd)
    
#Compute accuracy on the test set
test_error_knn_TFIDF_W2V_KD_10fold_cv = 1 - clf_TFIDFW2V_kd_10foldcv.score(standardized_data_sparse_tsvd_TFIDFW2V_test_kd, y_test_kd)
    
# Printing  the Training and Test Error rate (BF AND 10-FOLD CV) 

print('\n****Training Error for {0}'.format(clf_TFIDFW2V_kd_10foldcv.best_params_),'with 10-FOLD CV is %.3f%%'%(train_error_knn_TFIDF_W2V_KD_10fold_cv))

print('\n****Test Error for {0}'.format(clf_TFIDFW2V_kd_10foldcv.best_params_), 'with 10-FOLD CV is %.3f%%'%(test_error_knn_TFIDF_W2V_KD_10fold_cv))


# In[ ]:


tablenew.add_row(["TFIDF-W2V-KNN with KD and 10-fold CV", clf_TFIDFW2V_kd_10foldcv.best_params_, train_error_knn_TFIDF_W2V_KD_10fold_cv, test_error_knn_TFIDF_W2V_KD_10fold_cv])

print (tablenew)


# 

# 

# 

# 

# 

# 

# 
