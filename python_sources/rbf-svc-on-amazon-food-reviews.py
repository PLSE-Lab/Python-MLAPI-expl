#!/usr/bin/env python
# coding: utf-8

# >    **KNN on Amazon Fine Foods Reviews**
# 
# **Objective** - Run KNN Algorithms on Amazon Fine Foods Review Dataset using BoW,TF-IDF,Avg Word2Vec and TF-IDF weighed Word2Vec vectorization methods. Also to report the metrics for each iteration. Time based splitting to be followed.
# 
# 
# **KNN Algorithms to be used** - KD TREE and BRUTE
# 
# **Kaggle Dataset Location** - https://www.kaggle.com/snap/amazon-fine-food-reviews/data
# 
# 

# **1. Importing required packages**

# In[ ]:


#!pip install -U gensim
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import sqlite3
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,classification_report,f1_score
from sklearn.decomposition import TruncatedSVD
from sklearn import cross_validation
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from scipy.stats import expon


# **2. Importing Dataset from database.sqlite and ignoring reviews with Score  = 3 as they represent a neutral view**

# # Code to read csv file into colaboratory:
# !pip install -U -q PyDrive
# from pydrive.auth import GoogleAuth
# from pydrive.drive import GoogleDrive
# from google.colab import auth
# from oauth2client.client import GoogleCredentials
# 
# # 1. Authenticate and create the PyDrive client.
# auth.authenticate_user()
# gauth = GoogleAuth()
# gauth.credentials = GoogleCredentials.get_application_default()
# drive = GoogleDrive(gauth)
# 
# #2. Get the file
# downloaded = drive.CreateFile({'id':'1aeCGYtXU-YcvPMJfM9xxgFy2vooP-Xdk'}) # replace the id with id of file you want to access
# downloaded.GetContentFile('database.sqlite')  
# 
# #3. Read file as panda dataframe
# #import pandas as pd
# #mnist_data = pd.read_csv('xyz.csv') 

# In[ ]:


# creating sql connection string
con = sqlite3.connect('../input/database.sqlite')


# In[ ]:


#Positive Review - Rating above 3
#Negative Review - Rating below 3
#Ignoring Reviews with 3 Rating

filtered_data = pd.read_sql_query('SELECT * from Reviews WHERE Score != 3',con)


# In[ ]:


filtered_data.head(5)


# In[ ]:


# mapping ratings above 3 as Positive and below 3 as Negative

actual_scores = filtered_data['Score']
positiveNegative = actual_scores.map(lambda x: 'Positive' if x>3 else 'Negative')
filtered_data['Score'] = positiveNegative


# In[ ]:


filtered_data.head(5)


# In[ ]:


# Sorting values according to Time for Time Based Slicing
sorted_values = filtered_data.sort_values('Time',kind = 'quicksort')


# **3. Data Preprocessing**

# In[ ]:


final = sorted_values.drop_duplicates(subset= { 'UserId', 'ProfileName', 'Time',  'Text'})


# In[ ]:


print('Rows dropped : ',filtered_data.size - final.size)
print('Percentage Data remaining after dropping duplicates :',(((final.size * 1.0)/(filtered_data.size * 1.0) * 100.0)))


# In[ ]:


# Dropping rows where HelpfulnessNumerator < HelpfulnessDenominator
final = final[final.HelpfulnessDenominator >= final.HelpfulnessNumerator]


# In[ ]:


print('Number of Rows remaining in the Dataset: ',final.size)


# In[ ]:


# Checking the number of positive and negative reviews
final['Score'].value_counts()


# In[ ]:


# Data Sampling
final = final.iloc[:5000,:]
print(final.shape)
print(final['Score'].value_counts())


# In[ ]:


# Function to Remove HTML Tags
def cleanhtml(sentence):
    cleaner = re.compile('<.*?>')
    cleantext = re.sub(cleaner,"",sentence)
    return cleantext


# In[ ]:


# Function to clean punctuations and special characters

def cleanpunct(sentence):
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    return  cleaned


# In[ ]:


#import nltk
#nltk.download()


# In[ ]:


# Initialize Stop words and PorterStemmer and Lemmetizer
stop = set(stopwords.words('english'))
sno = SnowballStemmer('english')


print(stop)
print('*' * 100)
print(sno.stem('tasty'))


# In[ ]:


# Cleaning HTML and non-Alphanumeric characters from the review text
i=0
str1=' '
final_string=[]
all_positive_words=[] # store words from +ve reviews here
all_negative_words=[] # store words from -ve reviews here.
s=''
for sent in final['Text'].values:
    filtered_sentence=[]
    #print(sent);
    sent=cleanhtml(sent) # remove HTMl tags
    for w in sent.split():
        for cleaned_words in cleanpunct(w).split():
            if((cleaned_words.isalpha()) & (len(cleaned_words)>2)):    
                if(cleaned_words.lower() not in stop):
                    s=(sno.stem(cleaned_words.lower())).encode('utf8')
                    filtered_sentence.append(s)
                    if (final['Score'].values)[i] == 'Positive': 
                        all_positive_words.append(s) #list of all words used to describe positive reviews
                    if(final['Score'].values)[i] == 'Negative':
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


final['CleanedText']=final_string
final.head(5)


# **4.  KNN with KD Tree and Brute Force Algorithm**

# In[ ]:


#Split data into Train and Test Set
X_Train,X_Test,y_train,y_test = train_test_split(final['CleanedText'],final['Score'],random_state = 0,test_size = 0.3)


# In[ ]:


# Function to run SVC with GridSearchCV and RandomSearchCV
def RunSVC(X_Train,X_Test,y_train,y_test,Search_Type):    
    lb_make = LabelEncoder()
    
    y_train_encoded = lb_make.fit_transform(y_train)
    y_test_encoded = lb_make.fit_transform(y_test)
    
    
    if (Search_Type == 'grid'):
        grid_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]}]
        model = GridSearchCV(SVC(),grid_parameters,cv = 5,scoring = 'f1')
        model.fit(X_Train,y_train_encoded)
        print(model.best_estimator_)
        print('The Score with '+ Search_Type+ 'search CV is: '+ str(model.score(X_Test, y_test_encoded)))
    elif (Search_Type == 'random'):
        random_parameters = dict(C=[1, 10, 100, 1000],gamma=[1e-3, 1e-4])  
        model = RandomizedSearchCV(SVC(),random_parameters,cv = 5,scoring = 'f1',n_jobs= 1)
        model.fit(X_Train,y_train_encoded)
        print(model.best_estimator_)
        print('The Score with '+ Search_Type+ 'search CV is: ' + str(model.score(X_Test, y_test_encoded)))


# **4.1 Using Bag of Words**

# In[ ]:


# BoW Vectorization

vect = CountVectorizer().fit(X_Train)
X_Train_vectorised = vect.transform(X_Train)
X_Test_vectorised = vect.transform(X_Test)


RunSVC(X_Train_vectorised,X_Test_vectorised,y_train,y_test,'grid')
RunSVC(X_Train_vectorised,X_Test_vectorised,y_train,y_test,'random')


# **4.2 Using TF-IDF**

# X_Train.head()

# In[ ]:


# Applying TFIDF

vect_tfidf = TfidfVectorizer(min_df = 5).fit(X_Train)
X_Train_vectorised = vect_tfidf.transform(X_Train)
X_Test_vectorised = vect_tfidf.transform(X_Test)
RunSVC(X_Train_vectorised,X_Test_vectorised,y_train,y_test,'grid')
RunSVC(X_Train_vectorised,X_Test_vectorised,y_train,y_test,'random')


# 
# **4.3 Using Average Word2Vec**

# #Split data into Train and Test Set
# X_Train,X_Test,y_train,y_test = train_test_split(final['Text'],final['Score'],random_state = 0,test_size = 0.3)

# # Train your own Word2Vec model using your own text corpus
# 
# i=0
# list_of_sent=[]
# for sent in X_Train:
#     filtered_sentence=[]
#     sent=cleanhtml(sent)
#     for w in sent.split():
#         for cleaned_words in cleanpunct(w).split():
#             if(cleaned_words.isalpha()):    
#                 filtered_sentence.append(cleaned_words.lower())
#             else:
#                 continue 
#     list_of_sent.append(filtered_sentence)
#     

# In[ ]:


'''print(final['Text'].values[0])
print("*****************************************************************")
'''
print(list_of_sent[0])


# In[ ]:


w2v_model=gensim.models.Word2Vec(list_of_sent,min_count=5,size=50, workers=4)    
words = list(w2v_model.wv.vocab)
#print(len(words))


# # average Word2Vec
# # compute average word2vec for each review.
# sent_vectors = []; # the avg-w2v for each sentence/review is stored in this list
# for sent in list_of_sent: # for each review/sentence
#     sent_vec = np.zeros(50) # as word vectors are of zero length
#     cnt_words =0; # num of words with a valid vector in the sentence/review
#     for word in sent: # for each word in a review/sentence
#         try:
#             vec = w2v_model.wv[word]
#             sent_vec += vec
#             cnt_words += 1
#         except:
#             pass
#     sent_vec /= cnt_words
#     sent_vectors.append(sent_vec)
# #print(len(sent_vectors))
# #print(len(sent_vectors[0]))
# 
# '''X_1, X_test, y_1, y_test = cross_validation.train_test_split(sent_vectors, final['Score'], random_state = 0,test_size = 0.3)
# #print('X_train first entry: \n\n', X_1[0])
# #print('\n\nX_train shape: ', X_1.shape)
# 
# # split the train data set into cross validation train and cross validation test
# X_tr, X_cv, y_tr, y_cv = cross_validation.train_test_split(X_1, y_1, test_size=0.3)'''
# 
# 
# #runKNN(X_tr_vectorized,x_cv_vectorized,y_tr,y_cv,'Average Word2Vec')

# **4.4 Using TF-IDF Weighted Word2Vec**

# # TF-IDF weighted Word2Vec
# tfidf_feat = vect_tfidf.get_feature_names() # tfidf words/col-names
# # final_tf_idf is the sparse matrix with row= sentence, col=word and cell_val = tfidf
# 
# tfidf_sent_vectors = []; # the tfidf-w2v for each sentence/review is stored in this list
# row=0;
# for sent in list_of_sent: # for each review/sentence
#     sent_vec = np.zeros(50) # as word vectors are of zero length
#     weight_sum =0; # num of words with a valid vector in the sentence/review
#     for word in sent: # for each word in a review/sentence
#         try:
#             vec = w2v_model.wv[word]
#             # obtain the tf_idfidf of a word in a sentence/review
#             tfidf = vect_tfidf[row, tfidf_feat.index(word)]
#             sent_vec += (vec * tf_idf)
#             weight_sum += tf_idf
#         except:
#             pass
#     
#     #print(type(sent_vec))
#     try:
#         sent_vec /= weight_sum
#     except:
#         pass
#     
#     tfidf_sent_vectors.append(sent_vec)
#     row += 1
#     
# 
#     
# X_1, X_test, y_1, y_test = cross_validation.train_test_split(tfidf_sent_vectors, final['Score'], random_state = 0,test_size = 0.3)
# #print('X_train first entry: \n\n', X_1[0])
# #print('\n\nX_train shape: ', X_1.shape)
# 
# # split the train data set into cross validation train and cross validation test
# X_tr, X_cv, y_tr, y_cv = cross_validation.train_test_split(X_1, y_1, test_size=0.3)
# 
# runKNN(X_tr_vectorized,x_cv_vectorized,y_tr,y_cv,'TF-IDF weighted Word2Vec')
#     

# In[ ]:





# In[ ]:





# In[ ]:




