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
#import nltk
#nltk.download()


# In[ ]:



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


# **2. Importing Dataset from database.sqlite and ignoring reviews with Score  = 3 as they represent a neutral view**

# In[ ]:


# creating sql connection string
con = sqlite3.connect('../input/database.sqlite')


# In[ ]:


#Positive Review - Rating above 3
#Negative Review - Rating below 3
#Ignoring Reviews with 3 Rating

filtered_data = pd.read_sql_query('SELECT * from Reviews WHERE Score != 3',con)


# In[ ]:


# mapping ratings above 3 as Positive and below 3 as Negative

actual_scores = filtered_data['Score']
positiveNegative = actual_scores.map(lambda x: 'Positive' if x>3 else 'Negative')
filtered_data['Score'] = positiveNegative


# **3. Data Preprocessing**

# In[ ]:


final = filtered_data.drop_duplicates(subset= { 'UserId', 'ProfileName', 'Time',  'Text'})


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
final = final.iloc[:50000,:]


# In[ ]:


# Checking the number of positive and negative reviews

Class_Count  = final['Score'].value_counts()

plt.figure()
flatui = ["#15ff00", "#ff0033"]
sns.set_palette(flatui)
sns.barplot(Class_Count.index, Class_Count.values, alpha=0.8 )
plt.title('Positive Class Count vs Negative Class Count')
plt.ylabel('Count', fontsize=12)
plt.xlabel('Class', fontsize=12)
plt.show()

print(final['Score'].value_counts())


# In[ ]:


# Sorting values according to Time for Time Based Slicing
final = final.sort_values('Time',kind = 'quicksort')


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


# Initialize Stop words and PorterStemmer and Lemmetizer
stop = set(stopwords.words('english'))
sno = SnowballStemmer('english')


#print(stop)
#print('*' * 100)
#print(sno.stem('tasty'))


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


#Dictionary for storing Metrics
Final_Metrics =pd.DataFrame()


# In[ ]:



# Function for KNN
def runKNN(X_Train,X_Test,y_train,y_test,VectorizationType,algo):
    global Final_Metrics
    cv_scores = []
    k_value = []
    Train_Scores = []
    Test_Scores = []
    algorithm = ['kd_tree','brute']
    Cnf_Mtx = []
    Algo_Temp = 'Dummy'
    
        #print(algo)
        # kd_tree cannot consume Sparse Matrix. Converting Sparse Matrix to Dense using Truncated SVD.
    if algo == 'kd_tree':
      Algo_Temp = algo
      svd = TruncatedSVD()
      X_Train = svd.fit_transform(X_Train)
      X_Test = svd.fit_transform(X_Test)
           
            
    print('*' * 300)        
    j=0
    for i in range(1,30,2):
    # instantiate learning model (k = 30)
      knn = KNeighborsClassifier(n_neighbors=i,algorithm = algo)
      scores = cross_val_score(knn, X_Train, y_train, cv=10, scoring='accuracy')
      cv_scores.append(scores.mean())
      k_value.append(i)
      
      print('For K = ', i,'Accuracy Score = ', cv_scores[j])
      j+=1
       
    plt.plot(k_value,cv_scores,'-o')
    plt.xlabel('K-Value')
    plt.ylabel('CV-Scores')
    plt.title('K-Value vs CV-Scores')
    print('*' * 300)
               
        #print(cv_scores)
        #print(max(cv_scores))
    k_optimum = k_value[cv_scores.index(max(cv_scores))]

    knn = KNeighborsClassifier(n_neighbors=k_optimum,algorithm = algo)
        # fitting the model on crossvalidation train
    knn.fit(X_Train, y_train)

        # predict the response on the crossvalidation train
    pred = knn.predict(X_Test)
    knn.fit(X_Train, y_train).score(X_Train, y_train)
    Train_Scores.append(knn.score(X_Train, y_train))
    Test_Scores.append(knn.score(X_Test, y_test))
      
    Temp_List = [algo,VectorizationType,k_optimum,knn.score(X_Train, y_train)*100,knn.score(X_Test, y_test)*100]
        #print(Temp_List)
    Final_Metrics = Final_Metrics.append({'Algorithm': algo,'Vectorization':VectorizationType,'HyperParameter':k_optimum,
                                              'Training Accuracy Score': knn.score(X_Train, y_train)*100,
                                              'Testing Accuracy Score':knn.score(X_Test, y_test)*100},
                                            ignore_index=True)

        # evaluate CV accuracy
        #acc = accuracy_score(y_cv_input, pred, normalize=True) * float(100)
        
    print('\nDetails for ',VectorizationType,'Vectorization:')
        
      #print('Accuracy for',algo,' algorithm with alpha =',alpha_optimum,' is ' ,np.round((accuracy_score(y_cv_input, pred)*100),decimals = 2))
    print('Accuracy for',algo,' algorithm with K =',k_optimum,' is ' ,np.round((accuracy_score(y_test, pred)*100),decimals = 2))
    print('F1 score for',algo,' algorithm with K =',k_optimum,' is ' , np.round((f1_score(y_test, pred,average= 'macro')*100),decimals = 2))
    print('Recall for',algo,' agorithm with K =',k_optimum,' is ' , np.round((recall_score(y_test, pred,average= 'macro')*100),decimals = 2))
    print('Precision for',algo,' algorithm with K =',k_optimum,' is ' , np.round((precision_score(y_test, pred,average= 'macro')*100),decimals = 2))
    print ('\n Clasification report for',algo,' algorithm with K =',k_optimum,' is \n ' , classification_report(y_test,pred))
        #print ('\n Confusion matrix for',algo,' algorithm with K =',k_optimum,' is \n' ,confusion_matrix(y_test, pred))
    Cnf_Mtx = [pred]
    
    plt.figure()
    confusion_matrix_Plot = confusion_matrix(y_test,pred)
    heatmap = sns.heatmap(confusion_matrix_Plot, annot=True, fmt="d")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    print('The Confusion Matrix for ',algo,' Algorithm')
             


# **4.1 Using Bag of Words**

# In[ ]:


#Splitting into Training and Testing Set, and using only Training set for Word2Vec Training
X_Train,X_Test,y_train,y_test = train_test_split(final['CleanedText'],final['Score'])


# In[ ]:


# BoW Vectorization

vect = CountVectorizer().fit(X_Train)
X_Train = vect.transform(X_Train)
X_Test = vect.transform(X_Test)


# In[ ]:


runKNN(X_Train,X_Test,y_train,y_test,'Bag of Words','kd_tree')


# In[ ]:


runKNN(X_Train,X_Test,y_train,y_test,'Bag of Words','brute')


# **4.2 Using TF-IDF**

# In[ ]:


#Splitting into Training and Testing Set, and using only Training set for Word2Vec Training
X_Train,X_Test,y_train,y_test = train_test_split(final['Text'],final['Score'])


# In[ ]:


# TF-IDF weighted Word2Vec
vect_tfidf = TfidfVectorizer(min_df = 5).fit(X_Train)
tfidf_feat = vect_tfidf.get_feature_names() # tfidf words/col-names
#print(tfidf_feat)
# final_tf_idf is the sparse matrix with row= sentence, col=word and cell_val = tfidf

tfidf_sent_vectors = []; # the tfidf-w2v for each sentence/review is stored in this list
row=0;
for sent in X_Train: # for each review/sentence
    sent_vec = np.zeros(50) # as word vectors are of zero length
    weight_sum =0; # num of words with a valid vector in the sentence/review
    for word in sent: # for each word in a review/sentence
        try:
            vec = w2v_model.wv[word]
            # obtain the tf_idfidf of a word in a sentence/review
            tfidf = vect_tfidf[row, tfidf_feat.index(word)]
            sent_vec += (vec * tf_idf)
            weight_sum += tf_idf
        except:
            pass
    
    #print(type(sent_vec))
    try:
        sent_vec /= weight_sum
    except:
        pass
    
    tfidf_sent_vectors.append(sent_vec)
    row += 1
X_train_Vectorised = tfidf_sent_vectors


tfidf_sent_vectors = []; # the tfidf-w2v for each sentence/review is stored in this list
row=0;
for sent in X_Test: # for each review/sentence
    sent_vec = np.zeros(50) # as word vectors are of zero length
    weight_sum =0; # num of words with a valid vector in the sentence/review
    for word in sent: # for each word in a review/sentence
        try:
            vec = w2v_model.wv[word]
            # obtain the tf_idfidf of a word in a sentence/review
            tfidf = vect_tfidf[row, tfidf_feat.index(word)]
            sent_vec += (vec * tf_idf)
            weight_sum += tf_idf
        except:
            pass
    
    #print(type(sent_vec))
    try:
        sent_vec /= weight_sum
    except:
        pass
    
    tfidf_sent_vectors.append(sent_vec)
    row += 1

X_test_Vectorised = tfidf_sent_vectors
    
X_train_Vectorised = np.nan_to_num(X_train_Vectorised)
X_test_Vectorised = np.nan_to_num(X_test_Vectorised)
    

    


# In[ ]:


runKNN(X_train_Vectorised,X_test_Vectorised,y_train,y_test,'TF-IDF Weighted Word2Vec','kd_tree')


# In[ ]:


runKNN(X_train_Vectorised,X_test_Vectorised,y_train,y_test,'TF-IDF Weighted Word2Vec','brute')


# **4.3 Using Average Word2Vec**

# In[ ]:


#Splitting into TRaining and Testing Set, and using only Training set for Word2Vec Training
X_Train,X_Test,y_train,y_test = train_test_split(final['Text'],final['Score'])


# Train your own Word2Vec model using your own text corpus

i=0
list_of_sent=[]
for sent in X_Train.values:
    filtered_sentence=[]
    sent=cleanhtml(sent)
    for w in sent.split():
        for cleaned_words in cleanpunct(w).split():
            if(cleaned_words.isalpha()):    
                filtered_sentence.append(cleaned_words.lower())
            else:
                continue 
    list_of_sent.append(filtered_sentence)
    


# In[ ]:


'''print(final['Text'].values[0])
print("*****************************************************************")
print(list_of_sent[0])'''


# In[ ]:


w2v_model=gensim.models.Word2Vec(list_of_sent,min_count=5,size=50, workers=4)    
words = list(w2v_model.wv.vocab)
#print(len(words))


# In[ ]:


# average Word2Vec
# compute average word2vec for each review.
sent_vectors = []; # the avg-w2v for each sentence/review is stored in this list
for sent in X_Train: # for each review/sentence
    sent_vec = np.zeros(50) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sent: # for each word in a review/sentence
        try:
            vec = w2v_model.wv[word]
            sent_vec += vec
            cnt_words += 1
        except:
            pass
    sent_vec /= cnt_words
    sent_vectors.append(sent_vec)

X_train_Vectorised = sent_vectors



sent_vectors = []; # the avg-w2v for each sentence/review is stored in this list
for sent in X_Test: # for each review/sentence
    sent_vec = np.zeros(50) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sent: # for each word in a review/sentence
        try:
            vec = w2v_model.wv[word]
            sent_vec += vec
            cnt_words += 1
        except:
            pass
    sent_vec /= cnt_words
    sent_vectors.append(sent_vec)

X_test_Vectorised = sent_vectors
print(len(X_train_Vectorised))
print(len(X_test_Vectorised))

#X_1, X_test, y_1, y_test = cross_validation.train_test_split(sent_vectors, final['Score'], random_state = 0,test_size = 0.3)
#print('X_train first entry: \n\n', X_1[0])
#print('\n\nX_train shape: ', X_1.shape)

# split the train data set into cross validation train and cross validation test
#X_tr, X_cv, y_tr, y_cv = cross_validation.train_test_split(X_1, y_1, test_size=0.3)

np.where(np.isnan(X_test_Vectorised))
X_train_Vectorised = np.nan_to_num(X_train_Vectorised)
X_test_Vectorised = np.nan_to_num(X_test_Vectorised)
#np.nan_to_num(X_test_Vectorised)


# In[ ]:


runKNN(X_train_Vectorised,X_test_Vectorised,y_train,y_test,'Average Word2Vec','kd_tree')


# In[ ]:


runKNN(X_train_Vectorised,X_test_Vectorised,y_train,y_test,'Average Word2Vec','brute')


# **4.4 Using TF-IDF Weighted Word2Vec**

# In[ ]:


#Splitting into TRaining and Testing Set, and using only Training set for Word2Vec Training
X_Train,X_Test,y_train,y_test = train_test_split(final['Text'],final['Score'])

# TF-IDF weighted Word2Vec
vect_tfidf = TfidfVectorizer(min_df = 5).fit(X_Train)
tfidf_feat = vect_tfidf.get_feature_names() # tfidf words/col-names
#print(tfidf_feat)
# final_tf_idf is the sparse matrix with row= sentence, col=word and cell_val = tfidf

tfidf_sent_vectors = []; # the tfidf-w2v for each sentence/review is stored in this list
row=0;
for sent in X_Train: # for each review/sentence
    sent_vec = np.zeros(50) # as word vectors are of zero length
    weight_sum =0; # num of words with a valid vector in the sentence/review
    for word in sent: # for each word in a review/sentence
        try:
            vec = w2v_model.wv[word]
            # obtain the tf_idfidf of a word in a sentence/review
            tfidf = vect_tfidf[row, tfidf_feat.index(word)]
            sent_vec += (vec * tf_idf)
            weight_sum += tf_idf
        except:
            pass
    
    #print(type(sent_vec))
    try:
        sent_vec /= weight_sum
    except:
        pass
    
    tfidf_sent_vectors.append(sent_vec)
    row += 1
X_train_Vectorised = tfidf_sent_vectors


tfidf_sent_vectors = []; # the tfidf-w2v for each sentence/review is stored in this list
row=0;
for sent in X_Test: # for each review/sentence
    sent_vec = np.zeros(50) # as word vectors are of zero length
    weight_sum =0; # num of words with a valid vector in the sentence/review
    for word in sent: # for each word in a review/sentence
        try:
            vec = w2v_model.wv[word]
            # obtain the tf_idfidf of a word in a sentence/review
            tfidf = vect_tfidf[row, tfidf_feat.index(word)]
            sent_vec += (vec * tf_idf)
            weight_sum += tf_idf
        except:
            pass
    
    #print(type(sent_vec))
    try:
        sent_vec /= weight_sum
    except:
        pass
    
    tfidf_sent_vectors.append(sent_vec)
    row += 1

X_test_Vectorised = tfidf_sent_vectors
    
X_train_Vectorised = np.nan_to_num(X_train_Vectorised)
X_test_Vectorised = np.nan_to_num(X_test_Vectorised)
    

    


# In[ ]:


runKNN(X_train_Vectorised,X_test_Vectorised,y_train,y_test,'TF-IDF Weighted Word2Vec','kd_tree')


# In[ ]:


runKNN(X_train_Vectorised,X_test_Vectorised,y_train,y_test,'TF-IDF Weighted Word2Vec','brute')


# **Conclusion**

# In[ ]:


Final_Metrics


# In[ ]:





# 

# In[ ]:




