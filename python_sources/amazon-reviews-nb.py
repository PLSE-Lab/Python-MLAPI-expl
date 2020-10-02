#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
print(os.listdir("../input"))


# In[ ]:


# Import all the required libraries

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

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV


from collections import Counter

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report

# Tutorial about Python regular expressions: https://pymotw.com/2/re/
import re
import string
import nltk.corpus
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pickle




# In[ ]:


# using the SQLite Table to read data. (KAGGLE)
import sqlite3
show_tables = "select tbl_name from sqlite_master where type = 'table'" 
conn = sqlite3.connect('../input/database.sqlite') 
pd.read_sql(show_tables,conn)


# In[ ]:


#filtering only positive and negative reviews i.e. not taking into consideration those reviews with Score=3
filtered_data = pd.read_sql_query("""SELECT * FROM Reviews WHERE Score != 3""", conn) 


# In[ ]:


# Give reviews with Score>3 a positive rating, and reviews with a score<3 a negative rating.
def partition(x):
    if x < 3:
        return 'Negative'
    return 'Positive'

#changing reviews with score less than 3 to be positive and vice-versa
actualScore = filtered_data['Score']

positiveNegative = actualScore.map(partition) 

filtered_data['Polarity'] = positiveNegative

filtered_data['Class_Label']= filtered_data['Polarity'].apply(lambda x : 1 if x == 'Positive' else 0)


# In[ ]:


filtered_data.head()


# In[ ]:


# Data Cleaning: Deduplication, clearing records whereHelpfulnessNumerator is greater than HelpfulnessDenominator 
sorted_data=filtered_data.sort_values('ProductId', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
final=sorted_data.drop_duplicates(subset={"UserId","ProfileName","Time","Text"}, keep='first', inplace=False)

display= pd.read_sql_query("""
SELECT *
FROM Reviews
WHERE Score != 3 AND Id=44737 OR Id=64422
ORDER BY ProductID
""", conn)
final=final[final.HelpfulnessNumerator<=final.HelpfulnessDenominator]

#How many positive and negative reviews are present in our dataset?
final['Class_Label'].value_counts()


# In[ ]:


## Text Preprocessing: Stemming, stop-word removal and Lemmatization.

# find sentences containing HTML tags (DATASET FOR BRUTEFORCE)
import re
i=0;
for sent in final['Text'].values:
    if (len(re.findall('<.*?>', sent))):
        print(i)
        print(sent)
        break;
    i += 1;

# find sentences containing HTML tags (DATASET FOR KD_TREE)
import re
i=0;
for sent in final['Text'].values:
    if (len(re.findall('<.*?>', sent))):
        print(i)
        print(sent)
        break;
    i += 1;

# Remove Stop-Words

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


#  GET THE TRAINING AND TEST DATA-SET 

#pre-processing: agegate all positive and negative words
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
        for cleaned_words in cleanpunc(w).split():
            if((cleaned_words.isalpha()) & (len(cleaned_words)>2)):    
                if(cleaned_words.lower() not in stop):
                    s=(sno.stem(cleaned_words.lower())).encode('utf8')
                    filtered_sentence.append(s)
                    if (final['Polarity'].values)[i] == 'Positive': 
                        all_positive_words.append(s) #list of all words used to describe positive reviews
                    if(final['Polarity'].values)[i] == 'Negative':
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
    
#  cleaned-up columns

final['CleanedText']=final_string #adding a column of CleanedText which displays the data after pre-processing of the review 
final['CleanedText']=final['CleanedText'].str.decode("utf-8")

# define column names
names = ['Time', 'Text','CleanedText', 'Polarity']


# create design matrix X and target vector y
X_NB =  final[names]
y_NB = final['Class_Label']

X_train_NB, X_test_NB, y_train_NB, y_test_NB = model_selection.train_test_split(X_NB, y_NB, test_size=0.2, random_state=0)


# # ASSIGNMENT- PART 1:  NB  CLASSIFIER  ON BOW  VECTOR

# # STEP 1) Computing the Bag of Words (BoW)

# In[ ]:


# Get the BoW matrix

from sklearn.feature_extraction.text import TfidfVectorizer

count_vect = CountVectorizer() 

bow_NB = count_vect.fit(X_train_NB['CleanedText'].values)

bow_train_NB = bow_NB.transform(X_train_NB['CleanedText'].values)

bow_test_NB = bow_NB.transform(X_test_NB['CleanedText'].values)


# In[ ]:


# Colum Standardization of the Bag of Words vector

from sklearn.preprocessing import StandardScaler
scalar = StandardScaler(with_mean=False)
scalar.fit(bow_train_NB)
bow_train_NB_vectors = scalar.transform(bow_train_NB)
bow_test_NB_vectors = scalar.transform(bow_test_NB)


# # Step 2) Naive Bayes Classifier for BOW

# In[ ]:


# 10FOLD CV  to get the best Alpha (Hyper-Parameter)  for BOW model

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

#parameters = {"alpha":  np.array([1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,0,1])}

parameters = {"alpha":  np.array( [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10] )}

n_folds = 10

cv_timeSeries = TimeSeriesSplit(n_splits=n_folds)
    
model = MultinomialNB()

my_cv = TimeSeriesSplit(n_splits=n_folds).split(bow_train_NB_vectors)
    
gsearch_cv = GridSearchCV(estimator=model, param_grid=parameters, cv=my_cv,scoring='f1')

gsearch_cv.fit(bow_train_NB_vectors, y_train_NB)


# In[ ]:


#Plot the Scores for each Alpha used in CRoss-Validation
# source ref URL:  https://glowingpython.blogspot.com/2014/04/parameters-selection-with-cross.html

res = list(zip(*[(f1m,f1s.std(), p['alpha']) 
            for p, f1m, f1s in gsearch_cv.grid_scores_]))

plt.plot(res[2],res[0],'-o', color="g",)
plt.xlabel('values of Alpha (Hyper-Parameter)')
plt.ylabel('Average Score (Better Score implies Lesser Error)')

plt.show()


# In[ ]:


# Display the details for the  Hyper-parametrized BOW model

NB_OPTIMAL_classifier_for_BOW = gsearch_cv.best_estimator_
print("Best estimator for {} model : ".format("BOW"), NB_OPTIMAL_classifier_for_BOW)

NB_OPTIMAL_score_for_BOW = gsearch_cv.best_score_
print("Best Score for {} model : ".format("BOW"), NB_OPTIMAL_score_for_BOW)

OPTIMAL_MODEL_for_BOW= gsearch_cv.best_params_
for alpha in OPTIMAL_MODEL_for_BOW:
    print("Optimal Alpha for {} model : ".format("BOW"),'{:f}'.format(OPTIMAL_MODEL_for_BOW[alpha]))



# In[ ]:


# Plotting the ROC Curve for the Best Classifier

# Ref-Source-URL:  https://datamize.wordpress.com/2015/01/24/how-to-plot-a-roc-curve-in-scikit-learn/

from sklearn.metrics import roc_curve, auc
Y_score = NB_OPTIMAL_classifier_for_BOW.predict_proba(bow_test_NB_vectors)
fpr, tpr, thresholds = roc_curve(y_test_NB,Y_score[:, 1])
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:


# Display Performance of the  Hyper-parametrized BOW model on TEST data

nb_classifier = NB_OPTIMAL_classifier_for_BOW

y_pred = nb_classifier.predict(bow_test_NB_vectors)
    
#Evaluate the model accuracy on TEST data

test_accuracy = accuracy_score(y_test_NB, y_pred, normalize=True) * 100
points = accuracy_score(y_test_NB, y_pred, normalize=False)

# Display the classification report
print(classification_report(y_test_NB, y_pred,digits=4))

#Display the model accuracy on TEST data
print('\nThe number of accurate predictions out of {} data points on TEST data is {}'.format(bow_test_NB_vectors.shape[0], points))
print('Accuracy of the {} model on TEST data is {} %'.format("BOW", '{:f}'.format(np.round(test_accuracy,2))))
     
# Display the confusion matrix
import scikitplot.metrics as sciplot
sciplot.plot_confusion_matrix(y_test_NB, y_pred)
    
    


# In[ ]:



    # '''Get top 50 features displayed from both the negative and the positive review classes.'''
    # Reference URL: https://stackoverflow.com/questions/50526898/how-to-get-feature-importance-in-naive-bayes#50530697
    
    neg_class_prob_sorted = (-NB_OPTIMAL_classifier_for_BOW.feature_log_prob_[0, :]).argsort()               #Note : Putting a - sign indicates the indexes will be sorted in descending order.
    pos_class_prob_sorted = (-NB_OPTIMAL_classifier_for_BOW.feature_log_prob_[1, :]).argsort()
    
    neg_class_features = np.take(bow_NB.get_feature_names(), neg_class_prob_sorted[:50])
    pos_class_features = np.take(bow_NB.get_feature_names(), pos_class_prob_sorted[:50])
    
    print("The top 50 most frequent words from the positive class are :\n")
    print(pos_class_features)
    
    print("\nThe top 50 most frequent words from the negative class are :\n")
    print(neg_class_features)
    
    del(neg_class_prob_sorted, pos_class_prob_sorted, neg_class_features, pos_class_features)


# # ASSIGNMENT- PART 2:  Naive-Bayes on TFIDF vector

# In[ ]:


# getting the base TFIDF vector

tf_idf_vect_NB = TfidfVectorizer(ngram_range=(1,1))

tfidf_NB = tf_idf_vect_NB.fit(X_train_NB['CleanedText'].values)

tfidf_train_NB = tfidf_NB.transform(X_train_NB['CleanedText'].values)

tfidf_test_NB  = tfidf_NB.transform(X_test_NB['CleanedText'].values)


# In[ ]:


#Colum Standardization of the TFIDF vector 

from sklearn.preprocessing import StandardScaler
scalar = StandardScaler(with_mean=False)
scalar.fit(tfidf_train_NB)
TFIDF_train_NB_vectors = scalar.transform(tfidf_train_NB)
TFIDF_test_NB_vectors = scalar.transform(tfidf_test_NB)


# In[ ]:


# 10-Fold Cross Validation to find the best Alpha for TFIDF model 

from sklearn.model_selection import TimeSeriesSplit
from sklearn.naive_bayes  import MultinomialNB

parameters = {"alpha":  np.array( [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10] )}

n_folds = 10

cv_timeSeries = TimeSeriesSplit(n_splits=n_folds)
    
model = MultinomialNB()

my_cv = TimeSeriesSplit(n_splits=n_folds).split(TFIDF_train_NB_vectors)
    
gsearch_cv_TFIDF = GridSearchCV(estimator=model, param_grid=parameters, cv=my_cv, scoring='f1')
    
gsearch_cv_TFIDF.fit(TFIDF_train_NB_vectors, y_train_NB)


# In[ ]:


#Plot the Scores for each Alpha used in CRoss-Validation
# source ref URL:  https://glowingpython.blogspot.com/2014/04/parameters-selection-with-cross.html

res_tfidf = list(zip(*[(f1m,f1s.std(), p['alpha']) 
            for p, f1m, f1s in gsearch_cv_TFIDF.grid_scores_]))

plt.plot(res_tfidf[2],res_tfidf[0],'-o', color="g",)
plt.xlabel('values of Alpha (Hyper-Parameter)')
plt.ylabel('Average Score (Better Score implies Lesser Error)')

plt.show()


# In[ ]:





# In[ ]:


# Display  the details of the hyper-parametrized NB classifer (TFIDF)

NB_OPTIMAL_classifier_for_TFIDF = gsearch_cv_TFIDF.best_estimator_
print("Best estimator for {} model : ".format("TFIDF"), NB_OPTIMAL_classifier_for_TFIDF)

NB_OPTIMAL_score_for_TFIDF = gsearch_cv_TFIDF.best_score_
print("Best Score for {} model : ".format("TFIDF"), NB_OPTIMAL_score_for_TFIDF)

OPTIMAL_MODEL_for_TFIDF= gsearch_cv_TFIDF.best_params_
for alpha in OPTIMAL_MODEL_for_TFIDF:
    print("Optimal Alpha for {} model : ".format("TFIDF"), '{:f}'.format(OPTIMAL_MODEL_for_TFIDF[alpha]))


# In[ ]:


# Plotting the ROC Curve for the Best Hyper-parametrized classifier using TFIDF vector

            # Ref-Source-URL:  https://datamize.wordpress.com/2015/01/24/how-to-plot-a-roc-curve-in-scikit-learn/

from sklearn.metrics import roc_curve, auc
Y_score = NB_OPTIMAL_classifier_for_TFIDF.predict_proba(bow_test_NB_vectors)
fpr, tpr, thresholds = roc_curve(y_test_NB,Y_score[:, 1])
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:



# Display  the performance of  TFIDF model on TEST data

#Predict the labels for the test set
y_pred_TFIDF = NB_OPTIMAL_classifier_for_TFIDF.predict(TFIDF_test_NB_vectors)
    
#Evaluate the accuracy of the model on TEST data
test_accuracy_TFIDF = accuracy_score(y_test_NB, y_pred_TFIDF, normalize=True) * 100
points_TFIDF = accuracy_score(y_test_NB, y_pred_TFIDF, normalize=False)

#Display the classification_report
print(classification_report(y_test_NB, y_pred_TFIDF,digits=4))

#Display the  accuracy of the model on TEST data
print('\nThe number of accurate predictions out of {} data points on unseen data is {}'.format(TFIDF_test_NB_vectors.shape[0], points_TFIDF))
print('Accuracy of the {} model on unseen data is {} %'.format("TFIDF", np.round(test_accuracy_TFIDF,2)))

#Display the  confusion matrix
import scikitplot.metrics as sciplot
sciplot.plot_confusion_matrix(y_test_NB, y_pred_TFIDF)


# In[ ]:



 # '''Get top 50 features displayed from both the negative and the positive review classes for the TF-IDF 
    # Reference URL: https://stackoverflow.com/questions/50526898/how-to-get-feature-importance-in-naive-bayes#50530697
    
    neg_class_prob_sorted_TFIDF = (-NB_OPTIMAL_classifier_for_TFIDF.feature_log_prob_[0, :]).argsort()              
    pos_class_prob_sorted_TFIDF = (-NB_OPTIMAL_classifier_for_TFIDF.feature_log_prob_[1, :]).argsort()
    
    neg_class_features_TFIDF = np.take(tfidf_NB.get_feature_names(), neg_class_prob_sorted_TFIDF[:50])
    pos_class_features_TFIDF = np.take(tfidf_NB.get_feature_names(), pos_class_prob_sorted_TFIDF[:50])
    
    print("The top 50 most frequent words from the positive class are :\n")
    print(pos_class_features_TFIDF)
    
    print("\nThe top 50 most frequent words from the negative class are :\n")
    print(neg_class_features_TFIDF)


# In[ ]:


# Clearing the memory space for faster processing
del(neg_class_prob_sorted_TFIDF, pos_class_prob_sorted_TFIDF, neg_class_features_TFIDF, pos_class_features_TFIDF)


# 

# # ASSIGNMENT- PART 3:  Naive-Bayes on BI-GRAMS vector

# # computing the  BI-GRAMS matrix

# In[ ]:


# BI Grams matrix

TFIDF_vect_BIGRAMS = TfidfVectorizer(ngram_range=(1,2) )  # here we are taking BIGRAMS only 

BIGRAMS_NB = TFIDF_vect_BIGRAMS.fit(X_train_NB['CleanedText'].values)

BIGRAMS_train_NB = BIGRAMS_NB.transform(X_train_NB['CleanedText'].values)

BIGRAMS_test_NB = BIGRAMS_NB.transform(X_test_NB['CleanedText'].values)


# In[ ]:


# Colum Standardization of the Bigrams vector

from sklearn.preprocessing import StandardScaler
scalar = StandardScaler(with_mean=False)
scalar.fit(BIGRAMS_train_NB)
BIGRAMS_train_NB_vectors = scalar.transform(BIGRAMS_train_NB)
BIGRAMS_test_NB_vectors = scalar.transform(BIGRAMS_test_NB)


#  ##  Running the NB Classifier on BIGRAMS data

# In[ ]:


# 10 fold CV to get the Optimal Alpha for BIGRAMS model 

from sklearn.model_selection import TimeSeriesSplit
from sklearn.naive_bayes  import MultinomialNB

parameters = {"alpha": np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10])}

n_folds = 10

cv_timeSeries = TimeSeriesSplit(n_splits=n_folds)
    
model = MultinomialNB()

my_cv = TimeSeriesSplit(n_splits=n_folds).split(BIGRAMS_train_NB_vectors)
    
gsearch_cv_BIGRAMS = GridSearchCV(estimator=model, param_grid=parameters, cv=my_cv, scoring='f1')
    
gsearch_cv_BIGRAMS.fit(BIGRAMS_train_NB_vectors, y_train_NB)

    


# In[ ]:


#Plot the Scores for the Alphas used in Cross-Validation  for the BIGRAMS vector
            # source ref URL:  https://glowingpython.blogspot.com/2014/04/parameters-selection-with-cross.html

res_BIGRAMS = list(zip(*[(f1m,f1s.std(), p['alpha']) 
            for p, f1m, f1s in gsearch_cv_BIGRAMS.grid_scores_]))

plt.plot(res_BIGRAMS[2],res_BIGRAMS[0],'-o', color="g",)
plt.xlabel('values of Alpha (Hyper-Parameter)')
plt.ylabel('Average Score (Better Score implies Lesser Error)')

plt.show()


# In[ ]:


# Display  the Hyper-parametrized BIGRAMS model details

NB_OPTIMAL_classifier_for_BIGRAMS = gsearch_cv_BIGRAMS.best_estimator_
print("Best estimator for {} model : ".format("BIGRAMS"), NB_OPTIMAL_classifier_for_BIGRAMS)

NB_OPTIMAL_score_for_BIGRAMS = gsearch_cv_BIGRAMS.best_score_
print("Best Score for {} model : ".format("BIGRAMS"), NB_OPTIMAL_score_for_BIGRAMS)

OPTIMAL_MODEL_for_BIGRAMS= gsearch_cv_BIGRAMS.best_params_
for alpha in OPTIMAL_MODEL_for_BIGRAMS:
    print("Optimal Alpha for {} model : ".format("BIGRAMS"), '{:f}'.format(OPTIMAL_MODEL_for_BIGRAMS[alpha]))



# In[ ]:



# Plotting the ROC Curve for the Hyper-parametrized BIGRAMS model

# Ref-Source-URL:  https://datamize.wordpress.com/2015/01/24/how-to-plot-a-roc-curve-in-scikit-learn/

from sklearn.metrics import roc_curve, auc
Y_score = NB_OPTIMAL_classifier_for_BOW.predict_proba(bow_test_NB_vectors)
fpr, tpr, thresholds = roc_curve(y_test_NB,Y_score[:, 1])
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:


# Display  the performance of  BIGRAMS model on TEST data

#Predict the labels of TEST data

y_pred = NB_OPTIMAL_classifier_for_BIGRAMS.predict(BIGRAMS_test_NB_vectors)
    
#Get the accuracy of the model on TEST data
test_accuracy_BIGRAMS = accuracy_score(y_test_NB, y_pred, normalize=True) * 100
points_BIGRAMS = accuracy_score(y_test_NB, y_pred, normalize=False)

# Display the classification_report
print(classification_report(y_test_NB, y_pred,digits=4))

#Display the model accuracy of the model on TEST data
print('\nThe number of accurate predictions out of {} data points on unseen data is {}'.format(BIGRAMS_test_NB_vectors.shape[0], points_BIGRAMS))
print('Accuracy of the {} model on unseen data is {} %'.format("BIGRAMS", np.round(test_accuracy_BIGRAMS,2)))

#Display the confusion matrix
import scikitplot.metrics as sciplot
sciplot.plot_confusion_matrix(y_test_NB, y_pred)
    
    


# In[ ]:



    # '''Get top 50 features displayed from both the negative and the positive review classes for the BIGRAMS 
    # Reference URL: https://stackoverflow.com/questions/50526898/how-to-get-feature-importance-in-naive-bayes#50530697
    
    neg_class_prob_sorted_BIGRAMS = (-NB_OPTIMAL_classifier_for_BIGRAMS.feature_log_prob_[0, :]).argsort()               #Note : Putting a - sign indicates the indexes will be sorted in descending order.
    pos_class_prob_sorted_BIGRAMS = (-NB_OPTIMAL_classifier_for_BIGRAMS.feature_log_prob_[1, :]).argsort()
    
    neg_class_features_BIGRAMS = np.take(BIGRAMS_NB.get_feature_names(), neg_class_prob_sorted_BIGRAMS[:50])
    pos_class_features_BIGRAMS = np.take(BIGRAMS_NB.get_feature_names(), pos_class_prob_sorted_BIGRAMS[:50])
    
    print("The top 50 most frequent words from the positive class are :\n")
    print(pos_class_features_BIGRAMS)
    
    print("\nThe top 50 most frequent words from the negative class are :\n")
    print(neg_class_features_BIGRAMS)
    
    


# In[ ]:


# Clearing the memory space for faster processing
del(neg_class_prob_sorted_BIGRAMS, pos_class_prob_sorted_BIGRAMS, neg_class_features_BIGRAMS, pos_class_features_BIGRAMS)


# 1. # ASSIGNMENT- PART 4:  Naive-Bayes on TRI-GRAMS vector

# # computing the  TRI-GRAMS matrix

# In[ ]:


# TRI Grams matrix

TFIDF_vect_TRIGRAMS = TfidfVectorizer(ngram_range=(2,3) )  # here we r taking only TRI-RAMS

TFIDF_NB_TRIRAMS = TFIDF_vect_TRIGRAMS.fit(X_train_NB['CleanedText'].values)

TRIGRAMS_train_NB = TFIDF_NB_TRIRAMS.transform(X_train_NB['CleanedText'].values)

TRIGRAMS_test_NB = TFIDF_NB_TRIRAMS.transform(X_test_NB['CleanedText'].values)


# In[ ]:


# #Colum Standardization of the TRigrams vector created using cleaned data

from sklearn.preprocessing import StandardScaler
scalar = StandardScaler(with_mean=False)
scalar.fit(TRIGRAMS_train_NB)
TRIGRAMS_train_NB_vectors = scalar.transform(TRIGRAMS_train_NB)
TRIGRAMS_test_NB_vectors = scalar.transform(TRIGRAMS_test_NB)


#  ##  Running the NB Classifier on TRIGRAMS data

# In[ ]:


# Running 10 fold CV to get the Hyper-Parameter Alpha for TRIGRAMS model

from sklearn.model_selection import TimeSeriesSplit
from sklearn.naive_bayes  import MultinomialNB

parameters = {"alpha": np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10])}

n_folds = 10

cv_timeSeries = TimeSeriesSplit(n_splits=n_folds)
    
model = MultinomialNB()

my_cv = TimeSeriesSplit(n_splits=n_folds).split(TRIGRAMS_train_NB_vectors)
    
gsearch_cv_TRIGRAMS = GridSearchCV(estimator=model, param_grid=parameters, cv=my_cv, scoring='f1')
    
gsearch_cv_TRIGRAMS.fit(TRIGRAMS_train_NB_vectors, y_train_NB)


# In[ ]:


#Plot the Scores for each Alpha used in Cross-Validation
                    # source ref URL:  https://glowingpython.blogspot.com/2014/04/parameters-selection-with-cross.html

res_TRIGRAMS = list(zip(*[(f1m,f1s.std(), p['alpha']) 
            for p, f1m, f1s in gsearch_cv_TRIGRAMS.grid_scores_]))

plt.plot(res_TRIGRAMS[2],res_TRIGRAMS[0],'-o', color="g",)
plt.xlabel('values of Alpha (Hyper-Parameter)')
plt.ylabel('Average Score (Better Score implies Lesser Error)')

plt.show()


# In[ ]:


# Display the details of the Hyper-parametrized (alpha) TRIGRAMS model

NB_OPTIMAL_classifier_for_TRIGRAMS = gsearch_cv_TRIGRAMS.best_estimator_
print("Best estimator for {} model : ".format("TRIGRAMS"), NB_OPTIMAL_classifier_for_TRIGRAMS)

NB_OPTIMAL_score_for_TRIGRAMS = gsearch_cv_TRIGRAMS.best_score_
print("Best Score for {} model : ".format("TRIGRAMS"), NB_OPTIMAL_score_for_TRIGRAMS)

OPTIMALMODEL= gsearch_cv_TRIGRAMS.best_params_
for alpha in OPTIMALMODEL:
    print("Optimal Alpha for {} model : ".format("TRIGRAMS"), '{:f}'.format(OPTIMALMODEL[alpha]))


# In[ ]:



# Plotting the ROC Curve for the Hyper-parametrized (alpha) TRIGRAMS model

# Ref-Source-URL:  https://datamize.wordpress.com/2015/01/24/how-to-plot-a-roc-curve-in-scikit-learn/

from sklearn.metrics import roc_curve, auc
Y_score = NB_OPTIMAL_classifier_for_BOW.predict_proba(bow_test_NB_vectors)
fpr, tpr, thresholds = roc_curve(y_test_NB,Y_score[:, 1])
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()



# In[ ]:


# Display  the performance of  TRIGRAMS model on TEST data

#Predict the labels for the test set

y_pred_TRIGRAMS = NB_OPTIMAL_classifier_for_TRIGRAMS.predict(TRIGRAMS_test_NB_vectors)
    
#Get the accuracy of the model on TEST data
test_accuracy_TRIGRAMS = accuracy_score(y_test_NB, y_pred_TRIGRAMS, normalize=True) * 100
points_TRIGRAMS = accuracy_score(y_test_NB, y_pred_TRIGRAMS, normalize=False)

#Display the classification_report
print(classification_report(y_test_NB, y_pred_TRIGRAMS,digits=4))

#Display the accuracy of the model on TEST data
print('\nThe number of accurate predictions out of {} data points on unseen data is {}'.format(TRIGRAMS_test_NB_vectors.shape[0], points_TRIGRAMS))
print('Accuracy of the {} model on unseen data is {} %'.format("TRIGRAMS", np.round(test_accuracy_TRIGRAMS,2)))

#Display the confusion_matrix
import scikitplot.metrics as sciplot
sciplot.plot_confusion_matrix(y_test_NB, y_pred_TRIGRAMS)


# In[ ]:



    # '''Get top 50 features displayed from both the negative and the positive review classes for the TRI-GRAMS 
    # Reference URL: https://stackoverflow.com/questions/50526898/how-to-get-feature-importance-in-naive-bayes#50530697
    
    neg_class_prob_sorted_TRIGRAMS = (-NB_OPTIMAL_classifier_for_TRIGRAMS.feature_log_prob_[0, :]).argsort()               #Note : Putting a - sign indicates the indexes will be sorted in descending order.
    pos_class_prob_sorted_TRIGRAMS = (-NB_OPTIMAL_classifier_for_TRIGRAMS.feature_log_prob_[1, :]).argsort()
    
    neg_class_features_TRIGRAMS = np.take(TFIDF_NB_TRIRAMS.get_feature_names(), neg_class_prob_sorted_TRIGRAMS[:50])
    pos_class_features_TRIGRAMS = np.take(TFIDF_NB_TRIRAMS.get_feature_names(), pos_class_prob_sorted_TRIGRAMS[:50])
    
    print("The top 50 most frequent words from the positive class are :\n")
    print(pos_class_features_TRIGRAMS)
    
    print("\nThe top 50 most frequent words from the negative class are :\n")
    print(neg_class_features_TRIGRAMS)


# In[ ]:


# Clearing the memory space for faster processing
del(neg_class_prob_sorted_TRIGRAMS, pos_class_prob_sorted_TRIGRAMS, neg_class_features_TRIGRAMS, pos_class_features_TRIGRAMS)


# #  Summarizing the Results obtained from the various models:
# 

# In[ ]:


# Summary of Results
        #  Ref URL:  https://stackoverflow.com/questions/9535954/printing-lists-as-tabular-data

def print_table(table):
    longest_cols = [
        (max([len(str(row[i])) for row in table]) + 3)
        for i in range(len(table[0]))
    ]
    row_format = "".join(["{:>" + str(longest_col) + "}" for longest_col in longest_cols])
    for row in table:
        print(row_format.format(*row))

table = [
    ["Model", "OPTIMAL_ALPHA", "BEST_CV_SCORE", "TEST_ACCURACY"],
    ["BOW-Model", round(OPTIMAL_MODEL_for_BOW[alpha],5), NB_OPTIMAL_score_for_BOW, round(test_accuracy,2)],
    ["TFIDF-Model", round(OPTIMAL_MODEL_for_TFIDF[alpha],5), NB_OPTIMAL_score_for_TFIDF, round(test_accuracy_TFIDF,2)],
    ["BIGRAMS-Model", round(OPTIMAL_MODEL_for_BIGRAMS[alpha],5), NB_OPTIMAL_score_for_BIGRAMS,round(test_accuracy_BIGRAMS,2)],
    ["TRIGRAMS-Model",round(OPTIMALMODEL[alpha],5), NB_OPTIMAL_score_for_TRIGRAMS,round(test_accuracy_TRIGRAMS,2)]]
 
print_table(table)

