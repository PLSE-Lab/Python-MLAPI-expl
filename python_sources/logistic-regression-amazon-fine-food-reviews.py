#!/usr/bin/env python
# coding: utf-8

# # Amazon Fine Food Reviews Analysis
# 

# #### Data Source: https://www.kaggle.com/snap/amazon-ne-food-reviews  
# #### EDA: https://nycdatascience.com/blog/student-works/amazon-ne-foods-visualization/ 
# #### The Amazon Fine Food Reviews dataset consists of reviews of ne foods from Amazon. 
# #### Number of reviews: 568,454 
# #### Number of users: 256,059 
# #### Number of products: 74,258 
# #### Timespan: Oct 1999 - Oct 2012 
# #### Number of Attributes/Columns in data:10 Attribute Information:
# #### 1. Id
# #### 2. ProductId - unique identier for the product 
# #### 3. UserId - unqiue identier for the user 
# #### 4. ProleName 
# #### 5. HelpfulnessNumerator - number of users who found the review helpful 
# #### 6. HelpfulnessDenominator - number of users who indicated whether they found the review helpful or not 
# #### 7. Score - rating between 1 and 5 
# #### 8. Time - timestamp for the review  
# #### 9. Summary - brief summary of the review 
# #### 10. Text - text of the review
# 

# ## Objective:
# 
# Given a review, determine whether the review is positive (rating of 4 or 5) or negative (rating of 1 or 2).

# [Q] How to determine if a review is positive or negative? 
#  
# [Ans] We could use Score/Rating. A rating of 4 or 5 can be cosnidered as a positive review. A rating of 1 or 2 can be considered as negative one. A review of rating 3 is considered nuetral and such reviews are ignored from our analysis. This is an approximate and proxy way of determining the polarity (positivity/negativity) of a review

# ##  [1]. Reading Data
# 
#   ###     [1.1] Loading the data
#   
#   The dataset is available in two forms
# 1. .csv le
# 2. SQLite Database
# In order to load the data, We have used the SQLITE dataset as it is easier to query the data and visualise the data eciently.  Here as we only want to get the global sentiment of the recommendations (positive or negative), we will purposefully ignore all Scores equal to 3. If the score is above 3, then the recommendation wil be set to "positive". Otherwise, it will be set to "negative".
# 

# In[ ]:


# using the SQLite Table to read data.
con = sqlite3.connect('../input/database.sqlite')
#con = sqlite3.connect('database.sqlite') 

#filtering only positive and negative reviews i.e. 
# not taking into consideration those reviews with Score=3
filtered_data = pd.read_sql_query("""SELECT * FROM Reviews WHERE Score != 3 LIMIT 100000""", con) 

# Give reviews with Score>3 a positive rating, and reviews with a score<3 a negative rating.
def partition(x):
    if x < 3:
        return 0
    return 1

#changing reviews with score less than 3 to be positive and vice-versa
actualScore = filtered_data['Score']
positiveNegative = actualScore.map(partition) 
filtered_data['Score'] = positiveNegative
print("Number of data points in our data", filtered_data.shape)
filtered_data.head(5)


# In[ ]:


display = pd.read_sql_query(""" SELECT UserId, ProductId, ProfileName, Time, Score, Text, COUNT(*) FROM Reviews GROUP BY UserId HAVING COUNT(*)>1 """, con)
print(display.shape)
display.head()


# In[ ]:


display[display['UserId']=='AZY10LLTJ71NX']


# In[ ]:


display['COUNT(*)'].sum()


# ## [2] Exploratory Data Analysis
# 
# ### [2.1] Data Cleaning: Deduplication
# It is observed (as shown in the table below) that the reviews data had many duplicate entries. Hence it was necessary to remove duplicates in order to get unbiased results for the analysis of the data. Following is an example: 

# In[ ]:


display= pd.read_sql_query(""" SELECT * FROM Reviews WHERE Score != 3 AND UserId="AR5J8UI46CURR" ORDER BY ProductID """, con)
display.head()


# As can be seen above the same user has multiple reviews of the with the same values for HelpfulnessNumerator, HelpfulnessDenominator, Score, Time, Summary and Text and on doing analysis it was found that 
# 
# ProductId=B000HDOPZG was Loacker Quadratini Vanilla Wafer Cookies, 8.82-Ounce Packages (Pack of 8)
# 
# ProductId=B000HDL1RQ was Loacker Quadratini Lemon Wafer Cookies, 8.82-Ounce Packages (Pack of 8) and so on
# 
# It was inferred after analysis that reviews with same parameters other than ProductId belonged to the same product just having different flavour or quantity. Hence in order to reduce redundancy it was decided to eliminate the rows having same parameters.
# 
# The method used for the same was that we first sort the data according to ProductId and then just keep the first similar product review and delelte the others. for eg. in the above just the review for ProductId=B000HDL1RQ remains. This method ensures that there is only one representative for each product and deduplication without sorting would lead to possibility of different representatives still existing for the same product.
# 

# In[ ]:


#Sorting data according to ProductId in ascending order
sorted_data = filtered_data.sort_values('ProductId', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')


# In[ ]:


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
from sklearn.preprocessing import StandardScaler

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
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score 
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix 
import seaborn as sns


# In[ ]:


#Deduplication of entries
final = sorted_data.drop_duplicates(subset = {"UserId","ProfileName","Time","Text"}, keep ='first', inplace=False)
final.shape


# In[ ]:


# Checking to see how much % of data still remains
(final['Id'].size*1.0)/(filtered_data['Id'].size*1.0)*100


# ### Observation:
# It was also seen that in two rows given below the value of HelpfulnessNumerator is greater than HelpfulnessDenominator which is not practically possible hence these two rows too are removed from calcualtions

# In[ ]:


con = sqlite3.connect('../input/database.sqlite')
#con = sqlite3.connect('database.sqlite') 
display= pd.read_sql_query(""" SELECT * FROM Reviews WHERE Score != 3 AND Id=44737 OR Id=64422 ORDER BY ProductID """, con)
display.head()


# In[ ]:


final=final[final.HelpfulnessNumerator<=final.HelpfulnessDenominator]


# In[ ]:


#Before starting the next phase of preprocessing lets see the number of entries left
print(final.shape)

#How many positive and negative reviews are present in our dataset?
print(final['Score'].value_counts())
final['Score'].value_counts().plot(kind='bar')


# ### [3] Preprocessing
# #### [3.1]. Preprocessing Review Text
# 
# ### Text Preprocessing: Stemming, stop-word removal and Lemmatization
# 
# Now that we have finished deduplication our data requires some preprocessing before we go on further with analysis and making the prediction model.
# 
# Hence in the Preprocessing phase we do the following in the order below:-
# 
# 1.Begin by removing the html tags
# 
# 2.Remove any punctuations or limited set of special characters like , or . or # etc.
# 
# 3.Check if the word is made up of english letters and is not alpha-numeric
# 
# 4.Check to see if the length of the word is greater than 2 (as it was researched that there is no adjective in 2-letters)
# 
# 5.Convert the word to lowercase
# 
# 6.Remove Stopwords
# 
# 7.Finally Snowball Stemming the word (it was obsereved to be better than Porter Stemming)
# 
# After which we collect the words used to describe positive and negative reviews

# In[ ]:


final['Time']=pd.to_datetime(final['Time'],unit='s')
final=final.sort_values(by='Time')
final.head(5)


# In[ ]:


# printing some random reviews
sent_0 = final['Text'].values[0]
print(sent_0)
print("="*50)
sent_1000 = final['Text'].values[1000]
print(sent_1000) 
print("="*50)
sent_1500 = final['Text'].values[1500]
print(sent_1500) 
print("="*50)
sent_4900 = final['Text'].values[4900]
print(sent_4900)
print("="*50)


# In[ ]:


sent_0 = re.sub(r"http\S+", "", sent_0)
sent_1000 = re.sub(r"http\S+", "", sent_1000)
sent_150 = re.sub(r"http\S+", "", sent_1500)
sent_4900 = re.sub(r"http\S+", "", sent_4900) 
print(sent_0)


# In[ ]:


from bs4 import BeautifulSoup
soup = BeautifulSoup(sent_0, 'lxml') 
text = soup.get_text() 
print(text) 
print("="*50)

soup = BeautifulSoup(sent_1000, 'lxml') 
text = soup.get_text() 
print(text) 
print("="*50) 

soup = BeautifulSoup(sent_1500, 'lxml')
text = soup.get_text()
print(text)
print("="*50) 

soup = BeautifulSoup(sent_4900, 'lxml')
text = soup.get_text()
print(text)


# In[ ]:


import re
def decontracted(phrase): 
     # specific 
    phrase = re.sub(r"won't", "will not", phrase) 
    phrase = re.sub(r"can\'t", "can not", phrase) 
    # general 
    phrase = re.sub(r"n\'t", " not", phrase) 
    phrase = re.sub(r"\'re", " are", phrase)  
    phrase = re.sub(r"\'s", " is", phrase)   
    phrase = re.sub(r"\'d", " would", phrase) 
    phrase = re.sub(r"\'ll", " will", phrase)  
    phrase = re.sub(r"\'t", " not", phrase)   
    phrase = re.sub(r"\'ve", " have", phrase)  
    phrase = re.sub(r"\'m", " am", phrase)   
    return phrase
sent_1500 = decontracted(sent_1500) 
print(sent_1500)
print("="*50)


        


# In[ ]:


#remove words with numbers python: https://stackoverflow.com/a/18082370/4084039 
sent_0 = re.sub("\S*\d\S*", "", sent_0).strip()
print(sent_0)


# In[ ]:


#remove spacial character: https://stackoverflow.com/a/5843547/4084039
sent_1500 = re.sub('[^A-Za-z0-9]+', ' ', sent_1500)
print(sent_1500)


# In[ ]:


# https://gist.github.com/sebleier/554280
# we are removing the words from the stop words list: 'no', 'nor', 'not'
# <br /><br /> ==> after the above steps, we are getting "br br"
# we are including them into stop words list
# instead of <br /> if we have <br/> these tags would have revmoved in the 1st ste
stop = set(stopwords.words('english')) #set of stopwords


# In[ ]:


# Combining all the above stundents 
from tqdm import tqdm 
preprocessed_reviews = [] 
# tqdm is for printing the status bar 
for sentance in tqdm(final['Text'].values): 
    sentance = re.sub(r"http\S+", "", sentance)
    sentance = BeautifulSoup(sentance, 'lxml').get_text()
    sentance = decontracted(sentance)
    sentance = re.sub("\S*\d\S*", "", sentance).strip()
    sentance = re.sub('[^A-Za-z]+', ' ', sentance)
    # https://gist.github.com/sebleier/554280
    sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stop)
    preprocessed_reviews.append(sentance.strip())
preprocessed_reviews[1500] 

 


# In[ ]:


print(len(preprocessed_reviews))
final.shape


# In[ ]:


final ['preprocessed_reviews']= preprocessed_reviews
final.head(5)


# # store final table into an SQlLite table for future.

# In[ ]:


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
from sklearn.preprocessing import StandardScaler

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
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score 
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix 
import seaborn as sns


# In[ ]:



conn = sqlite3.connect('final.sqlite')
c=conn.cursor()
conn.text_factory = str
final.to_sql('Reviews', conn,  schema=None, if_exists='replace',index=True, index_label=None, chunksize=None, dtype=None)
conn.close()
#Loading data
conn = sqlite3.connect('final.sqlite')
data=pd.read_sql_query("""select * from Reviews""",conn)


# In[ ]:





# # splitting data into Train, C.V and Test

# In[ ]:



X_train, X_test, y_train, y_test = train_test_split(data ['preprocessed_reviews'], data['Score'], test_size=0.33) 
X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size=0.33)
print("Train:",X_train.shape,y_train.shape)
print("CV:",X_cv.shape,y_cv.shape)
print("Test:",X_test.shape,y_test.shape)


# In[ ]:


vectorizer = CountVectorizer(ngram_range=(1,2))
vectorizer.fit(X_train)
#vectorizer.fit(X_train) # fit has to happen only on train data
# we use the fitted CountVectorizer to convert the text to vector
X_train_bow = vectorizer.fit_transform(X_train)
X_cv_bow = vectorizer.transform(X_cv)
X_test_bow = vectorizer.transform(X_test) 
print("After vectorizations")
print(X_train_bow.shape, y_train.shape) 
print(X_cv_bow.shape, y_cv.shape)
print(X_test_bow.shape, y_test.shape) 

print("*************************")
print("Standardization")
X_train=StandardScaler(with_mean=False).fit_transform(X_train_bow)
X_cv=StandardScaler(with_mean=False).fit_transform(X_cv_bow)
X_test=StandardScaler(with_mean=False).fit_transform(X_test_bow)
print(X_train.shape, y_train.shape) 
print(X_cv.shape, y_cv.shape)
print(X_test.shape, y_test.shape) 




# In[ ]:


lst=[10**i for i in range(-5,5)]
print(lst)


# In[ ]:




train_auc_l1 = [] 
train_auc_l2 = [] 
cv_auc_l1 = []
cv_auc_l2 = []
penalty=['l1','l2']
hyper_param=[10**i for i in range(-5,5)]
for i in penalty:
    for j in  hyper_param:
        clf = LogisticRegression(C=j, penalty= i)
        clf.fit(X_train, y_train)
        y_train_pred =  clf.predict_proba(X_train)[:,1] 
        y_cv_pred =  clf.predict_proba(X_cv)[:,1]
        if i == 'l1':
            train_auc_l1.append(roc_auc_score(y_train,y_train_pred)) 
            cv_auc_l1.append(roc_auc_score(y_cv, y_cv_pred))
        else:
            train_auc_l2.append(roc_auc_score(y_train,y_train_pred))
            cv_auc_l2.append(roc_auc_score(y_cv, y_cv_pred))
#Error plots with penaly L1
plt.plot(np.log(hyper_param), train_auc_l1, label='Train AUC-L1') 
plt.plot(np.log(hyper_param), cv_auc_l1, label='CV AUC-L1')
plt.legend()
plt.xlabel("Lamda: hyperparameter")
plt.ylabel("AUC")
plt.title("ERROR PLOTS")
plt.show()
#Error plots with penaly L2
plt.plot(np.log(hyper_param), train_auc_l2, label='Train AUC-L2') 
plt.plot(np.log(hyper_param), cv_auc_l2, label='CV AUC-L2')
plt.legend()
plt.xlabel("Lamda: hyperparameter")
plt.ylabel("AUC")
plt.title("ERROR PLOTS")
plt.show()
#Cv auc scores with penalty L1
print("Cv auc scores with penalty L1")
print(cv_auc_l1)
print("Maximun Auc value :",max(cv_auc_l1))
print("Index",cv_auc_l1.index(max(cv_auc_l1)))
#Cv auc scores with penalty L2
print("--------------------------")
print("Cv auc scores with penalty L2")
print(cv_auc_l2)
print("Maximun Auc value :",max(cv_auc_l2))
print("Index",cv_auc_l2.index(max(cv_auc_l2)))


# # Function to get important features

# ## [4] Featurization
# ### [4.1] BAG OF WORD

# In[ ]:


optimal_lamda(X_train_bow,y_train,X_cv_bow,y_cv)


# In[ ]:


#Testing with test data
clf = LogisticRegression(penalty='l1',C=0.01)
clf.fit(X_train_bow, y_train)
# roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the p 
# not the predicted outputs 
train_fpr,train_tpr,thresholds = roc_curve(y_train, clf.predict_proba(X_train_bow)[:,1] )
test_fpr,test_tpr,thresholds = roc_curve(y_test, clf.predict_proba(X_test_bow)[:,1]) 
plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr))) 
plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))                                             
plt.legend()
plt.xlabel("C_values: hyperparameter") 
plt.ylabel("AUC") 
plt.title("ERROR PLOTS") 
plt.show()  


# ## Confusion Matrix

# In[ ]:


from sklearn.metrics import confusion_matrix 
print("Train confusion matrix") 
print(confusion_matrix(y_train, clf.predict(X_train_bow))) 
print("Test confusion matrix")
print(confusion_matrix(y_test, clf.predict(X_test_bow)))
cm_test=confusion_matrix(y_test, clf.predict(X_test_bow))
import seaborn as sns
class_label = ["negative", "positive"]
df_cm = pd.DataFrame(cm_test, index = class_label, columns = class_label)
sns.heatmap(df_cm, annot = True, fmt = "d")
plt.title("Confusiion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# ## Important features from positive and negative classes

# In[ ]:


top_features(vectorizer, clf,10)


# ## Pertubation test

# In[ ]:


#Weights before adding noise
weights1=clf.coef_
print(weights1.shape)


# In[ ]:


import copy
Noise_data=copy.deepcopy(X_train_bow)
e=np.random.normal(0,0.001)
Noise_data.data += e
print(Noise_data.shape)


# In[ ]:


#Calculating weights after adding noise
model = LogisticRegression(C= 0.01, penalty= 'l1')
model.fit(Noise_data,y_train)
weights2=model.coef_
print(weights2.shape)


# In[ ]:


#Adding small noise to avoid zero divide error
e=np.random.normal(0,0.001)
weights1+=e
weights2+=e


# In[ ]:


def top_features(vectorizer, clf, n):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    print("\t\t\tNegative\t\t\t\t\t\tPositive")
    print("________________________________________________________________________________________________")
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print("\t%.4f\t%-15s\t\t\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))


# In[ ]:


weights_difference = np.abs((weights2-weights1)/weights1)*100
#print(weights_difference)
print(weights_difference.max())
print(weights_difference.min())
print(weights_difference.std())


# In[ ]:



percentage_change=[]
collinear_features=[]

for i in range(1,101):
    f=np.where(weights_difference > i)[1].size
    percentage_change.append(i)
    collinear_features.append(f)
    
plt.xlabel('percentage change of weight vectors')
plt.ylabel('no. of multicollinear features')
plt.plot(percentage_change,collinear_features)


# In[ ]:


feat = vectorizer.get_feature_names()
print("No of features have weight changes greater than 30%: ", weights_difference[np.where(weights_difference > 30)].size)
fe=[]
print("\nHence below features are collinear:")
for i in np.where(weights_difference > 1)[1]:
    fe.append(feat[i])
print(fe)


# # 4.2 TF-IDF

# In[ ]:


vect = TfidfVectorizer(ngram_range=(1,2))
tf_idf_vect = vect.fit(X_train)
# we use the fitted CountVectorizer to convert the text to vector
X_train_tfidf = tf_idf_vect.transform(X_train)
X_cv_tfidf = tf_idf_vect.transform(X_cv)
X_test_tfidf = tf_idf_vect.transform(X_test) 
print("After vectorizations")
print(X_train_tfidf.shape, y_train.shape) 
print(X_cv_tfidf.shape, y_cv.shape)
print(X_test_tfidf.shape, y_test.shape) 
print("Standardization")
X_train_tfidf=StandardScaler(with_mean=False).fit_transform(X_train_tfidf)
X_cv_tfidf=StandardScaler(with_mean=False).fit_transform(X_cv_tfidf)
X_test_tfidf=StandardScaler(with_mean=False).fit_transform(X_test_tfidf)
print(X_train_tfidf.shape, y_train.shape) 
print(X_cv_tfidf.shape, y_cv.shape)
print(X_test_tfidf.shape, y_test.shape)


# In[ ]:


optimal_lamda(X_train_tfidf,y_train,X_cv_tfidf,y_cv)


# In[ ]:


#Testing with test data
clf = LogisticRegression(penalty='l1',C=0.01)
clf.fit(X_train_tfidf, y_train)
# roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the p 
# not the predicted outputs 
train_fpr,train_tpr,thresholds = roc_curve(y_train, clf.predict_proba(X_train_tfidf)[:,1] )
test_fpr,test_tpr,thresholds = roc_curve(y_test, clf.predict_proba(X_test_tfidf)[:,1]) 
plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr))) 
plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))                                             
plt.legend()
plt.xlabel("C_values: hyperparameter") 
plt.ylabel("AUC") 
plt.title("ERROR PLOTS") 
plt.show()  


# # Confusion Matrix

# In[ ]:


from sklearn.metrics import confusion_matrix 
print("Train confusion matrix") 
print(confusion_matrix(y_train, clf.predict(X_train_tfidf))) 
print("Test confusion matrix")
print(confusion_matrix(y_test, clf.predict(X_test_tfidf)))
cm_test=confusion_matrix(y_test, clf.predict(X_test_tfidf))
import seaborn as sns
class_label = ["negative", "positive"]
df_cm = pd.DataFrame(cm_test, index = class_label, columns = class_label)
sns.heatmap(df_cm, annot = True, fmt = "d")
plt.title("Confusiion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# ## Important features from positive and negative classes

# In[ ]:


top_features(vect, clf,10)


# ## No. of non Zero elements when penalty ='l1'

# In[ ]:


print(np.count_nonzero(clf.coef_))


# # 4.3 Avg W2V

# In[ ]:


#Word2Vec
#train
i=0 
list_of_sent=[] 
for sentance in X_train:    
    list_of_sent.append(sentance.split())
w2v_model=Word2Vec(list_of_sent,min_count=5,size=50, workers=4)
w2v_words = list(w2v_model.wv.vocab)


# In[ ]:


sent_vectors = []; # the avg-w2v for each sentence/review is stored in this list
for sent in tqdm(list_of_sent): # for each review/sentence
    sent_vec = np.zeros(50) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sent: # for each word in a review/sentence
        if word in w2v_words:
            vec = w2v_model.wv[word]
            sent_vec += vec
            cnt_words += 1
    if cnt_words != 0:
        sent_vec /= cnt_words
    sent_vectors.append(sent_vec)
print(len(sent_vectors))
print(len(sent_vectors[0]))


# In[ ]:


X_train_w2v=sent_vectors
print(len(X_train_w2v))


# In[ ]:


#cv
i=0 
list_of_sent=[] 
for sentance in X_cv:    
    list_of_sent.append(sentance.split())
w2v_model=Word2Vec(list_of_sent,min_count=5,size=50, workers=4)
w2v_words = list(w2v_model.wv.vocab)
sent_vectors = []; # the avg-w2v for each sentence/review is stored in this list
for sent in tqdm(list_of_sent): # for each review/sentence
    sent_vec = np.zeros(50) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sent: # for each word in a review/sentence
        if word in w2v_words:
            vec = w2v_model.wv[word]
            sent_vec += vec
            cnt_words += 1
    if cnt_words != 0:
        sent_vec /= cnt_words
    sent_vectors.append(sent_vec)
print(len(sent_vectors))
print(len(sent_vectors[0]))


# In[ ]:


X_cv_w2v=sent_vectors
print(len(X_cv_w2v))


# In[ ]:


#test
i=0 
list_of_sent=[] 
for sentance in X_test:    
    list_of_sent.append(sentance.split())
w2v_model=Word2Vec(list_of_sent,min_count=5,size=50, workers=4)
w2v_words = list(w2v_model.wv.vocab)
sent_vectors = []; # the avg-w2v for each sentence/review is stored in this list
for sent in tqdm(list_of_sent): # for each review/sentence
    sent_vec = np.zeros(50) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sent: # for each word in a review/sentence
        if word in w2v_words:
            vec = w2v_model.wv[word]
            sent_vec += vec
            cnt_words += 1
    if cnt_words != 0:
        sent_vec /= cnt_words
    sent_vectors.append(sent_vec)
print(len(sent_vectors))
print(len(sent_vectors[0]))


# In[ ]:


X_test_w2v=sent_vectors
print(len(X_test_w2v))


# In[ ]:


optimal_lamda(X_train_w2v,y_train,X_cv_w2v,y_cv)


# # Testing with test data

# In[ ]:



clf = LogisticRegression(penalty='l2',C=0.01)
clf.fit(X_train_w2v, y_train)
# roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the p 
# not the predicted outputs 
train_fpr,train_tpr,thresholds = roc_curve(y_train, clf.predict_proba(X_train_w2v)[:,1] )
test_fpr,test_tpr,thresholds = roc_curve(y_test, clf.predict_proba(X_test_w2v)[:,1]) 
plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr))) 
plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))                                             
plt.legend()
plt.xlabel("C_values: hyperparameter") 
plt.ylabel("AUC") 
plt.title("ERROR PLOTS") 
plt.show()  


# # Confusion Matrix

# In[ ]:


from sklearn.metrics import confusion_matrix 
print("Train confusion matrix") 
print(confusion_matrix(y_train, clf.predict(X_train_w2v))) 
print("Test confusion matrix")
print(confusion_matrix(y_test, clf.predict(X_test_w2v)))
cm_test=confusion_matrix(y_test, clf.predict(X_test_w2v))
import seaborn as sns
class_label = ["negative", "positive"]
df_cm = pd.DataFrame(cm_test, index = class_label, columns = class_label)
sns.heatmap(df_cm, annot = True, fmt = "d")
plt.title("Confusiion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# #  TFIDF weighted W2v

# In[ ]:


#train
i=0 
list_of_sentance=[] 
for sentance in X_train:    
    list_of_sentance.append(sentance.split())
model = TfidfVectorizer()
tf_idf_matrix = model.fit_transform(X_train)
# we are converting a dictionary with word as a key, and the idf as a value 
dictionary = dict(zip(model.get_feature_names(), list(model.idf_)))
# TF-IDF weighted Word2Vec
tfidf_feat = model.get_feature_names() # tfidf words/col-names 
# final_tf_idf is the sparse matrix with row= sentence, col=word and cell_val = tfidf 
tfidf_sent_vectors = []; # the tfidf-w2v for each sentence/review is stored in this list 
row=0; 
for sent in tqdm(list_of_sentance): # for each review/sentence 
    sent_vec = np.zeros(50) # as word vectors are of zero length 
    weight_sum =0; # num of words with a valid vector in the sentence/revie 
    for word in sent: # for each word in a review/sentenc
        if word in w2v_words and word in tfidf_feat:
            vec = w2v_model.wv[word]
            tf_idf = dictionary[word]*(sent.count(word)/len(sent))
            sent_vec += (vec * tf_idf)
            weight_sum += tf_idf
    if weight_sum != 0:
        sent_vec /= weight_sum
    tfidf_sent_vectors.append(sent_vec) 
    row += 1
       

    


# In[ ]:


X_train_tfidf_w2v=tfidf_sent_vectors
print(len(X_train_tfidf_w2v))


# In[ ]:


#cv
i=0 
list_of_sentance=[] 
for sentance in X_cv:    
    list_of_sentance.append(sentance.split())
model = TfidfVectorizer()
tf_idf_matrix = model.fit_transform(X_cv)
# we are converting a dictionary with word as a key, and the idf as a value 
dictionary = dict(zip(model.get_feature_names(), list(model.idf_)))
# TF-IDF weighted Word2Vec
tfidf_feat = model.get_feature_names() # tfidf words/col-names 
# final_tf_idf is the sparse matrix with row= sentence, col=word and cell_val = tfidf 
tfidf_sent_vectors = []; # the tfidf-w2v for each sentence/review is stored in this list 
row=0; 
for sent in tqdm(list_of_sentance): # for each review/sentence 
    sent_vec = np.zeros(50) # as word vectors are of zero length 
    weight_sum =0; # num of words with a valid vector in the sentence/revie 
    for word in sent: # for each word in a review/sentenc
        if word in w2v_words and word in tfidf_feat:
            vec = w2v_model.wv[word]
            tf_idf = dictionary[word]*(sent.count(word)/len(sent))
            sent_vec += (vec * tf_idf)
            weight_sum += tf_idf
    if weight_sum != 0:
        sent_vec /= weight_sum
    tfidf_sent_vectors.append(sent_vec) 
    row += 1
       

    


# In[ ]:


X_cv_tfidf_w2v=tfidf_sent_vectors
print(len(X_cv_tfidf_w2v))


# In[ ]:


#test
i=0 
list_of_sentance=[] 
for sentance in X_test:    
    list_of_sentance.append(sentance.split())
model = TfidfVectorizer()
tf_idf_matrix = model.fit_transform(X_test)
# we are converting a dictionary with word as a key, and the idf as a value 
dictionary = dict(zip(model.get_feature_names(), list(model.idf_)))
# TF-IDF weighted Word2Vec
tfidf_feat = model.get_feature_names() # tfidf words/col-names 
# final_tf_idf is the sparse matrix with row= sentence, col=word and cell_val = tfidf 
tfidf_sent_vectors = []; # the tfidf-w2v for each sentence/review is stored in this list 
row=0; 
for sent in tqdm(list_of_sentance): # for each review/sentence 
    sent_vec = np.zeros(50) # as word vectors are of zero length 
    weight_sum =0; # num of words with a valid vector in the sentence/revie 
    for word in sent: # for each word in a review/sentenc
        if word in w2v_words and word in tfidf_feat:
            vec = w2v_model.wv[word]
            tf_idf = dictionary[word]*(sent.count(word)/len(sent))
            sent_vec += (vec * tf_idf)
            weight_sum += tf_idf
    if weight_sum != 0:
        sent_vec /= weight_sum
    tfidf_sent_vectors.append(sent_vec) 
    row += 1
       

    


# In[ ]:


X_test_tfidf_w2v=tfidf_sent_vectors
print(len(X_test_tfidf_w2v))


# In[ ]:


optimal_lamda(X_train_tfidf_w2v,y_train,X_cv_tfidf_w2v,y_cv)


# # Testing with test data

# In[ ]:



clf = LogisticRegression(penalty='l1',C=1)
clf.fit(X_train_tfidf_w2v, y_train)
# roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the p 
# not the predicted outputs 
train_fpr,train_tpr,thresholds = roc_curve(y_train, clf.predict_proba(X_train_tfidf_w2v)[:,1] )
test_fpr,test_tpr,thresholds = roc_curve(y_test, clf.predict_proba(X_test_tfidf_w2v)[:,1]) 
plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr))) 
plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))                                             
plt.legend()
plt.xlabel("C_values: hyperparameter") 
plt.ylabel("AUC") 
plt.title("ERROR PLOTS") 
plt.show()  


# # Confusion Matrix

# In[ ]:


from sklearn.metrics import confusion_matrix 
print("Train confusion matrix") 
print(confusion_matrix(y_train, clf.predict(X_train_tfidf_w2v))) 
print("Test confusion matrix")
print(confusion_matrix(y_test, clf.predict(X_test_tfidf_w2v)))
cm_test=confusion_matrix(y_test, clf.predict(X_test_tfidf_w2v))
import seaborn as sns
class_label = ["negative", "positive"]
df_cm = pd.DataFrame(cm_test, index = class_label, columns = class_label)
sns.heatmap(df_cm, annot = True, fmt = "d")
plt.title("Confusiion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# # Result :

# In[ ]:


Data=[["Bag of Words",0.01,0.923,0.928,"2198","2436","393","23939"],["TFIDF",0.01,0.937,0.935,"2348","2286","366","23966"],["Avgw2v",0.01,0.846,0.819,"484","4150","162","24170"],["Tf-Idf-w2v",1,0.855,0.854,"1497","3137","799","23533"]]

result=pd.DataFrame(Data,columns=["Featurization","Hyper parameter(c)","CV-AUC","Test-Auc","TNR","FPR","FNR","TPR"])
result


# # Conclusion :

# * TFIDF  Featurization performs best with CV-AUV of  0.937 and TEST AUC of 0.935.
# * TPR is maximun when tarined the model by tfidf.
# * FNR is low when tarined the model by Avg2v
# * Sparsity increases as we increase lambda or decrease C when L1 Regularizer is used.
# . 
