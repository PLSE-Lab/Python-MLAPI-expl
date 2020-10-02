#!/usr/bin/env python
# coding: utf-8

# # STOCK SENTIMENT ANALYSIS USING NEWS HEADLINES
# 
# IN THIS NOTEBOOK WE WILL CLASSIFY WHETHER THE STOCKS OF THE COMPANY WILL GO UP OR GO DOWN
# 
# ON THE BASIS OF THE TOP 25 HEADLINES ABOUT THE COMPANY.
# 
# WE WILL BE DOING TEXT PREPROCESSING AND THEN WE WILL USE RANDOM-FOREST CLASSIFIER AND NAIVE BAYES CLASSIFIER FOR THE PURPOSE OF CLASSIFICATION.
# 
# WE WILL USE BOTH 'BAG OF WORDS MODEL' AND 'TF-IDF VECTORIZER' TO CONVERT TEXT INTO VECTORS.
# 
# 1---> STOCKS WILL GO UP
# 
# 0---> STOCKS WILL GO DOWN

# In[ ]:


# IMPORTING LIBRARY FOR IMPORTING DATASET
import pandas as pd


# In[ ]:


# WE USE THIS ENCODING 'ISO-8859-1' TO READ SOME SPECIAL CHARACTERS IN OUR DATASET
df= pd.read_csv('../input/stock-sentiment-analysis/Stock_Dataa.csv',encoding='ISO-8859-1')


# In[ ]:


# LOOKING AT TOP 5 RECORDS OF DATASET
df.head()


# In[ ]:


# LOOKING AT LAST 5 RECORDS OF THE DATASET
df.tail()


# In[ ]:


# DIVIDING THE DATASET INTO TRAINING AND TEST SETS ACCORDING TO DATE
train= df[df['Date']< '20150101' ]
test=  df[df['Date']> '20141231']


# # Data Preprocessing Required for Text Data

# In[ ]:


# REMOVING PUNCTUATIONS
data= train.iloc[:,2:27]
data.replace("[^a-zA-Z]"," ",regex=True, inplace=True)

# RENAMING COLUMN NAME FOR EASE OF ACCESS
list1=[i for i in range(25)]
new_Index=[str(i) for i in list1]
data.columns= new_Index
data.head()


# In[ ]:


# CONVERTING THE HEADLINES INTO LOWER CASE
# SO THAT FOR EXAMPLE, united and United ARE TREATD EQUALLY
for index in new_Index:
    data[index]=data[index].str.lower()
data.head(1)


# In[ ]:


# COMBINING THE TOP 25 HEADLINES OF 1ST RECORD
' '.join(str(x) for x in data.iloc[1,0:25])


# In[ ]:


# COMBINING THE TOP 25 HEADLINES FOR EACH RECORD IN THE DATASET SO THAT WE COULD CONVER THEM INTO VECTORS
headlines=[]
for row in range(0,len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,0:25]))


# In[ ]:


# EXAMPLE- COMBINED ALL THE HEADLINES OF 0th RECORD
headlines[0] 


# # NATURAL LANGUAGE PROCESSING
# 
# WE WILL BE USING BAG OF WORDS MODEL & TF-IDF VECTORIZER FOR CONVERTING TEXT DATA INTO VECTORS
# 
# 
# FOR BOTH RANDOM FOREST CLASSIFIER & NAIVE BAYES CLASSIFIER

# In[ ]:


# Count vectorizer 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


## IMPLEMENTING BAG OF WORDS MODEL
countvector= CountVectorizer(ngram_range=(2,2))
traindataset= countvector.fit_transform(headlines) # CONVERTING ALL THE HEADLINES INTO VECTORS


# In[ ]:


# DATA CONVERTS INTO SPARSE MATRIX
traindataset[0]


# # RANDOM FOREST CLASSIFIER
# 
# FIRST WE WILL USE BAG OF WORDS MODEL WITH RANDOM FOREST CLASSIFIER

# In[ ]:


# Implement RandomForestClassifier on traindataset
random_classifier= RandomForestClassifier(n_estimators=200,criterion='entropy')
random_classifier.fit(traindataset,train['Label'])


# In[ ]:


## PREDICTING FOR TEST DATASET
# WE WILL BE PERFORMING SAME STEPS FOR TEST DATA ALSO.

test_transform=[]
for row in range(0,len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset= countvector.transform(test_transform)
predictions= random_classifier.predict(test_dataset)


# In[ ]:


# LOOKING AT OUR PREDICTIONS
predictions


# In[ ]:


# FOR CHECKING ACCURACY
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report


# In[ ]:


matrix= confusion_matrix(test["Label"],predictions)
print(matrix)
score= accuracy_score(test["Label"],predictions)
print(score)
report= classification_report(test['Label'],predictions)
print(report)


# **OVERALL ACCURACY= 83.86%**

# ## USING RANDOM FOREST CLASSIFIER WITH TF-IDF VECTORIZER

# In[ ]:


# USING RANDOM FOREST CLASSIFIER WITH TF-IDF VECTORIZER
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


## IMPLEMENTING TF-IDF VECTORIZER
tfidf= TfidfVectorizer(ngram_range=(2,2))
traindataset= tfidf.fit_transform(headlines) # CONVERTING ALL THE HEADLINES INTO VECTORS using TF-IDF technique


# In[ ]:


# Implement RandomForestClassifier on traindataset
random_classifier= RandomForestClassifier(n_estimators=200,criterion='entropy')
random_classifier.fit(traindataset,train['Label'])


# In[ ]:


## PREDICTING FOR TEST DATASET
# WE WILL BE PERFORMING SAME STEPS FOR TEST DATA ALSO.

test_transform=[]
for row in range(0,len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset= tfidf.transform(test_transform)
predictions= random_classifier.predict(test_dataset)


# In[ ]:


# ACCURACY AFTER USING TF-IDF VECTORIZER
matrix= confusion_matrix(test["Label"],predictions)
print(matrix)
score= accuracy_score(test["Label"],predictions)
print(score)
report= classification_report(test['Label'],predictions)
print(report)


# **OVERALL ACCURACY= 84.39%**

# # NAIVE BAYES CLASSIFIER
# 
# FIRST WE WILL USE 'BAG OF WORDS MODEL' TO CONVERT TEXT INTO VECTORS

# In[ ]:


from sklearn.naive_bayes import MultinomialNB
naive= MultinomialNB()


# In[ ]:


# WE WILL FIRST USE BAG OF WORDS MODEL FOR CONVERTING TEXT INTO VECTORS
## IMPLEMENTING BAG OF WORDS MODEL
countvector= CountVectorizer(ngram_range=(2,2))
traindataset= countvector.fit_transform(headlines) # CONVERTING ALL THE HEADLINES INTO VECTOR


# In[ ]:


# FITTING TRAIN DATA INTO  NAIVE BAYES CLASSIFIER 
naive.fit(traindataset,train['Label'])


# In[ ]:


## PREDICTING FOR TEST DATASET
# WE WILL BE PERFORMING SAME STEPS FOR TEST DATA ALSO.

test_transform=[]
for row in range(0,len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset= countvector.transform(test_transform)
predictions= naive.predict(test_dataset)


# In[ ]:


# LOOKING AT OUR PREDICTIONS
predictions


# In[ ]:


# FOR CHECKING ACCURACY
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report


# In[ ]:


matrix= confusion_matrix(test["Label"],predictions)
print(matrix)
score= accuracy_score(test["Label"],predictions)
print(score)
report= classification_report(test['Label'],predictions)
print(report)


# **OVERALL  ACCURACY= 84.65%**

# ## USING NAIVE BAYES CLASSIFIER WITH TF-IDF VECTORIZER

# In[ ]:


# NOW WE WILL USE TF-IDF VECTORIZER WITH NAIVE BAYES CLASSIFIER
traindataset= tfidf.fit_transform(headlines) # CONVERTING ALL THE HEADLINES INTO VECTORS using TF-IDF technique


# In[ ]:


naive.fit(traindataset,train['Label'])


# In[ ]:


## PREDICTING FOR TEST DATASET
# WE WILL BE PERFORMING SAME STEPS FOR TEST DATA ALSO.

test_transform=[]
for row in range(0,len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset= countvector.transform(test_transform)
predictions= naive.predict(test_dataset)


# In[ ]:


predictions


# In[ ]:


# ACCURACY AFTER USING TF-IDF VECTORIZER IN NAIVE BAYES CLASSIFIER
matrix= confusion_matrix(test["Label"],predictions)
print(matrix)
score= accuracy_score(test["Label"],predictions)
print(score)
report= classification_report(test['Label'],predictions)
print(report)


# **OVERALL ACCURACY= 85.185%**
# 
# 
# WE CAN SEE THAT NAIVE BAYES CLASSIFIER WITH TF-IDF VECTORIZER IS GIVING THE HIGHEST ACCURACY i.e 85.185%
