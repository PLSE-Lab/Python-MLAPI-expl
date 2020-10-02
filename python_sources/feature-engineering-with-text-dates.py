#!/usr/bin/env python
# coding: utf-8

# **Feature Engineering with Text & Dates**
# * Tokenization, stemming, and normalizing of text
# * TFIDF Vectorization
# * Feature engineering (length of SMS text)
# * Training classification model using Random Forest
# * Evaluating and Compare Performance of the trained classification model

# In[ ]:


# Import the required libraries 
import scipy as sp
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
import os
print(os.listdir("../input"))


# In[ ]:


# Load CSV data
spam_df = pd.read_csv('../input/sms-spam-collection-dataset/spam.csv', encoding='latin-1')
spam_df.head()


# In[ ]:


# Only get the label and text feature
spam_df = spam_df[['v1','v2']]
spam_df.columns = ['label','text']

# Convert spam/ham label into binary output
class_labels = ["ham",'spam']
spam_df['label'] = spam_df['label'].apply(class_labels.index)

spam_df.head()


# In[ ]:


# Check class distrubtion of target - About 14% spam and 86% ham
spam_df.label.value_counts() 


# In[ ]:


# Plot the class distribution
spam_df["label"].value_counts().plot(kind = 'pie', explode = [0, 0.1], figsize = (6, 6), shadow = True)
plt.ylabel("Spam vs Ham")
plt.legend(["Ham", "Spam"])
plt.show()


# **Train/Test Split**

# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(spam_df,spam_df['label'],test_size=0.20, random_state=123, stratify=spam_df['label'])


# **Pre-Processing Pipeline**
# * Tokenization
# * Lower casing
# * Stopword removal
# * Stemming
# * Transformmation (TFIDFVectorizer)

# **Tokenization**

# In[ ]:


# Examine the text reviews
text1 = x_train['text'][100]
text1


# In[ ]:


from nltk.tokenize import word_tokenize
tokens = word_tokenize(text1)
print(tokens)


# **Text Pre-Processing Pipeline**
# 

# In[ ]:


# Load Stemming Library (PorterStemmer)
from nltk.stem.porter import PorterStemmer
# Load TFIDFVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# **1. Stemming Words**

# In[ ]:


porter = PorterStemmer()
x_train['text'] = x_train['text'].apply(porter.stem)


# **2. Stopword Removal & Lowercasing**

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
# Initialize TFIDFVectorizer with stop word removal and lowercasing of text
cv = TfidfVectorizer(stop_words='english', lowercase=True)


# **3. Apply TFIDF**

# In[ ]:


x_train_CV = cv.fit_transform(x_train['text'])


# **Text Pre-Process Pipeline**

# In[ ]:


# A function that cleans and performs a TFIDF transformation to our text data
tfidf = TfidfVectorizer(stop_words='english', lowercase=True)
def tfidf_pipeline(txt, flag):
    if flag == "train":
        txt = txt.apply(porter.stem) # Apply Stemming on train set
        x = tfidf.fit_transform(txt) # Apply Vectorizer, Stopword Removal, & Lowercasing on train set
    else:
        txt = txt.apply(porter.stem) # Apply Stemming on test set
        x = tfidf.transform(txt) # Apply Vectorizer, Stopword Removal, & Lowercasing on test set
    return x 


# In[ ]:


x_train_TFIDF = tfidf_pipeline(x_train['text'], flag="train")
x_test_TFIDF = tfidf_pipeline(x_test['text'], flag="test")


# In[ ]:


#original vs preprocessed data
original = x_train.shape
preprocessed = x_train_TFIDF.shape
print ("Our original training set shape: " + str(original))
print ("Our preprocessed training set shape: " + str(preprocessed))


# **Modeling**

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score


# **Logistic Regression Model**

# In[ ]:


rf = RandomForestClassifier(n_estimators=100, class_weight='balanced')
rf_model = rf.fit(x_train_TFIDF,y_train)


# **Cross Validiation**
# 
# Apply 10-Fold Cross Validation

# In[ ]:


rf_result = cross_val_score(rf_model, x_train_TFIDF, y_train, cv=5, scoring='accuracy')
rf_result.mean()


# **Evaluate**

# In[ ]:


y_pred = rf_model.predict(x_test_TFIDF)
precision_1, recall_1, fscore_1, support_1 = score(y_test, y_pred, pos_label=1, average ='binary')
print('Classification Report (With Only Text) | Precision : {} / Recall : {} / fscore : {} / Accuracy: {}'.format(round(precision_1,3),round(recall_1,3),round(fscore_1,3),round((y_pred==y_test).sum()/len(y_test),3)))


# **Confusion Matrix**

# In[ ]:


confusion_matrix(y_test, y_pred)


# **Feature Engineering**

# In[ ]:


# Import additional libraries
from scipy.sparse import csr_matrix, hstack
# Examine New Dataset with 'length' feature
# Add a column for the lenght of the SMS text
spam_df['length'] = spam_df['text'].str.len()
spam_df.head()


# In[ ]:


# Plot the distribution of text length (spam vs ham)
plt.figure(figsize=(15,6))
plt.hist(spam_df[spam_df['label']==1]['length'],bins = np.linspace(0,200,num=40),alpha=0.4,label='spam',normed=True)
plt.hist(spam_df[spam_df['label']==0]['length'],bins = np.linspace(0,200,num=40),alpha =0.4,label ='ham', normed=True)
plt.legend(loc ='upper left')
plt.title('Length Distribution of Spam VS Ham')
plt.show()


# **Train and Test Split of the new dataframe**
# 

# In[ ]:


x_train2, x_test2, y_train2, y_test2 = train_test_split(spam_df,spam_df['label'],test_size=0.20, random_state=123, stratify=spam_df['label'])


# **Transform and Merge into one matrix**
# 

# In[ ]:


# Get TFIDF Transformed Matrix on 'text'
x_train_TFIDF = tfidf_pipeline(x_train2['text'], flag="train")
x_test_TFIDF = tfidf_pipeline(x_test2['text'], flag="test")

# Convert length column into csr_matrix
x_train_len = csr_matrix(x_train2[['length']])
x_test_len =  csr_matrix(x_test2[['length']])

# Merge matrices together
x_train_merge = hstack((x_train_TFIDF, x_train_len)).tocsr()
x_test_merge = hstack((x_test_TFIDF, x_test_len)).tocsr()


# ** Random Forest model to the new data**
# 

# In[ ]:


rf_model2 = rf.fit(x_train_merge,y_train2)


# **10-Fold Cross Validation**

# In[ ]:


# Apply 10-Fold Cross Validation and Examine Accuracy Score
rf_result2 = cross_val_score(rf_model2, x_train_merge, y_train2, cv=10, scoring='accuracy')
rf_result2.mean()


# In[ ]:


y_pred2 = rf_model2.predict(x_test_merge)
precision_2, recall_2, fscore_2, support_2 = score(y_test2, y_pred2, pos_label=1, average ='binary')


# **Classification Report**

# In[ ]:


print(' Classification Report (With Text) \n Precision : {} / Recall : {} / fscore : {} / Accuracy: {}'.format(round(precision_1,3),round(recall_1,3),round(fscore_1,3),round((y_pred==y_test).sum()/len(y_test),3)))
print('\n Classification Report (With Text & Length)\n Precision : {} / Recall : {} / fscore : {} / Accuracy: {}'.format(round(precision_2,3),round(recall_2,3),round(fscore_2,3),round((y_pred2==y_test2).sum()/len(y_test2),3)))


# **Confusion Matrix**

# In[ ]:


print('Confusion Matrix (With Text)')
print(confusion_matrix(y_test, y_pred))
print('Confusion Matrix (With Text & Length)')
print(confusion_matrix(y_test2, y_pred2))
print("Made 6 less false negative errors")


# **Examine misclassified SMS Text**

# In[ ]:


# Examine misclassified SMS text 
missclassified = np.nonzero(y_pred2!=y_test2)[0]
ind_miss = y_test2.index[missclassified]
spam_df.iloc[ind_miss,:]


# **Feature Engineering with Dates**

# In[ ]:


alexa_df = pd.read_csv('../input/amazon-alexa-reviews/amazon_alexa.tsv', sep = '\t')
alexa_df = alexa_df[['date']]
alexa_df.head()


# **Extracting Date, Month, Year, and Quarter**

# In[ ]:


# Convert the column into a date time object
alexa_df['date'] = pd.to_datetime(alexa_df['date'])

# Create a new 'year' feature by extracting the year value from date
alexa_df['year'] = alexa_df.date.dt.year

# Create a new 'month' feature by extracting the month value from date
alexa_df['month'] = alexa_df.date.dt.month

# Create a new 'day' feature by extracting the day value from date
alexa_df['day'] = alexa_df.date.dt.day

# Create a new 'Qtr' feature by extracting the monthly quarter from date
alexa_df['Qtr'] = alexa_df.date.dt.quarter


# In[ ]:


# Examine new data frame
alexa_df.head()


# **Extracting Holidays**

# In[ ]:


# Create holiday dates
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
cal = calendar()
holidays = cal.holidays()

# Examine holiday dates
cal.holidays()


# In[ ]:


# Create New Column 'Holiday'
alexa_df['Holiday'] = alexa_df['date'].isin(holidays)
alexa_df.tail()


# **Calculating Days Before Two Given Dates**

# In[ ]:


def calculate_xmas(date):
    xmas = pd.to_datetime(pd.Series('2018-12-25'))
    diff = xmas - date
    return diff

alexa_df['days_before_xmas'] = alexa_df.date.apply(calculate_xmas)
alexa_df.head()

