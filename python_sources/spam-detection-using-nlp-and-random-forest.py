#!/usr/bin/env python
# coding: utf-8

# # Spam Detection using NLP and Random Forest

# This Project focusses on the following areas : 
# 
# 1. Data Cleaning Pipeline : 
#     - Removing Punctuation
#     - Tokenizing
#     - Removing Stop Words
# 2. Applying NLYKs - TF-IDF on the Cleaned Dataset
# 3. Feature Engineering
#     - Length of Text Messages
#     - Percent of punctuation used in each corpus
#     - Number finding in each corpus
#     - Web links in each corpus
# 4. Understanding the Features
# 5. Making a new Dataframe having all the required features for prediction
#    and have the TF-IDF data
# 6. Splitting the Dataset into Test and Train to build and apply the model
# 7. Building model on Train set using Random Forest
# 8. Understanding the relative importance of features used
# 9. Applying the model on Test set
# 10. Tweaking hyperparameters using Grid Search
# 

# 

# # Importing NLP's nltk library
# # Downloading Stopwords to remove from the dataset later

# In[ ]:


import nltk
#nltk.download()
from nltk.corpus import stopwords


# # 1. Data Cleaning Pipeline

# In[ ]:


import pandas as pd
add = "../input/spam.csv"
data = pd.read_csv(add, encoding='latin-1')
data.head(5)
data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
pd.set_option('display.max_colwidth', 0)
data.columns = ['label','text']

data.head(5)

#print(data.head(5))


# ### Applying the Data Cleaning Pipeline  - followed by TF_IDF

# The stopwords can be in many languages, for this particular dataset we'll use the English Language words

# In[ ]:


import string
import re
stopword = nltk.corpus.stopwords.words('english')


# We'll apply Stemming so as to remove any redundancy among words, i.e words which have the same root eg. running and run etc would be converted into run using stemming. 
# 
# NOTE :  For the same task,another technique named lemmatizing can also be used, which also considers the grammatical sense of the word in the context in which it has been used.
# 
# As we're using stemming here, we'll use the portstemmer as below

# In[ ]:


ps = nltk.PorterStemmer()


# We'll Remove  all the Punctuation, StopWords from the dataset and finally apply stemming on the resulting dataset to make clean, tidy and ready for processing

# In[ ]:


def clean_text(text):
    remove_punct = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+',remove_punct)
    noStop = ([ps.stem(word) for word in tokens if word not in stopword])
    return noStop


# # 2. Applying NLYKs - TF-IDF on the Cleaned Dataset

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_Vector = TfidfVectorizer(analyzer= clean_text)
Xtfidf_Vector = tfidf_Vector.fit_transform(data['text'])


# # 3. Feature Engineering

# #### Feature for Length of Text messages & Punctuation Percentage

# In[ ]:


import string


# In[ ]:


def punct_percent(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/ (len(text) - text.count(" ")),3)*100
data['punct_%'] = data['text'].apply(lambda x: punct_percent(x))
data['length'] = data['text'].apply(lambda x: len(x) - x.count(" "))


# In[ ]:


pd.set_option('display.max_colwidth', 0)

print(data.head(5))


# #### Features for Rows having numbers with number.length > =5

# In[ ]:


import re

#def find_num(text):
#    return re.findall('\d{7,}',text)

data['number'] = pd.DataFrame(data['text'].apply(lambda x: len(re.findall('\d{5,}',x))))


# In[ ]:


data.head(5)


# #### Feature for Currency

# In[ ]:



#def get_currency_symbol(text):
#    pattern = r'(\D*)\d*\.?\d*(\D*)'
#    result = re.match(pattern,text).group()
#    return result
#data['currency']= pd.DataFrame(data['text'].apply(lambda x: len(get_currency_symbol(x))))


# In[ ]:


#print(data.head(5))


# #### Feature for web address

# In[ ]:


def web_address(t):
    if(len(re.findall('www|http|https|.co',t)) > 0):
        return 1
    else:
        return 0
    
data['url'] = pd.DataFrame(data['text'].apply(lambda x: web_address(x)))
print(data.head(5))   


# # 4. Understand the Feartures

# In[ ]:


import numpy as np
from matplotlib import pyplot
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


bins = np.linspace(0,200,40)
pyplot.hist(data[data['label'] == 'spam']['length'],bins,alpha = 0.5,normed = True,label = 'spam')
pyplot.hist(data[data['label'] == 'ham']['length'],bins,alpha = 0.5,normed = True, label = 'ham')
pyplot.legend(loc = 'upper right')
pyplot.figure(figsize = (1000,400), dpi = 1000)
pyplot.show()


# In[ ]:


bins = np.linspace(0,50,40)
pyplot.hist(data[data['label'] == 'spam']['punct_%'], bins, alpha = 0.5,normed = True, label = 'spam')
pyplot.hist(data[data['label'] == 'ham']['punct_%'], bins, alpha = 0.5,normed = True, label = 'ham')
pyplot.legend(loc = 'upper right')
pyplot.show()


# In[ ]:


bins = np.linspace(0,5,10)
pyplot.hist(data[data['label'] == 'spam']['number'], bins,alpha = 0.5, label = 'spam')
pyplot.hist(data[data['label'] == 'ham']['number'], bins, alpha = 0.5, label = 'ham')
pyplot.legend(loc = 'upper right')
pyplot.show()


# In[ ]:


bins = np.linspace(0,5,100)
pyplot.hist(data[data['label'] == 'spam']['url'], bins,alpha = 0.5, label = 'spam')
pyplot.hist(data[data['label'] == 'ham']['url'], bins, alpha = 0.5, label = 'ham')
pyplot.legend(loc = 'upper right')
pyplot.show()


# #### Drawing pyplot for Datawith url as feature for data

# In[ ]:


data['url'].value_counts()


# ####  0 values : 78.4%
# #### 1 valyes  : 21.6%

# # 5. Making new Dataframe with all features for prediction and TF-IDF data

# In[ ]:


Xfeatures_data = pd.concat([data['length'],data['punct_%'],data['number'],data['url'], pd.DataFrame(Xtfidf_Vector.toarray())], axis = 1)
Xfeatures_data.head(5)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
print(dir(RandomForestClassifier))
print(RandomForestClassifier())


# # 6. Splitting the Dataset into Test and Train to build and apply the model
# 

# In[ ]:


from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(Xfeatures_data, data['label'], test_size = 0.2)


# # 7. Building the model on Train Data
# # 8. Understanding the relative importance of features added

# In[ ]:


rf = RandomForestClassifier(n_estimators= 50, max_depth= 20, n_jobs = -1)
rf_model = rf.fit(X_train,y_train)
sorted(zip(rf.feature_importances_,X_train.columns), reverse= True)[0:10]


# This shows that our original assumption about < numbers, length, url > in a row being good predictors of ham/spam class was correct

# # 9. Applying the model on Test set

# In[ ]:


y_pred = rf_model.predict(X_test)
precision,recall,fscore,support = score(y_test,y_pred,pos_label = 'spam', average = 'binary')
print('Precision: {}/ Recall: {}/ Accuracy: {}'.format(round(precision,3), round(recall,3), (y_pred == y_test).sum()/len(y_pred)))


# # 10. Tweaking hyperparameters using Grid Search

# In[ ]:


def train_rf(n_est, depth):
    rf = RandomForestClassifier(n_estimators= n_est, max_depth= depth, n_jobs = -1)
    rf_model = rf.fit(X_train,y_train)
    y_pred = rf_model.predict(X_test)
    precision,recall,fscore,support = score(y_test,y_pred,pos_label= 'spam',average= 'binary')
    print('Est: {}/ Depth: {}/ Precision: {}/ Recall: {}/ Accuracy : {}'.format(n_est,depth, round(precision,3), round(recall,3), (y_pred == y_test).sum()/len(y_pred)))


# In[ ]:


for n_est in [10,30,50,70]:
    for depth in [20,40,60,80, None]:
        train_rf(n_est,depth)


# # Conclusion :
# 
# 
# 
# So our best results are :  Est: 70/ Depth: None/ Precision: 1.0/ Recall: 0.908/ Accuracy : 0.9883303411131059

# In[ ]:




