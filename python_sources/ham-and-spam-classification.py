#!/usr/bin/env python
# coding: utf-8

# # First import of required lybraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings('ignore')


# Let's read the data from csv file

# In[ ]:


data= pd.read_csv('../input/sms-spam-collection-dataset/spam.csv',delimiter=',',encoding='latin-1')


# In[ ]:


data.head() #First five row


# In[ ]:


data.shape


# In[ ]:


data.isna().sum()


# As the preview of data shows there are three useless columns, we need to delete these columns and Rename the columns v1 and v2 to 'Labels' and 'Message'

# In[ ]:


data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1,inplace=True)
data = data.rename(columns={'v1':'Labels','v2':'Message'}) 


# In[ ]:


data.info()


# Let's calculate the percentage of ham and spam messages

# In[ ]:


sns.countplot(data.Labels)
plt.title('No. of ham and spam messages')
print('{:0.2f}% of the ham massages'.format(100*(data.Labels.value_counts()[0])/len(data)))
print('{:0.2f}% of the spam massages'.format(100*(data.Labels.value_counts()[1])/len(data)))


# 86% of data consists of ham message

# In[ ]:


pd.set_option('display.max_colwidth',2000)


# In[ ]:


data['message_len']=data['Message'].apply(len)


# In[ ]:


data.describe()


# In[ ]:


data.loc[data['message_len'].max()][1]


# Now define our text precessing function,comman preprocessing includes removing punctuation and stopwords (i.e. "and" "or" these words are not giving useful meaning). Also the characters are changed to lower case 

# In[ ]:


data['Text'] = data['Message'].map(lambda word :''.join([w for w in word if not w in string.punctuation]))
data['Text'] = data['Text'].map(lambda text : word_tokenize(text.lower()))
stopword = set(stopwords.words('english'))
data['Text'] = data['Text'].map(lambda token : [w for w in token if not w in stopword])


# In[ ]:


data.head()


# After all that stems each word.Stemming is the process of reducing inflected words to their word stem, base or root form generally a written word form. Here, i am using 'SnowballStemmer'(this means that it replaces a word with the root of that word, for example "tasted" or "tasting" would become "taste").

# In[ ]:


from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer("english")
data['Text'] = data['Text'].map(lambda text : [stemmer.stem(w)for w in text])


# In[ ]:


data['Text'] = data['Text'].map(lambda text : ' '.join(text))


# In[ ]:


data.head()


# Lets convert our clean text into a representation that a machine learning model can understand. I'll use the Tfifd for this

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
x = data['Text']
y = data['Labels']


# In[ ]:


tfidf = TfidfVectorizer()
tfidf_dtm = tfidf.fit_transform(x)
tfidf_data=pd.DataFrame(tfidf_dtm.toarray())
tfidf_data.head()


# Let's split our features into train and test test

# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(tfidf_data,y,test_size=0.2)


# Import machine model and make functions to fit our classifiers and make predictions

# In[ ]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
nb = MultinomialNB(alpha=0.2)
nb.fit(x_train,y_train)
pred = nb.predict(x_test)
accuracy_score(pred,y_test)


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
lr = LogisticRegression()
lr.fit(x_train,y_train)
pred = lr.predict(x_test)
accuracy_score(pred,y_test)


# Naive bayes gives the best result.

# 
