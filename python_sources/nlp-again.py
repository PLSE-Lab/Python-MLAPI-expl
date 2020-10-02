#!/usr/bin/env python
# coding: utf-8

# In[55]:


corpus = [
    "This is my first corpus",
    "Processing it for ML",
    "Doing ML is awesome",
    "This is fun to look at",
    "ML is life, ML is interest"
]
corpus


# In[56]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()


# In[57]:


X = cv.fit_transform(corpus)
X


# In[58]:


X.toarray()


# In[59]:


cv.get_feature_names()


# In[60]:


cv.vocabulary_


# In[61]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[62]:


df = pd.DataFrame({'Text':['This is my first text! on today', 'I am working oon text data:)', 'I am creating, text data frame'],'Label':[1,1,0]})
df


# In[63]:


df['Text'].str.get_dummies(' ')


# In[64]:


a = cv.fit_transform(df['Text']).toarray()


# In[65]:


b = cv.get_feature_names()


# In[66]:


pd.DataFrame(a,columns=b)


# In[67]:


df['Text'] = df['Text'].str.lower().str.replace('[^a-z]',' ').str.split()
df


# In[68]:


import nltk
from nltk.corpus import stopwords


# In[69]:


df['Text'] = df['Text'].apply(lambda x: [word for word in x if word not in set(stopwords.words('english'))])
df


# In[70]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
ps


# In[71]:


df['Text'] = df['Text'].apply(lambda x: ' '.join([ps.stem(word) for word in x]))
df


# In[72]:


pd.DataFrame(cv.fit_transform(df['Text']).toarray(),columns=cv.get_feature_names())


# In[73]:


spam_df = pd.read_csv('spam.csv',encoding='ISO-8859-1',engine='c')
spam_df.head()


# In[74]:


spam_df = spam_df.loc[:,['v1','v2']]
spam_df.head()


# In[75]:


spam_df.rename(columns={'v1':'Target','v2':'Text'},inplace=True)


# In[76]:


spam_df.head()


# In[77]:


from wordcloud import WordCloud


# In[78]:


spam_list = spam_df[spam_df['Target']=='spam']['Text'].unique().tolist()
spam_list[:2]


# In[79]:


spam = ' '.join(spam_list)
spam[:100]


# In[80]:


spam_wc = WordCloud().generate(spam)
spam_wc


# In[81]:


plt.figure()
plt.imshow(spam_wc)
plt.show()


# In[82]:


from nltk.stem.porter import PorterStemmer


# In[83]:


import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


ps = PorterStemmer()

spam_df['Text'] = spam_df['Text'].str.lower().str.replace('[^a-z]', ' ').str.split()
spam_df['Text']


# In[84]:


spam_df['Text'] = spam_df['Text'].apply(lambda x: ' '.join([ps.stem(word) for word in x if word not in set(stopwords.words('english'))]))
spam_df.head()


# In[90]:


X = cv.fit_transform(spam_df.Text).toarray()


# In[93]:


X.shape


# In[94]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[95]:


y= le.fit_transform(spam_df.Target)
y[:5]


# In[96]:


y.shape


# In[97]:


le.classes_


# In[98]:


from sklearn.naive_bayes import MultinomialNB


# In[100]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = .25, random_state = 0)


# In[108]:


clf = MultinomialNB()
clf


# In[109]:


clf.fit(X_train,y_train)


# In[110]:


y_pred = clf.predict(X_test)
y_pred


# In[111]:


from sklearn.metrics import accuracy_score


# In[112]:


accuracy_score(y_test,y_pred)


# In[116]:


test_data = spam_df.head(10).Text.tolist()+spam_df.tail(10).Text.tolist()
pred_data = spam_df.head(10).Target.tolist()+spam_df.tail(10).Target.tolist()


# In[118]:


test_pred = clf.predict(cv.transform(test_data))
test_pred


# In[120]:


i = 0
for sms, label in zip(test_data,pred_data):
    print(str(test_data[i][:50])+"("+str(pred_data[i])+")=>"+str(test_pred[i]))
    i+=1


# In[ ]:




