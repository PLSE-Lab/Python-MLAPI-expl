#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#This is my first participation in kaggle competitions
#I wish evaluation and support
#thank forll all 
#this solution Divided into two main parts first clean the data and some visualisation
#then using gaussian naive bayes classifier  
#with evaliting the train data with confusion_matrix and root mean square error 


# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer 
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from wordcloud import WordCloud,STOPWORDS
from numpy import nan
from bs4 import BeautifulSoup    
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from math import sqrt


# In[2]:


train_data = pd.read_csv('../input/train.csv')


# In[3]:


train_data.head()


# In[ ]:


#cleaning the data


# In[4]:


train_data.info()


# In[5]:


comment_text_train = []
for i in range(0,159571):
    review = re.sub('[^a-zA-Z]', ' ', train_data['comment_text'][i])
    review = review.lower()
    review = review.split()
    lemmatizer = WordNetLemmatizer()
    review = [lemmatizer.lemmatize(w) for w in review if not w in set(stopwords.words('english'))]
    review = ' '.join(review)
    comment_text_train.append(review)


# In[6]:


train_data['new_comment_text']=comment_text_train


# In[7]:


train_data.drop(['comment_text'],axis=1,inplace=True)


# In[8]:


train_data.head()


# In[9]:


train_data=train_data[['id','new_comment_text','toxic','severe_toxic','obscene','threat','insult','identity_hate']]


# In[ ]:


#here i divide the data for visualisation


# In[10]:


train_data.head()


# In[11]:


toxic=train_data[['new_comment_text','toxic']]


# In[12]:


toxic1=toxic[toxic['toxic']==(1)]


# In[13]:


severe_toxic=train_data[['new_comment_text','severe_toxic']]


# In[14]:


severe_toxic1=severe_toxic[severe_toxic['severe_toxic']==(1)]


# In[15]:


obscene=train_data[['new_comment_text','obscene']]


# In[16]:


obscene1=obscene[obscene['obscene']==(1)]


# In[17]:


threat=train_data[['new_comment_text','threat']]


# In[18]:


threat1=threat[threat['threat']==(1)]


# In[19]:


insult=train_data[['new_comment_text','insult']]


# In[20]:


insult1=insult[insult['insult']==(1)]


# In[21]:


identity_hate=train_data[['new_comment_text','identity_hate']]


# In[22]:


identity_hate1=toxic[identity_hate['identity_hate']==(1)]


# In[23]:


words = ' '.join(toxic1['new_comment_text'])
split_word = " ".join([word for word in words.split()])
wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color='black',
                      width=3000,
                      height=2500
                     ).generate(split_word)
plt.figure(1,figsize=(13, 13))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# In[24]:


words1 = ' '.join(severe_toxic1['new_comment_text'])
split_word1 = " ".join([word for word in words1.split()])
wordcloud1 = WordCloud(stopwords=STOPWORDS,
                      background_color='black',
                      width=3000,
                      height=2500
                     ).generate(split_word1)
plt.figure(1,figsize=(13, 13))
plt.imshow(wordcloud1)
plt.axis('off')
plt.show()


# In[26]:


words2 = ' '.join(obscene1['new_comment_text'])
split_word2 = " ".join([word for word in words2.split()])
wordcloud2 = WordCloud(stopwords=STOPWORDS,
                      background_color='black',
                      width=3000,
                      height=2500
                     ).generate(split_word2)
plt.figure(1,figsize=(13, 13))
plt.imshow(wordcloud2)
plt.axis('off')
plt.show()


# In[25]:


words3 = ' '.join(threat1['new_comment_text'])
split_word3 = " ".join([word for word in words3.split()])
wordcloud3 = WordCloud(stopwords=STOPWORDS,
                      background_color='black',
                      width=3000,
                      height=2500
                     ).generate(split_word3)
plt.figure(1,figsize=(13, 13))
plt.imshow(wordcloud3)
plt.axis('off')
plt.show()


# In[26]:


words4 = ' '.join(insult1['new_comment_text'])
split_word4 = " ".join([word for word in words4.split()])
wordcloud4 = WordCloud(stopwords=STOPWORDS,
                      background_color='black',
                      width=3000,
                      height=2500
                     ).generate(split_word4)
plt.figure(1,figsize=(13, 13))
plt.imshow(wordcloud4)
plt.axis('off')
plt.show()


# In[29]:


words5 = ' '.join(identity_hate1['new_comment_text'])
split_word5 = " ".join([word for word in words5.split()])
wordcloud5 = WordCloud(stopwords=STOPWORDS,
                      background_color='black',
                      width=3000,
                      height=2500
                     ).generate(split_word5)
plt.figure(1,figsize=(13, 13))
plt.imshow(wordcloud5)
plt.axis('off')
plt.show()


# In[ ]:


#prepare data for count most word in this data set 


# In[27]:


toxic_2=toxic1['new_comment_text']
severe_toxic2=severe_toxic1['new_comment_text']
obscene2=obscene1['new_comment_text']
threat2=threat1['new_comment_text']
insult2=insult1['new_comment_text']
identity_hate2=identity_hate1['new_comment_text']


# In[28]:


vectorizer1 = CountVectorizer(analyzer = "word",   
                             tokenizer = None,    
                             preprocessor = None, 
                             stop_words = None,   
                             max_features = 2000) 
vectorizer2 = CountVectorizer(analyzer = "word",   
                             tokenizer = None,    
                             preprocessor = None, 
                             stop_words = None,   
                             max_features = 2000) 
vectorizer3 = CountVectorizer(analyzer = "word",   
                             tokenizer = None,    
                             preprocessor = None, 
                             stop_words = None,   
                             max_features = 2000) 
vectorizer4 = CountVectorizer(analyzer = "word",   
                             tokenizer = None,    
                             preprocessor = None, 
                             stop_words = None,   
                             max_features = 2000) 
vectorizer5 = CountVectorizer(analyzer = "word",   
                             tokenizer = None,    
                             preprocessor = None, 
                             stop_words = None,   
                             max_features = 2000) 
vectorizer6 = CountVectorizer(analyzer = "word",   
                             tokenizer = None,    
                             preprocessor = None, 
                             stop_words = None,   
                             max_features = 2000) 


# In[29]:


toxic_feature = vectorizer1.fit_transform(toxic_2)
toxic_feature=toxic_feature.toarray()
toxic_feature.shape


# In[30]:


toxic_feature_vectorize= vectorizer1.get_feature_names()
toxic_feature_vectorize


# In[31]:


toxic_dist = np.sum(toxic_feature, axis=0)
for tag, count in zip(toxic_feature_vectorize, toxic_dist):
    print (tag,count)


# In[32]:


toxic_new= pd.DataFrame(toxic_dist)
toxic_new.columns=['word_count']
toxic_new['word'] = pd.Series(toxic_feature_vectorize, index=toxic_new.index)
toxic_new_1=toxic_new[['word','word_count']]
toxic_new_top_15=toxic_new_1.sort_values(['word_count'],ascending=[0])
toxic_new_top_15.head(15)


# In[33]:


severe_toxic_feature = vectorizer2.fit_transform(severe_toxic2)
severe_toxic_feature=severe_toxic_feature.toarray()
severe_toxic_feature.shape


# In[34]:


severe_feature_vectorize= vectorizer2.get_feature_names()
severe_feature_vectorize


# In[35]:


severe_dist = np.sum(severe_toxic_feature, axis=0)
for tag, count in zip(severe_feature_vectorize, severe_dist):
    print (tag,count)


# In[36]:


severe_new= pd.DataFrame(severe_dist)
severe_new.columns=['word_count']
severe_new['word'] = pd.Series(severe_feature_vectorize, index=severe_new.index)
severe_new_1=severe_new[['word','word_count']]
severe_new_top_15=severe_new_1.sort_values(['word_count'],ascending=[0])
severe_new_top_15.head(15)


# In[37]:


obscene_feature = vectorizer3.fit_transform(obscene2)
obscene_feature=obscene_feature.toarray()
obscene_feature.shape


# In[38]:


obscene_feature_vectorize= vectorizer3.get_feature_names()
obscene_feature_vectorize


# In[39]:


obscene_dist = np.sum(obscene_feature, axis=0)
for tag, count in zip(obscene_feature_vectorize,obscene_dist):
    print (tag,count)


# In[40]:


obscene_new= pd.DataFrame(obscene_dist)
obscene_new.columns=['word_count']
obscene_new['word'] = pd.Series(obscene_feature_vectorize, index=obscene_new.index)
obscene_new_1=obscene_new[['word','word_count']]
obscene_new_top_15=obscene_new_1.sort_values(['word_count'],ascending=[0])
obscene_new_top_15.head(15)


# In[41]:


threat_feature = vectorizer4.fit_transform(threat2)
threat_feature=threat_feature.toarray()
threat_feature.shape


# In[42]:


threat_feature_vectorize= vectorizer4.get_feature_names()
threat_feature_vectorize


# In[43]:


threat_dist = np.sum(threat_feature, axis=0)
for tag, count in zip(threat_feature_vectorize,threat_dist):
    print (tag,count)


# In[44]:


threat_new= pd.DataFrame(threat_dist)
threat_new.columns=['word_count']
threat_new['word'] = pd.Series(threat_feature_vectorize, index=threat_new.index)
threat_new_1=threat_new[['word','word_count']]
threat_new_top_15=threat_new_1.sort_values(['word_count'],ascending=[0])
threat_new_top_15.head(15)


# In[45]:


insult_feature = vectorizer5.fit_transform(insult2)
insult_feature=insult_feature.toarray()
insult_feature.shape


# In[46]:


insult_feature_vectorize= vectorizer5.get_feature_names()
insult_feature_vectorize


# In[47]:


insult_dist = np.sum(insult_feature, axis=0)
for tag, count in zip(insult_feature_vectorize,insult_dist):
    print (tag,count)


# In[48]:


insult_new= pd.DataFrame(insult_dist)
insult_new.columns=['word_count']
insult_new['word'] = pd.Series(insult_feature_vectorize, index=insult_new.index)
insult_new_1=insult_new[['word','word_count']]
insult_new_top_15=insult_new_1.sort_values(['word_count'],ascending=[0])
insult_new_top_15.head(15)


# In[49]:


identity_hate_feature = vectorizer6.fit_transform(identity_hate2)
identity_hate_feature=identity_hate_feature.toarray()
identity_hate_feature.shape


# In[50]:


identity_hate_feature_vectorize= vectorizer6.get_feature_names()
identity_hate_feature_vectorize


# In[51]:


identity_hate_dist = np.sum(identity_hate_feature, axis=0)
for tag, count in zip(identity_hate_feature_vectorize,identity_hate_dist):
    print (tag,count)


# In[52]:


identity_hate_new= pd.DataFrame(identity_hate_dist)
identity_hate_new.columns=['word_count']
identity_hate_new['word'] = pd.Series(identity_hate_feature_vectorize, index=identity_hate_new.index)
identity_hate_new_1=identity_hate_new[['word','word_count']]
identity_hate_new_top_15=identity_hate_new_1.sort_values(['word_count'],ascending=[0])
identity_hate_new_top_15.head(15)


# In[ ]:


#time for predction using gaussian naive bayes classifier  
#and  evaliting the train data with confusion_matrix and root mean square error 


# In[53]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
x__train = cv.fit_transform(comment_text_train).toarray()
#x__test= cv.fit_transform(comment_text_test).toarray()


# In[54]:


y1 = train_data.iloc[:, 2].values
y2 = train_data.iloc[:, 3].values
y3 = train_data.iloc[:, 4].values
y4 = train_data.iloc[:, 5].values
y5 = train_data.iloc[:, 6].values
y6 = train_data.iloc[:, 7].values


# In[55]:


from sklearn.cross_validation import train_test_split
X1_train, X1_test, y1_train, y1_test = train_test_split(x__train, y1, test_size = 0.40, random_state = 0)
X2_train, X2_test, y2_train, y2_test = train_test_split(x__train, y2, test_size = 0.40, random_state = 0)
X3_train, X3_test, y3_train, y3_test = train_test_split(x__train, y3, test_size = 0.40, random_state = 0)
X4_train, X4_test, y4_train, y4_test = train_test_split(x__train, y4, test_size = 0.40, random_state = 0)
X5_train, X5_test, y5_train, y5_test = train_test_split(x__train, y5, test_size = 0.40, random_state = 0)
X6_train, X6_test, y6_train, y6_test = train_test_split(x__train, y6, test_size = 0.40, random_state = 0)


# In[56]:



from sklearn.naive_bayes import GaussianNB
classifier1 = GaussianNB()
classifier2 = GaussianNB()
classifier3 = GaussianNB()
classifier4 = GaussianNB()
classifier5 = GaussianNB()
classifier6 = GaussianNB()

classifier1.fit(X1_train, y1_train)
classifier2.fit(X2_train, y2_train)
classifier3.fit(X3_train, y3_train)
classifier4.fit(X4_train, y4_train)
classifier5.fit(X5_train, y5_train)
classifier6.fit(X6_train, y6_train)

y1_pred = classifier1.predict(X1_test)
y2_pred = classifier2.predict(X2_test)
y3_pred = classifier3.predict(X3_test)
y4_pred = classifier4.predict(X4_test)
y5_pred = classifier5.predict(X5_test)
y6_pred = classifier6.predict(X6_test)


# In[57]:


from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y1_test, y1_pred)
cm2 = confusion_matrix(y2_test, y2_pred)
cm3 = confusion_matrix(y3_test, y3_pred)
cm4 = confusion_matrix(y4_test, y4_pred)
cm5 = confusion_matrix(y5_test, y5_pred)
cm6 = confusion_matrix(y6_test, y6_pred)


# In[58]:


print ("confusion_matrix of toxic is\n:" ,cm1)
print ("confusion_matrix of severe_toxic is\n:" ,cm2)
print ("confusion_matrix of obscene is \n:" ,cm3)
print ("confusion_matrix threat is  \n:" ,cm4)
print ("confusion_matrix of insult is \n:" ,cm5)
print ("confusion_matrix dentity_hate is  \n:" ,cm6)


# In[59]:


mse1 = ((y1_pred - y1_test) ** 2).mean()
mse2 = ((y2_pred - y2_test) ** 2).mean()
mse3 = ((y3_pred - y3_test) ** 2).mean()
mse4 = ((y4_pred - y4_test) ** 2).mean()
mse5 = ((y5_pred - y5_test) ** 2).mean()
mse6 = ((y6_pred - y6_test) ** 2).mean()


# In[60]:


print("toxic mean square error\n",mse1)
print("severe_toxic mean square error \n",mse2)
print("obscene  mean square error\n",mse3)
print("threat mean square error \n",mse4)
print("insult mean square error \n",mse5)
print("dentity_hate mean square error\n",mse6)


# In[61]:


rmse1 = sqrt(mse1)
rmse2 = sqrt(mse2)
rmse3 = sqrt(mse3)
rmse4 = sqrt(mse4)
rmse5 = sqrt(mse5)
rmse6 = sqrt(mse6)


# In[62]:


print("toxic  root mean square error \n",rmse1)
print("severe_toxic  root mean square error \n",rmse2)
print("obscene root  mean square error\n",rmse3)
print("threat root  mean square error\n",rmse4)
print("insult root  mean square error \n",rmse5)
print("dentity_hate  root mean square error \n",rmse6)


# In[ ]:


#last i will upload the test set with the results of my prediction
#thank you all
#resouresecs i use ths https://www.kaggle.com/c/word2vec-nlp-tutorial in kaggle for preapre and clean my data 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




