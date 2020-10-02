#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas_profiling
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df=pd.read_csv("../input/Womens-Clothing-E-Commerce-Reviews.csv")


# In[ ]:


df


# In[ ]:



    


# In[ ]:


df=pd.DataFrame(df)


# In[ ]:


df.info()


# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


df['Review Text']=df['Review Text'].astype(str)
df['Review Length']=df['Review Text'].apply(len)


# In[ ]:


g=sns.FacetGrid(data=df,col='Rating')
g.map(plt.hist,'Review Length',bins=50)


# In[ ]:


plt.figure(figsize=(10,10))
sns.boxplot(x='Rating', y='Review Length', data=df)


# In[ ]:


rating = df.groupby('Rating').mean()
rating.corr()


# In[ ]:


sns.heatmap(data=rating.corr(),annot=True)


# In[ ]:


df.head()


# In[ ]:


df.groupby(['Rating',pd.cut(df['Age'],np.arange(0,100,10))]).size().unstack(0).plot.bar(stacked=True)


# In[ ]:


plt.figure(figsize=(15,15))
df.groupby(['Department Name', pd.cut(df['Age'], np.arange(0,100,10))])       .size()       .unstack(0)       .plot.bar(stacked=True)


# In[ ]:


plt.figure(figsize=(15,15))
df.groupby(['Class Name', pd.cut(df['Age'], np.arange(0,100,10))])       .size()       .unstack(0)       .plot.bar(stacked=True)


# In[ ]:


z=df.groupby(by=['Department Name'],as_index=False).count().sort_values(by='Class Name',ascending=False)

plt.figure(figsize=(10,10))
sns.set_style("whitegrid")
ax=sns.barplot(x=z['Department Name'],y=z['Class Name'],data=z)
plt.xlabel("Department Name")
plt.ylabel=("Count")
plt.title("Counts vs Department Name")


# In[ ]:


z=df.groupby(by=['Division Name'],as_index=False).count().sort_values(by='Class Name',ascending=False)

plt.figure(figsize=(10,10))
sns.set_style("whitegrid")
ax=sns.barplot(x=z['Division Name'],y=z['Class Name'],data=z)
plt.xlabel("Division Name")
plt.ylabel=("Count")
plt.title("Counts vs Division Name")


# In[ ]:


from collections import Counter
from nltk.tokenize import RegexpTokenizer
import nltk
from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize
from wordcloud import WordCloud, STOPWORDS
import re


# In[ ]:


from nltk.corpus import stopwords
nltk.download('stopwords')
stopwords = stopwords.words('english')


# In[ ]:


top_N=100

a=df['Review Text'].str.lower().str.cat(sep=' ')
a=re.sub('[^A-Za-z]+', ' ', a)


# In[ ]:


word_tokens=word_tokenize(a)
filtr_sent=[w for w in word_tokens if w not in stopwords]
filtr_sent=[]
for w in word_tokens:
    if w not in stopwords:
        filtr_sent.append(w)


# In[ ]:


filtr_sent[:20]


# In[ ]:


without_sing_char=[word for word in filtr_sent if len(word)>2]


# In[ ]:


clean_data=[w for w in without_sing_char if not w.isnumeric()]


# In[ ]:


word_dist=nltk.FreqDist(clean_data)
rslt=pd.DataFrame(word_dist.most_common(top_N),columns=['word','Frequency'])


# In[ ]:


plt.figure(figsize=(10,10))
sns.set_style("whitegrid")
ax=sns.barplot(x="word",y="Frequency",data=rslt.head(7))


# In[ ]:


def wc(data,bgcolor,title):
    plt.figure(figsize=(100,100))
    wc=WordCloud(background_color=bgcolor,max_words=1000,max_font_size=50)
    wc.generate(' '.join(data))
    plt.imshow(wc)
    plt.axis('off')


# In[ ]:


wc(clean_data,'white','Most Used Words')


# In[ ]:


from textblob import TextBlob

bloblist=list()
df_review_str=df['Review Text'].astype(str)
for row in df_review_str:
    blob=TextBlob(row)
    bloblist.append((row,blob.sentiment.polarity,blob.sentiment.subjectivity))
    df_polarity=pd.DataFrame(bloblist,columns=['Review','Sentiment','polarity'])
    
def f(df_polarity):
    if df_polarity['Sentiment']>0:
        val="Positive Review"
    elif df_polarity['Sentiment']==0:
        val="Neutral Review"
    else:
        val="Negative Review"
    return val

df_polarity['Sentiment Type'] = df_polarity.apply(f,axis=1)


# In[ ]:


plt.figure(figsize=(10,10))
sns.set_style("whitegrid")
ax=sns.countplot(x="Sentiment Type",data=df_polarity)


# In[ ]:


positive_reviews=df_polarity[df_polarity['Sentiment Type']=='Positive Review']
negative_review=df_polarity[df_polarity['Sentiment Type']=='Negative Review']


# In[ ]:


wc(positive_reviews['Review'],'white','Most positive')


# In[ ]:


wc(negative_review['Review'],'black','Most Neegative')


# In[ ]:


df_polarity.head()


# In[ ]:


import string
def text_process(review):
    nopunc=[w for w in review if w not in string.punctuation]
    nopunc=''.join(nopunc)
    return [w for w in nopunc.split() if w.lower() not in 
           stopwords]


# In[ ]:


df['Review Text'].head(5).apply(text_process)


# In[ ]:


df=df.dropna(axis=0,how='any')
rating_class=df[(df['Rating']==1) | (df['Rating']==5)]
X_review=rating_class['Review Text']
y=rating_class['Rating']


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
bow_transformer=CountVectorizer(analyzer=text_process).fit(X_review)


# In[ ]:


print(len(bow_transformer.vocabulary_))


# In[ ]:


X_review = bow_transformer.transform(X_review)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_review, y, test_size=0.3, random_state=101)


# In[ ]:


from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train, y_train)


# In[ ]:


predict=nb.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, predict))
print('\n')
print(classification_report(y_test, predict))


# In[ ]:


rating_positive=df['Review Text'][4]
rating_positive


# In[ ]:


rating_positive_transformed = bow_transformer.transform([rating_positive])
nb.predict(rating_positive_transformed)[0]


# In[ ]:


rating_negative=df['Review Text'][61]
rating_negative


# In[ ]:


rating_positive_transformed = bow_transformer.transform([rating_negative])
nb.predict(rating_positive_transformed)[0]


# In[ ]:


x_recom=df['Review Text']
y_rec=df['Recommended IND']

bow_transformer=CountVectorizer(analyzer=text_process).fit(x_recom)

x_recom=bow_transformer.transform(x_recom)

X_train, X_test, y_train, y_test = train_test_split(x_recom, 
                        y_rec, test_size=0.3, random_state=101)

nb=MultinomialNB()
nb.fit(X_train, y_train)

predict_recom=nb.predict(X_test)

print(confusion_matrix(y_test, predict_recom))
print('\n')
print(classification_report(y_test, predict_recom))


# In[ ]:


rating_p_trans=bow_transformer.transform([rating_positive])
nb.predict(rating_p_trans)[0]


# In[ ]:


rating_negative=df['Review Text'][2]
rating_negative


# In[ ]:


rating_n_trans=bow_transformer.transform([rating_negative])
nb.predict(rating_n_trans)[0]


# In[ ]:


df.head()

