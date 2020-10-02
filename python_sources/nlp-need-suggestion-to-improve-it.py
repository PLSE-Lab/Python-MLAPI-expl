#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS


# In[ ]:


import re 
import nltk
nltk.download('stopwords')


# In[ ]:


from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB


# In[ ]:


sw=stopwords.words('english')
np.array(sw)


# In[ ]:


len(sw)


# In[ ]:


dataset=pd.read_csv('../input/Restaurant_Reviews.tsv', delimiter='\t' )


# In[ ]:


dataset.head(10)


# In[ ]:


dataset.Liked.value_counts()


# In[ ]:


senntiment_words=[]
for row in dataset['Liked']:
    if row==0:
        senntiment_words.append('Negative')
    else:
        senntiment_words.append('Positive')
dataset['senntiment_word']=senntiment_words


# In[ ]:


word_count=pd.value_counts(dataset['senntiment_word'].values, sort=False)
word_count


# In[ ]:


index=[1,2]
plt.figure(figsize=(15,6))
plt.bar(index, word_count, color='c')
plt.xticks(index, ['negative', 'positive'], rotation=45)
plt.xlabel("word")
plt.ylabel("Word counts")
plt.title("count of Moods")
plt.bar(index, word_count)
for a, b in zip(index, word_count):
    plt.text(a, b , str(b), color='red', fontweight='bold')
plt.show()


# In[ ]:


def review_to_words(raw_review):
    review=raw_review
    review=re.sub('[^a-zA-Z]', ' ', review)
    review=review.lower()
    review=review.split()
    lem=WordNetLemmatizer()
    review=[lem.lemmatize(w) for w in review if not w in set(stopwords.words('english'))]
    return ''.join(review)


# In[ ]:


dataset.shape


# In[ ]:


corpus=[]
for i in range(0, 1000):
    corpus.append(review_to_words(dataset['Review'][i]))


# In[ ]:


corpus1=[]
for i in range(0, 1000):
    corpus1.append(review_to_words(dataset['Review'][i]))


# In[ ]:


dataset['new_corpus']=corpus


# In[ ]:


dataset.head()


# In[ ]:


dataset.drop(['Review'], axis=1,inplace=True)


# In[ ]:


dataset.head()


# In[ ]:


positive=dataset[dataset['senntiment_word']==('Positive')]


# In[ ]:


word=' '.join(positive['new_corpus'])
split_word=" ".join([word for word in word.split()])


# In[ ]:


wordCloud=WordCloud(stopwords=STOPWORDS, background_color='black', width=2000, height=1500).generate(split_word)


# In[ ]:


plt.figure(figsize=(15,15))
plt.imshow(wordCloud)
plt.axis('off')
plt.show()


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
x_train=cv.fit_transform(corpus).toarray()
x_test=cv.fit_transform(corpus1).toarray()
y=dataset.iloc[:,1].values


# In[ ]:


from sklearn.cross_validation import train_test_split
X_train,X_test, y_train, y_test=train_test_split(x_train, y, test_size=0.4, random_state=0)


# In[ ]:


clf=GaussianNB()
clf.fit(X_train, y_train)


# In[ ]:


y_pred=clf.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix, consensus_score


# In[ ]:


cm=confusion_matrix(y_test, y_pred)


# In[ ]:


cm


# In[ ]:


plt.imshow(cm)
plt.show()


# In[ ]:


plt.matshow(cm)
plt.show()

