#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis of Restaurant Reviews 

# Here, our main target is to predict the number of positive and negative reviews (given as 1 or 0) based on sentiments by using different classification models.

# # importing the libraries

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # importing the dataset

# In[ ]:


dataset = pd.read_csv('/kaggle/input/Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)


# # some basic information regarding the dataset is as follows:

# In[ ]:


dataset.head()


# In[ ]:


dataset.shape


# In[ ]:


dataset.info()


# In[ ]:


dataset.describe()


# # Sentiment count
# here, we will see number of negative and positive reviews in the given data set, to make sure whether the datset is balanced or not.

# In[ ]:


fig=plt.figure(figsize=(5,5))
colors=["blue",'pink']
pos=dataset[dataset['Liked']==1]
neg=dataset[dataset['Liked']==0]
ck=[pos['Liked'].count(),neg['Liked'].count()]
legpie=plt.pie(ck,labels=["Positive","Negative"],
                 autopct ='%1.1f%%', 
                 shadow = True,
                 colors = colors,
                 startangle = 45,
                 explode=(0, 0.1))


# # importing libraries from nltk for cleaning the text.

# In[ ]:


import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
stop=stopwords.words('english')


# # Let's see positive and negative words by using WordCloud.

# In[ ]:


from wordcloud import WordCloud
positivedata = dataset[ dataset['Liked'] == 1]
positivedata =positivedata['Review']
negdata = dataset[dataset['Liked'] == 0]
negdata= negdata['Review']

def wordcloud_draw(dataset, color = 'white'):
    words = ' '.join(dataset)
    cleaned_word = " ".join([word for word in words.split()
                              if(word!='food' and word!='place')
                            ])
    wordcloud = WordCloud(stopwords=stop,
                      background_color=color,
                      width=2500,
                      height=2000
                     ).generate(cleaned_word)
    plt.figure(1,figsize=(10, 7))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    
print("Positive words are as follows")
wordcloud_draw(positivedata,'white')
print("Negative words are as follows")
wordcloud_draw(negdata)


# positive words are highlighted as great, delicious, amazing, nice etc.
# 
# negative words are highlighted as never, terrible, horrible, bad, disappointed etc.

# # Sentiment analysis
# here we first Remove special characters, then stemming the text and after that Remove the stopwords from the reviews.
# 
# 

# In[ ]:


corpus = []
for i in range(0, 1000):
  review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
  review = review.lower()
  review = review.split()
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  corpus.append(review)
print(corpus)


# # Creating the Bag of Words model
# here we will apply count vactorization process.

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values


# In[ ]:


X


# # Splitting the dataset into the Training set and Test set.

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# # Training the Naive Bayes model on the Training set

# In[ ]:


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# # Predicting the Test set results

# In[ ]:


y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# # Making the Confusion Matrix

# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[ ]:


from sklearn import metrics


# # accuracy for naive bayes model

# In[ ]:


print("Naive Bayes Accuracy:",metrics.accuracy_score(y_test, y_pred))


# # Training the Logistic Regression model on the Training set

# In[ ]:


from sklearn.linear_model import LogisticRegressionCV

classifier=LogisticRegressionCV(cv=6,scoring='accuracy',random_state=0,n_jobs=-1,verbose=3,max_iter=500).fit(X_train,y_train)

y_pred1 = classifier.predict(X_test)


# # Predicting the Test set results

# In[ ]:


print(np.concatenate((y_pred1.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# # Making the Confusion Matrix

# In[ ]:


from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_pred1)
print(cm)


# # accuracy for Logistic Regression Model

# In[ ]:


print("Logistic Regression Accuracy:",metrics.accuracy_score(y_test, y_pred1))


# # Training the Random Forest Classifier on the Training set

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier()
classifier.fit(X_train,y_train)
preds=classifier.predict(X_test)


# # Predicting the Test set results

# In[ ]:


print(np.concatenate((preds.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# # Making the Confusion Matrix

# In[ ]:


cm = confusion_matrix(preds,y_test)
print(cm)


# # Accuracy for Random Forest Classifier

# In[ ]:


print("Randon Forest Accuracy:",metrics.accuracy_score(preds,y_test))


# # Training the XGBoost Model on the training set

# In[ ]:


import xgboost as xgb
xgb=xgb.XGBClassifier()
xgb.fit(X_train,y_train)


# # Predicting the Test set results

# In[ ]:


preds2=xgb.predict(X_test)
print(np.concatenate((preds2.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# # Making the Confusion Matrix

# In[ ]:


cm = confusion_matrix(preds2,y_test)
print(cm)


# # Accuracy for XGBoost model

# In[ ]:


print("XGBoost model Accuracy:",metrics.accuracy_score(y_test, preds2))


# # Conclusion
# here we observed that both Logistic regression model and random forest model performed highest accuracy by getting 77.5% and 75%, whereas naive bayes and XGBoost model performed less accuracy.

# In[ ]:




