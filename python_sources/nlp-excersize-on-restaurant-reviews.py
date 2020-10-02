#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd
import nltk
import re
import matplotlib.pyplot as plt


# In[ ]:


comments = pd.read_csv("../input/restaurant_reviews.csv")
comments.head(10)


# In[ ]:


comment = re.sub('[^a-zA-Z]',' ', comments["Review"][0])
comment


# In[ ]:


comment = comment.lower()
comment


# In[ ]:


comment = comment.split()
comment


# In[ ]:


#nltk.download("stopwords") # stopwords for english.
from nltk.corpus import stopwords


# In[ ]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[ ]:


comment = [ps.stem(word) for word in comment if not word in set(stopwords.words('english'))]
comment


# In[ ]:


comment = ' '.join(comment)
comment


# In[ ]:


comments["Review"][0]
# Did you see difference? From old comment to processed comment...
# Anymore our preparing template is ready.


# In[ ]:


df = []
for i in range(1000):
    comment = re.sub('[^a-zA-Z]', ' ', comments["Review"][i])
    comment = comment.lower()
    comment = comment.split()
    comment = [ps.stem(word) for word in comment if not word in set(stopwords.words("english"))]
    comment = " ".join(comment)
    df.append(comment)


# In[ ]:


df[:10]


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2000)


# In[ ]:


X = cv.fit_transform(df).toarray()
y = comments.iloc[:,1].values


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20, random_state=0)


# In[ ]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 1000)
rf.fit(X_train, y_train)


# In[ ]:


y_pred = rf.predict(X_test)


# In[ ]:


cm = confusion_matrix(y_test,y_pred)
print(cm)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
k_range = range(1,50)
scores = {}
scores_list = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores[k] = metrics.accuracy_score(y_test,y_pred)
    scores_list.append(metrics.accuracy_score(y_test,y_pred))


# In[ ]:


plt.plot(k_range, scores_list)
plt.xlabel("Value of K for KNN")
plt.ylabel("Testing Accuracy")
plt.show()


# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 34)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
score = metrics.accuracy_score(y_test,y_pred)
score


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state=0)
dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print(cm)


# In[ ]:


from sklearn import svm
svc = svm.SVC(kernel="rbf", gamma="scale")
svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print(cm)

