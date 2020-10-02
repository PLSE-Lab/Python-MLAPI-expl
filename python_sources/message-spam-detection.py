#!/usr/bin/env python
# coding: utf-8

# # MESSAGE SPAM DETECTION 
# 
# The purpose of this project is to explore the results of applying machine learning techniques to
# spam detection.

# In[ ]:


import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt 

df = pd.read_csv('../input/spam.csv', encoding='latin-1')
df = df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
df = df.rename(columns = {'v1':'Class','v2':'Text'})
df.head()


# In[ ]:


df['count']=0
for i in np.arange(0,len(df.Text)):
    df.loc[i,'count'] = len(df.loc[i,'Text'])

df = df.replace(['ham','spam'],[0, 1]) 


# In[ ]:


print ("Unique values in the set: ", df.Class.unique())
df.info()


# In[ ]:


df.head()


# In[ ]:


ham  = df[df.Class == 0]
ham_count  = pd.DataFrame(pd.value_counts(ham['count'],sort=True).sort_index())

ax = plt.axes()
xline_ham = np.linspace(0, len(ham_count) - 1, len(ham_count))
ax.bar(xline_ham, ham_count['count'], width=2.2, color='r')
ax.set_title('SMS Ham by length of message')
plt.xlabel('length')
plt.ylabel('frequency')
plt.show()


# In[ ]:


spam = df[df.Class == 1]
spam_count = pd.DataFrame(pd.value_counts(spam['count'],sort=True).sort_index())

ax = plt.axes()
xline_spam = np.linspace(0, len(spam_count) - 1, len(spam_count))
ax.bar(xline_spam,spam_count['count'], width=0.75, color='b')
ax.set_title('SMS Spam by length of message')
plt.xlabel('length')
plt.ylabel('frequency')
plt.show()


# In[ ]:


from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

if False:
    nltk.download('stopwords')
    
if False:
    vectorizer = TfidfVectorizer()
    
if True:
    stopset = set(stopwords.words())
    vectorizer = TfidfVectorizer(stop_words=stopset)


# In[ ]:


X = vectorizer.fit_transform(df.Text)
y = df.Class


# In[ ]:


#!/usr/bin/env python -W ignore::DeprecationWarning
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, train_size=0.80, random_state=42)

print ("Training set has {} samples." .format(X_train.shape[0]))
print ("Testing set has {} samples." .format(X_test.shape[0]))


# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

objects = ('NB', 'DT', 'AdaBoost', 'KNeig', 'RF')


# In[ ]:


A = MultinomialNB(alpha=1.0,fit_prior=True)
A.fit(X_train, y_train)
A_pred = A.predict(X_test)
A = accuracy_score(y_test,A_pred)
print (A)


# In[ ]:


B = DecisionTreeClassifier(random_state=42)
B.fit(X_train, y_train)
B_pred = B.predict(X_test)
B = accuracy_score(y_test,B_pred)
print (B)


# In[ ]:


C = AdaBoostClassifier(n_estimators=100)
C.fit(X_train, y_train)
C_pred = C.predict(X_test)
C = accuracy_score(y_test,C_pred)
print (C)


# In[ ]:


D = KNeighborsClassifier(n_neighbors=3)
D.fit(X_train, y_train)
D_pred = D.predict(X_test)
D = accuracy_score(y_test,D_pred)
print (D)


# In[ ]:


E = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
E.fit(X_train, y_train)
E_pred = E.predict(X_test)
E = accuracy_score(y_test,E_pred)
print (E)


# In[ ]:


y_pos = np.arange(len(objects))
y_val = [ x-0.9 for x in [A,B,C,D,E]]
plt.bar(y_pos,y_val, align='center', alpha=0.6)
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy Score')
plt.title('Model')
plt.show()

