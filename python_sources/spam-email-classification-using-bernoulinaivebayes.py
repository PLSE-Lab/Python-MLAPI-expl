#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
docs= pd.read_csv('../input/sms-spam-collection-dataset/spam.csv',encoding='latin')
docs.head()


# In[ ]:


docs = docs[['v1','v2']]


# In[ ]:


docs.rename(columns = {'v1':'label','v2':'sms'},inplace=True)


# In[ ]:


docs.head()


# In[ ]:


#df.column_name.value_counts() - gives no. of unique inputs in that columns
docs.sms.value_counts()


# In[ ]:


ham_spam=docs.label.value_counts()
ham_spam


# In[ ]:


print("Spam % is ",(ham_spam[1]/float(ham_spam[0]+ham_spam[1]))*100)


# In[ ]:


# mapping labels to 1 and 0
docs['label'] = docs.label.map({'ham':0, 'spam':1})


# In[ ]:


docs.head()


# In[ ]:


X=docs.sms
y=docs.label


# In[ ]:


X = docs.sms
y = docs.label
print(X.shape)
print(y.shape)


# In[ ]:


# splitting into test and train

from sklearn.model_selection  import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


# In[ ]:


X_train.head()


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer

# vectorising the text
vect = CountVectorizer(stop_words='english')


# In[ ]:


vect.fit(X_train)


# In[ ]:


vect.vocabulary_


# In[ ]:


vect.get_feature_names()


# In[ ]:


# transform
X_train_transformed = vect.transform(X_train)
X_test_tranformed =vect.transform(X_test)


# In[ ]:


from sklearn.naive_bayes import BernoulliNB

# instantiate bernoulli NB object
bnb = BernoulliNB()

# fit 
bnb.fit(X_train_transformed,y_train)

# predict class
y_pred_class = bnb.predict(X_test_tranformed)

# predict probability
y_pred_proba =bnb.predict_proba(X_test_tranformed)

# accuracy
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_class)


# In[ ]:


bnb


# In[ ]:


metrics.confusion_matrix(y_test, y_pred_class)


# In[ ]:


confusion = metrics.confusion_matrix(y_test, y_pred_class)
print(confusion)
#[row, column]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]
TP = confusion[1, 1]


# In[ ]:


sensitivity = TP / float(FN + TP)
print("sensitivity",sensitivity)


# In[ ]:


specificity = TN / float(TN + FP)

print("specificity",specificity)


# In[ ]:


precision = TP / float(TP + FP)

print("precision",precision)
print(metrics.precision_score(y_test, y_pred_class))


# In[ ]:


print("precision",precision)
print("PRECISION SCORE :",metrics.precision_score(y_test, y_pred_class))
print("RECALL SCORE :", metrics.recall_score(y_test, y_pred_class))
print("F1 SCORE :",metrics.f1_score(y_test, y_pred_class))


# In[ ]:


y_pred_proba


# In[ ]:


from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred_proba[:,1])
roc_auc = auc(false_positive_rate, true_positive_rate)


# In[ ]:


print (roc_auc)


# In[ ]:


print(true_positive_rate)


# In[ ]:


print(false_positive_rate)


# In[ ]:


print(thresholds)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC')
plt.plot(false_positive_rate, true_positive_rate)

