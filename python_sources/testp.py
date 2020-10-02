#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
get_ipython().run_line_magic('pylab', 'inline')


# In[ ]:


#reading the data from the disk into memory
df = pd.read_csv("../input/train.csv")


# In[ ]:


df.head()


# In[ ]:


df.columns


# In[ ]:


df.Pclass


# In[ ]:


X = pd.DataFrame()
X['Pclass'] = df['Pclass']
X['survived'] = df['Survived']


# In[ ]:


X.head()


# In[ ]:


X = X.dropna(axis=0)


# In[ ]:


y = X['survived']
X = X.drop(['survived'], axis=1)


# In[ ]:


X.head()


# In[ ]:


from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


len(X_train)


# In[ ]:


len(X_test)


# In[ ]:


def base_rate_model(X):
    y = np.zeros(X.shape[0])
    return y


# In[ ]:


y_base_rate = base_rate_model(X_test)
from sklearn.metrics import accuracy_score
print ("Base rate accuracy is %2.2f" % accuracy_score(y_test, y_base_rate))


# In[ ]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(penalty='l2', C=1)


# In[ ]:


model.fit(X_train, y_train)


# In[ ]:


print ("Logistic accuracy is %2.2f" % accuracy_score(y_test,model.predict(X_test)))


# In[ ]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report


# In[ ]:


print ("---Base Model---")
#base rate AUC
base_roc_auc = roc_auc_score(y_test, base_rate_model(X_test))
print ("Base Rate AUC = %2.2f" % base_roc_auc)
print (classification_report(y_test,base_rate_model(X_test)))
print ("\n\n---Logistic Model---")
#logistic AUC
logit_roc_auc = roc_auc_score(y_test, model.predict(X_test))
print ("Logistic AUC = %2.2f" % logit_roc_auc)
print (classification_report(y_test, model.predict(X_test)))


# In[ ]:


from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])


# In[ ]:


plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# In[ ]:




