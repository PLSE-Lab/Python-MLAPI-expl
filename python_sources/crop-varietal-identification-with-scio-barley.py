#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


df = pd.read_csv("../input/Barley.data.csv")
df.head()


# In[ ]:


#Na Handling
df.isnull().values.any()


# In[ ]:


df=df.dropna()


# In[ ]:


from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[ ]:


X = df.drop(['Predictor'], axis=1)
X_col = X.columns


# In[ ]:


y = df['Predictor']


# In[ ]:



#Savitzky-Golay filter with second degree derivative.
from scipy.signal import savgol_filter 

sg=savgol_filter(X,window_length=11, polyorder=3, deriv=2, delta=1.0)


# In[ ]:


sg_x=pd.DataFrame(sg, columns=X_col)
sg_x.head()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(sg_x, y,
                                                    train_size=0.8,
                                                    random_state=1,stratify = y)


# In[ ]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components=15)  
X_train = lda.fit_transform(X_train, y_train)  
X_test = lda.transform(X_test)


# In[ ]:


pd.DataFrame(X_train).head()


# In[ ]:


from sklearn import svm
clf = svm.SVC(kernel="rbf")
clf.fit(X_train, y_train)  
y_pred = clf.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix  
from sklearn.metrics import accuracy_score

cm = confusion_matrix(y_test, y_pred)  
print(cm)  
print('Accuracy' + str(accuracy_score(y_test, y_pred))) 


# In[ ]:




