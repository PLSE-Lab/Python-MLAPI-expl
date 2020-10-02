#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[9]:


df=pd.read_csv("../input/ChurnData.csv")


# In[10]:


df.info()


# In[11]:


df.head()


# In[12]:


df.shape


# In[13]:


X_data=df[["tenure","age","address","income","ed","equip","callcard","wireless"]]


# In[14]:


X_data.head()


# In[56]:


X_data.corr()


# In[57]:


plt.hist(X_data.corr())


# In[15]:


Y_data=df["churn"]


# In[16]:


Y_data.head()


# Converting pandas  into numpy array

# In[17]:


XA=np.asanyarray(X_data)
YA=np.asanyarray(Y_data)


# In[18]:


XA.dtype


# In[19]:


XA[:3]


# Scaling the dataset

# In[20]:


from sklearn.preprocessing import StandardScaler
XA=StandardScaler().fit(XA).transform(XA)


# In[21]:


print(XA)


# In[24]:


m=XA.mean() # calculating mean
std=XA.std()#calculating standard deviation
print("mean of XA {} \nstd of XA {}".format(round(m),std))


# **Splitting the dataset into train and test set**

# In[25]:


X_train=XA[:170]
X_test=XA[170:]

Y_train=YA[:170]
Y_test=YA[170:]


# In[26]:


print(X_train)


# In[27]:


from sklearn.linear_model import LogisticRegression
clf=LogisticRegression(solver='liblinear')
mdl=clf.fit(X_train,Y_train)


# **Predicting the values**

# In[28]:


Yp=mdl.predict(X_test)
YA=Y_test
print(Yp)


# In[59]:


sns.boxplot(Y_test,Yp)

**Finding the probability of right and wrong predictionm ie: [Truth False]**
# In[30]:


Yp_prob=mdl.predict_proba(X_test)
print(Yp_prob)


# In[32]:


P_Y1X=Yp_prob[:,0] # give the array of truth
P_Y0X=Yp_prob[:,1] # give the array of false


# In[33]:


table=pd.DataFrame({"P(Y=1|X)":P_Y1X,"P(Y=0|X)":P_Y0X})
print(table)


# ** Accuracy score  / jakart metrics **

# In[35]:


from sklearn.metrics import accuracy_score , jaccard_similarity_score,confusion_matrix,roc_curve
jss=jaccard_similarity_score(YA,Yp)
acc=accuracy_score(YA,Yp)
print("jaccard is = {} \naccuracy_score  = {}".format(jss,acc))


# In[37]:


cm=confusion_matrix(YA,Yp)
print("Confusion matrix \n",cm)


# In[38]:


from sklearn.metrics import classification_report
print(classification_report(YA,Yp))


# In[39]:


plt.matshow(cm)
plt.title("Confusion matrix")
plt.xlabel("Predicted")
plt.ylabel("Real")


# In[46]:


my_con=pd.crosstab(YA,Yp)
print(my_con)


# In[48]:


import seaborn as sns
sns.heatmap(my_con)


# In[47]:


table1=pd.DataFrame({"Ya":YA,"Yp":Yp})
print(table1)


# In[45]:


import seaborn as sns
conmat=pd.crosstab(tables.Ya,tables.Yp,margins=True)
sns.heatmap(conmat,annot=True)
plt.show()


# In[49]:


from sklearn.metrics import roc_curve
Yppr=Yp_prob[:,1]
fpr,tpr,thr=roc_curve(YA,Yppr)


# In[50]:


print(fpr)


# In[51]:


print(tpr)


# In[52]:


print(thr)


# In[54]:


roc=roc_curve(YA,Yp)
print(roc)


# In[53]:


plt.plot(fpr,tpr,label="ROC CURVE")
plt.grid()

