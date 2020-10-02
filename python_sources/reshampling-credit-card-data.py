#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')


# # Data Understanding
# There is 284807 observation of 31 variable. Class is target variable where as others are predictor variable. Information given in data is sesitive so i think data has been preprocessed with technique such as PCA or Factor Analysis, So we need not to put extra effort on Data Cleaning and Wrangling. Out of 284807 only 492 observations are detected Fraud so this data is highly imbalanced we will use different sampling technique to increase accuracy.

# In[ ]:


df=pd.read_csv('../input/creditcardfraud/creditcard.csv')
df.head()


# In[ ]:


df.shape


# In[ ]:


all(df.isnull().any())


# **There is no missing value**

# In[ ]:


df['Class'].value_counts()


# In[ ]:


print((492/(284807+492))*100)


# In[ ]:


plt.figure(dpi=100)
sns.set_style('darkgrid')
sns.countplot('Class',data=df)
plt.xlabel('Target Class')
plt.ylabel('Count')
plt.xticks([0,1],['Not Fraud','Fraud'])
plt.show()


# **Dataset is highly imbalanced, only 0.17 % obseravations are detected as Fraud**

# In[ ]:


mask = np.triu(np.ones_like(df.corr(),dtype=bool))
plt.figure(dpi=100,figsize=(10,8))
sns.heatmap(df.corr(),yticklabels=True,mask=mask,cmap='viridis',annot=False, lw=1)
plt.show()


# **We can see there are only less variable which are weakly correalted with class, May be this is because data is already reduced to lower domension using PCA and other Feature engineering method and these varables are explaining significant variance in data**

# # Data Reshampling
# Here i am reshampling data using **SMOTE** method because dataset is imbalaned.

# In[ ]:


x=df.iloc[:,:-1]
y=df.iloc[:,-1]


# In[ ]:


print((x.shape,y.shape))


# In[ ]:


from imblearn.combine import SMOTETomek
smk=SMOTETomek(ratio=1,random_state=0)
x_new,y_new=smk.fit_sample(x,y)


# In[ ]:


print(x_new.shape,y_new.shape)


# In[ ]:



from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test=tts(x_new,y_new,test_size=0.80,random_state=0,stratify=y_new)


# In[ ]:



print(x_train.shape,x_test.shape)


# # Logistic Regression Model

# In[ ]:


from sklearn.linear_model import LogisticRegression
lrm=LogisticRegression(C=0.1,penalty='l1',n_jobs=-1)
lrm.fit(x_train,y_train)


# In[ ]:


y_pred=lrm.predict(x_test)


# # Model Evaluation

# In[ ]:


print("Train Set Accuracy is ==> ",metrics.accuracy_score(y_train,lrm.predict(x_train)))
print("Test Set Accuracy is ==> ",metrics.accuracy_score(y_test,y_pred))


# In[ ]:


print("Classification Report on Hold Out Dataset==>\n\n",metrics.classification_report(y_test,y_pred))


# In[ ]:


probs = lrm.predict_proba(x_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)
plt.figure(dpi=100)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# **Model is validated , we can see accuracy at both train and test set is almost sam which means model is not overfitting, ROC-AUC score is also enough good 0.98.**
# 
# ### Please Upvote this kernel, if it is useful for you :)
