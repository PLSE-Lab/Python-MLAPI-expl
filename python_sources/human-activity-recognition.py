#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df_train=pd.read_csv(r'../input/train.csv')
from sklearn.preprocessing import LabelEncoder


# In[ ]:


df_train.head()


# In[ ]:


df_train['Activity'].unique()


# In[ ]:


x_train=df_train.drop(['subject','Activity'],axis=1)


# In[ ]:


y_train=df_train['Activity']


# In[ ]:


activity_number=y_train.value_counts()


# In[ ]:


plt.pie(activity_number,labels=y_train.unique(),explode=[0.05,0.05,0.05,0.05,0.05,0.05],autopct='%1.2f%%')
plt.show()


# In[ ]:


#mean angles in x,y,z directions in various activities
pd.pivot_table(df_train,values=['angle(X,gravityMean)','angle(Y,gravityMean)','angle(Z,gravityMean)'],index='Activity',aggfunc=np.mean)


# In[ ]:


#bar-plot for mean angles in x,y,z directions in various activities

pd.pivot_table(df_train,values=['angle(X,gravityMean)','angle(Y,gravityMean)','angle(Z,gravityMean)'],index='Activity',aggfunc=np.mean).plot(kind='bar',figsize=(8,8))
plt.yticks(np.arange(-0.9,0.7,0.1))
plt.show()


# In[ ]:


df_test=pd.read_csv(r'../input/test.csv')


# In[ ]:


x_test=df_test.drop(['subject','Activity'],axis=1)
y_test=df_test['Activity']


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


lr=LogisticRegression(random_state=101)
rfc=RandomForestClassifier(n_estimators=50)
knn=KNeighborsClassifier()


# In[ ]:


lr.fit(x_train,y_train)


# In[ ]:


lr_score=lr.score(x_test,y_test)


# In[ ]:


rfc.fit(x_train,y_train)


# In[ ]:


rf_score=rfc.score(x_test,y_test)
print(rf_score)


# In[ ]:


knn.fit(x_train,y_train)


# In[ ]:


knn_score=knn.score(x_test,y_test)
print(knn_score)


# In[ ]:





# In[ ]:


pred_lr=lr.predict(x_test)


# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


confusion_matrix(y_test,pred_lr)


# In[ ]:


from sklearn.preprocessing import normalize
sns.heatmap(normalize(confusion_matrix(y_test,pred_lr)),xticklabels=y_test.unique(),yticklabels=y_test.unique(),cmap='coolwarm',annot=True)
plt.title('LOGISTIC REGRESSION NORMALIZED CONFUSION MATRIX')
plt.show()


# In[ ]:


from sklearn.preprocessing import normalize


# In[ ]:


from sklearn.decomposition import PCA


# In[ ]:


pca=PCA(0.95)


# In[ ]:


encode=LabelEncoder()


# In[ ]:


df_train_pca=pca.fit_transform(df_train.drop('Activity',axis=1))


# In[ ]:


df_train['Activity_encoded']=encode.fit_transform(df_train['Activity'])


# In[ ]:


clf=RandomForestClassifier()


# In[ ]:


clf.fit(df_train_pca,df_train['Activity_encoded'])


# In[ ]:


rf_pca_score=clf.score(df_train_pca,df_train['Activity_encoded'])


# In[ ]:


knn1=KNeighborsClassifier()
knn1.fit(df_train_pca,df_train['Activity_encoded'])
knn_pca_score=knn1.score(df_train_pca,df_train['Activity_encoded'])


# In[ ]:


score_series=pd.Series([lr_score,rf_score,knn_score,rf_pca_score,knn_pca_score],['lr_score','rf_score','knn_score','rf_pca_score','knn_pca_score'])


# In[ ]:


#comparison of different model accuracy scores
bars=plt.bar(score_series.index,score_series.values,color=['grey', 'red', 'green', 'blue', 'violet'])
for bar in bars:
    yval = round(bar.get_height(),4)
    plt.text(bar.get_x(), yval + .005, yval)
plt.tight_layout()
plt.show()


# In[ ]:


#VARIOUS MODEL PERFORMANCE ACCURACY SCORES
score_series

