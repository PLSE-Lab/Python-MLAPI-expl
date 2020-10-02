#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np 
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

df=pd.read_excel('../input/Dataset.xlsx')


# In[ ]:


df


# In[ ]:


df.shape


# In[ ]:


# Creating a base dataset for continous columns and categorical columns
df.isna().sum()


# In[ ]:


Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).all(axis=1)]
df.shape


# In[ ]:


from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

print(__version__) # requires version >= 1.9.0


# In[ ]:


import cufflinks as cf


# In[ ]:


# For Notebooks
init_notebook_mode(connected=True)


# In[ ]:


# For offline use
cf.go_offline()


# In[ ]:


df.iplot(kind='scatter',x='Domain Registration Length',y='Age of domain',mode='markers',size=10)


# In[ ]:


df.iplot(kind='bar',x='HTTPS',y='Page Rank')


# In[ ]:


df1=df[['Number of Links Pointing to Page','Page Rank']]


# In[ ]:


df1.iplot(kind='box')


# In[ ]:


df3 = pd.DataFrame({'Age of domain':[1,2,3,4,5],'HTTPS':[0,0,0,1,1],'Page Rank':[5,4,3,2,1]})
df3.iplot(kind='surface',colorscale='rdylbu')


# In[ ]:


df[['Age of domain','Page Rank']].iplot(kind='spread')


# In[ ]:


df['Page Rank'].iplot(kind='hist',bins=25)


# In[ ]:


from sklearn.cluster import KMeans
Kmeans=KMeans(n_clusters=3,random_state=0).fit(df)


# In[ ]:


labels=Kmeans.labels_


# In[ ]:


df['cluster']=labels


# In[ ]:


df


# In[ ]:


df_1=df[df['cluster']==0]
df_2=df[df['cluster']==1]
df_3=df[df['cluster']==2]


# In[ ]:


y=df_1['Statistical-Reports based Feature']
x=df_1.drop('Statistical-Reports based Feature',axis=1)


# In[ ]:


# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X1_train, X1_test, y1_train, y1_test = train_test_split(x, y, test_size=0.2,random_state=0)


# In[ ]:


# train a logistic regression model on the training set
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X1_train, y1_train)


# In[ ]:


# make class predictions for the testing set
y1_pred_class = logreg.predict(X1_test)


# In[ ]:


# calculate accuracy
from sklearn import metrics
lr1=metrics.accuracy_score(y1_test, y1_pred_class)


# In[ ]:


lr1


# In[ ]:



from sklearn import metrics
print(metrics.confusion_matrix(y1_test, y1_pred_class))


# In[ ]:


import seaborn as sns
sns.heatmap(metrics.confusion_matrix(y1_test,y1_pred_class))


# In[ ]:


from sklearn.model_selection import cross_val_score  
all_accuracies_LogisticRegression = cross_val_score(estimator=LogisticRegression(), X=X1_train, y=y1_train, cv=10)  


# In[ ]:


a1=pd.DataFrame(all_accuracies_LogisticRegression)


# In[ ]:


a1.plot()


# In[ ]:


from sklearn.tree import DecisionTreeClassifier

# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X1_train, X1_test, y1_train, y1_test = train_test_split(x, y, test_size=0.4,random_state=42)


# In[ ]:


# train a logistic regression model on the training set
clf = DecisionTreeClassifier(criterion='entropy',splitter='best',min_samples_split=3)
clf.fit(X1_train, y1_train)


# In[ ]:


# make class predictions for the testing set
y1_pred_class = clf.predict(X1_test)


# In[ ]:


# calculate accuracy
from sklearn import metrics
dt1=metrics.accuracy_score(y1_test, y1_pred_class)


# In[ ]:


dt1


# In[ ]:


# IMPORTANT: first argument is true values, second argument is predicted values
print(metrics.confusion_matrix(y1_test, y1_pred_class))


# In[ ]:


import seaborn as sns
sns.heatmap(metrics.confusion_matrix(y1_test,y1_pred_class))


# In[ ]:


from sklearn.model_selection import cross_val_score  
all_accuracies_DT = cross_val_score(estimator=DecisionTreeClassifier(), X=X1_train, y=y1_train, cv=10)  


# In[ ]:


a1=pd.DataFrame(all_accuracies_DT)


# In[ ]:


a1.plot()


# In[ ]:


# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X1_train, X1_test, y1_train, y1_test = train_test_split(x, y, test_size=0.2,random_state=0)


# In[ ]:


# train a logistic regression model on the training set
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X1_train, y1_train)


# In[ ]:


# make class predictions for the testing set
y1_pred_class = rf_clf.predict(X1_test)


# In[ ]:


# calculate accuracy
from sklearn import metrics
rf1=metrics.accuracy_score(y1_test, y1_pred_class)


# In[ ]:


rf1


# In[ ]:


# IMPORTANT: first argument is true values, second argument is predicted values
print(metrics.confusion_matrix(y1_test, y1_pred_class))


# In[ ]:


import seaborn as sns
sns.heatmap(metrics.confusion_matrix(y1_test,y1_pred_class))


# In[ ]:


from sklearn.model_selection import cross_val_score  
from sklearn.ensemble import RandomForestClassifier
all_accuracies_RF= cross_val_score(estimator=RandomForestClassifier(), X=X1_train, y=y1_train, cv=10)  


# In[ ]:


all_accuracies_RF


# In[ ]:


a1=pd.DataFrame(all_accuracies_RF)


# In[ ]:


a1.plot()


# In[ ]:


# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X1_train, X1_test, y1_train, y1_test = train_test_split(x, y, test_size=0.2,random_state=0)


# In[ ]:


# train a logistic regression model on the training set
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X1_train, y1_train)


# In[ ]:


# make class predictions for the testing set
y1_pred_class = nb.predict(X1_test)


# In[ ]:


# calculate accuracy
from sklearn import metrics
nb1=metrics.accuracy_score(y1_test, y1_pred_class)


# In[ ]:


nb1


# In[ ]:


# IMPORTANT: first argument is true values, second argument is predicted values
print(metrics.confusion_matrix(y1_test, y1_pred_class))


# In[ ]:


import seaborn as sns


# In[ ]:


sns.heatmap(metrics.confusion_matrix(y1_test,y1_pred_class))


# In[ ]:


from sklearn.model_selection import cross_val_score  
all_accuracies_GaussianNB = cross_val_score(estimator=GaussianNB(), X=X1_train, y=y1_train, cv=10)  


# In[ ]:


all_accuracies_GaussianNB=pd.DataFrame(all_accuracies_GaussianNB)


# In[ ]:


all_accuracies_GaussianNB.plot()


# In[ ]:




df_f=pd.DataFrame(columns=['acuracy_c1','acuracy_c2','acuracy_c3','cv1','cv2','cv3','cv4','cv5','cv6','cv7','cv8','cv9','cv10'],index=['LogisticRegression','DecisionTree','RandomForest','NaiveBayes'])


# In[ ]:


av1=[lr1,dt1,rf1,nb1]
df_f['acuracy_c1']=av


# In[ ]:


df_f


# In[ ]:


df_f=pd.DataFraticRegression','DecisionTree','RandomForest','NaiveBayes'])


# In[ ]:


pd.DataFrame()


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


y=df_2['Statistical-Reports based Feature']
x=df_2.drop('Statistical-Reports based Feature',axis=1)


# In[ ]:


# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X2_train, X2_test, y2_train, y2_test = train_test_split(x, y, test_size=0.2,random_state=0)


# In[ ]:


# train a logistic regression model on the training set
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X2_train, y2_train)


# In[ ]:


# make class predictions for the testing set
y2_pred_class = logreg.predict(X2_test)


# In[ ]:


# calculate accuracy
from sklearn import metrics
lr2=metrics.accuracy_score(y2_test, y2_pred_class)


# In[ ]:


# IMPORTANT: first argument is true values, second argument is predicted values
print(metrics.confusion_matrix(y2_test, y2_pred_class))


# In[ ]:


import seaborn as sns
sns.heatmap(metrics.confusion_matrix(y2_test,y2_pred_class))


# In[ ]:


from sklearn.model_selection import cross_val_score  
all_accuracies_LogisticRegression1 = cross_val_score(estimator=LogisticRegression(), X=X2_train, y=y2_train, cv=10) 


# In[ ]:


all_accuracies_LogisticRegression1=pd.DataFrame(all_accuracies_LogisticRegression)


# In[ ]:


all_accuracies_LogisticRegression.plot()


# In[ ]:


from sklearn.tree import DecisionTreeClassifier

# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X2_train, X2_test, y2_train, y2_test = train_test_split(x, y, test_size=0.4,random_state=42)


# In[ ]:


# train a logistic regression model on the training set
clf = DecisionTreeClassifier(criterion='entropy',splitter='best',min_samples_split=3)
clf.fit(X2_train, y2_train)


# In[ ]:


# make class predictions for the testing set
y2_pred_class = clf.predict(X2_test)


# In[ ]:


# calculate accuracy
from sklearn import metrics
df2=metrics.accuracy_score(y2_test, y2_pred_class)


# In[ ]:


df2


# In[ ]:


# IMPORTANT: first argument is true values, second argument is predicted values
print(metrics.confusion_matrix(y2_test, y2_pred_class))


# In[ ]:


import seaborn as sns
sns.heatmap(metrics.confusion_matrix(y2_test,y2_pred_class))


# In[ ]:


from sklearn.model_selection import cross_val_score  
all_accuracies_DecisionTreeClassifier = cross_val_score(estimator=DecisionTreeClassifier(), X=X2_train, y=y2_train, cv=10)  


# In[ ]:


all_accuracies_DecisionTreeClassifier=pd.DataFrame(all_accuracies_DecisionTreeClassifier)


# In[ ]:


all_accuracies_DecisionTreeClassifier.plot()


# In[ ]:


# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X2_train, X2_test, y2_train, y2_test = train_test_split(x, y, test_size=0.2,random_state=0)


# In[ ]:



from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X2_train, y2_train)


# In[ ]:


# make class predictions for the testing set
y2_pred_class = rf_clf.predict(X2_test)


# In[ ]:


# calculate accuracy
from sklearn import metrics
rf2=metrics.accuracy_score(y2_test, y2_pred_class)


# In[ ]:


rf2


# In[ ]:


# IMPORTANT: first argument is true values, second argument is predicted values
print(metrics.confusion_matrix(y2_test, y2_pred_class))


# In[ ]:




sns.heatmap(metrics.confusion_matrix(y2_test,y2_pred_class))


# In[ ]:



from sklearn.model_selection import cross_val_score  
from sklearn.ensemble import RandomForestClassifier
all_accuracies_RandomForestClassfier2 = cross_val_score(estimator=RandomForestClassifier(), X=X2_train, y=y2_train, cv=10) 


# In[ ]:


all_accuracies_RandomForestClassfier2=pd.DataFrame(all_accuracies_RandomForestClassfier)


# In[ ]:


all_accuracies_RandomForestClassfier2.plot()


# In[ ]:


# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X2_train, X2_test, y2_train, y2_test = train_test_split(x, y, test_size=0.2,random_state=0)


# In[ ]:


# train a logistic regression model on the training set
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X2_train, y2_train)


# In[ ]:


# make class predictions for the testing set
y2_pred_class = nb.predict(X2_test)


# In[ ]:


# calculate accuracy
from sklearn import metrics
print(metrics.accuracy_score(y2_test, y2_pred_class))


# In[ ]:


# IMPORTANT: first argument is true values, second argument is predicted values
print(metrics.confusion_matrix(y2_test, y2_pred_class))


# In[ ]:



sns.heatmap(metrics.confusion_matrix(y2_test,y2_pred_class))


# In[ ]:


from sklearn.model_selection import cross_val_score  
all_accuracies_GaussianNB = cross_val_score(estimator=GaussianNB(), X=X2_train, y=y2_train, cv=10)  


# In[ ]:


all_accuracies_GaussianNB=pd.DataFrame(all_accuracies_GaussianNB)


# In[ ]:


all_accuracies_GaussianNB.plot()


# In[ ]:


y=df_3['Statistical-Reports based Feature']
x=df_3.drop('Statistical-Reports based Feature',axis=1)


# In[ ]:


# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X3_train, X3_test, y3_train, y3_test = train_test_split(x, y, test_size=0.2,random_state=0)


# In[ ]:


# train a logistic regression model on the training set
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X3_train, y3_train)


# In[ ]:


# make class predictions for the testing set
y3_pred_class = logreg.predict(X3_test)


# In[ ]:


from sklearn import metrics
print(metrics.accuracy_score(y3_test, y3_pred_class))


# In[ ]:


print(metrics.confusion_matrix(y3_test, y3_pred_class))


# In[ ]:


sns.heatmap(metrics.confusion_matrix(y3_test, y3_pred_class))


# In[ ]:


from sklearn.model_selection import cross_val_score  
all_accuracies_LogisticRegression3 = cross_val_score(estimator=LogisticRegression(), X=X3_train, y=y3_train, cv=10)  


# In[ ]:


all_accuracies_LogisticRegression3=pd.DataFrame(all_accuracies_LogisticRegression3)


# In[ ]:


all_accuracies_LogisticRegression3.plot()


# In[ ]:


from sklearn.tree import DecisionTreeClassifier

# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X3_train, X3_test, y3_train, y3_test = train_test_split(x, y, test_size=0.4,random_state=42)


# In[ ]:


# train a logistic regression model on the training set
clf = DecisionTreeClassifier(criterion='entropy',splitter='best',min_samples_split=3)
clf.fit(X3_train, y3_train)


# In[ ]:


# make class predictions for the testing set
y3_pred_class = clf.predict(X3_test)


# In[ ]:


# calculate accuracy
from sklearn import metrics
print(metrics.accuracy_score(y3_test, y3_pred_class))


# In[ ]:


# IMPORTANT: first argument is true values, second argument is predicted values
print(metrics.confusion_matrix(y3_test, y3_pred_class))


# In[ ]:


sns.heatmap(metrics.confusion_matrix(y3_test, y3_pred_class))


# In[ ]:


from sklearn.model_selection import cross_val_score  
all_accuracies_DecisionTreeClassifier3 = cross_val_score(estimator=DecisionTreeClassifier(), X=X3_train, y=y3_train, cv=10)  


# In[ ]:


all_accuracies_DecisionTreeClassifier3=pd.DataFrame(all_accuracies_DecisionTreeClassifier3)


# In[ ]:


all_accuracies_DecisionTreeClassifier3


# In[ ]:


all_accuracies_DecisionTreeClassifier.plot()


# In[ ]:


# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X3_train, X3_test, y3_train, y3_test = train_test_split(x, y, test_size=0.2,random_state=0)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X3_train, y3_train)


# In[ ]:


#make class predictions for the testing set
y3_pred_class = rf_clf.predict(X3_test)


# In[ ]:


# calculate accuracy
from sklearn import metrics
print(metrics.accuracy_score(y3_test, y3_pred_class))


# In[ ]:


# IMPORTANT: first argument is true values, second argument is predicted values
print(metrics.confusion_matrix(y3_test, y3_pred_class))


# In[ ]:


sns.heatmap(metrics.confusion_matrix(y3_test, y3_pred_class))


# In[ ]:


from sklearn.model_selection import cross_val_score  
from sklearn.ensemble import RandomForestClassifier
all_accuracies_RandomForestClassfier3 = cross_val_score(estimator=RandomForestClassifier(), X=x, y=y, cv=10)  


# In[ ]:


all_accuracies_RandomForestClassfier3=pd.DataFrame(all_accuracies_RandomForestClassfier3)


# In[ ]:


all_accuracies_RandomForestClassfier3


# In[ ]:


all_accuracies_RandomForestClassfier3.plot()


# In[ ]:


# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X3_train, X3_test, y3_train, y3_test = train_test_split(x, y, test_size=0.2,random_state=0)


# In[ ]:


# train a logistic regression model on the training set
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X3_train, y3_train)


# In[ ]:


# make class predictions for the testing set
y3_pred_class = nb.predict(X3_test)


# In[ ]:


# calculate accuracy
from sklearn import metrics
print(metrics.accuracy_score(y3_test, y3_pred_class))


# In[ ]:


# IMPORTANT: first argument is true values, second argument is predicted values
print(metrics.confusion_matrix(y3_test, y3_pred_class))


# In[ ]:


sns.heatmap(metrics.confusion_matrix(y3_test, y3_pred_class))


# In[ ]:


from sklearn.model_selection import cross_val_score  
all_accuracies_GaussianNB = cross_val_score(estimator=GaussianNB(), X=x,y=y, cv=10)  


# In[ ]:


all_accuracies_GaussianNB=pd.DataFrame(all_accuracies_GaussianNB)


# In[ ]:


all_accuracies_GaussianNB


# In[ ]:


all_accuracies_GaussianNB.plot()


# In[ ]:



df1=pd.DataFrame(all_accuracies_LogisticRegression)
df2=pd.DataFrame(all_accuracies_DT)
df3=pd.DataFrame(all_accuracies_RF)
df4=pd.DataFrame(all_accuracies_GaussianNB)
frames=[df1,df2,df3,df4]
df_f=pd.concat(frames,axis=1)


# In[ ]:


df_f.columns=['LogisticRegression','DecisionTree','RandomForest','GaussianNB']


# In[ ]:


df_a


# In[ ]:




