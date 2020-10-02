#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/mobile-price-classification/train.csv')


# In[ ]:


df.head()


# In[ ]:


df.describe()


# # Data Exploration 

# In[ ]:


# show the count and type of each colum
df.info()


# In[ ]:


# show if the data have any null values in it
df.isnull().sum()


# In[ ]:


# show every value in each column
for uniqu in df.columns:
    print(f"{uniqu:15}{df[uniqu].unique()}\n")


# # Data visualization  

# In[ ]:


# show if the data is balanced or not with pie chart
plt.pie(df['price_range'].value_counts().values,labels=df['price_range'].unique(),autopct='%1.1f%%')
plt.title('price_range')
#plt.ylabel('wifi')
plt.show()


# In[ ]:


# show if the data is balanced or not with count plot
sns.countplot('price_range',data=df)


# In[ ]:


### We didn't use scatter plot because the data is categorical so we can use pie chart or bar chart or swarn plot
def newcircle(data,lab,title,fig=1):
    plt.subplot(1,4,fig)
    plt.pie(data.value_counts(),labels=lab,autopct='%1.1f%%')
    plt.title(title)


# In[ ]:


# how the touch screen availability in each category 
plt.figure(figsize=(15,15))
newcircle(df.loc[df['price_range']==3,'touch_screen'],['support touch screen','Not support touch screen'],'price_range',1)
newcircle(df.loc[df['price_range']==2,'touch_screen'],['support touch screen','Not support touch screen'],'price_range',2)
newcircle(df.loc[df['price_range']==1,'touch_screen'],['support touch screen','Not support touch screen'],'price_range',3)
newcircle(df.loc[df['price_range']==0,'touch_screen'],['support touch screen','Not support touch screen'],'price_range',4)


# In[ ]:


df[['touch_screen','price_range']].groupby(['price_range']).mean()


# In[ ]:


## Point plot to show relaction between ram and price
sns.pointplot(y="ram", x="price_range", data=df)


# In[ ]:


plt.figure(figsize=(20,20))
newcircle(df.loc[df['price_range']==0,'n_cores'],df['n_cores'].unique(),'price_range vs cores')


# In[ ]:


# how many cell phone support 3G
plt.figure(figsize=(20,20))
newcircle(df['three_g'],['support 3G','Not support 3G'],'3G')


# In[ ]:


sns.countplot('three_g',data=df)


# # Machine learning model

# In[ ]:


X=df.drop('price_range',axis=1)


# In[ ]:


y=df['price_range']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_tes, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)


# In[ ]:


from sklearn.pipeline import Pipeline


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# # uising RandomForestClassifier model and show what is the overfitting

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


pipe3 = Pipeline( [("RF", RandomForestClassifier())])


# In[ ]:


pipe3.fit(X_train, y_train)


# In[ ]:


print("Test score: {:.2f}".format(pipe3.score(X_tes, y_test)))


# In[ ]:


pr = pipe3.predict(X_tes)


# In[ ]:


print(classification_report(y_test,pr))


# In[ ]:


print("Train set score: {:.2f}".format(pipe3.score(X_train, y_train)))


# # Using LogisticRegression with scale the data  by using the pipeline 

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


pipe2 = Pipeline( [("scaler", MinMaxScaler()),("lR", LogisticRegression(C=100))])


# In[ ]:


pipe2.fit(X_train, y_train)


# In[ ]:


print("Test score: {:.2f}".format(pipe2.score(X_tes, y_test)))


# In[ ]:


pr = pipe2.predict(X_tes)


# In[ ]:


print(classification_report(y_test,pr))


# ## Using SVM model with tuning the parameter and cross validation using GridSearchCV 

# In[ ]:


parameters = {'svm__kernel':('linear', 'rbf'), 'svm__C':[0.001, 0.01, 0.1, 1, 10, 100],'svm__gamma': [0.001, 0.01, 0.1, 1, 10, 100]}


# In[ ]:


svc = SVC()
pipe = Pipeline( [("svm", svc)])


# In[ ]:


grid = GridSearchCV(pipe, param_grid=parameters,cv=5)
grid.fit(X_train, y_train)
print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
print("Test set score: {:.2f}".format(grid.score(X_tes, y_test)))
print("Best parameters: {}".format(grid.best_params_))


# In[ ]:


print("Train set score: {:.2f}".format(grid.score(X_train, y_train)))


# In[ ]:


print(classification_report(y_test,grid.predict(X_tes)))


# # Apply the SVM model on the Test data 

# In[ ]:


test_data = pd.read_csv('../input/mobile-price-classification/test.csv')


# In[ ]:


test_data.head()


# In[ ]:


test_data.drop('id',axis=1,inplace=True)


# In[ ]:


test_data.head()


# In[ ]:


predicted_price=grid.predict(test_data)


# In[ ]:


predicted_price


# In[ ]:


test_data['price_range']=predicted_price


# In[ ]:


test_data.head()


# In[ ]:




