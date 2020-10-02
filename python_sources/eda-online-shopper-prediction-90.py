#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv("../input/online-shoppers-intention/online_shoppers_intention.csv")
df.head()


# In[ ]:


plt.xlabel("Visitor Type")
sns.countplot(df['VisitorType'])


# In[ ]:


plt.rcParams['figure.figsize'] = (18,6)
plt.subplot(1,2,1)
plt.title("Weeked Day or Not")
sns.countplot(df['Weekend'])


plt.subplot(1,2,2)
plt.title("Revenue or Not")
sns.countplot(df['Revenue'])


# In[ ]:


labels = ['Returning_Visitor','New_Visitor','Other']
plt.title("Types of Vistors")
plt.pie(df['VisitorType'].value_counts(),labels=labels,autopct = '%.2f%%')
plt.legend()


# In[ ]:


plt.figure(figsize=(15,6))
plt.title("Types of Operating System used")
plt.xlabel("Operating System Used")
sns.countplot(df['Browser'])


# In[ ]:


plt.title("Region in Data")
plt.xlabel("Region")
sns.countplot(df['Region'])


# In[ ]:



sns.boxenplot(df['Revenue'], df['Informational_Duration'], palette = 'rainbow')
plt.title('Info. duration vs Revenue', fontsize = 30)
plt.xlabel('Info. duration', fontsize = 15)
plt.ylabel('Revenue', fontsize = 15)


# In[ ]:


pd.crosstab(df['TrafficType'],df['Revenue']).plot(kind = 'bar', stacked = True, figsize = (15, 5), color = ['lightpink', 'yellow'])


# In[ ]:


pd.crosstab(df['VisitorType'],df['Revenue']).plot(kind='bar',stacked=True)
plt.title('Visitor Type vs Revenue')
plt.show()


# In[ ]:


pd.crosstab(df['Region'],df['Revenue']).plot(kind='bar',stacked=True)


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier,BaggingClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,r2_score
from sklearn.preprocessing import LabelEncoder


# In[ ]:


lb = LabelEncoder()
df['Month'] = lb.fit_transform(df['Month'])
df['VisitorType'] = lb.fit_transform(df['VisitorType'])
df.dropna(inplace=True)


# In[ ]:


# removing the target column revenue from x
X = df.drop(['Revenue'], axis = 1)
y = df['Revenue']


# In[ ]:



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:


model = DecisionTreeClassifier()
model.fit(X_train,y_train)
predict = model.predict(X_test)
accuracy_score(y_test,predict)


# In[ ]:


model = LogisticRegression()
model.fit(X_train,y_train)
predict = model.predict(X_test)
accuracy_score(y_test,predict)


# In[ ]:


model = ExtraTreesClassifier()
model.fit(X_train,y_train)
predict = model.predict(X_test)
accuracy_score(y_test,predict)


# In[ ]:


model = GradientBoostingClassifier()
model.fit(X_train,y_train)
predict = model.predict(X_test)
accuracy_score(y_test,predict)


# In[ ]:





# In[ ]:




