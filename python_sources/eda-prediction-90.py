#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv("../input/indian-candidates-for-general-election-2019/LS_2.0.csv")
df.head()


# In[ ]:





# In[ ]:


df['PARTY'].value_counts()[:10].plot(kind='bar')


# In[ ]:


df.columns


# In[ ]:


df.columns = ["State","Constituency","Name","Winner","Party","Symbol","Gender","CrimeCases","Age","Category","Education","Assets","Liabilites","General Votes","Postal Votes","Total Votes","Over total Electors in Constituency","Over Total Votes Polled In Constituency","Total Electors"]


# In[ ]:


df.dropna(inplace=True)
df.head()


# In[ ]:


df['Education'].value_counts().plot.bar()


# In[ ]:


plt.figure(figsize=(15,7))
df['Age'].value_counts().plot.bar()


# In[ ]:


df['Category'].value_counts().plot.bar()


# In[ ]:


def changeAssetValue(x):
    try:
        temp = (x.split('Rs')[1].split('\n')[0].strip())
        temp2 = ''
        for i in temp.split(","):
            temp2 = i+temp2
        return temp2
    except:
        x = 0
        return x
def changeLiabilitesValue(x):
    try:
        temp = x.split('Rs')[1].split('\n')[0].strip()
        temp2 = ''
        for i in temp.split(","):
            temp2 = i+temp2
        return temp2
    except:
        x = 0
        return x
    


# In[ ]:


df['Assets'] = df['Assets'].apply(changeAssetValue)
df['Liabilites'] = df['Liabilites'].apply(changeLiabilitesValue)


# In[ ]:


df.head()


# In[ ]:


sns.countplot(df['Gender'])


# In[ ]:


df['Assets'] = df['Assets'].astype('int64')
print("Total Assets of Indian Policitions:",df['Assets'].sum())


# In[ ]:


df['Liabilites'] = df['Liabilites'].astype('int64')
print("Total Liabilites of Indian Policitions:",df['Liabilites'].sum())


# In[ ]:


df['Assets'].sum()-df['Liabilites'].sum()


# In[ ]:


df['Assets'].mean()
# Average 20 cr.


# In[ ]:


df['Liabilites'].mean()
# Average 2 cr.


# In[ ]:


pd.crosstab(df['Symbol'],df['Winner']).plot(figsize=(15,7))


# In[ ]:


pd.crosstab(df['CrimeCases'],df['Winner']).plot(figsize=(15,7))


# In[ ]:


plt.figure(figsize=(15,7))
sns.countplot(df['CrimeCases'],hue=df['Winner'])


# In[ ]:


plt.figure(figsize=(19,10))
sns.countplot(df['Education'],hue=df['Winner'])


# In[ ]:


plt.figure(figsize=(19,10))
sns.countplot(df['Category'],hue=df['Winner'])


# In[ ]:


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestNeighbors,KNeighborsClassifier
from sklearn.metrics import accuracy_score


# In[ ]:


l = df.select_dtypes('object').columns
lb = LabelEncoder()
for i in l:
    df[i] = lb.fit_transform(df[i])


# In[ ]:





# In[ ]:


df.drop(['Name'],axis=1,inplace=True)
X = df.drop('Winner',axis=1)
y = df['Winner']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:


models = [["LogisticRegression",LogisticRegression()],["DecisionTreeClassifier",DecisionTreeClassifier()],["ExtraTreeClassifier",ExtraTreeClassifier()],["GaussianNB",GaussianNB()],["KNeighborsClassifier",KNeighborsClassifier()]]
model_prediction = []


# In[ ]:


for i in models:
    model = i[1]
    model.fit(X_train,y_train)
    predict = model.predict(X_test)
    score = (accuracy_score(predict,y_test))
    model_prediction.append([i[0],score])


# In[ ]:


model_prediction = pd.DataFrame(model_prediction)
model_prediction.columns = ["Model Name","Score"]
model_prediction.sort_values(by='Score',ascending=False)


# In[ ]:




