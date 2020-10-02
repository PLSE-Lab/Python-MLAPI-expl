#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


df=pd.read_csv("../input/diabetes/diabetes.csv")


# In[ ]:


pd.set_option("max_columns",300)
df


# In[ ]:


df.info()


# In[ ]:


for i in df.columns:
    plt.scatter(df["Outcome"],df[i])
    plt.title(i)
    plt.show()


# In[ ]:


sns.pairplot(df)


# In[ ]:


df[df["SkinThickness"]==0]


# In[ ]:


df["SkinThickness"].mean()


# In[ ]:


df["Insulin"].mean()


# In[ ]:


df.corr()


# In[ ]:


plt.plot(df["Age"],df["Pregnancies"],"d")


# In[ ]:


for i in df.columns:
    sns.distplot(df[i])
    plt.title(i)
    plt.show()


# In[ ]:


for i in df.columns:
    sns.stripplot(df["Outcome"],df[i],jitter=True)
    plt.title(i)
    plt.show()


# In[ ]:


# replacing zero values with the mean of the column
df['BMI'] = df['BMI'].replace(0,df['BMI'].mean())
df['BloodPressure'] = df['BloodPressure'].replace(0,df['BloodPressure'].mean())
df['Glucose'] = df['Glucose'].replace(0,df['Glucose'].mean())
df['Insulin'] = df['Insulin'].replace(0,df['Insulin'].mean())
df['SkinThickness'] = df['SkinThickness'].replace(0,df['SkinThickness'].mean())


# In[ ]:


for i in df.columns:
    sns.distplot(df[i])
    plt.title(i)
    plt.show()


# In[ ]:


x=df.iloc[:,:8]
y=df["Outcome"]
x,y


# In[ ]:


from sklearn.preprocessing import StandardScaler 
scalar = StandardScaler()
X_scaled = scalar.fit_transform(x)


# In[ ]:


X_scaled


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size= 0.50, random_state = 355)


# In[ ]:


x_train


# In[ ]:


y_train


# In[ ]:


from sklearn.linear_model  import LogisticRegression
log=LogisticRegression()
log.fit(x_train,y_train)


# In[ ]:


y_pred = log.predict(x_test)
y_pred


# In[ ]:


y_prob=log.predict_proba(x_test)
y_prob=y_prob[:,1]


# In[ ]:


from sklearn.metrics import confusion_matrix , classification_report


# In[ ]:


print(classification_report(y_test,y_pred))


# In[ ]:


con_mat=confusion_matrix(y_test,y_pred)
con_mat


# In[ ]:


true_positive = con_mat[0][0]
false_positive = con_mat[0][1]
false_negative = con_mat[1][0]
true_negative = con_mat[1][1]


# In[ ]:


recall=true_positive/(true_positive+false_negative)
recall


# In[ ]:


precision=true_positive/(true_positive+false_positive)
precision


# In[ ]:


f1_score=2*recall*precision/(recall+precision)
f1_score


# The result is depend on thresold value which is normally be 0.5 , if we will change the thresold value then result may be 
# differ from the current one.

# In[ ]:




