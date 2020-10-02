#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns #visualisation
import matplotlib.pyplot as plt #visualisation


# In[ ]:


df = pd.read_csv("../input/iris-flower-dataset/IRIS.csv")
df.shape


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df.isnull().sum()


# In[ ]:


print(df.duplicated().sum())

duplicateRowsDF = df[df.duplicated(keep='last')]
duplicateRowsDF


# In[ ]:


df = df.drop_duplicates()


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


df['species'].value_counts()


# **Detecting Outliers**

# In[ ]:


sns.boxplot(x=df.petal_length)


# In[ ]:


sns.boxplot(x=df.petal_width)


# In[ ]:


sns.boxplot(x=df.sepal_length)


# In[ ]:


sns.boxplot(x=df.sepal_width)


# **Remove Outlier**

# In[ ]:


Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[ ]:


df = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
df.shape


# **Finding overall Relation**

# In[ ]:


plt.figure(figsize=(20,10))
correlation = df.corr()
sns.heatmap(correlation,cmap="BrBG",annot=True)
correlation


# **Finding Relation between variable**

# In[ ]:


sns.pairplot(df, hue="species", size=2, markers=["o", "s", "D"])
plt.tight_layout()


# In[ ]:


df['petal_area'] = df.apply(lambda row: (row['petal_length'] * row['petal_width']), axis=1)
df['sepal_area'] = df.apply(lambda row: (row['sepal_length'] * row['sepal_width']), axis=1)


# In[ ]:


df.head()


# In[ ]:


plt.figure(figsize=(20,10))
correlation = df[['petal_area', 'sepal_area']].corr()
sns.heatmap(correlation,cmap="BrBG",annot=True)
correlation


# In[ ]:


sns.pairplot(df, hue="species", size=2.5, markers=["o", "s", "D"], vars=["petal_area", "sepal_area"])
plt.tight_layout()


# **Predict Species using Logistic Regression**

# In[ ]:


from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[ ]:


X = df[['petal_width', 'petal_length', 'petal_area']]
y = df[["species"]]

print(X.shape)
print(y.shape)


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)


# In[ ]:


lr = LogisticRegression()
lr.fit(x_train, y_train)

print(lr.coef_)
print(lr.intercept_)


# In[ ]:


y_pred = lr.predict(x_test)
lr.predict_proba(x_test)


# In[ ]:


print(classification_report(y_test,y_pred))


# In[ ]:


cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
ax.figure.colorbar(im, ax=ax)

# We want to show all ticks...
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       # ... and label them with the respective list entries
       xticklabels=y.species.unique(), yticklabels=y.species.unique(),
       title="Confusion Matrix",
       ylabel='True label',
       xlabel='Predicted label')

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
fmt = 'd'
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], fmt),
        ha="center", va="center",
        color="white" if cm[i, j] > thresh else "black")
fig.tight_layout()

