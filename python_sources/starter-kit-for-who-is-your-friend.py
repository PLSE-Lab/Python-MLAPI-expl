#!/usr/bin/env python
# coding: utf-8

# # Please give it a vote up! if you find it insightfull

# this is just a basic overview of modelling

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# import libraries for plotting
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# ignore warnings
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df = pd.read_csv("/kaggle/input/whoisafriend/train.csv")


# lets check the shape of the dataset given

# In[ ]:


df.shape


# In[ ]:


df.info()


# this means that there are no null values

# In[ ]:


df.describe()


# In[ ]:


df["ID"].head()


# it is of no use as it's unique for every entry and has no significance

# In[ ]:


df.drop("ID",axis=1,inplace=True)


# checking the no of unique values in Person A column

# In[ ]:


len(df['Person A'].unique()) 


# Thus there are total 100 person in this column

# In[ ]:


df['Person A'].value_counts() 


# and every single person has more than 200-400 relations with others

# In[ ]:


df['Interaction Type'].unique() 


# there are 6 different type of Interaction

# In[ ]:



plt.figure(figsize=(20,6))
sns.countplot(df['Interaction Type'],hue=df['Friends'])


# In[ ]:


len(df['Moon Phase During Interaction'].unique()) 


# there are 8 different type of Moon phase 

# In[ ]:


plt.figure(figsize=(20,6))
sns.countplot(df['Moon Phase During Interaction'],hue=df['Friends'])
plt.show()


# # Setting feature vector and target variable 

# In[ ]:


X= df.drop(['Friends'],axis=1)


# In[ ]:


y = df['Friends']


# In[ ]:


X.head(4)


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# # Feature Engineering

# Encode categorical variables

# In[ ]:



from sklearn import preprocessing
categorical = ['Person A', 'Person B', 'Interaction Type', 'Moon Phase During Interaction']

for feature in categorical:
    le = preprocessing.LabelEncoder()
    X_train[feature] = le.fit_transform(X_train[feature])
    X_test[feature] = le.transform(X_test[feature])


# In[ ]:


X_train.head()


# # Feature Scaling

# In[ ]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X.columns)

X_test = pd.DataFrame(scaler.transform(X_test), columns = X.columns)


# In[ ]:


X_train.head()


# In[ ]:


X_test.head()


# # Logistic Regression model with all features

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

logreg = LogisticRegression(penalty='l2')
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print('Logistic Regression accuracy score with all 6 the features: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# In[ ]:


logreg.score(X_train, y_train)


# # PCA

# In[ ]:


from sklearn.decomposition import PCA
pca = PCA()
temp_X_train= X_train
temp_X_train = pca.fit_transform(temp_X_train)
pca.explained_variance_ratio_


# In[ ]:


plt.figure(figsize=(8,6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlim(0,5,1)
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()
# ploting cumulative explained variance ratio with number 
# of components to show how variance ratio varies with number of components.


# As we can see every feature adds significant variance except one
# droping one variable in this case is a good idea! 
