#!/usr/bin/env python
# coding: utf-8

# # MultiLayerPerceptron based classification

# ### Importing libraries

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Reading data from csv

# In[ ]:


df=pd.read_csv('../input/voice.csv')


# ### Exploratory Data Analysis

# In[ ]:


df.head()


# In[ ]:


df.info()


# no null values

# In[ ]:


df.describe()


# In[ ]:


df['label'].value_counts()


# #### Converting label column from str to int64 

# In[ ]:


le=LabelEncoder()


# In[ ]:


le.fit_transform(df['label'])


# In[ ]:


correlation = df.corr()
plt.figure(figsize=(15,15))
sns.heatmap(correlation, square=True, annot=True ,cmap='coolwarm')


# ### (meanfreq,centroid) and (maxdom,dfrange) show correlation value 1 thus removing those columns for simplification
# ### There are other high correlations in the data too like between (kurt,skew) and (meanfreq,median) but not removing those because a neural network deals with such correlations by adjusting it's weights and bias values accordingly.

# In[ ]:


features=df.drop(['label','centroid','dfrange'],axis=1)
target=df['label']

sc=StandardScaler()
sc.fit_transform(features)


# ### Training a MLP model with 2 hidden layers of 8 and 4 neurons respectively , max iterations of 2000 until convergence , with a hyperbolic tangent function as activation and Limited-memory BFGS optimization.
# above values are chosen via some trial and error

# In[ ]:


(X_train,X_test,Y_train,Y_test)=train_test_split(features,target,test_size=0.30)


# In[ ]:


mlp_classifier= MLPClassifier(activation='tanh',hidden_layer_sizes=(8,4),max_iter=2000,solver='lbfgs',random_state=1)
mlp_classifier.fit(X_train,Y_train)


# In[ ]:


mlp_classifier.score(X_test,Y_test)


# ### 98% accuracy in prediction achieved.

# In[ ]:




