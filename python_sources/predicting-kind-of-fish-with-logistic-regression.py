#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Basic Operations

# In[ ]:


df=pd.read_csv('../input/fish-market/Fish.csv')
df.head()


# In[ ]:


df.info()


# In[ ]:


df.isnull().sum()


# We have no empty data. All columns except 'Species' are float, which is in fact logically.

# In[ ]:


df.describe()


# In[ ]:


df.Species.value_counts()


# # Exploratory Data Analysis
# 
# Let's *pairplot* of all columns

# In[ ]:


px.pie(df,names='Species')


# In[ ]:


sns.pairplot(df,hue='Species')


# We can see from this graph that weight of the fish has linear correlation with its length and width

# In[ ]:


plt.figure(figsize=(12,5))
plt.plot(df['Length1'],label='Length Vertical')
plt.plot(df['Length2'],label='Length Diagonal')
plt.plot(df['Length3'],label='Length Cross')
plt.legend()


# Generally, pattern of vertical, diagonal and cross length do not differ, but mostly numerically.

# In[ ]:


plt.figure(figsize=(12,5))
#plt.plot(df['Weight'],label='Weight')
plt.plot(df['Width'],label='Width')
plt.plot(df['Height'],label='Height')
plt.legend()


# In[ ]:


px.scatter(df,x='Height',y='Length1',size='Weight',color='Species')


# From thorough analysis we can see that weight of a fish no matter its species has different values. However, species such as Smelt and Bream have avarage weight distribution.

# In[ ]:


px.scatter_3d(df,x='Length1',y='Height',z='Width',size='Weight',color='Species')


# # Logistic Regression
# 
# I will use Logistic Regression for classification, and further predicting kind of fish according to its weight, vertical, diagonal and corss lengths, height and width.

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[ ]:


X=df.loc[:,'Weight':'Width']
y=df['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[ ]:


print('Dimenstions of Train Set of features:',np.shape(X_train))
print('Dimensions of Train Set of the target:',np.shape(y_train))
print('Dimenstions of Test Set of features:',np.shape(X_test))
print('Dimenstions of Test Set of the target:',np.shape(y_test))


# In[ ]:


#first I will use default values in Logistic Regression
logreg=LogisticRegression(max_iter=100000).fit(X_train,y_train)


# In[ ]:


print('Model intercept: ', logreg.intercept_)
print('Model coefficients: ',logreg.coef_)


# In[ ]:


print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))


# Model has done pretty well.

# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


pred1=logreg.predict(X_test)
print(confusion_matrix(y_test,pred1))
print('\n')
print(classification_report(y_test,pred1))


# In[ ]:


logreg.predict([[400,26.0,30.0,37.0,12.3,5.0]])


# # Conclusion
# 
# Despite we have quite small dataset, our model had done very well with Training set score: 0.973 and Test set score: 0.938.
# 
# However, I should here make a small but important remark about performance of a model and dataset at all. Let's look at probability theory in order to undersdant the mislead with good performance.
# 
# If we had equally distributed number of species in our dataset, then probability to 'guess' correct one woul be 1/7. In contrast, we have different values of species, totatl comprising of 159 species. So for guessing 'Perch' probability will be 56/159, and other species 35/159, 20/159, 17/159,14/159, 11/159 and 6/159. Quite small number of species in order to have precise demonstration of Logistic Regression. In addition, having several distinc features such as weight, length etc. probability to find correct kind will significantly increase. One way to show high potential is having larger number of species.
# 
# If there is an error in my process, please note about that, because I am beginner and every advice will increase my experience!
# 
# Thank you!

# In[ ]:




