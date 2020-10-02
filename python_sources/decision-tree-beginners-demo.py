#!/usr/bin/env python
# coding: utf-8

# ### Decision Tree Classifier

# #### Import all the required packages

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


data = pd.read_csv("../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")


# In[ ]:


data.info()


# In[ ]:


data.shape


# In[ ]:


data.isnull().sum()


# In[ ]:


z=data['quality'].value_counts()
z


# In[ ]:


sns.set(style='darkgrid')
plt.figure(figsize=(15,6))
sns.barplot(x = 'quality', y = 'fixed acidity', data = data)
plt.xlabel("Quality")
plt.ylabel("Fixed Acidity")
plt.title("Quality vs Fixed Acidity")


# In[ ]:


data.columns


# In[ ]:


sns.set(style='darkgrid')
plt.figure(figsize=(15,6))
sns.scatterplot(x='fixed acidity', y='volatile acidity', data = data, hue='quality', palette='Set1')


# In[ ]:


sns.set(style='darkgrid')
plt.figure(figsize=(15,6))
sns.kdeplot(data['quality'], shade=True)


# ### Data Preprocessing

# In[ ]:


correlation = data.corr()
correlation


# In[ ]:


plt.figure(figsize=(15,10))
sns.heatmap(correlation, annot=True, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm')


# In[ ]:


plt.figure(figsize=(15,10))
sns.pairplot(data, hue='quality')


# ### Outlier Detection

# In[ ]:


from scipy import stats


# In[ ]:


sns.catplot(y='quality', data=data, kind='box')


# In[ ]:


sns.catplot(y='pH', data=data, kind='box')


# In[ ]:


sns.boxplot('quality', 'fixed acidity', data = data)


# In[ ]:


sns.boxplot('quality', 'volatile acidity', data = data)


# In[ ]:


sns.boxplot('quality', 'citric acid', data = data)


# In[ ]:


#next we shall create a new column called Review. This column will contain the values of 1,2, and 3. 
#1 - Bad
#2 - Average
#3 - Excellent
#This will be split in the following way. 
#1,2,3 --> Bad
#4,5,6,7 --> Average
#8,9,10 --> Excellent
#Create an empty list called Reviews
reviews = []
for i in data['quality']:
    if i >= 1 and i <= 3:
        reviews.append('1')
    elif i >= 4 and i <= 7:
        reviews.append('2')
    elif i >= 8 and i <= 10:
        reviews.append('3')
data['Reviews'] = reviews


# In[ ]:


data['Reviews'].unique()


# ### Reducing the Dimensions of the Data

# In[ ]:


x = data.iloc[:,:11]
y = data['Reviews']


# In[ ]:


from sklearn.decomposition import PCA


# In[ ]:


pca = PCA()


# In[ ]:


x_pca = pca.fit_transform(x)


# In[ ]:


x_pca.shape


# In[ ]:


plt.figure(figsize=(10,10))
plt.plot(np.cumsum(pca.explained_variance_ratio_), 'ro-')
plt.grid()


# In[ ]:


pca_new = PCA(n_components=8)
x_new = pca_new.fit_transform(x)


# In[ ]:


x_new.shape


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x_new, y, test_size = 0.20, random_state=40)


# In[ ]:


x_train.shape


# In[ ]:


x_test.shape


# In[ ]:


y_train.shape


# In[ ]:


y_test.shape


# In[ ]:


model = DecisionTreeClassifier()


# In[ ]:


model.fit(x_train, y_train)


# In[ ]:


predict = model.predict(x_test)


# In[ ]:


predict


# ### Performance Metrics

# After training the model and predicting the necessary information from that model, our next step is to find out how well our model works. So, for this, we have various performance metrics like Confusion Matrix, Accuracy and Classification Report.

# In[ ]:


confusion_matrix = confusion_matrix(y_test, predict)
confusion_matrix


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predict)
accuracy

