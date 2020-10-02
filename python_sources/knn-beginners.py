#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix


# ### Import the Dataset
# Here in this case, we have imported the Iris dataset from Kaggle.

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


# ### Data Exploration
# Now, in this step, you will need to identify and understand about various parameters in your dataset. Also plot some visualizations to understand about the distribution of your dataset.
# 1. Find the Dimensions of your Data
# 2. Print the head and tail of your data to understand various features of your dataset
# 3. Identify the information about your dataset (like column names, datatypes etc.)
# 4. Describe the summary features of your Dataset (Mean, Count, Quantiles etc.)
# 5. Plot all the required Visualizations.

# In[ ]:


data = pd.read_csv('../input/iris-classifier-with-knn/Iris.csv')
data.shape


# In[ ]:


data.columns


# In[ ]:


data.head()


# In[ ]:


data.tail()


# In[ ]:


data.describe()


# ### Data Preprocessing

# Now, Find out if your Dataset has any null values.Because your model will not give you proper accuracy when your data has NA's in them. 

# In[ ]:


data.isnull().sum()


# So, it looks like our dataset does not have any Missing Values. So now, we will look if there are any outliers in the dataset using the Boxplot Visualizations and identify if we have any outliers in the data.

# In[ ]:


sns.boxplot(y=data['SepalLengthCm'])


# In[ ]:


sns.boxplot(y=data['SepalWidthCm'])


# Now, Sepal Width Feature looks like it has some outliers in it's data. So, now we have to preprocess these outliers for proper funtioning of the dataset to create a Machine Learning Model

# In[ ]:


sns.boxplot(y=data['PetalLengthCm'])


# In[ ]:


sns.boxplot(y=data['PetalWidthCm'])


# Now, before treating the outliers, we just look at the Feature description of the target parameter "Species". This shows that there are 3 types of species and are of equal in number.
# 1. Versicolor = 50
# 2. Setosa = 50
# 3. Virginica = 50

# In[ ]:


data['Species'].value_counts()


# In[ ]:


data.shape


# Seperate the Predictor Variables and the Target Variables

# In[ ]:


features = data.drop('Species', axis=1)
target = data['Species']


# ### Outlier Detection
# Z-Score function in the statistics is used to identify the outliers in your dataset

# In[ ]:


z = np.abs(stats.zscore(features))
print(z)


# In[ ]:


threshold = 3
print(np.where(z>threshold))


# In[ ]:


data_new = features[z>threshold]
print(data_new)


# So, from our Exploration, it is observed that a datapoint with ID=16 is an outlier. So, after detecting the outlier you can either remove the outlier datapoint or replace the outlier value with some benchmark value. This process depends on the dataset you are choosing and the type of problem.

# ### Pair Plot
# Now, Let's identify the distribution of datapoints based on the multi-dimensional plots as shown below

# In[ ]:


sns.pairplot(data, hue='Species')


# From the visulaization data from Pairplot, we identified that "Setosa" species are clearly seperated from the other two. But There is some overlapping in the datapoints between "Versicolor" and "Virginica"

# ### Scaling
# It's time for the predictor variables to be scaled down to a common value. We use standard scaler technique because the distribution is Standard Normal Distribution.

# In[ ]:


scale = StandardScaler()


# In[ ]:


scale.fit(features)


# In[ ]:


scaled_features=scale.transform(features)


# In[ ]:


data_new = pd.DataFrame(scaled_features)
data_new.head(3)


# ### Split the Data
# So, finally we scaled down the data and the next step is to split the dataset into train and test sets

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(data_new, target, test_size=0.25, random_state=45)


# In[ ]:


x_train.shape


# In[ ]:


x_train.head()


# In[ ]:


x_test.head()


# In[ ]:


y_train.head()


# In[ ]:


y_test.head()


# ### Model Train
# Once after you have splitted the dataset into training and test dataset, our task is to train the model with your training data. Since, I dont know the value of K correctly, I estimate the value of K as 1 and train the model. Later on you can find the value of K using the Elbow method and replace it with K to improve your model accuracy.

# In[ ]:


model = KNeighborsClassifier(n_neighbors=1)


# In[ ]:


model.fit(x_train, y_train)


# In[ ]:


pred = model.predict(x_test)


# In[ ]:


pred


# Successfully, we have predicted the values of the test dataset and now, our next task is to know if our model has predicted correctly or not. For knowing this, you have to evaluate your model with various performance metrics like Confusion Matrix and Classification Report.

# ### Confusion Matrix

# In[ ]:


confusion_matrix(y_test, pred)


# In[ ]:


print(classification_report(y_test, pred))


# ### Elbow Method
# To identify the optimal value of 'K', we need to perform Elbow method.

# In[ ]:


error_rate = []
for i in range(1,40):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(x_train,y_train)
    pred_i = model.predict(x_test)
    error_rate.append(np.mean(pred_i!=y_test))


# In[ ]:


plt.figure(figsize=(15,6))
plt.plot(range(1,40),error_rate, color='red',linestyle='dashed', marker='o',markerfacecolor='blue', markersize=8)
plt.title("Elbow Graph")
plt.xlabel("K-Value")
plt.ylabel("Error Rate")


# In[ ]:


model = KNeighborsClassifier(n_neighbors=21)

model.fit(x_train,y_train)
pred = model.predict(x_test)

print('WITH K=21')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


# In[ ]:




