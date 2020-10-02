#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Predict the Song Popularity
# 
# ![spotify](https://www.ft.com/__origami/service/image/v2/images/raw/http%3A%2F%2Fcom.ft.imagepublish.upp-prod-us.s3.amazonaws.com%2F97fcb70e-447a-11ea-9a2a-98980971c1ff?fit=scale-down&source=next&width=700)
# 
# ### Spotify has become a recent music station preference. With recommendation system, it was better to search for your preferred music. The 50 most popular song is listed, and each song has their popularity. 
# 
# ### This data consisted of many parameters, and from the parameters we want to predict how popular the song is. The higher the value the more popular the song is.
# 
# ### This notebook consist of:
# ### 1. Exploratory and Data Analysis
# ### 2. Preprocessing and Feature Engineering
# ### 3. Machine learning modeling using SVM, decision tree, kNN, and Adaboost.
# ### 4. Some mathematical notes

# # 1. Loading Data, Exploratory, and Data Analysis
# 
# Here is popular spotify song. Then, see what inside the data and visualize it.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
data = pd.read_csv("/kaggle/input/top50spotify2019/top50.csv",encoding='ISO-8859-1')


# In[ ]:


data.head()


# ### Sorted by popularity

# In[ ]:


data.sort_values(by = ['Popularity'],ascending=False)


# In[ ]:


plt.figure(figsize = (15,7))
ax = sns.swarmplot(x="Genre", y="Popularity", data = data)
ax = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax = plt.title('Genre')


# ## Now, we want to know what genre is the most popular. We will make the genre plot.

# In[ ]:


plt.figure(figsize = (15,7))
ax = sns.countplot(x="Genre", data=data)
ax = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax = plt.title('Genre')


# In[ ]:


plt.figure(figsize = (15,7))
ax = sns.countplot(x="Artist.Name", data=data)
ax = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax = plt.title('Artist.Name')


# In[ ]:


data.columns


# In[ ]:


f, axes = plt.subplots(4, 2, figsize=(20,20))

sns.distplot(data["Beats.Per.Minute"],kde = False, ax=axes[0][0])
sns.distplot(data["Energy"],kde = False, ax=axes[0][1])
sns.distplot(data["Danceability"],kde = False, ax=axes[1][0])
sns.distplot(data["Loudness..dB.."],kde = False, ax=axes[1][1])
sns.distplot(data["Liveness"],kde = False, ax=axes[2][0])
sns.distplot(data["Valence."],kde = False, ax=axes[2][1])
sns.distplot(data["Length."],kde = False, ax=axes[3][0])
sns.distplot(data["Speechiness."],kde = False, ax=axes[3][1])


# In[ ]:


data.columns


# In[ ]:


import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set()

plt.figure(figsize=(20,20))
heatmap_data = data[['Beats.Per.Minute','Energy','Danceability','Loudness..dB..','Liveness','Valence.',
                   'Length.','Acousticness..','Speechiness.','Popularity']].corr()
ax = sns.heatmap(heatmap_data,annot=True)


# # 2. Feature Engineering
# 
# Preparing what we need for machine learning modeling. 
# We need:
# * Dummy variable for categorical data (Artist Name, Genre)
# * Treat track name as uniue value

# In[ ]:


data_ml = data.copy()


# In[ ]:


data_ml.head()


# * ### Dummy variable for artist and genre

# In[ ]:


data_ml['Artist.Name'] = data_ml['Artist.Name'].astype('category').cat.codes
data_ml['Genre'] = data_ml['Genre'].astype('category').cat.codes


# In[ ]:


data_ml.head()


# In[ ]:


X = data_ml.iloc[:,2:13].values
y = data_ml.iloc[:,13:].values


# In[ ]:


X


# In[ ]:


y


# ## Scaling
# 
# ![](https://i.stack.imgur.com/Z7ATR.png)
# 
# ![](https://scikit-learn.org/stable/_images/sphx_glr_plot_all_scaling_thumb.png)

# In[ ]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X = sc.fit_transform(X)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# # 3. Alright, then the data is ready to learn!

# ### 3.1 Support Vector Regression

# ![](https://scikit-learn.org/0.18/_images/sphx_glr_plot_svm_regression_001.png)

# In[ ]:


from sklearn.svm import SVC


# In[ ]:


clf = SVC(probability=True,kernel='poly',degree=4,gamma='auto')


# In[ ]:


clf.fit(X_train,y_train.ravel())


# In[ ]:


train_result = clf.predict(X_train)
test_result = clf.predict(X_test)


# In[ ]:


from sklearn.metrics import mean_squared_error

print('Training MSE: ', mean_squared_error(y_train, train_result))
print('Test MSE: ', mean_squared_error(y_test, test_result))


# In[ ]:


indices_train = np.arange(0,len(y_train),1)
indices_test = np.arange(0,len(y_test),1)


# In[ ]:


indices_train.shape


# In[ ]:


fig = plt.figure(figsize = (10,10))
plt.subplot(1, 2, 1)
ax1 = sns.scatterplot(indices_train, train_result.ravel(), label = 'train result')
ax1 = sns.scatterplot(indices_train, y_train.ravel(),label = 'actual data')
plt.title('SVR')
plt.subplot(1, 2, 2)
ax2 = sns.scatterplot(indices_test, test_result.ravel(), label = 'train result')
ax2 = sns.scatterplot(indices_test, y_test.ravel(),label = 'actual data')
plt.title('SVR')


# ## 3.2 KNN regression

# ![](https://miro.medium.com/max/405/0*BMFO6QFX55-oESwy.png)

# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
neigh = KNeighborsRegressor(n_neighbors=2)


# In[ ]:


knn_clf = neigh.fit(X_train, y_train.ravel())


# In[ ]:


train_result_knn = knn_clf.predict(X_train)
test_result_knn = knn_clf.predict(X_test)


# In[ ]:


from sklearn.metrics import mean_squared_error

print('Training MSE: ', mean_squared_error(y_train, train_result_knn))
print('Test MSE: ', mean_squared_error(y_test, test_result_knn))


# In[ ]:


fig = plt.figure(figsize = (10,10))
plt.subplot(1, 2, 1)
ax1 = sns.scatterplot(indices_train, train_result_knn.ravel(), label = 'train result')
ax1 = sns.scatterplot(indices_train, y_train.ravel(),label = 'actual data')
plt.title('KNN')
plt.subplot(1, 2, 2)
ax2 = sns.scatterplot(indices_test, test_result_knn.ravel(), label = 'train result')
ax2 = sns.scatterplot(indices_test, y_test.ravel(),label = 'actual data')
plt.title('KNN')


# ## 3.3 Decision tree regression

# ![](https://scikit-learn.org/stable/_images/sphx_glr_plot_tree_regression_001.png)

# In[ ]:


from sklearn.tree import DecisionTreeRegressor

dec_clf = DecisionTreeRegressor(max_depth=4)


# In[ ]:


dec_tree_clf = dec_clf.fit(X_train, y_train.ravel())


# In[ ]:


train_result_dec = dec_clf.predict(X_train)
test_result_dec = dec_clf.predict(X_test)


# In[ ]:


from sklearn.metrics import mean_squared_error

print('Training MSE: ', mean_squared_error(y_train, train_result_dec))
print('Test MSE: ', mean_squared_error(y_test, test_result_dec))


# In[ ]:


fig = plt.figure(figsize = (10,10))
plt.subplot(1, 2, 1)
ax1 = sns.scatterplot(indices_train, train_result_dec.ravel(), label = 'train result')
ax1 = sns.scatterplot(indices_train, y_train.ravel(),label = 'actual data')
plt.title('DT')
plt.subplot(1, 2, 2)
ax2 = sns.scatterplot(indices_test, test_result_dec.ravel(), label = 'train result')
ax2 = sns.scatterplot(indices_test, y_test.ravel(),label = 'actual data')
plt.title('DT')


# ## 3.4 AdaBoost Regression

# ![](https://scikit-learn.org/stable/_images/sphx_glr_plot_adaboost_regression_thumb.png)

# In[ ]:


from sklearn.ensemble import AdaBoostRegressor


# In[ ]:


regr = AdaBoostRegressor(random_state=0, n_estimators=100)


# In[ ]:


boost = regr.fit(X, y)


# In[ ]:


train_result_boost = boost.predict(X_train)
test_result_boost = boost.predict(X_test)


# In[ ]:


from sklearn.metrics import mean_squared_error

print('Training MSE: ', mean_squared_error(y_train, train_result_boost))
print('Test MSE: ', mean_squared_error(y_test, test_result_boost))


# In[ ]:


fig = plt.figure(figsize = (10,10))
plt.subplot(1, 2, 1)
ax1 = sns.scatterplot(indices_train, train_result_boost.ravel(), label = 'train result')
ax1 = sns.scatterplot(indices_train, y_train.ravel(),label = 'actual data')
plt.title('boost')
plt.subplot(1, 2, 2)
ax2 = sns.scatterplot(indices_test, test_result_boost.ravel(), label = 'train result')
ax2 = sns.scatterplot(indices_test, y_test.ravel(),label = 'actual data')
plt.title('boost')

