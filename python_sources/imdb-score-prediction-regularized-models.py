#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/movie_IMDB.csv')


# In[ ]:


data.shape


# In[ ]:


data.columns.tolist()


# In[ ]:


data.fillna(value = 0,axis = 1,inplace = True) #Filling NaN values with 0


# In[ ]:


data.head(2)


# In[ ]:


df_genres = pd.DataFrame(data['genres'])
df_genres[:3]


# In[ ]:


df_genres = pd.DataFrame(df_genres.genres.str.split('|').tolist(),columns = ["Genre_"+str(i) for i in  range(0,8)] )
data.drop('genres',inplace = True, axis = 1)


# In[ ]:


data = data.merge(df_genres,left_index = True,right_index = True)


# In[ ]:


data.director_name.nunique() # No of unique Directors


# In[ ]:


data.drop(axis = 1,labels = ['Genre_6','Genre_7'],inplace = True)


# In[ ]:


data.rename(columns = {'director_facebook_likes':'dir_fb_likes','actor_1_facebook_likes':'actor_1_fb','actor_3_facebook_likes':'actor_3_fb','actor_2_facebook_likes':'actor_2_fb'},inplace = True)


# In[ ]:


data.drop(axis = 1, labels = ['Genre_4','Genre_5'],inplace = True)


# In[ ]:


data['imdb_score'].describe()


# In[ ]:


sns.boxplot(x= 'imdb_score',data = data,orient = 'v',saturation = 1)
#sns.swarmplot(data = data['imdb_score'])


# In[ ]:


plt.xlabel('IMDB SCORE')
plt.ylabel('% of Movies')
plt.hist(data['imdb_score'],normed = 1,edgecolor='black',linewidth=1.2)


# In[ ]:



plt.scatter(data['dir_fb_likes'],data['imdb_score'])
plt.show()

plt.scatter(data['budget'],data['imdb_score'])
plt.show()


# In[ ]:


a = data.groupby('language').agg({'imdb_score':'mean'}).reset_index().sort_values(by = 'imdb_score',ascending = False)[:20]
plt.figure(figsize=(16,6))
sns.barplot(x="language", y="imdb_score", data=a)


# In[ ]:


data.columns.tolist()


# In[ ]:



c = data.groupby('director_name').agg({'gross':'mean'}).reset_index().sort_values(by = 'gross',ascending = False)[:10]
plt.figure(figsize=(15,6))
sns.barplot(x="director_name", y="gross", data=c)


#  **Linear Regression***

# In[ ]:


numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric_data = data.select_dtypes(include=numerics)
features = numeric_data[['num_critic_for_reviews',
 'duration',
 'dir_fb_likes',
 'actor_3_fb',
 'actor_1_fb',
 'gross',
 'facenumber_in_poster',
 'num_user_for_reviews',
 'budget',
 'actor_2_fb',
 'movie_facebook_likes',
 ]]
target = numeric_data['imdb_score']


# **KNN Regression (R-squared test score: 0.203)**

# In[ ]:


from sklearn.neighbors import KNeighborsRegressor

X_train, X_test, y_train, y_test = train_test_split(features, target, random_state = 0)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
knnreg = KNeighborsRegressor(n_neighbors = 5).fit(X_train_scaled, y_train)

print('R-squared test score: {:.3f}'
     .format(knnreg.score(X_test_scaled, y_test)))


# **Linear Regression [R-squared score (training): 0.151]**

# In[ ]:


from sklearn.linear_model import LinearRegression

linreg = LinearRegression().fit(X_train_scaled, y_train)
print('R-squared score (training): {:.3f}'
     .format(linreg.score(X_train_scaled, y_train)))
print('R-squared score (test): {:.3f}'
     .format(linreg.score(X_test_scaled, y_test)))


# **Ridge Regression**

# In[ ]:


from sklearn.linear_model import Ridge
ridge = Ridge(alpha=10.0).fit(X_train_scaled, y_train)
print('R-squared score (training): {:.3f}'
     .format(ridge.score(X_train_scaled, y_train)))
print('R-squared score (test): {:.3f}'
     .format(ridge.score(X_test_scaled, y_test)))


# In[ ]:


import seaborn as sns 
#draw correlation matrix of features

corr = train_x_std.corr()
#draw the heatmap of correlation to identify any features that can be dropped
plt.figure(figsize=(18,18))
sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 15},
            cmap= 'YlGnBu') 


# In[ ]:


high_score_data= data.loc[data['imdb_score']>=8] #only count of movies with score>= 8.0
sns.countplot(high_score_data['imdb_score'],label='imdb_score')


# In[ ]:




