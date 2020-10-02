#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


mov=pd.read_csv("../input/movie_metadata.csv")
mov.columns.values


# In[ ]:


labels=mov["imdb_score"]
mov.drop(["imdb_score", "aspect_ratio", "movie_imdb_link"], inplace=True, axis=1)


# In[ ]:


numeric_features=mov._get_numeric_data().columns.values.tolist()
print(numeric_features)
text_features=mov.columns.values.tolist()
text_features=[i for i in text_features if i not in numeric_features]
print(text_features)

numeric_features.remove("title_year") ### This  is categorical all the others can be considered continuous 
###(See about facenumber_in_poster tho)
string_features=["movie_title", "plot_keywords"]
categorical_features=[i for i in text_features if i not in string_features]
categorical_features.append("title_year")
print(categorical_features)


# **Taking care of numerical features (imputation and scaling)**

# In[ ]:


mov[numeric_features]


# In[ ]:


from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler 
## we use standard scaler to keep as much variance as possible (compared to minmax)
imp=Imputer(missing_values='NaN',strategy="most_frequent", axis=0)
mov[numeric_features]=imp.fit_transform(mov[numeric_features])

scl=StandardScaler()
mov[numeric_features]=scl.fit_transform(mov[numeric_features])

mov[numeric_features]


# **Taking care of categorical variables (creating dummies)**

# In[ ]:


for feat in categorical_features:
    mov=pd.concat([mov, pd.get_dummies(mov[feat], prefix=feat, dummy_na=True)],axis=1)
  


# In[ ]:


cat_dummies=[i for i in mov.columns.values.tolist() if i not in numeric_features]
cat_dummies=[i for i in cat_dummies if i not in text_features]
cat_dummies.remove("title_year")
cat_dummies[-5:]


# **Taking care of the text variables**

# In[ ]:


mov["movie_title"]


# import re
# for i in range(len(mov["movie_title"])):
#     mov["movie_title"][i]=re.sub("[^a-zA-Z]", " ", mov["movie_title"][i]) 
#     mov["movie_title"][i]=mov["movie_title"][i].lower()

# from sklearn.feature_extraction.text import CountVectorizer
# 
# cv=CountVectorizer(stop_words="english",max_features=500)
# movie_title_words=cv.fit_transform(mov["movie_title"])
# 
# movie_title_words=movie_title_words.toarray()
# 
# words = cv.get_feature_names()
# words=["Title_"+w for w in words]
# 
# words_title=pd.DataFrame(movie_title_words, columns=words)
# words_title

# mov["plot_keywords"]=mov["plot_keywords"].fillna("None")
# 
# def token(text):
#     return(text.split("|"))
# 
# cv=CountVectorizer(max_features=2000,tokenizer=token )
# plot_keywords_words=cv.fit_transform(mov["plot_keywords"])
# 
# plot_keywords_words=plot_keywords_words.toarray()
# 
# words = cv.get_feature_names()
# words=["Keyword_"+w for w in words]
# 
# keywords=pd.DataFrame(plot_keywords_words, columns=words)
# keywords

# In[ ]:



X=pd.concat([mov[numeric_features], mov[cat_dummies]], axis=1)


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.set(context="paper", font="monospace")
corrmat = mov[numeric_features].corr()
f, ax = plt.subplots(figsize=(12, 9))

# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=.8, square=True)


# In[ ]:


mov["reviews"]=(mov['num_critic_for_reviews']+mov["num_voted_users"]+mov["gross"]
                +mov["num_user_for_reviews"]+mov["movie_facebook_likes"])/5
mov["likes"]=(mov["cast_total_facebook_likes"]+mov["actor_1_facebook_likes"]+
              mov["actor_2_facebook_likes"]+mov["actor_3_facebook_likes"])/4

num_features_2=[x for x in numeric_features if x not in ["movie_facebook_likes","num_user_for_reviews","cast_total_facebook_likes","actor_1_facebook_likes","actor_2_facebook_likes",
                                      "actor_3_facebook_likes",'num_critic_for_reviews',"num_voted_users",
                                      "gross"]]
num_features_2.extend(["reviews", "likes"])
print(num_features_2)


# In[ ]:


sns.set(context="paper", font="monospace")
corrmat = mov[num_features_2].corr()
f, ax = plt.subplots(figsize=(12, 9))

# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=.8, square=True)


# In[ ]:


X=mov[num_features_2]


# In[ ]:


y=labels


# In[ ]:


from sklearn.cross_validation import train_test_split 
X_train, X_test, y_train, y_test = train_test_split( X, y,  test_size=0.2, random_state=42)


# In[ ]:


from sklearn.svm import SVR
svr=SVR()
svr.fit(X_train, y_train)


# In[ ]:


svr.score(X_test, y_test)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
RFR=RandomForestRegressor()
parameters={"n_estimators":[120, 300, 500, 800,1200], "max_depth":[5,8,15,25], 
            "min_samples_split":[1,2,5,10,15],"max_features":("log2", "sqrt") }


# In[ ]:


from sklearn.grid_search import GridSearchCV
clf = GridSearchCV(RFR, parameters)
clf.fit(X_train,y_train)
report(clf.grid_scores_)


# In[ ]:




