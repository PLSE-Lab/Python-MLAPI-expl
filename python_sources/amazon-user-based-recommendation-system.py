#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 


# In[ ]:


amazon= pd.read_csv(r"../input/amazon-movie-ratings/Amazon.csv")


# In[ ]:


amazon_pd = pd.DataFrame(amazon)


# In[ ]:


amazon.head()


# In[ ]:


amazon.shape


# In[ ]:


amazon.size


# In[ ]:


amazon.describe()


# In[ ]:


#maximum number of views 
amazon.describe().T["count"].sort_values(ascending = False)[0:6]


# In[ ]:


amazon.index


# In[ ]:


amazon.columns


# In[ ]:


Amazon_filtered = amazon.fillna(value=0)
Amazon_filtered


# In[ ]:



Amazon_filtered1 = Amazon_filtered.drop(columns='user_id')
Amazon_filtered1.head()


# In[ ]:


Amazon_filtered1.describe()


# In[ ]:


Amazon_max_views = Amazon_filtered1.sum()
Amazon_max_views


# In[ ]:


#finding maximum sum of ratings 
max(Amazon_max_views)


# In[ ]:



Amazon_max_views.head()
Amazon_max_views.tail()


# In[ ]:


Amazon_max_views.index


# In[ ]:


#finding which movie has maximum views/ratings
max_views= Amazon_max_views.argmax()
max_views


# In[ ]:


#checking whether that movie has max views/ratings or not 
Amazon_max_views['Movie127']


# In[ ]:


sum(Amazon_max_views)


# In[ ]:


len(Amazon_max_views.index)


# In[ ]:


#the average rating for each movie
Average_ratings_of_every_movie=sum(Amazon_max_views)/len(Amazon_max_views.index)
Average_ratings_of_every_movie


# In[ ]:


#the average rating for each movie (alternative way )
Amazon_max_views.mean()


# In[ ]:


Amazon_df = pd.DataFrame(Amazon_max_views)
Amazon_df.head()


# In[ ]:


Amazon_df.columns=['rating']


# In[ ]:


Amazon_df.index


# In[ ]:


Amazon_df.tail()


# In[ ]:


#top 5 movie ratings 
Amazon_df.nlargest(5,'rating')


# In[ ]:


#top 5 movies having least audience 
Amazon_df.nsmallest(5,'rating')


# In[ ]:


melt_df=amazon_pd.melt(id_vars= amazon.columns[0],value_vars=amazon.columns[1:],var_name='Movie',value_name='rating')


# In[ ]:


melt_df


# In[ ]:


melt_df.shape


# In[ ]:


melt_filtered = melt_df.fillna(0)
melt_filtered.shape


# In[ ]:


import surprise


# In[ ]:


from surprise import Reader
from surprise import Dataset
from surprise import SVD
from surprise.model_selection import train_test_split


# In[ ]:


reader = Reader(rating_scale=(-1,10))


data = Dataset.load_from_df(melt_df.fillna(0), reader=reader)


# In[ ]:


#Divide the data into training and test data
trainset, testset = train_test_split(data, test_size=0.25)


# In[ ]:


algo = SVD()


# In[ ]:


#Building a model
algo.fit(trainset)


# In[ ]:


#Make predictions on the test data
predict= algo.test(testset)


# In[ ]:


from surprise.model_selection import cross_validate


# In[ ]:


cross_validate(algo,data,measures=['RMSE','MAE'],cv=3,verbose=True)


# In[ ]:


user_id='A1CV1WROP5KTTW'
Movie='Movie6'
rating='5'
algo.predict(user_id,Movie,r_ui=rating)
print(cross_validate(algo,data,measures=['RMSE','MAE'],cv=3,verbose=True))

