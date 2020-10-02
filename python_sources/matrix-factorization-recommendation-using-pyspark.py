#!/usr/bin/env python
# coding: utf-8

# **Matrix factorization** is a class of collaborative filtering algorithms used in recommender systems. **Matrix factorization** approximates a given rating matrix as a product of two lower-rank matrices.
# It decomposes a rating matrix R(nxm) into a product of two matrices W(nxd) and U(mxd).
# 
# \begin{equation*}
# \mathbf{R}_{n \times m} \approx \mathbf{\hat{R}} = 
# \mathbf{V}_{n \times k} \times \mathbf{V}_{m \times k}^T
# \end{equation*}

# In[ ]:


get_ipython().system('pip install pyspark       #installing pyspark')


# #### Importing the necessary libraries

# In[ ]:


from pyspark import SparkContext, SQLContext   # required for dealing with dataframes
import numpy as np
from pyspark.ml.recommendation import ALS      # for Matrix Factorization using ALS 


# In[ ]:


sc = SparkContext()      # instantiating spark context 
sqlContext = SQLContext(sc) # instantiating SQL context 


# #### Step 1. Loading the data into a PySpark dataframe

# In[ ]:


jester_ratings_df = sqlContext.read.csv("/kaggle/input/jester-17m-jokes-ratings-dataset/jester_ratings.csv",header = True, inferSchema = True)


# In[ ]:


jester_ratings_df.show(5)


# In[ ]:


print("Total number of ratings: ", jester_ratings_df.count())
print("Number of unique users: ", jester_ratings_df.select("userId").distinct().count())
print("Number of unique jokes: ", jester_ratings_df.select("jokeId").distinct().count())


# #### Step 2. Splitting into train and test part

# In[ ]:


X_train, X_test = jester_ratings_df.randomSplit([0.9,0.1])   # 90:10 ratio


# In[ ]:


print("Training data size : ", X_train.count())
print("Test data size : ", X_test.count())


# In[ ]:


X_train.show(5)


# In[ ]:


X_test.show(5)


# #### Step 3. Fitting an ALS model

# In[ ]:


als = ALS(userCol="userId",itemCol="jokeId",ratingCol="rating",rank=5, maxIter=10, seed=0, )
model = als.fit(X_train)


# In[ ]:


model.userFactors.show(5, truncate = False)  # displaying the latent features for five users


# #### Step 4. Making predictions

# In[ ]:


predictions = model.transform(X_test[["userId","jokeId"]])  # passing userId and jokeId from test dataset as an argument 


# In[ ]:


# joining X_test and prediction dataframe and also dropping the records for which no predictions made
ratesAndPreds = X_test.join(other=predictions,on=['userId','jokeId'],how='inner').na.drop() 
ratesAndPreds.show(5)


# #### Step 5. Evaluating the model

# In[ ]:


# converting the columns into numpy arrays for direct and easy calculations 
rating = np.array(ratesAndPreds.select("rating").collect()).ravel()
prediction = np.array(ratesAndPreds.select("prediction").collect()).ravel()
print("RMSE : ", np.sqrt(np.mean((rating - prediction)**2)))


# #### Step 6. Recommending jokes

# In[ ]:


# recommending top 3 jokes for all the users with highest predicted rating 
model.recommendForAllUsers(3).show(5,truncate = False)

