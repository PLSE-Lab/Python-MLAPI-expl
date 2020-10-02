#!/usr/bin/env python
# coding: utf-8

# Here we are going to see how we can use the Random Forest Classifier on a dataset.
# A Random Forest Classifier is an extension of Decision Tree Classifier. In this model we use several decison trees to create a **forest** or a group of outcomes where the final outcome is taken on vote.
# Look at the below image to visualize the working of the model.
# ![](http://www.globalsoftwaresupport.com/wp-content/uploads/2018/02/ggff5544hh.png)
# Try to find out on what are the benefits of using the Random Forest Classifier and where the model suits best.

# In[ ]:


import pandas as pd
import numpy as np
np.random.seed(0)


# In[ ]:


df = pd.read_csv('../input/IRIS.csv')
df.head()


# In this kernal we will not use the train_test_split(), we will try a different approach to split the dataset into train and test.
# Initially we will add a column to label which rows will be taken as train and the rest as test.

# In[ ]:


df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
df.head(10)


# As you can see we have used the np.random.uniform() to achieve this. (0,1, len(df)) <=75 suggests that we will assign 75% of the data as train and the remaining as test.
# Let's split the data now and check the count for each.

# In[ ]:


train, test = df[df['is_train'] == True], df[df['is_train'] == False]
print("Size of training data: ", len(train))
print("Size of test data: ", len(test))


# To train the model we need make sure all values are numerical.
# Our species column contains string, which is also the target(y) we are going to predict.
# Let's replace these values with numercials. We will use the factorize() for this.

# In[ ]:


y = pd.factorize(train['species'])[0]
y


# All other features are numbers. So now let's bring in our model.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier as RFC


# In[ ]:


features = df.columns[:4]
features


# In[ ]:


clf = RFC(n_jobs = 2, random_state = 0)
clf.fit(train[features], y)


# As you can see we have fitted our train dataset to the model. The output shows several attributes you can alter. Check out the Documentaion to understand these values.

# Now let's try to predict the species type using our model. We will use the test data for this.

# In[ ]:


clf.predict(test[features])


# To understand how the model arrives at these results. Let's take a look at the probabilty for each species type given by the model for each input

# In[ ]:


clf.predict_proba(test[features])[10:20]


# Here each line shows 3 probabilty values for respective species type given by the model.

# In[ ]:


preds = df.species[clf.predict(test[features])]
preds[:5]


# If we check the accuracy of the model we will have a low value due to the small amount of data. Try using this model with a larger dataset. Random Forest Classifier works best for large datasets and for many features.

# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(test['species'], preds)

