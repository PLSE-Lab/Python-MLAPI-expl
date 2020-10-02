#!/usr/bin/env python
# coding: utf-8

# # Import H2o

# In[2]:


import h2o


# In[3]:


h2o.init(ip="localhost", port=54323)


# In[4]:


data = h2o.import_file('../input/creditcard/creditcard.csv')


# In[5]:


data.head()


# # Determine Predictors and Response Column

# In[37]:


x = data.columns


# In[7]:


y = "Class"# response


# In[38]:


x.remove(y)
x.remove("Id")
x.remove("Time")


# # Splitting Frames

# In[39]:


train,test,validation = data.split_frame(ratios=[0.7, 0.1])


# # Setting the response column as factor for binary classification

# In[42]:


train[y] = train[y].asfactor()
test[y] = test[y].asfactor()


# # Run H2o Auto-ML for 10 Minutes

# In[49]:


# Run AutoML for 10 minutes
from h2o.automl import H2OAutoML
aml = H2OAutoML(max_runtime_secs = 2*60)
aml.train(x = x, y = y,
          training_frame = train, leaderboard_frame = validation)


# # Show Leaderboard Models

# In[50]:


lb = aml.leaderboard
lb


# # Select Leader Model

# In[51]:


model = aml.leader


# # Predict Test Frame

# In[52]:


preds = aml.leader.predict(test)
preds


# # Plot Model Performance

# In[53]:


perf_test = model.model_performance(test_data=test)
perf_test.plot()


# In[54]:


perf_test.confusion_matrix()


# # Merge prediction and test frames

# In[55]:


merged = test.cbind(preds)
merged_pred = merged[:, ["Id", "Class","predict"]]
submission = merged_pred.as_data_frame()
submission.head()


# In[56]:


submission.to_csv('submission.csv', index=False)

