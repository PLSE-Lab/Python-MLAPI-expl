#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#For basic datasets, We don't need numpy and pandas explicitly. 
import h2o 
from h2o.automl import H2OAutoML
#Initializing H2O
h2o.init()


# In[ ]:


data = h2o.import_file("/kaggle/input/bank-note-authentication-uci-data/BankNote_Authentication.csv")


# In[ ]:


data.summary()


# In[ ]:


#class is our target in this query
#This is a Classification Problem and In Summary, We can spot class as integer type. 
#So, We will change it to enum type.
data['class'] = data['class'].asfactor()


# In[ ]:


#Splitting Test & Train in 80 & 20. 
train, test = data.split_frame(ratios=[0.8])


# In[ ]:


print("{} Rows in training frame & {} Rows in testing frame".format(train.nrows,test.nrows))


# In[ ]:


#Storing name of colums. 
x = train.columns #x will contain indepedent variable. 
y = "class" #y will contain dependent variable. 
x.remove(y)


# In[ ]:


# We will run AutoML for 300sec as it is a small dataset.
# Seed will help reproduce same models
aml = H2OAutoML(max_runtime_secs = 300, seed = 127)
aml.train(x=x, y=y, training_frame=train)


# In[ ]:


lb = aml.leaderboard
lb.head() #This will give us top 10 models. 


# In[ ]:


# The leader model is stored here
aml.leader
# Let's make prediction on leader
preds = aml.leader.predict(test)


# In[ ]:


preds


# In[ ]:


#We will see performance of Leader Model on test set
aml.leader.model_performance(test)


# #### Observations:
# + In Confusion Matrix, We can spot that we got nothing wrong. 
# + Rest variables are self explanatory. 

# In[ ]:


lead_id = aml.leader.model_id
print("Model ID of leader is {}".format(lead_id))


# In[ ]:


#Hyper-parameters of leader model
out = h2o.get_model(lead_id)
out.params


# In[ ]:


#Stopping Cluster
h2o.cluster().shutdown()

