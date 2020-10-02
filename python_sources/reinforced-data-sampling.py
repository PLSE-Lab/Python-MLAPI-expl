#!/usr/bin/env python
# coding: utf-8

# **This notebook serves as a demonstration for Reinforced Data Sampling (RDS).**
# 
# RDS provides a method to learn how to sample data effectively on the search for useful models and meaningful insights. By employing diverse base learners such as neural networks, decision trees, or support vector machines, it aims to maximize the learning potentials and optimum allocation of data sampling to disentangle dataset shift and evidence ambiguity. In the hope of saving a massive amount of computational resources and time, we design RDS as a viable alternative to simple randomization and stratification in train_test_split for various machine learning tasks such as classification and regression.
# 
# The Default of Credit Card Clients (DCCC) Data Set is only used for illustrative purposes.
# 
# https://github.com/probeu/RDS
# 

#  # 1. Load Libraries

# In[ ]:


get_ipython().system('pip install torchRDS')


# In[ ]:


from torchRDS.RDS import RDS
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


# # 2. Preparing Base Models

# In[ ]:


class LR:
    def run(self, state, action):
        data_x, data_y = state
        train_x, train_y, test_x = data_x[action == 1], data_y[action == 1, 1], data_x[action == 0]
        
        clf = LogisticRegression(solver='liblinear')
        clf.fit(train_x, train_y)
        return clf.predict_proba(test_x)
    
class KNN:
    def run(self, state, action):   
        data_x, data_y = state
        train_x, train_y, test_x = data_x[action == 1], data_y[action == 1, 1], data_x[action == 0]

        clf = KNeighborsClassifier(n_neighbors=3)
        clf.fit(train_x, train_y)

        return clf.predict_proba(test_x)

class RF:
    def run(self, state, action):   
        data_x, data_y = state
        train_x, train_y, test_x = data_x[action == 1], data_y[action == 1, 1], data_x[action == 0]

        clf = RandomForestClassifier(n_estimators=16, bootstrap=False, n_jobs=-1)
        clf.fit(train_x, train_y)

        return clf.predict_proba(test_x)


# # 3. Setup RDS

# In[ ]:


trainer = RDS(data_file="../input/dccc-dataset/default_of_credit_card_clients.csv", target=[0], task="classification", measure="auc", 
              models=[LR(), KNN(), RF()], learn="deterministic", ratio=0.6, delta=0.02, weight_iid=0.1, iters=100, device="cuda")


# Parameters:
# * data_file: *path to your csv file*
# * target: *indexes of your output columns*
# * task: *classification or regression*
# * measure: *auc, cross_entropy, mse, f1_micro, r2*
# * models: *initialized base models*
# * learn: *deterministic or stochastic*
# * ratio: *sampling ratio*
# * delta: *acceptable allocation error (e.g., 0.02 = +- 2% from the sampling ratio)*
# * weight_iid: *weight factor to ensure the same distribution of classes in both the training and test sets*
# * iters: *iterations to run*
# * device: *cuda or cpu*

# # 4. Run RDS

# In[ ]:


sample = trainer.train()


# # 5. Selection Results

# In[ ]:


print("Number of observations in the trainning set:", sum(sample))
print("Number of observations in the test set:", len(sample) - sum(sample))

