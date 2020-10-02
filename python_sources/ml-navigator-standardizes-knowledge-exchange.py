#!/usr/bin/env python
# coding: utf-8

# # ML Navigator
# 
# Github https://github.com/KI-labs/ML-Navigator<br>
# Documentation: https://ki-labs.github.io/ML-Navigator/index.html<br>
# Pypi: https://pypi.org/project/ML-Navigator/<br>
# 
# For more than one and a half year I have been organizing Kaggle Munich meetup in Munich. I met a lot of interesting, smart people who would like to start their career as data scientists. Most of the time I hear the same question, which is: where should I start?
# 
# The question inspired me to think of a solution that helps data scientists to exchange knowledge in a white-box mode. That is how I come up with my first open-source project: **ML-Navigator**
# 
# ML-Navigator is a tutorial-based informative Machine Learning framework. The main component of ML-Navigator is the flow. A flow is a collection of compact methods/functions that can be stuck together with guidance texts.
# 
# The flow functions as a map which shows the road from point A to point B. The guidance texts function as navigator instructions that help the user to figure out the next step after executing the current step.
# 
# Like the car navigator, the user is not forced to follow the path. At any point, the user can take a break to explore data, modify the features and make any necessary changes. The user can always come back to the main path which the flow defines.
# 
# The flows are created by the community for the community. Therefore, your contributions and constructive feedback are very welcome.
# 
# Big thanks to KI labs and my great team for supporting me at every step of this project.

# # Why ML-Navigator
# ML-Navigator standardizes exchanging knowledge among data scientists. Junior data scientists can learn and apply data science best-practice in a white-box mode. Senior data scientists can automate many repetitive processes and share experience effectively. Enterprises can standardize data science among different departments.

# ## Install ML-Navigator
# The normal way for installing ML-Navigator is use the `pip` tool:
# 
# `pip install ML-Navigator`
# 
# Since the Kaggle kernels don't support most of the recent versions of the Python libraries you need to install ML-Navigator from the Github repository.

# In[ ]:


get_ipython().system('git clone https://github.com/KI-labs/ML-Navigator.git ML_Navigator')


# Run the following commands to install ML-Navigator successfully and to ensure a successful kernel commit:

# In[ ]:


get_ipython().system('cp -fr ./ML_Navigator/* ./')
get_ipython().system('rm -r ./ML_Navigator')
get_ipython().system('rm -r ./docs')


# Run the following command to ensure that the recent Python libraries are installed

# In[ ]:


get_ipython().system('cat requirements.txt | xargs -n 1 pip install')


# ## Quick Start

# In[ ]:


from flows.flows import Flows


# In[ ]:


flow = Flows(3)


# The visualization of the flow helps to get a general overview about operations that are included in this flow.

# <img src="./tutorials/flow_3.png">

# How to apply the flow to your data? It is straightforward. You need to point to the location of your data and the name of the datasets after loading a particular flow. Currently, the framework supports only reading CSV files.

# In[ ]:


path = '/kaggle/input/ieee-fraud-detection/'
files_list = ['train_transaction.csv','test_transaction.csv']


# In case of large dataset, you can use `rows_amount` variable to limit the number of the rows that are loaded.

# In[ ]:


dataframe_dict, columns_set = flow.load_data(path, files_list, rows_amount=10000)


# As you can see above, the method `load_data` provides a summary of every dataset. It can figure out which type of data that each column contains. Moreover, it can find out which columns are the best candidates for the id and for the target that should be predicted. It figures out the type of problem that should be solved. In this example, it is a classification problem.

# In[ ]:


dataframe_dict["train_transaction"].head()


# Let us load the second dataset similar to the first dataset

# In[ ]:


files_list_2 = ['train_identity.csv','test_identity.csv']
dataframe_dict_identity, columns_set_identity = flow.load_data(path, files_list_2, rows_amount=10000)


# In[ ]:


dataframe_dict_identity["train_identity"].head()


# Here we will merge both datasets. This causes that we diverge from the flow that we load.

# In[ ]:


import pandas as pd
dataframe_train =  pd.merge(dataframe_dict["train_transaction"],
                            dataframe_dict_identity["train_identity"], how="left",
                            on='TransactionID') 
dataframe_test = pd.merge(dataframe_dict["test_transaction"],
                            dataframe_dict_identity["test_identity"], how="left",
                            on='TransactionID') 


# In[ ]:


print(dataframe_train.shape)
print(dataframe_test.shape)


# In[ ]:


dataframe_dict = {}
dataframe_dict["train"] = dataframe_train
dataframe_dict["test"] = dataframe_test


# Here we can come back to the flow by using one command only.

# In[ ]:


columns_set = flow.update_data_summary(dataframe_dict)


# The next step is to encode the categorical variables

# In[ ]:


dataframe_dict, columns_set = flow.encode_categorical_feature(dataframe_dict, print_results=10)


# For dropping columns with constant values and highly correlated columns, it is a good idea to exclude some important columns that we want to keep them in the dataset, e.g. the target.

# In[ ]:


ignore_columns = ['isFraud']


# In[ ]:


dataframe_dict, columns_set = flow.drop_columns_constant_values(dataframe_dict, ignore_columns)


# In[ ]:


dataframe_dict, columns_set = flow.drop_correlated_columns(dataframe_dict, ignore_columns)


# The ID and the target should not be scaled. Therefore, we update the `ignore_columns` variable as follows:

# In[ ]:


ignore_columns = ["TransactionID", "isFraud"]


# In[ ]:


dataframe_dict, columns_set = flow.scale_data(dataframe_dict, ignore_columns)


# Exploring the data and comparing the statistics properties of the features from different datasets should be helpful to understand the model quality variation between training and testing datasets.

# In[ ]:


flow.exploring_data(dataframe_dict, "train")


# Preparing the data for training by ensuring that all datasets contains the same features as the training dataset:

# In[ ]:


columns = dataframe_dict["train"].columns
total_columns = columns_set["train"]["continuous"] +columns_set["train"]["categorical_integer"]

train_dataframe = dataframe_dict["train"][
    [x for x in total_columns if x not in ignore_columns]]
test_dataframe = dataframe_dict["test"][
    [x for x in total_columns if x not in ignore_columns]]
train_target = dataframe_dict["train"]["isFraud"]


# Define the parmaters object for training the model using the LightGBM using K-Fold method. Five folds are used. If the `predict` key is defined, the model is used to calculate the target for test dataset.

# In[ ]:


parameters_lightgbm = {
    "data": {
        "train": {"features": train_dataframe, "target": train_target.to_numpy()},
    },
    "split": {
        "method": "kfold",  # "method":"kfold"
        "fold_nr": 5,  # foldnr:5 , "split_ratios": 0.8 # "split_ratios":(0.7,0.2)
    },
    "model": {"type": "lightgbm",
              "hyperparameters": dict(objective='binary', metric='cross-entropy', num_leaves=5,
                                      boost_from_average=True,
                                      learning_rate=0.05, bagging_fraction=0.99, feature_fraction=0.99, max_depth=-1,
                                      num_rounds=10000, min_data_in_leaf=10, boosting='dart')
              },
    "metrics": ["accuracy_score", "roc_auc_score"],
    "predict": {
        "test": {"features": test_dataframe}
    }
}


# The models are saved locally in `save_models_dir`.

# In[ ]:


model_index_list, save_models_dir, y_test = flow.training(parameters_lightgbm)


# # Summary
# ML-Navigator provides out of the box informative tools that help the user to get knowing the data before proceeding to the next steps. The flow helps the user to figure out how different Machine Learning components are working together. The user can modify the flow on the fly to adapt case-specific required changes. Last but not least, ML-Navigator helps to exchange experiences systemically and cleanly. Your feedback and your contributions are very welcome.

# In[ ]:




