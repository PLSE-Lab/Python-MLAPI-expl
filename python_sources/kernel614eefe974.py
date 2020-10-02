#!/usr/bin/env python
# coding: utf-8

# # ConocoPhillips Sensor Challenge -- OIL MEN Submission

# # Import Relevant Libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import RobustScaler, Imputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.manifold import TSNE
pd.set_option('display.max_columns', 200) # This allow us to see every column


# In[ ]:


data = pd.read_csv("../input/equipfails/equip_failures_training_set.csv")  # Read Data into a Pandas DataFrame for Analysis


# In[ ]:


data.describe(include = 'all')


# # Everything looks weird in the .describe() DataFrame, after taking a look at the data we found out that some numerical values where saves as strings and the 'na' string ultimately affects the computation

# # So we write a function that does all the cleaning (replaces the 'na' with the Numpy Nan Object), the function also removes all features (sensor reading), which have less than half of the total samples available (60,000). It also replace all missing data with the median value of its correspnding feature

# In[ ]:


def clean_data(df):
    df.replace('na', np.nan, inplace = True) #Replacing "na" with Numpy's Nan for Computation
    df = df.astype(float)
    r_c = []
    for i in range(df.shape[1]): #Removes features with more than 50% of the samples missing 
        if df[df.columns[i]].count() > (0.5*df.shape[0]):
            r_c.append(df.columns[i])
    imp_data = Imputer(strategy = 'median').fit_transform(df[r_c]) #Imputs missing data with median value of corresponding feature
    ret_data = pd.DataFrame(imp_data, columns = r_c)
    return ret_data, r_c #The Index kept are stored for use when readying the deployment dataset (i.e the test dataset)


# In[ ]:


data_fresh, index_in = clean_data(data)


# In[ ]:


#The Total Data is split into features (X) and a target set, the feature set correponds to a matrix of every eligible sensor reading
#y corresponds to a failure (1) or normal conditions(0)
X = data_fresh.iloc[:,2:] 
y = np.array(data_fresh.iloc[:,1])


# # Scaling Data is an important step in the ML workflow as it removes the effect of dissimilar scales and is generally best practice

# # We selected the Robust scaler which centers each feature (subtracts by its means) and divides by a user selected quantile range of that feature. The purpose of this is to ensure our scaling proedure is not affected by extreme values at either end which are likely to occur in sensor data

# In[ ]:


def scale(df,method): #This function scales a dataset by the method set by the user
    scale_d = method.fit_transform(df)
    return scale_d


# In[ ]:


robust = RobustScaler(quantile_range=(2.5,97.5)) #Defining the Robust Scaler


# In[ ]:


X_scale = scale(X,robust) #Scaling Feature Set


# # To ensure we dont have a "static" feature in the dataset we remove every feature with zero variance using the Sklearn VarianceThreshold function

# In[ ]:


var = VarianceThreshold()


# In[ ]:


var.fit(X_scale) 


# In[ ]:


var_feat = var.get_support() #This gets all index with non-zero variance, we stored it for use for the readying the deployment dataset


# In[ ]:


X_new_feat = X_scale[:,var_feat] #Extract all features with non-zero variance


# # The data is a bit high dimensional, at least high dimensional enough for us not to be able to visualize it using conventional methods. We used the tSNE method to reduce the data to a 2-dimensional space visualization

# # From the above 2-D plot we can see that most of the faulty data most exist at the edge of the 2D space, obviously this a very low dimensional representation of the data and we believe that in higher dimensional space the delineations are clearer

# # We did perform some dimensionality reduction techniques to try to reduce the dimensions to aid the performance of the models but this generally lead to poorer results so we decided to stick with high dimensional data for training

# ## We split our data into a Train and Test set to ensure we are not overfitting on our dataset

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_new_feat,y, test_size = 0.3, random_state = 0, stratify = y)


# In[ ]:


def train_model(X_train,X_test,y_train,y_test, model): # This function trains the model
    model.fit(X_train,y_train)
    y_hat = model.predict(X_test)
    return ('The accuracy score is: ' + str(model.score(X_test, y_test)) +
            ' The F1 score is: ' + str(f1_score(y_test, y_hat)))


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression


# In[ ]:


#Initiate our models, all hyperparameters used were based on extensive GridSearch and trial and error
lr = LogisticRegression()
rfc = RandomForestClassifier(n_estimators=200, n_jobs = -1, max_features = 15)
xgb = XGBClassifier(n_jobs=-1)


# # The models selected were based on the high dimensionality of the dataset and non-linearity, we believed that the random forest and Gradient boost were the best models to control bias and variance of the model. The SVC was considered but took too long to train. The Logistic Regression was used as something of a base model

# In[ ]:


train_model(X_train, X_test, y_train, y_test, lr)


# In[ ]:


train_model(X_train, X_test, y_train, y_test, rfc)


# In[ ]:


train_model(X_train, X_test, y_train, y_test, xgb)


# # Import Test Data and Refine the feature the same way as was done with the TRAIN set

# In[ ]:


test_data = pd.read_csv("../input/equipfails/equip_failures_test_set.csv")


# In[ ]:


test_d = test_data.iloc[:,1:] #Select all features except id


# In[ ]:


test_in = test_d[index_in[2:]] #Filter based on features


# In[ ]:


test_in.replace('na',np.nan, inplace = True) #Replace na with Nan as was done with TRAIN data


# In[ ]:


test_imp = Imputer(strategy = 'median').fit_transform(test_in) #Replace missing values with with median of feature


# In[ ]:


test_final = test_imp[:,var_feat]


# In[ ]:


test_final = robust.fit_transform(test_final) #Scale data using Robust scaler


# In[ ]:


rfc.fit(X_new_feat, y)


# In[ ]:


y_pred = rfc.predict(test_final) # We chose the Random Forest Classifier as we believe its less likely to overfit compared to Random Forest


# In[ ]:


export = test_data[['id']]


# In[ ]:


export['target'] = y_pred


# In[ ]:


print(export)


# In[ ]:




