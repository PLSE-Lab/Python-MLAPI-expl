#!/usr/bin/env python
# coding: utf-8

# Hi everyone ! My brand new Python package for Auto Machine Learning is now available on github/PyPI/Kaggle kernels ! :)
# 
# **https://github.com/AxeldeRomblay/MLBox**
# 
# - It is very easy to use (see **documentation** on github)
# - It provides state-of-the-art algorithms and technics such as deep learning/entity embedding, stacking, leak detection, parallel processing, hyper-parameters optimization...
# - It has already been tested on Kaggle and performs well (see Kaggle "Two Sigma Connect: Rental Listing Inquiries" | Rank : **85/2488**)
# 
# **Please put a star on github and fork the script if you like it !** 
# 
# Enjoy :) 

# # Inputs & imports : that's all you need to give !

# In[1]:


from mlbox.preprocessing import *
from mlbox.optimisation import *
from mlbox.prediction import *


# In[2]:


paths = ["../input/nyc-taxi-trip-duration/train.csv", "../input/nyc-taxi-trip-duration/test.csv"]
target_name = "trip_duration"


# # Now let MLBox do the job ! 

# In[3]:


time.sleep(30)


# ## ... to read and clean all the files 

# In[4]:


rd = Reader(sep = ",")
df = rd.train_test_split(paths, target_name)   #reading and preprocessing (dates, ...)


# **adding OSRM features and distances**

# Here you can create your own features... Then MLBox will do the rest for you !

# In[5]:


cols = [u'id', u'starting_street', u'end_street', u'total_distance',u'total_travel_time', u'number_of_steps']
extra_train = pd.read_csv("../input/new-york-city-taxi-with-osrm/fastest_route_train.csv", usecols=cols)
extra_test = pd.read_csv("../input/new-york-city-taxi-with-osrm/fastest_route_test.csv", usecols=cols)

df['train'] = pd.merge(df['train'], extra_train, on ='id', how='left')
df['test'] = pd.merge(df['test'], extra_test, on ='id', how='left')


# In[ ]:


df['train']["N2"] = ((df['train']["dropoff_longitude"]-df['train']["pickup_longitude"]).apply(lambda x: x**2) + (df['train']["dropoff_latitude"]-df['train']["pickup_latitude"]).apply(lambda x: x**2)).apply(lambda x: np.sqrt(x))
df['test']["N2"] = ((df['test']["dropoff_longitude"]-df['test']["pickup_longitude"]).apply(lambda x: x**2) + (df['test']["dropoff_latitude"]-df['test']["pickup_latitude"]).apply(lambda x: x**2)).apply(lambda x: np.sqrt(x))

df['train']["N1"] = (df['train']["dropoff_longitude"]-df['train']["pickup_longitude"]).apply(lambda x: np.abs(x)) + (df['train']["dropoff_latitude"]-df['train']["pickup_latitude"]).apply(lambda x: np.abs(x))
df['test']["N1"] = (df['test']["dropoff_longitude"]-df['test']["pickup_longitude"]).apply(lambda x: np.abs(x)) + (df['test']["dropoff_latitude"]-df['test']["pickup_latitude"]).apply(lambda x: np.abs(x))

df['train']["pickup_distance_center"] = ((df['train']["pickup_longitude"].mean()-df['train']["pickup_longitude"]).apply(lambda x: x**2) + (df['train']["pickup_latitude"].mean()-df['train']["pickup_latitude"]).apply(lambda x: x**2)).apply(lambda x: np.sqrt(x))
df['test']["pickup_distance_center"] = ((df['test']["pickup_longitude"].mean()-df['test']["pickup_longitude"]).apply(lambda x: x**2) + (df['test']["pickup_latitude"].mean()-df['test']["pickup_latitude"]).apply(lambda x: x**2)).apply(lambda x: np.sqrt(x))


# **drift**

# In[ ]:


dft = Drift_thresholder()
df = dft.fit_transform(df)   #removing non-stable features (like ID,...)


# ## ... to tune all the hyper-parameters

# In[ ]:


df['target'] = df['target'].apply(lambda x: np.log1p(x))   #evaluation metric: rmsle

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

opt = Optimiser(scoring = make_scorer(rmse, greater_is_better=False), n_folds=2)


# **XGBoost**

# In[ ]:


space = {
     
        'est__strategy':{"search":"choice",
                                  "space":["XGBoost"]},    
        'est__n_estimators':{"search":"choice",
                                  "space":[300]},    
        'est__colsample_bytree':{"search":"uniform",
                                  "space":[0.78,0.82]},   
        'est__colsample_bylevel':{"search":"uniform",
                                  "space":[0.78,0.82]},    
        'est__subsample':{"search":"uniform",
                                  "space":[0.82,0.88]},
        'est__max_depth':{"search":"choice",
                                  "space":[10,11]},
        'est__learning_rate':{"search":"choice",
                                  "space":[0.075]} 
    
        }

params = opt.optimise(space, df, 1)  #only 1 iteration because it takes a long time otherwise :) 


# But you can also tune the whole Pipeline ! Indeed, you can choose:
# 
# * different strategies to impute missing values
# * different strategies to encode categorical features (entity embeddings, ...)
# * different strategies and thresholds to select relevant features (random forest feature importance, l1 regularization, ...)
# * to add stacking meta-features !
# * different models and hyper-parameters (XGBoost, Random Forest, Linear, ...)

# ## ... to predict

# In[ ]:


prd = Predictor()
prd.fit_predict(params, df)


# ### Formatting for submission

# In[ ]:


submit = pd.read_csv("../input/nyc-taxi-trip-duration/sample_submission.csv",sep=',')
preds = pd.read_csv("save/"+target_name+"_predictions.csv")

submit[target_name] =  preds[target_name+"_predicted"].apply(lambda x: np.exp(x)-1).values

submit.to_csv("mlbox.csv", index=False)


# # That's all !!
# 
# If you like my new auto-ml package, please **put a star on github and fork/vote the Kaggle script :)**
