#!/usr/bin/env python
# coding: utf-8

# **Hello!**
# 
# This Kernel is for the PetFinder.my Adoption Prediction Kaggle competition submission. The goal is to assign each pet a value from 1-4 which corresponds to the amount of time it takes to get adopted. The higher the number, the longer it takes. The full challenge can be found at this [link](https://www.kaggle.com/c/petfinder-adoption-prediction). Here I take a simple approach to defining features, joining multiple datasets, comparing a few different algorithms, and making predcitons based on the best models. Let's get started by importing all the right packages.
# 
# I will include some factors from the the sentiment metadata, so I'll need a JSON package. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json
import warnings
warnings.filterwarnings("ignore")


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# The first thing I want to do is extract the "Magnitude" of the "documentSentiment" from both sentiment metadata files (testing and training). I'll store this data in a dictionary. In lieu of a cuteness metrric, I will use this quantity as a possibly predictive factor. 

# In[ ]:


## -- Training Sentiment Metedata -- ##
json_path = "../input/train_sentiment"
json_list = os.listdir(json_path)

jdict_train = {}
for js in json_list:
    with open(os.path.join(json_path, js), encoding="utf8") as json_file:
        json_text = json.load(json_file)
        jdict_train[js[0:-5]] = json_text["documentSentiment"]["magnitude"]
## -- Testing Sentiment Metedata -- ##
json_path = "../input/test_sentiment"
json_list = os.listdir(json_path)

jdict_test = {}
for js in json_list:
    with open(os.path.join(json_path, js), encoding="utf8") as json_file:
        json_text = json.load(json_file)
        jdict_test[js[0:-5]] = json_text["documentSentiment"]["magnitude"]

print(len(jdict_train))
print(len(jdict_test))


# The next step is to import both training and testing datasets, join the *Magnitude* metadata to it, and finally concatenate it to a single dataframe. I perform this concatenation so that I can appropriately one-hot encode the categorical data as to have the same shape for testing/training inputs. 
# 
# The joins are performed on the left to enure no loss of actual training or testing observations. Since Magnitude has some null values, I impute the mean before combining the dataset as to reduce data *leakage*.
# 
# A new column is added called "PredictionType" to distinguish between the testing and training sets. 
# 
# On the testing set, the column "AdoptionSpeed" is missing, so I add it with values of 100 which are unrealistic. These values will be overwritten by the predicted values later. 

# In[ ]:


df_train_data = pd.read_csv("../input/train/train.csv", header=0, index_col="PetID")
df_test_data = pd.read_csv("../input/test/test.csv", header=0, index_col="PetID")

df_sent_train = pd.DataFrame.from_dict(jdict_train, orient="index", columns=["Magnitude"])
df_sent_test = pd.DataFrame.from_dict(jdict_test, orient="index", columns=["Magnitude"])

joined_train = df_train_data.merge(df_sent_train, how="left", left_index=True, right_index=True)
joined_train["PredictionType"] = "Training"
joined_train["Magnitude"] = joined_train["Magnitude"].fillna(value=joined_train["Magnitude"].mean())
                                                             
joined_test = df_test_data.merge(df_sent_test, how="left", left_index=True, right_index=True)
joined_test["PredictionType"] = "Testing"
joined_test["AdoptionSpeed"] = 100
joined_test["Magnitude"] = joined_test["Magnitude"].fillna(value=joined_test["Magnitude"].mean())

full_df =  joined_train.append(joined_test)

print(full_df.info())


# Now a full dataframe has been constructed with training and testing sets with an extra column from the sentiment metadata. My next step is to convert all the columns into their proper type and drop those that might be irrelevant (like Name). 
# 
# My approach is one that prioritizes readability. 
# 

# In[ ]:


data_type_dict = {
        "category":["Type", # 1 = Dog, 2 = Cat
                    "Breed1", # Primary breed of pet (Refer to BreedLabels dictionary)
                    "Breed2", # Secondary, if mixed
                    "Gender", # 1 = Male, 2 = Female, 3 = Mixed, if profile represents group of pets
                    "Color1", # Color 1 of pet (Refer to ColorLabels dictionary)
                    "Color2", 
                    "Color3", 
                    "MaturitySize", # Size at maturity (1 = Small, 2 = Medium, 3 = Large, 4 = Extra Large, 0 = Not Specified)
                    "FurLength", # Fur length (1 = Short, 2 = Medium, 3 = Long, 0 = Not Specified)
                    "Vaccinated", # Pet has been vaccinated (1 = Yes, 2 = No, 3 = Not Sure)
                    "Dewormed", # Pet has been dewormed (1 = Yes, 2 = No, 3 = Not Sure)
                    "Sterilized", # Pet has been spayed / neutered (1 = Yes, 2 = No, 3 = Not Sure)
                    "Health", # Health Condition (1 = Healthy, 2 = Minor Injury, 3 = Serious Injury, 0 = Not Specified)
                    "State" # State location in Malaysia (Refer to StateLabels dictionary)								
                    ],
        "float":["Age", # Months
                 "Quantity", # Number of pets represented in profile
                 "Fee", # Adoption fee (0=Free)
                 "VideoAmt", # Total uploaded videos for this pet
                 "PhotoAmt", # Total uploaded photos for this pet
                 "Magnitude", # From the Sentiment metadata
                 ],
        "int8":["AdoptionSpeed"], # Categorical speed of adoption. Lower is faster. This is the value to predict.]
        "object":["PredictionType"] # Custom indeifyer of testing/training set"
                 }

output_var = "AdoptionSpeed"


# df_train = pd.DataFrame()
df_learn = pd.DataFrame()
for typ,col in data_type_dict.items():
    for c in col:
        df_learn[c] = full_df[c].astype(typ)

print(df_learn.info())


# Now is the excited step of preparing the data for use in a machine learning algorithm. In this step I will one-hot encode the categorical factors, then split the full dataset into a "whole" training set,  then subset the "whole" training set further for test/train validation. 

# In[ ]:


df_learn = pd.get_dummies(df_learn)

X_train_whole = df_learn[df_learn["PredictionType_Training"] == 1].drop(output_var, axis=1)
y_train_whole = df_learn[df_learn["PredictionType_Training"] == 1][output_var]

from sklearn.model_selection import train_test_split
rnd = 0
X_train, X_test, y_train, y_test = train_test_split(X_train_whole, y_train_whole, 
                                                    test_size=0.20, 
                                                    random_state=rnd).copy()


# Now I will choose from three different models with default parameters for a quick comparison. I'll use the default score method to assess them. 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC


models = {"Random Forest":RandomForestClassifier(random_state=rnd),
          "K-neighbors":KNeighborsClassifier(),
          "Linear SVC":LinearSVC(random_state=rnd)
         }

for desc, mod in models.items():
    print(desc)
    mod.fit(X_train, y_train)
    print(mod.score(X_test, y_test))


# It appears that Random Forest gives the best scores out of the box, so I'll choose this one to proceed with tuning. I encourage you to fork this Kernel and try for yourself different parameters. 
# 
# Below is one that I found works best at about 0.401 accuracy score with 120 estimators. Now to retrain on the entire training set.

# In[ ]:


mod = RandomForestClassifier(n_estimators=120, random_state=rnd)
mod.fit(X_train_whole, y_train_whole)


# Great! It's now time to make a prediction on our testing data (complete with testing sentiment metadata ) and prepare the submission file.

# In[ ]:


pred = mod.predict(df_learn[df_learn["PredictionType_Testing"] == 1].drop(output_var, axis=1))

df_test_data["AdoptionSpeed"] = pred
submission = df_test_data["AdoptionSpeed"]
print(submission.head(10).to_string())


# In[ ]:


submission.to_csv("submission.csv", index=True, index_label="PetID", header=["AdoptionSpeed"])


# **Thank you for reading**. As always, please drop a comment or upvote this Kernel if it helped you at all. 
# 
# Michael Greene

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




