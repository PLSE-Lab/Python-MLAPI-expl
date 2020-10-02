# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 14:56:36 2018

@author: Diwas.Tiwari
"""

import numpy as np
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

train_file = pd.read_json(open("../input/train.json", "r"))
test_file =  pd.read_json(open("../input/test.json", "r"))

train_file.describe()
train_file.info()
numeric_headers = ["bathrooms","bedrooms","latitude","longitude","price"]

## Extracting information out of the other dataframes features ##
## Train features ##
train_file["photo_count"] = train_file["photos"].apply(len)
train_file["feature_count"] = train_file["features"].apply(len)
train_file["words_in_description"] = train_file["description"].apply(len)
train_file["words_in_adress"] = train_file["display_address"].apply(len)

train_file["created"] = pd.to_datetime(train_file["created"], format = '%Y-%m-%d %H:%M:%S')
#train_file["created"] = train_file["created"].astype(str)
train_file.info()
train_file["year_created"] = train_file["created"].dt.year
train_file["month_created"] = train_file["created"].dt.month
train_file["day_created"] = train_file["created"].dt.day

features_extracted = ["photo_count", "feature_count", "words_in_description", "words_in_adress",
                      "year_created", "month_created", "day_created"]
## Test Features ##
test_file["photo_count"] = test_file["photos"].apply(len)
test_file["feature_count"] = test_file["features"].apply(len)
test_file["words_in_description"] = test_file["description"].apply(len)
test_file["words_in_adress"] = test_file["display_address"].apply(len)

test_file["created"] = pd.to_datetime(test_file["created"], format = '%Y-%m-%d %H:%M:%S')
#train_file["created"] = train_file["created"].astype(str)
#train_file.info()
test_file["year_created"] = test_file["created"].dt.year
test_file["month_created"] = test_file["created"].dt.month
test_file["day_created"] = test_file["created"].dt.day

feature_of_intrest_train = numeric_headers+features_extracted


x = train_file[feature_of_intrest_train]
y = train_file["interest_level"]

feature_of_intrest_test = numeric_headers+["photo_count", "feature_count", "words_in_description", "words_in_adress",
                                           "year_created", "month_created", "day_created"]
x_test = test_file[feature_of_intrest_test]

x.info()
x.head()
y.head()

from sklearn.model_selection import train_test_split
def TrainValSplit(data, input_features, output_features, split_ratio):
    train_x, test_x, train_y, test_y = train_test_split(data[input_features],
                                                        data[output_features], train_size = split_ratio)
    return train_x, test_x, train_y, test_y

from sklearn.ensemble import RandomForestClassifier

def RF_Classifier(input_data, output_data):
    model = RandomForestClassifier(n_estimators = 2000, min_samples_leaf = 2, min_samples_split = 11,
                                   max_features = 4)
    model.fit(input_data,output_data)
    return model

from sklearn.svm import SVC

def SVC_Classifier(input_data, output_data):
    model = SVC()
    model.fit(input_data, output_data)
    return model
## Function for evaluating performance of GridSearch Cross-Validation and Evaluation 
#of Accuracy Improvement ##

#def GridSearchEvaluation(model, test_input,test_features):
#    prediction = model.predict(test_input)
#    err = abs(prediction-test_features)
#    percent_err = np.mean(err/test_features)
#    accuracy = 100-percent_err
#    print ('Accuracy = {:0.2f}%.'.format(accuracy))
#    return accuracy
def AccuracyImprovment(grid_acc, base_acc):
    acc_improved = 100*(grid_acc-base_acc)/base_acc
    return acc_improved

#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import accuracy_score   
from sklearn.metrics import log_loss    
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from time import time
def main():
    
    train_x, test_x, train_y, test_y = train_test_split(x,y,train_size = 0.8)
    ## Major Hyperparameters for RF Classifier are: - Max Depth, Max features, Min Samples Leaf,
    #Min Samples Split, n Estimators ###
    params_grid = {
        'max_depth': [70,85,90], 'max_features':[5,9,4], 'min_samples_leaf':[3,4],
        'min_samples_split':[6,8], 'n_estimators':[500,600,800]
        }
    
    
    model = RF_Classifier(train_x,train_y)
    pprint(model.get_params())
    y_pred = model.predict_proba(test_x)
    print(log_loss(test_y, y_pred))
    
    y_test = model.predict_proba(x_test)
    ## Labelling the 3 categorical values ##
    labels2idx = {label: i for i, label in enumerate(model.classes_)}
    labels2idx
    
    ## Writing the output in .csv format ##
    sub = pd.DataFrame()
    sub["listing_id"] = test_file["listing_id"]
    for label in ["high", "medium", "low"]:
        sub[label] = y_test[:, labels2idx[label]]
        #path = "W:\Diwas\Rental_Listing_Enquires"
    sub.to_csv("submission_output.csv", index=False)
    
#    search_iter = 3
#    RandSearch = RandomizedSearchCV(model,param_distributions = params_grid, n_iter = search_iter)
#    RandSearch.fit(train_x,train_y)

## Writing the File in the .csv Folder ##
if __name__ == "__main__": 
    
    main()
    
#    for i in xrange(0,15):
#        print "Actual_Outcome & Predicted Outcome are: {}".format(list(test_y)[i],y_pred[i])
#        
        
        
        
    


    
    
