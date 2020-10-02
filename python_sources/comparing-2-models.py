# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))

import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV,Ridge,RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score,learning_curve,train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import PolynomialFeatures

import warnings
warnings.filterwarnings("ignore")

raw_data=pd.read_csv('../input/kc_house_data.csv')
print(raw_data.describe())

#checking for Null values

def check_null(data):
    null_col_list=[]
    for col in data:
        if data[col].isnull().sum() >0:
            null_col_list.append(col)
        else:
            pass
    return null_col_list

null_col_list=check_null(raw_data)
print("Columns with Null Values = {0}".format(null_col_list))

#filling null values in columns
def fill_null_columns(null_col_list,data):
    for each in null_col_list:
        data[each]=data[each].fillna(data[each].mean())
    return
fill_null_columns(null_col_list,raw_data)

#again checking for null values
null_col_list=check_null(raw_data)
print("Columns with Null Values = {0}".format(null_col_list))

#feature data and target data
"""
raw_data columns=Index(['id', 'date', 'price', 'bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long', 'sqft_living15', 'sqft_lot15'],
      dtype='object')

removing id and date
"""
feature_data=raw_data.drop(['id','date','price'],axis=1)
target_data = raw_data['price']
#here we will make 2 models and will compare there efficiency
#######################################################################
############################   model 1  ##############################
#####################################################################
#In this we will use LassoCV for removing some features
model1_feature=feature_data
model1_target=target_data



#printing learning curve for model
def learning_curve_plot(model,feature,target):
    train_sizes,train_scores,test_scores=learning_curve(model,feature,target,train_sizes=np.linspace(0.1,1.0,6),cv=10)
    train_scores_mean=np.mean(train_scores,axis=1)
    test_scores_mean=np.mean(test_scores,axis=1)
    plt.figure()
    plt.xlabel("Training Data")
    plt.ylabel("Scores")
    plt.title("Learning Curve")
    plt.plot(train_sizes,train_scores_mean,'o-',color='r')
    plt.plot(train_sizes,test_scores_mean,'o-',color='g')
    plt.ylim(-.1,1.1)
    plt.show()
    return
    
def feature_selection():
    clf=LassoCV()
    sfm=SelectFromModel(clf,threshold=0.1)
    sfm.fit(model1_feature,model1_target)
    return sfm.get_support() #return in boolean form list

def remove_feature(bool_feature_list):
    i=0
    feature_remove_list=[] #storing the index of the columns that are to be removed
    for bol in bool_feature_list:
        if bol == False:
            feature_remove_list.append(i)
        else:
            pass
        i=i+1
    return model1_feature.drop(model1_feature.columns[feature_remove_list],axis=1)
    
            
            
bool_feature_list=feature_selection()
model1_feature=remove_feature(bool_feature_list)

#for alpha value of Ridge 
def for_alpha(model_feature,model_target):
    poly=PolynomialFeatures(2)
    rcv=RidgeCV([1,10,100,60,70],cv=10)
    x=poly.fit_transform(model_feature)
    rcv.fit(x,model_target)
    return rcv.alpha_

x=for_alpha(model1_feature,model1_target)
model1=make_pipeline(PolynomialFeatures(2),Ridge(alpha=x))



train,test,train_label,test_label=train_test_split(model1_feature,model1_target,test_size=0.30)
model1_fit=model1.fit(train,train_label)
predict=model1_fit.predict(test)


learning_curve_plot(model1,model1_feature,model1_target)
print("Generalized model accuracy {0}".format(cross_val_score(model1,model1_feature,model1_target,cv=10).mean()*100))
print("Accuracy on data : {0}".format(r2_score(test_label,predict)))

#######################################################################
############################   model 2  ##############################
#####################################################################
#In this we will be removing some features by hypothesis
model2_feature=feature_data
model2_target=target_data



#removing features(waterfront and view were containing values = 0 for most data)
model2_feature=model2_feature.drop(['waterfront', 'view','lat','long','zipcode'],axis=1)

#for alpha value of Ridge line 110(function)

x=for_alpha(model2_feature,model2_target)
model2=make_pipeline(PolynomialFeatures(2),Ridge(alpha=x))



train,test,train_label,test_label=train_test_split(model2_feature,model2_target,test_size=0.30)
model2_fit=model2.fit(train,train_label)
predict=model2_fit.predict(test)

#ploting learning curve for model 2 see line(73)
learning_curve_plot(model2,model2_feature,model2_target)
print("Generalized model accuracy {0}".format(cross_val_score(model2,model2_feature,model2_target,cv=10).mean()*100))
print("Accuracy on data : {0}".format(r2_score(test_label,predict)))


