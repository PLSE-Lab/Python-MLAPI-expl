# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import csv
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split, cross_val_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))
#print("CURR B : ",os.listdir("."))
if os.path.exists("submissionRFG.csv"):
  os.remove("submissionRFG.csv")
if os.path.exists("submission.csv"):
  os.remove("submission.csv")

#print("CURR A : ",os.listdir("."))
# Any results you write to the current directory are saved as output.




#A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
def rmsle(y, y_pred, lstRmv):
	assert len(y) == len(y_pred)
	j = len(y)
	i = 1
	terms_to_sum = 0
	while i<j:
	    #print ("in rmsle, y_pred : ", y_pred[i], " & y : ", y[i] )
	    if len(lstRmv) != 0:
	        for item in lstRmv:
	            if item == i:
	                print("Skipping removed index: ", i, " len: ", len(lstRmv))
	                lstRmv.remove(i)
	                print("After skipping removed index: ", i, " len: ", len(lstRmv))
	                break
	            else:
	                terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0]
	    else:
	        terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0]
	    
	    i=i+1
	print ("i : ", i)
	return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5

#def rmsle(y, y_pred): TRYING TO GET RMSLE WITH RANDOM 50% OF DATA..
#	assert len(y) == len(y_pred)
#	j = len(y)
#	k = 1
#	terms_to_sum = 0
#	
#	for k in range(int(j/2)):
	#while i<j:
#	    i = random.randint(1,j-1)
#	    #print ("in rmsle, y_pred : ", y_pred[i], " & y : ", y[i] )
#	    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0] # for i,pred in enumerate(y_pred)]
#	    k=k+1
#	return (sum(terms_to_sum) * (1.0/int(j/2))) ** 0.5

##function getMAE..
def getMAE():
    submission_data_fromCSV = pd.read_csv("submission.csv")
    #submission_data_fromCSV = pd.read_csv("./Output/submissionRFG.csv")
    #submission_data = submission_data_fromCSV.fillna(0)

    test_data_fromCSV = pd.read_csv("../input/test.csv")
    test_data = test_data_fromCSV.fillna(0)
    test_data_dummies = pd.get_dummies(test_data)            

    testSubmission_All = pd.merge(test_data_dummies,submission_data_fromCSV, on="Id")
    
    testSubmission = testSubmission_All.head(730)

    features = test_data_dummies.axes[1]
    X = testSubmission[features]
    y = testSubmission.SalePrice


    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

    my_model = RandomForestRegressor(random_state=1, n_estimators=100)
    my_model.fit(train_X, train_y)
    predictedSalePrices = my_model.predict(val_X)
    print("Mean Absolute Error using RandomForestRegressor: ", mean_absolute_error(val_y, predictedSalePrices))


    train_X2, val_X2, train_y2, val_y2 = train_test_split(X, y, random_state=1)

    my_pipeline = make_pipeline(SimpleImputer(), XGBRegressor(random_state=1))
    my_pipeline.fit(train_X2, train_y2)
    predictedSalePrices2 = my_pipeline.predict(val_X)

    print("Mean Absolute Error using XGBRegressor: ", mean_absolute_error(val_y2, predictedSalePrices2))
    
    scores = cross_val_score(my_pipeline, X, y, scoring='neg_mean_absolute_error', cv=5)
    print('Mean Absolute Error %2f' %(-1 * scores.mean()))
    return
##############


################
def missingValues(df):
    total_num = df.isnull().sum().sort_values(ascending=False)
    perc = df.isnull().sum()/df.isnull().count() *100
    perc1 = (round(perc,2).sort_values(ascending=False))
    # Creating a data frame:
    df_miss = pd.concat([total_num, perc1], axis =1 , keys =["Total Missing Values", "Percentage %"]).sort_values(by ="Percentage %", ascending = False)
    top_mis = df_miss[df_miss["Percentage %"]>80]
    top_mis.reset_index(inplace=True)
    #print ("Columns with top missing values \n :  ", top_mis.head(10))
    return top_mis
#####################



#### STEP 1
#### Read training data & test data from their respective CSV files - BEGIN


training_data_fromCSV = pd.read_csv("../input/train.csv")
#training_data_fromCSV = training_data_fromCSV[training_data_fromCSV.YrSold < 2008]
training_data_fromCSV_TOMV = training_data_fromCSV.copy(deep=True)
top_mis = missingValues(training_data_fromCSV_TOMV)

#training_data = training_data_fromCSV.fillna(0)
training_data_fromCSV_dropped = training_data_fromCSV.drop(columns=['PoolQC', 'Utilities'], axis = 1)
training_data = training_data_fromCSV_dropped.fillna(0)

test_data_fromCSV = pd.read_csv("../input/test.csv")
test_data_fromCSV_TOMV = test_data_fromCSV.copy(deep=True)
top_mis = missingValues(test_data_fromCSV_TOMV)

#test_data = test_data_fromCSV.fillna(0)
test_data_fromCSV_dropped = test_data_fromCSV.drop(columns=['PoolQC', 'Utilities'], axis = 1)
test_data = test_data_fromCSV_dropped.fillna(0)
test_data_id = test_data.copy(deep=True)
#### Read training data & test data from their respective CSV files - END

training_data = training_data.drop(columns=['Id'])
test_data = test_data.drop(columns=['Id'])
#### Make y for the model from training data - BEGIN
training_data_price = training_data.SalePrice
#print ("\n\n\ntraining_data_price : ", training_data_price.describe())

#print ("\n\n\tmp_t_d_p. : ", training_data_price.describe(percentiles=[.03]))
#print ("\n\n\tmp_t_d_p.10% : ", training_data_price.describe(percentiles=[.03]).iloc[4] )
#remPrcntl = training_data_price.describe(percentiles=[.03]).iloc[4]

print (training_data.GrLivArea.describe()) #1515.463699
print (training_data_price.describe()) #1515.463699
i = 0
lstRmv = []

while i<1460:
    
    if ((training_data.GrLivArea[i]) > 3500):
        training_data = training_data.drop(i)
        training_data_price = training_data_price.drop(i)
        lstRmv.append(i)

    i = i+1
print (training_data.GrLivArea.describe()) #1515.463699
print (training_data_price.describe()) #1515.463699
print (lstRmv)

#i=0
#print ("\n\n\n\n")
#while i<1460:
    #if ( training_data_price.loc[i] <= 40000):
     #   print ("training_data_price. valuesin loop : ", training_data_price.loc[i] )
      #  training_data_price.loc[i] = 180921.195890
    #i=i+1

#print ("\n\n\training_data_price. : ", training_data_price.describe(percentiles=[.03, .1]))

#### Make y for the model from training data - END

#### STEP 2

#### Convert categorical variables into dummy/indicator variables - BEGIN
### get_dummies is used to get one-hot encoding
training_data_dummies = pd.get_dummies(training_data)            
test_data_dummies = pd.get_dummies(test_data)
#### Convert categorical variables into dummy/indicator variables - BEGIN


#### STEP 3

#### Get features common to both training data & test data. - BEGIN

## Get axes to check the common features - BEGIN
list_training_data_dummies_axes = training_data_dummies.axes
list_test_data_dummies_axes = test_data_dummies.axes
## Get axes to check the common features - END

#print("list_training_data_xSet_dummies_axes - 1 : ", len(list_training_data_dummies_axes[1]))
#print("list_test_data_xSet_dummies_axes - 1 : ", len(list_test_data_dummies_axes[1]))

features_list = [];

#comparing value of two lists
for item in list_training_data_dummies_axes[1]:
    #print("item : ", item)
    for item1 in list_test_data_dummies_axes[1]:
        if item == item1:
             features_list.append(item)

#print ("\n\n\n\nfeatures_list length : ", len(features_list))
#### Get features common to both training data & test data. - END


#### STEP 4

#### Make X for the model - BEGIN
training_data_dummies_xSet = training_data_dummies [features_list]
test_data_dummies_xSet = test_data_dummies [features_list]
#### Make X for the model - END


#impute the data // Imputer didn't made any difference.. MAE was same in both cases when tried to predict with training data only
#imputer = SimpleImputer()
#imp_train_X = pd.DataFrame(imputer.fit_transform(training_data_dummies_xSet))
#imp_test_X = pd.DataFrame(imputer.fit_transform(test_data_dummies_xSet))
imp_train_X = training_data_dummies_xSet
imp_test_X = test_data_dummies_xSet

#### Create model, fit training data with the common features - BEGIN
my_model = RandomForestRegressor(random_state=1, n_estimators=100)

#from sklearn.model_selection import train_test_split #TEMP
#train_X, val_X, train_y, val_y = train_test_split(imp_train_X, training_data_price, random_state=1) #TEMP
#my_model.fit(train_X, train_y) #TEMP
#print("Mean Absolute Error using RandomForestRegressor: \n ", mean_absolute_error(val_X, val_y)) #TEMP

my_model.fit(imp_train_X, training_data_price)
#### Create model, fit training data with the common features - END


#### STEP 5

#### Predict SalePrice, using test data with the common features - BEGIN
predictedSalePrices = my_model.predict(imp_test_X)

#print("Mean Absolute Error from Imputation: \n ", mean_absolute_error(training_data_price, predictedSalePrices))

##############print("\n\n\n : predictedSalePrice : ", predictedSalePrices)


my_submission = pd.DataFrame({'Id': test_data_id.Id, 'SalePrice': predictedSalePrices})
my_submission.to_csv('submissionRFG.csv', index=False)

#### Predict SalePrice, using test data with the common features - END

#my_modelXGB = XGBRegressor()
#my_modelXGB = XGBRegressor(random_state=1, n_estimators=1000, early_stopping_rounds = 5, learning_rate=0.05)
my_modelXGB = XGBRegressor(max_depth=3, gamma=0, reg_alpha=0.1, random_state=1, n_estimators=300, reg_lambda=0.7, max_delta_step=0, min_child_weight=1, colsample_bytree=0.5, colsample_bylevel=0.2, scale_pos_weight=1, booster='gbtree', learning_rate=0.1)

my_modelXGB.fit(imp_train_X, training_data_price, verbose=False)
predictions = my_modelXGB.predict(imp_test_X)

my_submission2 = pd.DataFrame({'Id': test_data_id.Id, 'SalePrice': predictions})
my_submission2.to_csv('submission.csv', index=False)
#print ("rmsle b4 : ", training_data_price[1:3:1].dtype, " P : ", predictions[1:3:1].dtype )
#print ("rmsle b422222222 \n : ", training_data_price, " P : ", predictions.astype(int) )
#print ("\n\n\n\n c422222222 \n : ", training_data_price[1], " P : ", predictions[1].astype(int) )
#print ("\n\n\n\n c422222222 \n : ", training_data_price[0], " P : ", predictions[0].astype(int), "\n\n\n\n" )

#i = 0
#while i<4:
#    print ("training_data_price.. : ", training_data_price[i])
#    print ("pred.. : ", predictions[i])
#    i=i+1
print ("need dtype : ", predictions.dtype)
pred_for_rmsle = predictions.copy()
pred_for_rmsle = np.delete(pred_for_rmsle, lstRmv)
len_rmsle = len(pred_for_rmsle)
print (len(predictions))
print (len(pred_for_rmsle))
print ("rmsle: ", rmsle(training_data_price[1:len_rmsle:1], pred_for_rmsle[1:len_rmsle:1], lstRmv))
getMAE()