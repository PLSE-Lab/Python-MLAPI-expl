# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from  sklearn.preprocessing import LabelEncoder

print ("Loading the data into Pandas Dataframe...")
data_frame = pd.read_csv('../input/train.csv') ## import the training data into Pandas data frame
data_frame_test=pd.read_csv('../input/test.csv') ## import the test data into Pandas data frame

categorical_fields=['MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle',
                    'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
                    'BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual',
                    'GarageCond','PavedDrive','PoolQC','Fence','MiscFeature','SaleType','SaleCondition'
                    ]

numerical_fields=['MSSubClass','LotFrontage','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',
                  'TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',
                  'TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea',
                  'MiscVal','MoSold','YrSold']

ally_dict={'NA':0,'Grvl':1,'Pave':2} ## dictionary to digitize the field : Alley

## Preprocess the data. fill the "NA" with "default". Encode the categorical fields to numeric values.
def pre_process_data():
    print ("Preprocessing the Data...")
    for col in categorical_fields:
        data_frame[col].fillna('default',inplace=True)
        data_frame_test[col].fillna('default',inplace=True)

    for col in numerical_fields:
        data_frame[col].fillna(0,inplace=True)
        data_frame_test[col].fillna(0,inplace=True)

    encode=LabelEncoder()
    for col in categorical_fields:
        data_frame[col]=encode.fit_transform(data_frame[col])
        data_frame_test[col]=encode.fit_transform(data_frame_test[col])
    data_frame['SalePrice'].fillna(0,inplace=True)


## Get the imporance of the features using Random Forest Regressor.
def get_feature_importance(list_of_features):
    n_estimators=10000
    random_state=0
    n_jobs=5
    print ("Running random forest to get the feature imporance with the paramers.. n_estimators:{} random_state :{} n_jobs :{}".format(n_estimators,random_state,n_jobs))
    x_train=data_frame[list_of_features]
    y_train=data_frame.iloc[:,-1]
    feat_labels= data_frame.columns[1:]
    forest = RandomForestRegressor(n_estimators=n_estimators,random_state=random_state,n_jobs=n_jobs) ## Create a Random Forest Classifier object
    forest.fit(x_train,y_train) ## Fit the data into the model
    importances=forest.feature_importances_ ## get the feature importance
    # print("Original ",np.argsort(importances))
    indices = np.argsort(importances)[::-1]
    # print (" importances ",importances)
    # print (" indices ",indices)

    for f in range(x_train.shape[1]):
        print("%2d) %-*s %f" % (f+1,30,feat_labels[indices[f]],
                                        importances[indices[f]]))

    ## Plot the feature importance in the bar chart.
    plt.title("Feature Importance")
    plt.bar(range(x_train.shape[1]),importances[indices],color='lightblue',align='center')
    plt.xticks(range(x_train.shape[1]),feat_labels[indices],rotation=90)
    plt.xlim([-1,x_train.shape[1]])
    plt.tight_layout()
    #plt.show()  ## Remove the  comment to display the feature imporance in clumn chart.


def get_features():
    print ("Getting the features to build the model...")
    l1=list(data_frame.columns.values)
    l2=list(data_frame_test.columns.values)
    fields=list(set(l1)& set(l2))
    fields.remove('Id')
    list_of_features=sorted(fields)
    print ("List of Features :",list_of_features)
    return list_of_features

#### Develop the regression Model
def create_model(list_of_features):

    n_estimators=10000 # Number of Trees in the forest
    n_jobs=5 ## Number of parallel jobs
    x_train=data_frame[list_of_features] ## Training set of features
    y_train=data_frame.iloc[:,-1]## target in traunung set
    x_test=data_frame_test[list_of_features] ## Test set of feature for validation
    random_state=0
    ## Model creation.

    print ("Running random forest to create the model with parameters.. n_estimators:{} random_state :{} n_jobs :{}".format(n_estimators,random_state,n_jobs))
    forest=RandomForestRegressor(n_estimators=n_estimators,random_state=random_state, n_jobs=n_jobs)
    forest.fit(x_train[list_of_features],y_train) ## Fit the model
    Y_pred=forest.predict(data_frame_test[list_of_features].as_matrix()) ## Get the prediction on test data

    i=0
    file=open('submission.csv','w')
    header="Id,SalePrice"
    header=header+'\n'
    file.write(header)
    for id in (data_frame_test['Id']):
        str="{},{}".format(id,Y_pred[i])
        str=str+'\n'
        #print(str)
        file.write(str)
        i+=1




def main():
    pre_process_data()
    list_of_features=get_features()
    get_feature_importance(list_of_features)
    create_model(list_of_features)


if __name__==main():
    main()
