# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
iowa_file_path="../input/home-data-for-ml-course/train.csv"
home_data=pd.read_csv(iowa_file_path)
#------------------------------------------------------------------------------------------------------------------------------------------
#select features in train data, including SalePrice
features_pre=['MSSubClass','MSZoning','LotArea','LotShape','LandContour','Utilities','LandSlope',
              'Neighborhood','Condition1','BldgType','HouseStyle','OverallQual','OverallCond',
              'YearBuilt','YearRemodAdd','RoofStyle','RoofMatl','Exterior1st','ExterQual','ExterCond',
              'Foundation','Heating','HeatingQC','1stFlrSF','2ndFlrSF','BsmtFullBath','BedroomAbvGr',
              'KitchenQual','Functional','GarageArea','OpenPorchSF','YrSold','SaleType',
              'SaleCondition','SalePrice']
#------------------------------------------------------------------------------------------------------------------------------------------
#select features at test data, including Id
features_test=['MSSubClass','MSZoning','LotArea','LotShape','LandContour','Utilities','LandSlope',
              'Neighborhood','Condition1','BldgType','HouseStyle','OverallQual','OverallCond',
              'YearBuilt','YearRemodAdd','RoofStyle','RoofMatl','Exterior1st','ExterQual','ExterCond',
              'Foundation','Heating','HeatingQC','1stFlrSF','2ndFlrSF','BsmtFullBath','BedroomAbvGr',
              'KitchenQual','Functional','GarageArea','OpenPorchSF','YrSold','SaleType',
              'SaleCondition','Id']
#------------------------------------------------------------------------------------------------------------------------------------------
#subroutine to obtain best leaf nodes
def best_leaf_nodes (train_X, val_X, train_y, val_y):

        candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
        result={i:get_mae(i, train_X,val_X,train_y,val_y) for i in candidate_max_leaf_nodes}
        best_tree_size=min(result, key=result.get)
        return best_tree_size
#------------------------------------------------------------------------------------------------------------------------------------------
#soubrutine to obtain criteria (mae) to get  best leaf node
def get_mae(current_leaf_nodes, train_X, val_X, train_y, val_y):
    model = RandomForestRegressor(max_leaf_nodes=current_leaf_nodes, random_state=1)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)
#------------------------------------------------------------------------------------------------------------------------------------------
#apply features and del rows wit Nan values on them, later split into X & y
home_data_post=home_data[features_pre].dropna(axis=0)
y=home_data_post.SalePrice
X=home_data_post.drop(['SalePrice'],axis=1)



check_path="../input/home-data-for-ml-course/test.csv"
check_data=pd.read_csv(check_path)
#apply features and del rows wit Nan values on them, later split into X & y
check_data=check_data[features_test]
check_y=check_data['Id']
check_X=check_data.drop(['Id'],axis=1)
check_X.fillna(0, inplace=True)

#pass variables from categorical to numeric
train_dumm_X=pd.get_dummies(X, dtype=int)
check_dumm_X=pd.get_dummies(check_X, dtype=int)


#after compare train and check columns, drop columns that there aren't at check Dframe and avoid different columns numbers for matching train vs valid Dframes
#seems that train & test data changes sometimes
train_dumm_X.drop(columns= ['Utilities_NoSeWa','HouseStyle_2.5Fin','RoofMatl_ClyTile','RoofMatl_Membran','RoofMatl_Metal',
                            'RoofMatl_Roll','Exterior1st_ImStucc','Exterior1st_Stone','Heating_Floor','Heating_OthW'], axis=1, inplace=True)
check_dumm_X.drop(columns=['MSZoning_0','Utilities_0','Exterior1st_0','KitchenQual_0','Functional_0','SaleType_0'], axis=1, inplace=True)



#split train data for geting best leaf nodes
train_test_dumm_X, val_test_dumm_X, train_test_dumm_y, val_test_dumm_y = train_test_split(train_dumm_X, y, random_state=1)

#create model  and prediction trhough Random Forest
rf_model1=RandomForestRegressor(max_leaf_nodes=best_leaf_nodes(train_test_dumm_X, val_test_dumm_X, train_test_dumm_y, val_test_dumm_y ),random_state=1)
rf_model1.fit(train_dumm_X,y)
rf_test_predict1=rf_model1.predict(check_dumm_X)

#create output
output = pd.DataFrame({'Id': check_y,
                       'SalePrice':rf_test_predict1})
output.to_csv('submission.csv', index=False,float_format='%.0f')