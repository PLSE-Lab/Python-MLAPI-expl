#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np #for linear algebra and scientific computing
import pandas as pd #data analysis and manipulation

# Input data files are available in the read-only "../input/" directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


#data visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


from sklearn.model_selection import train_test_split #split into training and testing data
from sklearn.metrics import mean_squared_error #RMSE for evaluation
from sklearn.model_selection import GridSearchCV #for exhaustive grid search(hyperparameter tuning)

#encoders for categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import category_encoders as ce


# In[ ]:


cars_Train = pd.read_csv("/kaggle/input/used-cars-price-prediction/train-data.csv", index_col=0)
cars_Test = pd.read_csv("/kaggle/input/used-cars-price-prediction/test-data.csv", index_col=0)


# In[ ]:


cars = cars_Train.copy()


# ### Let's explore the data

# In[ ]:


cars.head()


# In[ ]:


cars.info()


# In[ ]:


cars.describe()


# In[ ]:


#1   Location
plt.xticks(rotation = 90)
sns.countplot(cars.Location)


# In[ ]:


#2   Year
plt.xticks(rotation = 90)
sns.countplot(cars.Year)


# In[ ]:


#3   Kilometers_Driven
sns.distplot(cars[cars.Kilometers_Driven<500000].Kilometers_Driven, kde=False)


# *We can observe outliers in the data as kilometers driven cannot be so large*

# In[ ]:


#setting the outliers as nan
cars.loc[cars.Kilometers_Driven > 400000, "Kilometers_Driven"] = np.nan


# In[ ]:


#4   Fuel_Type
sns.countplot(cars.Fuel_Type)


# In[ ]:


#5   Transmission
sns.countplot(cars.Transmission, palette="Reds_r")


# In[ ]:


#6   Owner_Type
sns.countplot(cars.Owner_Type, order=['First','Second','Third','Fourth & Above'])


# In[ ]:


#7   Mileage
print("Null values:", cars.Mileage.isnull().sum())
print("Outliers:",(cars.Mileage==0).sum())


# *The null values in `Mileage` is because `Fuel_Type` is `Electric`.*

# In[ ]:


#Removing units and extracting numerical data from mileage
cars.Mileage = cars.Mileage.str.split(expand=True)[0].astype("float64")


# In[ ]:


#set the outliers as null
cars[cars.Mileage==0].Mileage = np.nan


# In[ ]:


sns.distplot(cars.Mileage, kde=False)


# In[ ]:


#7   Engine
print("Total null values:",cars.Engine.isnull().sum())
cars[cars.Engine.isnull()].head()


# In[ ]:


#Removing units and extracting numerical data from Engine
cars.Engine = cars.Engine.str.split(expand=True)[0].astype("float64")


# In[ ]:


sns.distplot(cars.Engine, kde=False)


# In[ ]:


#8 Power
print("Total null values:",cars.Power.isnull().sum())
cars[cars.Power.isnull()].head()


# In[ ]:


#Removing units and extracting numerical data from Power
cars.Power = cars.Power.apply(lambda s: np.nan if "null" in str(s) else s).str.split(expand=True)[0].astype("float64")


# In[ ]:


sns.distplot(cars.Power, kde=False)


# In[ ]:


#9 Seats
print("Total null values:",cars.Seats.isnull().sum())
cars[cars.Seats.isnull()].head()


# In[ ]:


cars.loc[cars.Seats<1,"Seats"] = np.nan


# In[ ]:


sns.distplot(cars.Seats, kde=False)


# *Note: We will impute Engine, Power and seats based on Brand(extract brand during feature engineering).*

# In[ ]:


#10 New_Price
print("Total null values:",cars.New_Price.isnull().sum())
cars[cars.New_Price.isnull()].head()


# In[ ]:


cars.New_Price = cars.New_Price.apply(lambda s: float(s.split()[0])*100 if "Cr" in str(s) else str(s).split()[0]).astype("float64")


# In[ ]:


print("Total null values:",cars.New_Price.isnull().sum())
sns.distplot(cars.New_Price, kde=False)


# In[ ]:


#sns.pairplot(cars)


# In[ ]:


sns.heatmap(cars.corr(), cmap="coolwarm")


# ***Some Feature Engineering and Preprocessing***

# ***Extracting brand and model names from name***

# In[ ]:


carnames = cars.Name.str.split(expand=True)[[0,1,2]]


# In[ ]:


carnames.rename(columns={0:'Brand',1:'Model',2:'Type'}, inplace=True)


# In[ ]:


cars = cars.join(carnames)
cars = cars.drop("Name", axis=1)


# ***Creating new features using combinations of categorical columns***

# In[ ]:


from itertools import combinations

object_cols = cars.select_dtypes("object").columns
low_cardinality_cols = [col for col in object_cols if cars[col].nunique() < 15]
low_cardinality_cols.append("Brand")
interactions = pd.DataFrame(index=cars.index)

# Iterate through each pair of features, combine them into interaction features
for features in combinations(low_cardinality_cols,2):
    
    new_interaction = cars[features[0]].map(str)+"_"+cars[features[1]].map(str)
    
    encoder = LabelEncoder()
    interactions["_".join(features)] = encoder.fit_transform(new_interaction)


# In[ ]:


cars = cars.join(interactions) #append to the dataset


# In[ ]:


cars.head(5)


# In[ ]:


# cars.info()


# *Now, we need to impute the missing values*
# 
# *And before that we need to do train test split to prevent data leakage*

# In[ ]:


features = cars.drop(["Price"], axis=1)
target = cars["Price"]
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=0)


# *Now, let's see those missing values*

# In[ ]:


X_train.isnull().sum()


# *We need to fill missing values in Engine, Power and Seats only. Missing values in Subtype and Subtype_2 would provide a feature.*
# 
# *Note: XGBoost and Light GBM have an inbuilt feature to handle missing values. So, we can also leave missing values as it is.*
# 
# *Let's fill missing values based on brand column.*
# 

# In[ ]:


num_cols = X_train.drop('New_Price',1).select_dtypes("number")
null_num_cols = num_cols.columns[num_cols.isnull().any()]

for cols in null_num_cols:
    X_train.loc[:,cols] = X_train.loc[:,cols].fillna(X_train.groupby('Brand')[cols].transform('mean'))
    X_train.loc[:,cols] = X_train.loc[:,cols].fillna(X_train[cols].mean())

    X_test.loc[:,cols] = X_test.loc[:,cols].fillna(X_test.groupby('Brand')[cols].transform('mean'))
    X_test.loc[:,cols] = X_test.loc[:,cols].fillna(X_test[cols].mean())


# In[ ]:


# #Binning #didn't provide improvement to results
# #Year
# X_train=X_train.drop('Year',1).join(pd.cut(X_train.Year, range(1996,2021,4), False, range(6)).astype('int64'))
# X_test=X_test.drop('Year',1).join(pd.cut(X_test.Year, range(1996,2021,4), False, range(6)).astype('int64'))

# #Kilometers_Driven
# X_train=X_train.drop('Kilometers_Driven',1).join(pd.cut(X_train.Kilometers_Driven, range(0,300001,10000), labels= range(30)).astype('int64'))
# X_test=X_test.drop('Kilometers_Driven',1).join(pd.cut(X_test.Kilometers_Driven, range(0,300001,10000), labels= range(30)).astype('int64'))


# In[ ]:


cars.select_dtypes("object").nunique()


# ### Encoding the categorical columns

# *One-hot encoding*

# In[ ]:


OHE_cat_features = ["Fuel_Type","Transmission", "Location", "Owner_Type", "Brand"]
OH_encoder = OneHotEncoder(sparse=False,handle_unknown='ignore')

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[OHE_cat_features]))
OH_cols_test = pd.DataFrame(OH_encoder.transform(X_test[OHE_cat_features]))

OH_cols_train.index = X_train.index
OH_cols_test.index = X_test.index

OH_cols_train.columns = OH_encoder.get_feature_names(OHE_cat_features)
OH_cols_test.columns = OH_encoder.get_feature_names(OHE_cat_features)

X_train_enc = X_train.join(OH_cols_train)
X_test_enc = X_test.join(OH_cols_test)


# *Drop OHE encoded columns*

# In[ ]:


X_train_enc.drop(OHE_cat_features, axis=1, inplace = True)
X_test_enc.drop(OHE_cat_features, axis=1, inplace = True)


# *Target encoding*

# In[ ]:


target_cat_features = X_train_enc.select_dtypes('object').columns
target_enc = ce.TargetEncoder(cols=target_cat_features)
target_enc.fit(X_train[target_cat_features], y_train)
X_train_enc = X_train_enc.join(target_enc.transform(X_train[target_cat_features]).add_suffix('_enc'))
X_test_enc = X_test_enc.join(target_enc.transform(X_test[target_cat_features]).add_suffix('_enc'))


# *Drop categorical columns(dtype: object)*

# In[ ]:


object_cols = X_train_enc.select_dtypes('object')
X_train_enc.drop(object_cols, axis=1, inplace = True)
X_test_enc.drop(object_cols, axis=1, inplace = True)


# *Conversion of all columns into int64*

# In[ ]:


# X_train_enc=X_train_enc.astype('int64')
# X_test_enc=X_test_enc.astype('int64')


# In[ ]:


X_train_enc.info()


# ***Feature selection based on pearson correlation***

# In[ ]:


pcorr = X_train_enc.join(y_train).corr()
imp_corr_cols = pcorr[['Price']][pcorr['Price']>-0.25].iloc[:-1].index

X_train_enc = X_train_enc[imp_corr_cols]
X_test_enc = X_test_enc[imp_corr_cols]


# In[ ]:


from xgboost import XGBRegressor


# ***XGB model***

# In[ ]:


base_xgbr = XGBRegressor(objective='reg:squarederror', tree_method='gpu_hist')
base_xgbr.fit(X_train_enc,y_train)

base_xgbr.score(X_test_enc,y_test) #r2 score


# *Visualizing the test results*

# In[ ]:


yhat_xgbr = base_xgbr.predict(X_test_enc)
print(mean_squared_error(y_test, yhat_xgbr, squared=False))
sns.kdeplot(y_test)
sns.kdeplot(yhat_xgbr)


# *Feature importance based on the XGBoost model*

# In[ ]:


feat_imp = pd.DataFrame(base_xgbr.feature_importances_, index=X_train_enc.columns)
sns.heatmap(feat_imp, cmap='Purples')


# ***Feature selection based on XGBoost model***

# In[ ]:


impfeat = pd.Series(base_xgbr.feature_importances_, index=X_train_enc.columns)
impcols = impfeat[impfeat>0.005].index
X_train_enc = X_train_enc[impcols]
X_test_enc = X_test_enc[impcols]


# *Visualizing pearson correlation of finally selected columns*

# In[ ]:


sns.heatmap(X_train_enc.join(y_train).corr()[['Price']], cmap='Reds')


# In[ ]:


#testing xgbr model
# param_grid = {
#     "learning_rate": [0.05],
#     "max_depth": [6,8,10,12],
#     "min_child_weight": [5],
#     "n_estimators": [350,400,450,500],
#     "subsample": [0.55]
# }
# gscv = GridSearchCV(estimator=base_xgbr, param_grid=param_grid, n_jobs=-1, verbose=5, cv=4)


# In[ ]:


# gscv.fit(X_train_enc, y_train)


# In[ ]:


#the best params from the given parameter grid
# gscv.best_params_
# gscv.score(X_test_enc,y_test) #r2 score


# In[ ]:


# tuned_xgbr = XGBRegressor(objective = 'reg:squarederror',
#                     learning_rate = 0.05, max_depth = 12, min_child_weight = 5,
#                     n_estimators = 500, subsample = 0.55)
# tuned_xgbr.fit(X_train_enc,y_train)

# tuned_xgbr.score(X_test_enc,y_test) #r2 score


# In[ ]:


# yhat_xgbr = tuned_xgbr.predict(X_test_enc)
# print(mean_squared_error(y_test, yhat_xgbr, squared=False))
# sns.kdeplot(y_test)
# sns.kdeplot(yhat_xgbr)


# ### Base LGBM model

# In[ ]:


from lightgbm import LGBMRegressor


# In[ ]:


base_lgbmr = LGBMRegressor()


# In[ ]:


base_lgbmr.fit(X_train_enc, y_train)
base_lgbmr.score(X_test_enc,y_test)


# In[ ]:


yhat_lgbmr = base_lgbmr.predict(X_test_enc)
print(mean_squared_error(y_test, yhat_lgbmr, squared=False))
sns.kdeplot(y_test)
sns.kdeplot(yhat_lgbmr)


# In[ ]:


#feature importance
#pd.Series(base_lgbmr.feature_importances_, index=X_train_enc.columns)


# In[ ]:


base_lgbmr.get_params()


# In[ ]:


#initial grid search
param_grid = {
    "learning_rate": [0.15],
    "max_depth": [5,8,10,12],
    "min_child_weight": [3,5,6,8],
    "n_estimators": [300,500,800,1000,1200],
    "num_leaves": [20,25,40,50],
    "subsample": [0.3,0.5]
}
# gscv_lgbm = GridSearchCV(estimator=base_lgbmr, param_grid=param_grid, n_jobs=-1, verbose=5, cv=4)


# In[ ]:


# gscv_lgbm.fit(X_train_enc, y_train)


# In[ ]:


# gscv_lgbm.best_params_


# In[ ]:


# gscv_lgbm.score(X_test_enc,y_test) #r2 score


# In[ ]:


param_grid2 = {
    "learning_rate": [0.15],
    "max_depth": [8],
    "n_estimators": [1500,1800],
    "num_leaves": [25,27],
    'reg_alpha': [0,0.001,0.01],
    'reg_lambda': [0,0.001,0.01]
}
gscv_lgbm2 = GridSearchCV(estimator=base_lgbmr, param_grid=param_grid2, n_jobs=-1, verbose=5, cv=4)


# In[ ]:


gscv_lgbm2.fit(X_train_enc, y_train)


# In[ ]:


print(gscv_lgbm2.best_params_)
print(gscv_lgbm2.score(X_test_enc,y_test)) #r2 score


# ### Tuned LGBM model

# In[ ]:


tuned_lgbmr = LGBMRegressor(**gscv_lgbm2.best_params_)
tuned_lgbmr.fit(X_train_enc, y_train)
tuned_lgbmr.score(X_test_enc,y_test)


# In[ ]:


yhat_lgbmr = tuned_lgbmr.predict(X_test_enc)
print(mean_squared_error(y_test, yhat_lgbmr, squared=False))
sns.kdeplot(y_test)
sns.kdeplot(yhat_lgbmr)


# *The Tuned LGBM Regressor model will be used as the final model for our predictions*

# In[ ]:


# Custom Label Encoder for handling unknown values
class LabelEncoderExt(object):
    def __init__(self):
        """
        It differs from LabelEncoder by handling new classes and providing a value for it [Unknown]
        Unknown will be added in fit and transform will take care of new item. It gives unknown class id
        """
        self.label_encoder = LabelEncoder()
        # self.classes_ = self.label_encoder.classes_

    def fit(self, data_list):
        """
        This will fit the encoder for all the unique values and introduce unknown value
        :param data_list: A list of string
        :return: self
        """
        self.label_encoder = self.label_encoder.fit(list(data_list) + ['Unknown'])
        self.classes_ = self.label_encoder.classes_

        return self

    def transform(self, data_list):
        """
        This will transform the data_list to id list where the new values get assigned to Unknown class
        :param data_list:
        :return:
        """
        new_data_list = list(data_list)
        for unique_item in np.unique(data_list):
            if unique_item not in self.label_encoder.classes_:
                new_data_list = ['Unknown' if x==unique_item else x for x in new_data_list]

        return self.label_encoder.transform(new_data_list)


# ***Let's preprocess our original training data.***

# In[ ]:


X_train = cars_Train.drop('Price',1)
y_train = cars.Price
X_test = cars_Test


# *Let's apply the feature engineering and preprocessing to the training and testing data*

# In[ ]:


carnames = X_train.Name.str.split(expand=True)[[0,1,2]]
carnames_test = X_test.Name.str.split(expand=True)[[0,1,2]]

carnames.rename(columns={0:'Brand',1:'Model',2:'type'}, inplace=True)
carnames_test.rename(columns={0:'Brand',1:'Model',2:'type'}, inplace=True)

X_train = X_train.join(carnames)
X_train = X_train.drop("Name", axis=1)
X_test = X_test.join(carnames_test)
X_test = X_test.drop("Name", axis=1)


# In[ ]:


object_cols = X_train.select_dtypes("object").columns
low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 15]
low_cardinality_cols.append("Brand")
interactions = pd.DataFrame(index=X_train.index)
interactions_test = pd.DataFrame(index=X_test.index)

for features in combinations(low_cardinality_cols,2):
    
    new_interaction = X_train[features[0]].map(str)+"_"+X_train[features[1]].map(str)
    new_interaction_test = X_test[features[0]].map(str)+"_"+X_test[features[1]].map(str)
    
    encoder = LabelEncoderExt()
    encoder.fit(new_interaction)
    interactions["_".join(features)] = encoder.transform(new_interaction)
    interactions_test["_".join(features)] = encoder.transform(new_interaction_test)


# In[ ]:


X_train = X_train.join(interactions)
X_test = X_test.join(interactions_test)


# In[ ]:


num_cols = X_train.drop('New_Price',1).select_dtypes("number")
null_num_cols = num_cols.columns[num_cols.isnull().any()]

for cols in null_num_cols:
    X_train.loc[:,cols] = X_train.loc[:,cols].fillna(X_train.groupby('Brand')[cols].transform('mean'))
    X_train.loc[:,cols] = X_train.loc[:,cols].fillna(X_train[cols].mean())

    X_test.loc[:,cols] = X_test.loc[:,cols].fillna(X_test.groupby('Brand')[cols].transform('mean'))
    X_test.loc[:,cols] = X_test.loc[:,cols].fillna(X_test[cols].mean())


# In[ ]:


num_cols = X_train.select_dtypes("number")
null_num_cols = num_cols.columns[num_cols.isnull().any()]

for cols in null_num_cols:
    X_train.loc[:,cols] = X_train.loc[:,cols].fillna(X_train.groupby('Brand')[cols].transform('mean'))
    X_train.loc[:,cols] = X_train.loc[:,cols].fillna(X_train[cols].mean())

    X_test.loc[:,cols] = X_test.loc[:,cols].fillna(X_test.groupby('Brand')[cols].transform('mean'))
    X_test.loc[:,cols] = X_test.loc[:,cols].fillna(X_test[cols].mean())


# In[ ]:


OHE_cat_features = ["Fuel_Type","Transmission", "Location", "Owner_Type", "Brand"]
OH_encoder = OneHotEncoder(sparse=False,handle_unknown='ignore')

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[OHE_cat_features]))
OH_cols_test = pd.DataFrame(OH_encoder.transform(X_test[OHE_cat_features]))

OH_cols_train.index = X_train.index
OH_cols_test.index = X_test.index

OH_cols_train.columns = OH_encoder.get_feature_names(OHE_cat_features)
OH_cols_test.columns = OH_encoder.get_feature_names(OHE_cat_features)

X_train_enc = X_train.join(OH_cols_train)
X_test_enc = X_test.join(OH_cols_test)


# In[ ]:


X_train_enc.drop(OHE_cat_features, axis=1, inplace = True)
X_test_enc.drop(OHE_cat_features, axis=1, inplace = True)


# In[ ]:


target_cat_features = X_train_enc.select_dtypes('object').columns
target_enc = ce.TargetEncoder(cols=target_cat_features)
target_enc.fit(X_train[target_cat_features], y_train)
X_train_enc = X_train_enc.join(target_enc.transform(X_train[target_cat_features]).add_suffix('_enc'))
X_test_enc = X_test_enc.join(target_enc.transform(X_test[target_cat_features]).add_suffix('_enc'))


# In[ ]:


object_cols = X_train_enc.select_dtypes('object')
X_train_enc.drop(object_cols, axis=1, inplace = True)
X_test_enc.drop(object_cols, axis=1, inplace = True)


# In[ ]:


pcorr = X_train_enc.join(y_train).corr()
imp_corr_cols = pcorr[['Price']][pcorr['Price']>-0.25].iloc[:-1].index

X_train_enc = X_train_enc[imp_corr_cols]
X_test_enc = X_test_enc[imp_corr_cols]


# In[ ]:


xgbr = XGBRegressor(objective='reg:squarederror', tree_method='gpu_hist')
xgbr.fit(X_train_enc,y_train)


# In[ ]:


impfeat = pd.Series(xgbr.feature_importances_, index=X_train_enc.columns)
impcols = impfeat[impfeat>0.005].index
X_train_enc = X_train_enc[impcols]
X_test_enc = X_test_enc[impcols]


# *Model Training*

# In[ ]:


lgbmr = LGBMRegressor(**gscv_lgbm2.best_params_)

lgbmr.fit(X_train_enc, y_train)


# ***Let's predict***

# In[ ]:


preds_test = lgbmr.predict(X_test_enc)


# In[ ]:


output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)

