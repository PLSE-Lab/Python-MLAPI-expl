# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 23:13:55 2019

@author: alexysp
"""


import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import gc


from sklearn.preprocessing import StandardScaler, Imputer, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
import time
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
import seaborn as sns
#import imblearn
import itertools
import lightgbm as lgb

#declarando caminho dos arquivos csv
caminho_bases = "C:\\Users\\alexysp\\Desktop\\python\\competicao_03\\bases\\"
train = "dataset_treino.csv"
test = "dataset_teste.csv"
stores = "lojas.csv"
variavel_target = "Sales"
plot_charts = False
submission_version = 1

#importando os csvs
dataset_train = pd.read_csv(caminho_bases + train, date_parser = "Date")
dataset_test = pd.read_csv(caminho_bases + test, date_parser = "Date")
df_stores = pd.read_csv(caminho_bases + stores)


#allocating columns names
time_series_features = dataset_train.columns.values
stores_features = df_stores.columns.values

#converting to date time index
dataset_train["Date"] = pd.to_datetime(dataset_train["Date"])
dataset_test["Date"] = pd.to_datetime(dataset_test["Date"])

#merging store infos into the datasets
dataset_train = dataset_train.merge(df_stores, how = "left", on= "Store")
dataset_test = dataset_test.merge(df_stores, how = "left", on= "Store")



#-------Begin exploratory analysis------------
dataset_train.info()


#create year and month columns
dataset_train["year"] = dataset_train["Date"].map(lambda x: pd.to_datetime(x).year)
dataset_train["month"] = dataset_train["Date"].map(lambda x: pd.to_datetime(x).month)
dataset_train["day"] = dataset_train["Date"].map(lambda x: pd.to_datetime(x).day)

dataset_train["year_month"] = dataset_train["Date"].map(lambda x: str(pd.to_datetime(x).year).zfill(4)+str(pd.to_datetime(x).month).zfill(2))
dataset_train["CompetitionOpenSinceYear"] = dataset_train["CompetitionOpenSinceYear"].fillna(0)
dataset_train["CompetitionOpenSinceMonth"] = dataset_train["CompetitionOpenSinceMonth"].fillna(0)

temp1 = dataset_train["CompetitionOpenSinceYear"].map(lambda x: str(int(x)).zfill(4))
temp2 = dataset_train["CompetitionOpenSinceMonth"].map(lambda x: str(int(x)).zfill(2))
dataset_train["year_month_comp_since"] = temp1+temp2



dataset_train["sales_per_customer"] = dataset_train["Sales"]/dataset_train["Customers"]
dataset_train["sales_per_customer"] = dataset_train["sales_per_customer"].fillna(0)

dataset_test["year"] = dataset_test["Date"].map(lambda x: pd.to_datetime(x).year)
dataset_test["month"] = dataset_test["Date"].map(lambda x: pd.to_datetime(x).month)
dataset_test["day"] = dataset_test["Date"].map(lambda x: pd.to_datetime(x).day)

dataset_test["year_month"] = dataset_test["Date"].map(lambda x: str(pd.to_datetime(x).year).zfill(4)+str(pd.to_datetime(x).month).zfill(2))
dataset_test["year_month"] = dataset_test["year_month"].map(lambda x: int(x))
temp1 = dataset_train["CompetitionOpenSinceYear"].map(lambda x: str(int(x)).zfill(4))
temp2 = dataset_train["CompetitionOpenSinceMonth"].map(lambda x: str(int(x)).zfill(2))
dataset_train["year_month_comp_since"] = temp1+temp2

del temp1, temp2
gc.collect()

for year in np.unique(dataset_train["year"].values):
    filter_year = dataset_train["year"] == year
    week_number = dataset_train.loc[filter_year,"Date"].dt.week
    dataset_train.loc[filter_year,"num_week"] = week_number

for year in np.unique(dataset_test["year"].values):
    filter_year = dataset_test["year"] == year
    week_number = dataset_test.loc[filter_year,"Date"].dt.week
    dataset_test.loc[filter_year,"num_week"] = week_number
    
del filter_year
gc.collect()

#null values
null_aux = pd.DataFrame(dataset_train.isnull().sum(), columns = ["num_null"])
null_aux["total_obs"] = len(dataset_train)
null_aux["perc_null"] = null_aux["num_null"]/null_aux["total_obs"]

#total yearlly sales plot
plt.title("Yearly total sales")
sns.barplot(x = "year", y = variavel_target, estimator = np.sum, data = dataset_train)
plt.show()

#daily sales distribution
plt.title("Sales distribution")
sns.distplot(dataset_train[variavel_target], kde = True)
plt.show()

#boxplot of sales
plt.title("Boxplot of sales")
sns.boxplot(data = dataset_train, x = variavel_target)
plt.show()

aux = dataset_train["Store"].value_counts()
aux = aux.sort_values()
store_issue = aux < max(aux)
store_issue.sum()
store_issue = aux[store_issue ].index
store_issue = list(store_issue)

df_store_issue = dataset_train.copy()
df_store_issue =df_store_issue.loc[dataset_train["Store"].isin(store_issue),:]
np.unique(df_store_issue["Store"])

df_store_issue.to_csv(caminho_bases + "issue.csv", index = False)


all_dates = np.unique(dataset_train["Date"])
import datetime
list_append = []

for store in store_issue:
    filter_store = dataset_train["Store"] == store  
    print(store)
    df_ajuste = dataset_train.copy()
    df_ajuste = df_ajuste.loc[filter_store,:]
    df_ajuste_dates = np.unique(df_ajuste["Date"])
    mean_sales_dates = df_ajuste["Sales"].mean()
    dates_issue = filter(lambda x: x not in df_ajuste_dates, all_dates)
    dates_issue = np.array(list(dates_issue))
    dates_solve = pd.to_datetime(dates_issue) - datetime.timedelta(6*30)
    mean_sales_solve = df_ajuste.loc[df_ajuste["Date"].isin(dates_solve),"Sales"].mean()
#    print(dates_issue)
    
    filtro_date = df_ajuste["Date"].isin(dates_solve)
    
    df_temp = df_ajuste.copy()
    df_temp = df_temp.loc[filtro_date,:]
    
    df_temp["Date"] = dates_issue[0:len(df_temp)]
    df_temp["DayOfWeek"] = df_temp["Date"].dt.day
    df_temp["num_week"] = df_temp["Date"].dt.week
    df_temp["Sales"]  = df_temp["Sales"]*mean_sales_dates/mean_sales_solve
    list_append.append(df_temp)

df_append = pd.concat(list_append)

df_append.loc[df_append["Open"]==0, "Sales"] = 0

#dataset_train2 = pd.concat([dataset_train, df_append])
#dataset_train_orig = dataset_train.copy()
#dataset_train = dataset_train2.copy()
#
#del df_append, list_append, dataset_train2



#--------there are outliers, they should be substituted por the previous sales values

dataset_train.loc[dataset_train[variavel_target] == 0,"Open"].value_counts()
#---------- the zeroes are the closed stores days
zeroes_df = dataset_train.loc[dataset_train[variavel_target] == 0,:]
zeroes_df = zeroes_df.loc[zeroes_df["Open"] == 1,:] 


#monthly total sales per year
plt.title("Sales per Month")
sns.lineplot(x = "month", y = variavel_target, data = dataset_train, hue = "year", estimator = "sum")
plt.show()



#monthly total sales per year per type
for region in dataset_train.loc[:, "StoreType"].unique():
    plt.title("Sales per Month - Store Type %s"%(region))
    sns.lineplot(x = "month", y = variavel_target, data = dataset_train.loc[dataset_train.loc[:, "StoreType"] == region,:], hue = "year", estimator = "sum")
    plt.show()

if plot_charts == True:
    #monthly total sales per year per type with promo distinction
    for region in np.sort(dataset_train.loc[:, "StoreType"].unique()):
        plt.title("Sales per Month - Store Type %s"%(region))
        sns.lineplot(x = "year_month", y = variavel_target, data = dataset_train.loc[dataset_train.loc[:, "StoreType"] == region,:], hue = "Promo2", estimator = "mean")
        plt.show()
        
        plt.title("Customer per Month - Store Type %s"%(region))
        sns.lineplot(x = "year_month", y = "Customers", data = dataset_train.loc[dataset_train.loc[:, "StoreType"] == region,:], hue = "Promo2", estimator = "mean")
        plt.show()
        
        plt.title("Sales per Customer per Month - Store Type %s"%(region))
        sns.lineplot(x = "year_month", y = "sales_per_customer", data = dataset_train.loc[dataset_train.loc[:, "StoreType"] == region,:], hue = "Promo2", estimator = "mean")
        plt.show()   
        
        plt.title("Sales per Month & assortment - Store Type %s"%(region))
        sns.lineplot(x = "year_month", y = variavel_target, data = dataset_train.loc[dataset_train.loc[:, "StoreType"] == region,:], hue = "Assortment", estimator = "mean")
        plt.show()
        
        plt.title("Sales per Month & Dayofweek - Store Type %s"%(region))
        sns.lineplot(x = "year_month", y = variavel_target, data = dataset_train.loc[dataset_train.loc[:, "StoreType"] == region,:], hue = "DayOfWeek", estimator = "mean")
        plt.show()
        
        plt.title("Sales per Month & StateHoliday - Store Type %s"%(region))
        sns.barplot(x = "StateHoliday", y = variavel_target, data = dataset_train.loc[dataset_train.loc[:, "StoreType"] == region,:], hue = "year_month", estimator = np.mean)
        plt.show()
        
        plt.title("Sales per Customer per Month - Store Type %s"%(region))
        sns.lineplot(x = "Date", y = "Sales", data = dataset_train.loc[dataset_train.loc[:, "StoreType"] == region,:], hue = "Promo", estimator = "mean")
        plt.show() 
        
        plt.title("Sales per Customer per Month - Store Type %s"%(region))
        sns.lineplot(x = "Date", y = "Sales", data = dataset_train.loc[dataset_train.loc[:, "StoreType"] == region,:], hue = "SchoolHoliday", estimator = "mean")
        plt.show() 
        
        plt.title("Sales per Month & Competition - Store Type %s"%(region))
        sns.lineplot(x = "year_month", y = variavel_target, data = dataset_train.loc[dataset_train.loc[:, "StoreType"] == region,:], hue = "DayOfWeek", estimator = "mean")
        plt.show()

    
#Promointerval variable provides differentiation in stores a e d   
#Assortments provide differentiation as well
#promo augments the sales
#day of the week also has a correlation
#    state holiday has correlation with certain public
    
#variables for keeping:
# "Date", "month", "year", "day", "StoreType",  "Promo2", "Promo", "DayOfWeek", "Assortment", "StateHoliday", "Open"     

#correlation analysis
   
correl = dataset_train.corr()

plt.title("Sales correlation")
correl["Sales"].sort_values().plot(kind = "barh")
plt.show()

def collinear_features(dataset, variables, threshold):
    x = dataset.copy()
    x = x.loc[:, variables]
    correl = x.corr()
    correl = correl.reset_index()
    list_collinear = []
    contador = 1
    for _, row in correl.loc[:,["index"]+variables].iterrows():
        row_name = row["index"]
        colnames = correl.columns.values
        colnames = correl.loc[:,["index"]+variables].columns.values
        
        for col in colnames[contador+1:-1]:
            if col == row_name:
                next
            else:
                if abs(row[col]) > threshold:
                    list_collinear.append(col)
                    
        contador = contador +1
    
    
    return np.unique(list_collinear)
    del x
    gc.collect()

    
collinear = collinear_features(dataset_train,list(dataset_train.columns.values), 0.5)
print(collinear)


def rsmpe(y_true, y_pred):
    y_true = y_true.copy()
    y_true = y_true.reshape(len(y_true),1)
    y_pred = y_pred.copy()
    y_pred = y_pred.reshape(len(y_pred),1)
    
    conc = np.concatenate([y_true, y_pred],axis = 1)
    
    def calc_diff(y_true, y_pred):
        delta = y_true - y_pred
        
        if (delta == 0) & (y_true == 0):
            return 0
        elif (delta != 0) & (y_true == 0):
            return 1
        else:
            return delta/y_true
    
    delta = map(lambda x: calc_diff(x[0],x[1]),conc)
    delta = np.array(list(delta))
    
    mean_square_delta = np.mean(delta**2)
    mean_root_delta = np.sqrt(mean_square_delta)
    return mean_root_delta 

rmspe_scoring = make_scorer(rsmpe, greater_better = False)
dataset_train = dataset_train.sort_values(by = ["Store", "Date"]).reset_index()

#----------Begin variable treatment-----------------
#final_features = ["Store","month", "year", "day", "StoreType",  "Promo2", "Promo", "DayOfWeek", "Assortment", "StateHoliday", "Open"]
#final_features = ["Store","month", "year", "day", "StoreType",  "Promo", "DayOfWeek", "Assortment", "StateHoliday", "Open"]
#final_features = ["Store","month", "year", "num_week", "StoreType",  "Promo", "DayOfWeek", "Assortment", "StateHoliday", "Open", "Promo2SinceWeek", "flag_holiday"]

final_features = ["Store","month", "year", "num_week", "StoreType", \
                  "Promo", "DayOfWeek", "Assortment","Open", "Promo2SinceWeek",  "Promo2",\
                  "CompetitionOpenSinceMonth","CompetitionOpenSinceYear","CompetitionDistance"]
#final_features = ["Store","month", "year", "day", "num_week", "StoreType",  "Promo", "Assortment","Open", "Promo2SinceWeek"]


dataset_train_2 = dataset_train.copy()
dataset_test_2 = dataset_test.copy()
dataset_train_2["flag_holiday"] = dataset_train_2["StateHoliday"].map(lambda x: 0 if str(x).strip()== "0" else 1)
dataset_train_2["Promo2SinceWeek"] = dataset_train_2["Promo2SinceWeek"].fillna(0)
dataset_train_2["CompetitionOpenSinceMonth"] = dataset_train_2["CompetitionOpenSinceMonth"].fillna(0)
dataset_train_2["CompetitionOpenSinceYear"] = dataset_train_2["CompetitionOpenSinceYear"].fillna(1900)
dataset_train_2["CompetitionDistance"] = dataset_train_2["CompetitionDistance"].fillna(0)
 

dataset_test_2["flag_holiday"] = dataset_test_2["StateHoliday"].map(lambda x: 0 if str(x).strip()== "0" else 1)
dataset_test_2["Promo2SinceWeek"] = dataset_test_2["Promo2SinceWeek"].fillna(0) 
dataset_test_2["CompetitionOpenSinceMonth"] = dataset_test_2["CompetitionOpenSinceMonth"].fillna(0)
dataset_test_2["CompetitionOpenSinceYear"] = dataset_test_2["CompetitionOpenSinceYear"].fillna(1900)
dataset_test_2["CompetitionDistance"] = dataset_test_2["CompetitionDistance"].fillna(0)
 


#create original sales variable
dataset_train_2[variavel_target+"_orig"] =dataset_train_2[variavel_target]

#treat 0 value for sales when the store was open
filter_zero = (dataset_train_2["Open"] == 1) & (dataset_train_2[variavel_target+"_orig"] == 0)
dataset_train_2.loc[filter_zero, variavel_target] = np.nan
dataset_train_2[variavel_target] = dataset_train_2[variavel_target].interpolate(method = "linear")

#treating the outliers
first_quartile = dataset_train_2.loc[dataset_train_2["Open"] == 1,variavel_target+"_orig"].quantile(0.25)
third_quartile = dataset_train_2.loc[dataset_train_2["Open"] == 1,variavel_target+"_orig"].quantile(0.75)
iqr = third_quartile - first_quartile
upper_bound = third_quartile +3*iqr
lower_bound = max(first_quartile -3*iqr, 0)
upper_bound
lower_bound

def fill_outlier(x, lower_bound , upper_bound, iqr):
    if x < lower_bound:
        return lower_bound
    elif x > upper_bound:
        return upper_bound
    else:
        return x

dataset_train_2.loc[dataset_train_2["Open"] == 1,variavel_target] = dataset_train_2.loc[dataset_train_2["Open"] == 1,variavel_target+"_orig"].map(lambda x: fill_outlier(x,lower_bound, upper_bound,iqr))

#plotting the new adjusted sales versus the original
plt.title("Sales Distribution Adjusted x Original")
sns.distplot(dataset_train_2[variavel_target], hist = True, label = "Adjusted")
sns.distplot(dataset_train_2[variavel_target+"_orig"], hist = True, label = "original")
plt.show()


for col in final_features:
    temp = dataset_train_2.copy()
    temp = temp[col]
    
    print("\nvarible %s - num missing = %i"%(col, temp.isnull().sum()))
    print(temp.value_counts())

dataset_train_2["StateHoliday"] = dataset_train_2["StateHoliday"].map(lambda x: str(x).strip())
dataset_test_2["StateHoliday"] = dataset_test_2["StateHoliday"].map(lambda x: str(x).strip())
dataset_test_2["Open"] = dataset_test_2["Open"].fillna(1)


col_for_encoding = ["StoreType", "Assortment", "StateHoliday"]

for col in col_for_encoding:
    encoder = LabelEncoder()
    encoder.fit(dataset_train_2[col].values)
    dataset_train_2[col] = encoder.transform(dataset_train_2[col])
    dataset_test_2[col] = encoder.transform(dataset_test_2[col])

dataset_train_2["Date"] = pd.to_datetime(dataset_train_2["Date"])
dataset_test_2["Date"] = pd.to_datetime(dataset_test_2["Date"])


dataset_train_2["flag_train_test"],dataset_test_2["flag_train_test"] = "train", "test"

dataset_aux = pd.concat([dataset_train_2, dataset_test_2])

groupby_list = ["Open" ,"Store", "DayOfWeek","Promo"]

dataset_aux = dataset_aux.sort_values(by =  ["Date"] + groupby_list )

#moving_60 = dataset_aux.set_index("Date").groupby(by = groupby_list)["Sales"].rolling("60d").mean().fillna(0)

moving_90 = dataset_aux.set_index("Date").groupby(by = groupby_list)["Sales"].rolling("90d").mean().fillna(0)
moving_120 = dataset_aux.set_index("Date").groupby(by = groupby_list)["Sales"].rolling("120d").mean().fillna(0)
moving_180 = dataset_aux.set_index("Date").groupby(by = groupby_list)["Sales"].rolling("180d").mean().fillna(0)
moving_360 = dataset_aux.set_index("Date").groupby(by = groupby_list)["Sales"].rolling("360d").mean().fillna(0)


#dataset_aux["moving_60"] = moving_60.values
dataset_aux["moving_90"] = moving_90.values
dataset_aux["moving_120"] = moving_120.values
dataset_aux["moving_180"] = moving_180.values
dataset_aux["moving_360"] = moving_360.values


dataset_train_2 = dataset_aux.loc[dataset_aux["flag_train_test"] == "train",:]
dataset_test_2 = dataset_aux.loc[dataset_aux["flag_train_test"] == "test",:]

final_features = final_features + ["moving_90", "moving_120", "moving_180", "moving_360"]

months_testing = np.arange(1,13)   
years_testing = [2013,2014,2015]
#months_testing = [1,2,3,4,5,6,7,12] 
#num_days_prediction = len(dataset_test_2["Date"].unique())



params = {"objective": ["regression"], # for linear regression
          "boosting " : ["gbdt"],   # use tree based models 
          "eta": [ 0.03],   # learning rate
          "max_depth": [7,8],    # maximum depth of a tree
          "metric": ["root_mean_squared_error"],
          "num_leaves":[150]
          }

def combinations_dictionary(dictionary):
    
    keys = dictionary.keys()
    values = (dictionary[key] for key in keys)
    combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
    return combinations

parameter_list = combinations_dictionary(params)

parameter_list
list_df_validation =[]
list_df_results = []
list_df_treino = []
#training_size = np.arange(0.5,1,0.1)
train_size = 1
contador = 0
num_boost_round = 15000

for params in parameter_list:
    inicio = time.time()
    
    num_days_prediction = len(dataset_test_2["Date"].unique())
#    train_size = 0.75
    
    filter_month = dataset_train_2["month"].isin(months_testing) & dataset_train_2["year"].isin(years_testing)
    days_training = np.sort(dataset_train_2.loc[filter_month,"Date"].unique())
    num_days_training = len(days_training)
    
    range_days_training = days_training[0: (num_days_training -num_days_prediction )]
    begin = len(range_days_training)*(1-train_size)
    begin = int(begin )
    range_days_training = range_days_training[begin:]
    len(range_days_training)
    
    range_days_training_validation = days_training[(num_days_training -num_days_prediction ):]
    
    filter_x_train = (dataset_train_2["Date"].isin(range_days_training)) & (dataset_train_2["Open"] != 0) & (dataset_train_2["Sales"] != 0)
    x_train = dataset_train_2.copy()
    x_train = x_train.loc[filter_x_train,:]
    y_train = x_train.loc[:, variavel_target]
    x_train = x_train.loc[:,final_features]
    
    filter_x_train_validation = dataset_train_2["Date"].isin(range_days_training_validation) & (dataset_train_2["Open"] != 0) &  (dataset_train_2["Sales_orig"] != 0)
    filter_x_train_validation.sum()
    x_train_validation = dataset_train_2.copy()
    x_train_validation = x_train_validation.loc[filter_x_train_validation,:]
    y_train_validation =  x_train_validation.loc[:, variavel_target+"_orig"]
    x_train_validation = x_train_validation.loc[:,final_features]
    
    dataset_for_validation = dataset_train_2.copy()
    dataset_for_validation = dataset_for_validation.loc[filter_x_train_validation,:]
    dataset_for_training = dataset_train_2.copy()
    dataset_for_training = dataset_for_training.loc[filter_x_train,:]
    
    del filter_x_train, filter_x_train_validation
    gc.collect()
    
    
    def rmspe_number(y, yhat):
        return np.sqrt(np.mean((yhat/y-1) ** 2))
    
    
    def rmspe_lgb(yhat, y):
    # define eval metrics

        y = np.expm1(y.get_label())
        yhat = np.expm1(yhat)
        return "rmspe", rmspe_number(y,yhat), False
   
    
    dtrain_lgb = lgb.Dataset(x_train.values, np.log1p(y_train.values))
    dvalid_lgb = lgb.Dataset(x_train_validation.values, np.log1p(y_train_validation.values))
    
    model_lgb = lgb.train(params, dtrain_lgb,num_boost_round, dvalid_lgb,  early_stopping_rounds= 100,\
                          feval = rmspe_lgb)
    
    prediction_lgbm = model_lgb.predict(x_train_validation.values)
    prediction_lgbm = np.expm1(prediction_lgbm)
    pred_train = model_lgb.predict(x_train.values)
    pred_train = np.expm1(pred_train)
    
    dataset_for_validation["forecast"] = prediction_lgbm
    dataset_for_training["forecast"] = pred_train
    
    plt.title("forecast x target")
    sns.lineplot(x = "Date", y = "forecast", data = dataset_for_validation, label = "forecast", color = "b")
    sns.lineplot(x = "Date", y = variavel_target+"_orig", data = dataset_for_validation, label = "target", color = "r")
    plt.show()
    
    plt.title("forecast x target")
    sns.distplot(dataset_for_validation["forecast"], label = "forecast", color = "b")
    sns.distplot(dataset_for_validation[variavel_target+"_orig"],label = "target", color = "r")
    plt.show()
    
    def weight_correction(prediction, y_train_validation, weights):
        errors = []
        for w in weights:
            error = rmspe_number(prediction*w, y_train_validation)
            errors.append(error)
        
        errors = np.array(errors)
        weights = np.array(weights)
        plt.title("Errors")
        sns.pointplot( x = weights, y = errors)
        plt.show()
        
        return {"error":errors, "weight":weights}
    
    weights = np.linspace(0.9,1.1,200)      
    dicionario_weights = weight_correction(prediction_lgbm, y_train_validation,weights)
    df_weights = pd.DataFrame(dicionario_weights)
    df_weights = df_weights.sort_values(by = "error", ascending = True).reset_index()
    weight_for_correction = df_weights.loc[0,"weight"]
    
    dataset_for_validation["forecast_weight"] = dataset_for_validation["forecast"]*weight_for_correction
    dataset_for_validation["train_size"] = train_size
    dataset_for_validation["weight_for_correction"] = weight_for_correction
    
    plt.title("forecast x target")
    sns.lineplot(x = "Date", y = "forecast", data = dataset_for_validation, label = "forecast", color = "b")
    sns.lineplot(x = "Date", y = "forecast_weight", data = dataset_for_validation, label = "forecast wieght", color = "y")
    sns.lineplot(x = "Date", y = variavel_target+"_orig", data = dataset_for_validation, label = "target", color = "r")
    plt.show()
    
    plt.title("forecast x target")
    sns.lineplot(x = "year_month", y = "forecast", data = dataset_for_validation, label = "forecast", color = "b")
    sns.lineplot(x = "year_month", y = "forecast_weight", data = dataset_for_validation, label = "forecast wieght", color = "y")
    sns.lineplot(x = "year_month", y = variavel_target+"_orig", data = dataset_for_validation, label = "target", color = "r")
    plt.show()
    
    plt.title("forecast x target")
    sns.distplot(dataset_for_validation["forecast"], label = "forecast", color = "b")
    sns.distplot(dataset_for_validation["forecast_weight"], label = "forecast_weight", color = "y")
    
    sns.distplot(dataset_for_validation[variavel_target+"_orig"],label = "target", color = "r")
    plt.show()
    
    rsmpe1 = rsmpe(dataset_for_validation["forecast_weight"].values, dataset_for_validation[variavel_target+"_orig"].values)
    rsmpe2 = rsmpe(dataset_for_validation["forecast"].values, dataset_for_validation[variavel_target+"_orig"].values)
    
    print("Forecast weight %f = %f"%(weight_for_correction, rsmpe1))
    print("Forecast no weight = %f"%(rsmpe2))
    
    dataset_for_validation["rsmpe"] = rsmpe2
    dataset_for_validation["rsmpe_weight"] = rsmpe1
    dataset_for_validation["idmodelo"] = contador
    
    dataset_for_training["idmodelo"] = contador
    
    
    
   

    begin = len(days_training)*(1-train_size)
    range_days_training = days_training[int(begin):]
    filter_x_train_xgb = dataset_train_2["Date"].isin(range_days_training) & (dataset_train_2["Sales"] != 0)
    x_train_final_xgb = dataset_train_2.copy()
    x_train_final_xgb = x_train_final_xgb.loc[filter_x_train_xgb,:]   
    y_train_final_xgb = x_train_final_xgb.loc[:,variavel_target]
    x_train_final_xgb  = x_train_final_xgb.loc[:,final_features]   
    x_test = dataset_test_2.copy()
    x_test = x_test.loc[:,final_features]
    
    dtrain = lgb.Dataset(x_train_final_xgb, np.log1p(y_train_final_xgb))
    dtest = lgb.Dataset(x_test.values)
    model = lgb.train(params, dtrain, num_boost_round)
    
    prediction = model.predict(x_test.values)
    prediction = np.expm1(prediction)
    data_out = dataset_test_2.copy()
    data_out["Sales"] = prediction
    df_weights = df_weights.sort_values(by = "error", ascending = True).reset_index()
    weight =  df_weights.loc[0,"weight"]
    data_out.loc[data_out["Open"] == 0, "Sales"] = 0
    data_out["Sales_weight"] =data_out["Sales"]*weight
    data_out["train_size"] = train_size
    data_out["rsmpe"] = rsmpe2
    data_out["rsmpe_weight"] = rsmpe1

    sns.barplot(x = "Open", y= "Sales", data = data_out)
    sns.barplot(x = "Open", y= "Sales_weight", data = data_out)    
    sns.distplot(data_out["Sales"], hist = False, kde = True, label = "xgb", color = "r" )
    sns.distplot(data_out["Sales_weight"], hist = False, kde = True, label = "xgb_weight", color = "y" )
    dataset_for_validation["param_id"] = contador
    data_out["param_id"] = contador
    
    list_df_validation.append(dataset_for_validation)
    list_df_results.append(data_out)
    list_df_treino.append(dataset_for_training)
    fim = time.time()
    
    data_out.to_csv(caminho_bases +"lgbm_ensemble01_"+str(contador)+".csv",index = False)
    
    delta = fim - inicio
    delta = delta/60
    print("tempo execucação = %f"%(delta))
    contador = contador+1
    

df_train = pd.concat(list_df_treino)
df_val = pd.concat(list_df_validation)
df_results = pd.concat(list_df_results)
