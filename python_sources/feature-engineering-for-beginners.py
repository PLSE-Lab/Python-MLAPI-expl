#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import stats
import pandas_profiling


# In[ ]:


# import scikit learn modules
from sklearn.model_selection import train_test_split # split the data into train and test
from sklearn.linear_model import LinearRegression # import linear regression model
import sklearn.metrics as metrics # rsquare RMSE, etc.
import statsmodels.formula.api as sm # To build linear regression model using stats models
from patsy import dmatrices


# In[ ]:


data = pd.read_csv("../input/Car_sales.csv")
data.head()


# In[ ]:


data.columns


# In[ ]:


data["Latest_Launch"] = pd.to_datetime(data["Latest_Launch"])


# In[ ]:


data.tail()


# In[ ]:


data.dtypes


# In[ ]:


data["month_name"] = data.Latest_Launch.dt.month_name()
data["weekday"] = data.Latest_Launch.dt.weekday_name
data["date"] = data.Latest_Launch.dt.day.astype(str)
data.head()


# In[ ]:


# pandas_profiling.ProfileReport(data)
# dropping the correlated variables.
data = data.drop(columns=["Power_perf_factor","Price_in_thousands"])


# In[ ]:


numeric_var_names = [key for key, val in dict(data.dtypes).items() if val in ["float64","int64","float32","int32"]]
cat_var_names = [key for key, val in dict(data.dtypes).items() if val in ["object"]]
print(numeric_var_names)
print(cat_var_names)


# In[ ]:


data_num = data[numeric_var_names]
data_cat = data[cat_var_names]


# In[ ]:


def outlier_capping(x):
    x = x.clip_upper(x.quantile(0.99))
    x = x.clip_lower(x.quantile(0.01))
    return x
data_num = data_num.apply(outlier_capping)


# In[ ]:


def Missing_imputation(x):
    x = x.fillna(x.median())
    return x
data_num = data_num.apply(Missing_imputation)


# In[ ]:


def cat_missing_imputation(x):
    x = x.fillna(x.mode())
    return x
data_cat = data_cat.apply(cat_missing_imputation)


# In[ ]:


# converting to numerical 
def create_dummies(df, colname):
    col_dummies = pd.get_dummies(df[colname], prefix = colname, drop_first= True)
    df = pd.concat([df, col_dummies], axis = 1)
    df.drop(colname, axis = 1, inplace = True)
    return df

for c_feature in data_cat.columns:
    data_cat[c_feature] = data_cat[c_feature].astype("category")
    data_cat = create_dummies(data_cat, c_feature)


# In[ ]:


data_cat.columns = data_cat.columns.str.replace(" ", "_")
data_cat.columns = data_cat.columns.str.replace("-", "_")


# In[ ]:


# concatenating to make it a single data_frame 

data_new = pd.concat([data_num, data_cat], axis = 1)
data_new.head()


# In[ ]:


sns.distplot(np.log(data_new.Sales_in_thousands))


# In[ ]:


data_new["ln_carSales"] = np.log(data_new["Sales_in_thousands"])
data_new.head()


# In[ ]:


# sns.heatmap(data_new.corr())
# corrm = data_new.corr()
# corrm.to_csv("corrm.csv")


# In[ ]:


features = data_new[data_new.columns.difference(["ln_carSales","Sales_in_thousands"])]
target = data_new["ln_carSales"]


# In[ ]:


## Feature reduction
data_new.head()


# In[ ]:


features.columns.shape


# ### Applying RFE (Recursive Feature Elimination)

# In[ ]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import itertools
lm = LinearRegression()
rfe = RFE(lm, n_features_to_select=20)
rfe = rfe.fit(features, target)


# In[ ]:


for feature, select in zip(features.columns, rfe.get_support()):
    if select == True:
        print(feature, select)


# In[ ]:


RFE_features = features.columns[rfe.get_support()]
RFE_features


# ### F Regression

# In[ ]:


# Feature Selection based on importance
from sklearn.feature_selection import f_regression
F_values, p_values = f_regression(features, target)


# In[ ]:


f_reg_results = [(i,v,z) for i, v, z in zip(features.columns, F_values, ["%.3f" %p for p in p_values])]
f_reg_results = pd.DataFrame(f_reg_results, columns=["variable", "F_Value", "P_Value"])
f_reg_results


# In[ ]:


f_reg_res = f_reg_results.sort_values(by = "F_Value", ascending = False).reset_index(drop = True).variable
f_reg_res


# In[ ]:


list_vars = set(f_reg_res[:20]).union(set(RFE_features))
list_vars


# ### VIF

# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices


# In[ ]:


all_columns = "+".join(list(list_vars))
my_formula = "ln_carSales ~ " + all_columns
print(my_formula)


# In[ ]:


my_formula = "ln_carSales ~ Model_A8+Model_Accord+date_20+Model_Grand_Cherokee+Horsepower+Model_Cavalier+Model_3000GT+Model_Civic+Model_GTI+Manufacturer_Mercedes_B+Length+Model_SW+Model_CL500+Model_GS400+Manufacturer_Volvo+Model_Carrera_Coupe+Vehicle_type_Passenger+date_28+Model_Diamante+Model_Malibu+Model_Cutlass+Model_Eldorado+Model_Carrera_Cabrio+Model_SLK230+Model_Prowler+Manufacturer_Ford+Model_Viper+Model_Avenger"


# In[ ]:


# get y and x dataframes based on this regression
y, x = dmatrices(my_formula, data_new, return_type = "dataframe")


# In[ ]:


variance_inflation_factor(x.values, 1)


# In[ ]:


# variance for each x, calculate and save in dataframe
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
vif["features"] = x.columns
vif.round(1)


# ### Sampling

# In[ ]:


train, test = train_test_split(data_new, test_size = 0.3, random_state = 123)
print(len(train))
print(len(test))


# In[ ]:


my_formula = "ln_carSales ~ Model_A8+Model_Grand_Cherokee+Horsepower+Model_3000GT+Model_GTI+Length+Model_SW+Model_CL500+Manufacturer_Volvo+Vehicle_type_Passenger+date_28+Model_SLK230+Manufacturer_Ford+Model_Avenger"


# In[ ]:


model = sm.ols(my_formula, data = train)
model = model.fit()


# In[ ]:


print(model.summary())


# In[ ]:


np.exp(model.predict(test))


# In[ ]:


train["pred"] = pd.DataFrame(np.exp(model.predict(train)))
test["pred"] = pd.DataFrame(np.exp(model.predict(test)))


# In[ ]:


train.head()


# ### Accuracy metrics

# In[ ]:


train.head()


# In[ ]:


# Train data
MAPE_train = np.mean(np.abs(train.Sales_in_thousands - train.pred)/train.Sales_in_thousands)
print(MAPE_train)

RMSE_train = metrics.mean_squared_error(train.Sales_in_thousands, train.pred)
print(RMSE_train)

Corr_train = stats.pearsonr(train.Sales_in_thousands, train.pred)
print(Corr_train)

# Test 
MAPE_test = np.mean(np.abs(test.Sales_in_thousands - test.pred)/test.Sales_in_thousands)
print(MAPE_test)

RMSE_test = metrics.mean_squared_error(test.Sales_in_thousands, test.pred)
print(RMSE_test)

Corr_test = stats.pearsonr(test.Sales_in_thousands, test.pred)
print(Corr_test)


# ### Decile analysis
# 

# In[ ]:


train["Decile"] = pd.qcut(train["pred"], 10, labels = False)
train.head()


# In[ ]:


avg_actual = train[["Decile", "Sales_in_thousands"]].groupby(train.Decile).mean().sort_index(ascending = False)["Sales_in_thousands"]
avg_pred = train[["Decile", "pred"]].groupby(train.Decile).mean().sort_index(ascending = False)["pred"]


# In[ ]:


Decile_analysis_train = pd.concat([avg_actual, avg_pred], axis = 1)


# In[ ]:


Decile_analysis_train


# The pattern should be similar for actual and predicted. Hence, the features are not picked correctly.
