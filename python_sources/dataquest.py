#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr


# In[ ]:



# /kaggle/input/dataquest2020/energy_test.csv
# /kaggle/input/dataquest2020/energy_train.csv
# /kaggle/input/dataquest2020/sample_submission.csv

df_train = pd.read_csv('/kaggle/input/dataquest2020/energy_train.csv')
df_test = pd.read_csv("/kaggle/input/dataquest2020/energy_test.csv")
df_sub = pd.read_csv("/kaggle/input/dataquest2020/sample_submission.csv")


# **Data Analysis**

# In[ ]:



df_train.columns


# In[ ]:



df_train['WattHour'].describe()


# In[ ]:


df_train.head()


# In[ ]:


sns.distplot(df_train['WattHour']);


# In[ ]:


sns.boxplot(df_train['WattHour']);


# In[ ]:



print("Skewness: %f" % df_train['WattHour'].skew())
print("Kurtosis: %f" % df_train['WattHour'].kurt())


# In[ ]:



corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);


# In[ ]:


corrmat.WattHour 


# In[ ]:



k = 10 
cols = corrmat.nlargest(k, 'WattHour')['WattHour'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[ ]:


df_train.isnull().sum()


# In[ ]:


df_train.describe()


# In[ ]:


WattHour_scaled = StandardScaler().fit_transform(df_train['WattHour'][:,np.newaxis]);
low_range = WattHour_scaled[WattHour_scaled[:,0].argsort()][:10]
high_range= WattHour_scaled[WattHour_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)


# In[ ]:


var = 'luminousity'
data = pd.concat([df_train['WattHour'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='WattHour', ylim=(0,1800));


# In[ ]:


var = 'Pressure'
data = pd.concat([df_train['WattHour'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='WattHour', ylim=(0,1800));


# In[ ]:


var = 'moisture_out'
data = pd.concat([df_train['WattHour'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='WattHour', ylim=(0,1800));


# In[ ]:


#bivariate analysis saleprice/grlivarea
var = 'Wind'
data = pd.concat([df_train['WattHour'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='WattHour', ylim=(0,1800));


# In[ ]:


var = 'Clarity'
data = pd.concat([df_train['WattHour'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='WattHour', ylim=(0,1800));


# In[ ]:


values = data.values
i = 1
groups=[1]
plt.figure()
for group in groups:
    plt.subplot(len(groups), 1, i)
    plt.plot(values[:, group])
    plt.title(data.columns[group], y=1, loc='right')
    i += 1
plt.show()


# **Data Preprocessing**

# In[ ]:


df_test['WattHour'] = df_sub['WattHour']
new_train = pd.concat((df_train,
                      df_test))
df_test = df_test.drop(['WattHour'],1)
wat = new_train['WattHour']
new_train = new_train.set_index("WattHour")
new_train = new_train.drop(0, axis=0)
new_train = new_train.reset_index()
new_train = new_train.drop(columns= ["random_variable_1","random_variable_2"])
df_train_1 = new_train
df_test_1 = df_test
df_test_1 = df_test_1.drop(columns = ["random_variable_1","random_variable_2"])
all_data = pd.concat((df_train_1.drop(['WattHour'],1),
                      df_test_1))


# In[ ]:


import datetime
all_data['date'] = [datetime.datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in all_data["date"]]
all_data['Date'] = [datetime.datetime.date(d) for d in all_data['date']] 
all_data['Time'] = [datetime.datetime.time(d) for d in all_data['date']]


# In[ ]:


all_data = all_data.drop(columns = ["date"])


# In[ ]:


df_train_1["WattHour"] = np.log1p(df_train_1["WattHour"])

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
print(numeric_feats)
skewed_feats = df_train_1[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index
all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
all_data = pd.get_dummies(all_data)
all_data = all_data.fillna(all_data.mean())


# **Model Training**

# In[ ]:


X_train = all_data[:df_train_1.shape[0]]
X_test = all_data[df_train_1.shape[0]:]
y = df_train_1.WattHour


# In[ ]:


from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)


# In[ ]:


model_ridge = Ridge()


# In[ ]:


alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
# alphas = [15]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 
            for alpha in alphas]


# In[ ]:


cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")


# In[ ]:


cv_ridge.min()


# In[ ]:


rmse_cv(model_ridge).mean()


# In[ ]:


model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)


# In[ ]:


rmse_cv(model_lasso).mean()


# In[ ]:


coef = pd.Series(model_lasso.coef_, index = X_train.columns)


# In[ ]:


print("Lasso picked" + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")


# In[ ]:


imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])


# In[ ]:


matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")


# In[ ]:


matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)

preds = pd.DataFrame({"preds":model_lasso.predict(X_train), "true":y})
preds["residuals"] = preds["true"] - preds["preds"]
preds.plot(x = "preds", y = "residuals",kind = "scatter")


# In[ ]:


import xgboost as xgb


# In[ ]:


dtrain = xgb.DMatrix(X_train, label = y)
dtest = xgb.DMatrix(X_test)

params = {"max_depth":2, "eta":0.1}
model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)


# In[ ]:


model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()


# In[ ]:


model_xgb = xgb.XGBRegressor(n_estimators=1500, max_depth=3, learning_rate=0.1)
model_xgb.fit(X_train, y)


# **Predicting Output**

# In[ ]:


xgb_preds = np.expm1(model_xgb.predict(X_test))
lasso_preds = np.expm1(model_lasso.predict(X_test))


# In[ ]:


np.expm1(xgb_preds)


# In[ ]:


predictions = pd.DataFrame({"xgb":xgb_preds, "lasso":lasso_preds})
predictions.plot(x = "xgb", y = "lasso", kind = "scatter")


# In[ ]:


preds1 = 0.5*lasso_preds + 0.5*xgb_preds
preds2 = 0.7*lasso_preds + 0.3*xgb_preds
preds3 = 0.3*lasso_preds + 0.7*xgb_preds


# In[ ]:


predictions = pd.DataFrame({"xgb":preds3, "lasso":preds1})
predictions.plot(x = "xgb", y = "lasso", kind = "scatter")


# **Creating Submission file**

# In[ ]:


re=pd.read_csv("/kaggle/input/dataquest2020/sample_submission.csv")
y_pred=[]
for i in preds1 : 
    i  = int(i)
    y_pred.append(i)

for i in range(len(re.WattHour)):
    if re.WattHour[i]!= 0:
        y_pred[i] = re.WattHour[i]    
        
df=pd.DataFrame(y_pred)


re=re.drop(['WattHour'],axis=1)
re.insert(1,"WattHour",df)
re.to_csv("result6.csv", index = False)

