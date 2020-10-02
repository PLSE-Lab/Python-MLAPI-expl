#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Input data files are available in the read-only "../input/" directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# <h1 align="center"><u><b> Employee Attrition Rate Predictor</b></u></h1>
# <h3 align="center"> Hackerearth machine learning challenge </h3>
# 
# ### Problem Statement:
# Employees are the most important part of an organization. Successful employees meet deadlines, make sales, and build the brand through positive customer interactions.
# 
# Employee attrition is a major cost to an organization and predicting such attritions is the most important requirement of the Human Resources department in many organizations. In this problem, your task is to predict the attrition rate of employees of an organization.

# In[ ]:


train_dataset_path = "/kaggle/input/Dataset/Train.csv"
test_dataset_path  = "/kaggle/input/Dataset/Test.csv"


# In[ ]:


# Data manipulation libraries
import pandas as pd
import numpy as np

# Data visualistaion libraries
import seaborn as sns
import matplotlib.pyplot as plt
pd.plotting.register_matplotlib_converters()
get_ipython().run_line_magic('matplotlib', 'inline')

# Machine learning libraries
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score

# Machine Learning Models
from sklearn.ensemble import RandomForestRegressor


# In[ ]:


# Both training and testing data
training_dataframe = pd.read_csv(train_dataset_path,index_col="Employee_ID")
testing_dataframe  = pd.read_csv(test_dataset_path,index_col ="Employee_ID")


# ### To gain a insight into the data

# In[ ]:


training_dataframe.describe()


# In[ ]:


## Data Distribution
training_dataframe.hist(bins = 50, figsize = (20,15))


# In[ ]:


print(training_dataframe)


# In[ ]:


print(training_dataframe.Travel_Rate)


# In[ ]:


# All the columns
print(training_dataframe.columns)


# #### First and last few rows

# In[ ]:


training_dataframe.head()


# In[ ]:


training_dataframe.tail()


# ## Preprocessing and Data Wrangling

# To deal with columns having missing values we will perform imputation on the column with missing data. We might have dropped the column entirely but in that case the model might lost a lot of data, unless most values of the dropped column are missing.
# We will first of all find out the colunms with missing data.

# ### Missing Values

# In[ ]:


## Columns with missing values
cols_with_missing = [col for col in training_dataframe.columns if training_dataframe[col].isnull().any()]
print(cols_with_missing)


# We will try to perform imputation on the missing values. But before imputation we will have to check for categorical values.

# ### Categorical Values
# To deal with categorical values we can drop the column if not necessary, label encode the variables or perform ohe hot encoding.

# In[ ]:


# Get a list of categorical columns

s = (training_dataframe.dtypes == 'object')
object_cols = list(s[s].index)
print(object_cols)


# In[ ]:


# Remove rows with missing target, separate target from predictors in training data
X_train_full = training_dataframe

X_train_full.dropna(axis=0, subset=['Attrition_rate'], inplace=True, how = "any")
y_train_full = X_train_full.Attrition_rate
# X_train_full.drop(['Attrition_rate'], axis=1, inplace=True)


X_train_full.head()


# Now we will label encode the categorical columns. Fitting a label encoder to a column in the training data creates a corresponding integer-valued label for each unique value that appears in the training data. In case, the validation data contains values that don't also appear in the training data, the encoder will throw an error, because these values won't have an integer assigned to them.

# In[ ]:


X_valid_full = testing_dataframe

labelEncoder = LabelEncoder()
for col in object_cols:
    X_train_full[col] = labelEncoder.fit_transform(X_train_full[col])
    X_valid_full[col] = labelEncoder.transform(X_valid_full[col])
    


# In[ ]:


X_train_full.head()


# In[ ]:


X_valid_full.head()


# ## Imputation

# In[ ]:


si = SimpleImputer(strategy='most_frequent')
X_train_imputed = pd.DataFrame(si.fit_transform(X_train_full))
# It is important to fit_transform the training data


# Imputation removed column names; they have to be put back
X_train_imputed.columns = X_train_full.columns
# X_train_imputed.index = X_train_full.index
print("shape of X_train_imputed = ",X_train_imputed.shape)
X_train_imputed.head()


# ***Imputation removes column names so we had to put them back***

# In[ ]:


print("shape of y_train_full = ",y_train_full.shape)


# In[ ]:


print("Shape of X_valid_full = ",X_valid_full.shape )


# In[ ]:


X_train_imputed.head()


# ## Data Visualization and analysis

# ### 1. First let's checkout the relation between attrition rate and Age

# In[ ]:


data = X_train_imputed
ind_var = y_train_full

plt.figure(figsize=(20,6))
plt.title("Attrition Rate Vs Age")

sns.lineplot(y=data["Attrition_rate"],x=data["Age"])


# **From this trend it is evident that employee attrition rate is more profound between the age of 35 to 40**. This may be due to a variety of factors such as low work life ballance or less job satisfaction

# ### 2. Let us consider the factor of time since last promotion which might help us gain further insights to justify our previous assumption.

# In[ ]:


plt.figure(figsize=(20,6))
plt.title("Attrition Rate Vs time_since_promotion")
sns.lineplot(y=data["Attrition_rate"],x=data["Time_since_promotion"])


# **We see that the attrition rate linearly dependent on the time since last promotion but increases a little bit if gap increases more than 3.5 years**

# ### 3. **Next let's see the effect of time of sevice on the attrition rate of the employee**

# In[ ]:


plt.figure(figsize=(20,6))
plt.title("Attrition Rate Vs time_of_service")
sns.lineplot(y=data["Attrition_rate"],x=data["Time_of_service"])


# **This relationship may be better understood using a 2D KDE plot**

# In[ ]:


sns.jointplot(x=data["Time_of_service"],y=data["Attrition_rate"],kind="kde")


# **This plot shows a strong relation between the two variables. Atrrition rate remains very low for low years of service but changes on increasing the term of service. This will be a very nice feature variable for our prediction**

# ### 4. **Lets us also compare the atrition rates for male and female employees.**

# In[ ]:


plt.figure(figsize= (8,6))
plt.title("Attrition Rate Vs Gender")
sns.barplot(y=data["Attrition_rate"],x=data["Gender"])
plt.legend(title='Gender', loc='lower left', labels=['Male : 1', 'Female : 0'])


# **From this bar plot it is evident that that Male are more likely to leave an organisation than their female counterparts. Still, the difference is not huge enough to have any signifigant contribution to the attrition rates.**

# ### 5. **Next let us understand the effects of eductaion level on our target variable i.e attrition rate. We may expect a linear realtionship.**

# In[ ]:


plt.figure(figsize=(20,6))
plt.title("Attrition Rate Vs eductaion_level")
sns.lineplot(y=data["Attrition_rate"],x=data["Education_Level"])


# **From the plot it is evident that attrition rates increases with a low level of education but the curve flatens out with an increase in the level of education. This might an important feature for predicting our target variable.**

# ### 6. **Let us understand the most important factor in prediction the attrition rates in a company. This may be the work life balance score. We may assume the trend to be linear. Let's see what the stats have to say about this**

# In[ ]:


plt.figure(figsize= (15,10))
plt.title("Attrition Rate Vs Work life balance")
sns.barplot(y=data["Attrition_rate"],x=data["Work_Life_balance"])


# In[ ]:


plt.figure(figsize=(20,6))
plt.title("Attrition Rate Vs work_life_balance")
sns.lineplot(y=data["Attrition_rate"],x=data["Work_Life_balance"])


# **According to our assumption work life balance does count towards the attrition rates of employees. There is a slight increase in the rates with an increase in work_life_balance value. There is a linear relation between the two**

# ### 7. Now we should analyze the growth rate of an employee in an organisation. Let's assume that the attrition rates will be inversely proportional to the growth rate

# In[ ]:


plt.figure(figsize=(20,6))
plt.title("Attrition Rate Vs growth rate(%)")
sns.lineplot(y=data["Attrition_rate"],x=data["growth_rate"])


# **This plot shows a non linear relation between attrition Rate and growth rate. This is in sharp contrast to our earlier assumption.This can be further understood using a 2D Kde plot**

# In[ ]:


sns.jointplot(x=data["growth_rate"],y=data["Attrition_rate"],kind="kde")


# **From this 2D KDE plot it is evident that attrition rate is maximum between a growth rate of 20 - 40 and 50 - 70. Hence this would be an important variable in the prediction of the the target.**

# ### 8. **Relationship between attrition rates and the job unit.**

# In[ ]:


plt.figure(figsize= (15,10))
plt.title("Attrition Rate Vs Job Unit")
sns.barplot(y=data["Attrition_rate"],x=data["Unit"])


# **Thus the job department or unit such as IT, logistics, Quality control and HR is a good feature for prediction of our target**

# ### 9. **Next in our list is a very important factor: Pay Scale.**

# In[ ]:


plt.figure(figsize=(20,6))
plt.title("Attrition Rate Vs Pay scale")
sns.lineplot(y=data["Attrition_rate"],x=data["Pay_Scale"])


# In[ ]:


plt.figure(figsize= (15,10))
plt.title("Attrition Rate Vs Pay_scale")
sns.barplot(y=data["Attrition_rate"],x=data["Pay_Scale"])


# **Initially there is a increase in attrition rates with respect to pay scale (till 3) then it slowly decreases and becomes stable over rest of the values and decreases beyond 8. Pay scale will be an interesting feature for our model.**

# ### 10. Next is the realtionship with decision making skills of the employess.

# In[ ]:


plt.figure(figsize= (15,10))
plt.title("Attrition Rate Vs decision making skills")
sns.barplot(y=data["Attrition_rate"],x=data["Decision_skill_possess"])


# **As we see there is not much variation with the attrition rates in comparision with the decision making skills like conceptual, analytical and directive(Label encoded) of the employees.**

# ### 11. Next let's check the effect of travel on attrition rates

# In[ ]:


plt.figure(figsize=(20,6))
plt.title("Attrition Rate Vs travel rate")
sns.lineplot(y=data["Attrition_rate"],x=data["Travel_Rate"])


# In[ ]:


plt.figure(figsize= (15,10))
plt.title("Attrition Rate Vs Travel rate")
sns.barplot(y=data["Attrition_rate"],x=data["Travel_Rate"])


# ## 12. Relationship with Compensation and benefits

# In[ ]:


plt.figure(figsize= (15,10))
plt.title("Attrition Rate Vs Compensation_and_Benefits")
sns.barplot(y=data["Attrition_rate"],x=data["Compensation_and_Benefits"])


# **This shows that compensation and benefits are an important factor in deciding employee attrition rates**

# ### 13. Now taking into account the various Anominised variables (VAR1, VAR2, VAR3, VAR4, VAR5, VAR6, VAR7)
# 

# In[ ]:


plt.title("Attrition Rate Vs Anominised variables")
sns.lineplot(y=data["Attrition_rate"],x=data["VAR1"])


# In[ ]:


sns.lineplot(y=data["Attrition_rate"],x=data["VAR2"])


# In[ ]:


sns.lineplot(y=data["Attrition_rate"],x=data["VAR3"])


# In[ ]:


sns.lineplot(y=data["Attrition_rate"],x=data["VAR4"])


# In[ ]:


sns.lineplot(y=data["Attrition_rate"],x=data["VAR5"])


# In[ ]:


sns.lineplot(y=data["Attrition_rate"],x=data["VAR6"])


# In[ ]:


sns.lineplot(y=data["Attrition_rate"],x=data["VAR7"])


# **We need to plot the correlation matrix to study the Anominised variables**

# ### Let's develop the Correlation Matrix with heatmap
# Correlation states how the features are related to each other or the target variable. Heatmap makes it easy to identify which features are most related to the target variable

# ### Looking for correaltions

# In[ ]:


corr_matrix = data.corr()
corr_matrix["Attrition_rate"].sort_values(ascending = False)


# The values which are close to +1 show a very strong positive correaltion with the target variable,i.e directly proportional and for values close to -1 it shows a strong negative correlation ,i.e inversely proportional. For values lying near zero show that there is no linear correlation with the variable.

# In[ ]:


sample_data = data
plt.figure(figsize = (30,10))
corr = sample_data.corr()
ax = sns.heatmap(corr,vmin = -0.03,vmax = 0.03, center = 0,cmap=sns.diverging_palette(20, 220, n=200), square=True, linewidths = 0.5)

ax.set_xticklabels( ax.get_xticklabels(),rotation=45, horizontalalignment='right')


# ### Fianlly the columns selected for our model would be:
# ***From the last row of the correlation heat map we see the columns highly correlated to Attrition_rate are:***
# * Gender
# * Relationship_Status
# * Hometown
# * Unit
# * Decision_skill_possess
# * Time_since_promotion
# * growth_rate
# * Post_Level
# * Work_Life_balance
# * Compensation_and_Benefits
# * VAR 2
# * VAR 7
# * Time of service
# * Pay scale
# * age
# * Travel Rate

# ## Building the ML model

# **Let's split the training_dataframe into training and testing subsets.**

# In[ ]:


features_1 = ["Compensation_and_Benefits","Travel_Rate","Pay_Scale",
              "Unit","growth_rate","Education_Level","Time_of_service","Age"]

features_2 = ["Gender","Relationship_Status","Hometown","Unit",
              "Decision_skill_possess","Time_since_promotion",
              "growth_rate","Post_Level","Work_Life_balance"]     #Strong positive correaltion

features_3 = data.columns[:-1]

features_4 =["Gender","Unit","Work_Life_balance","Decision_skill_possess","Post_Level","growth_rate","Time_since_promotion","Travel_Rate",                 
"VAR4","Age","Pay_Scale", "VAR7","Time_of_service", "VAR2","Compensation_and_Benefits"]      # Strong negative and positive correaltion

features_5 =["Gender","Unit","Work_Life_balance","Decision_skill_possess","Post_Level","growth_rate","Time_since_promotion","Travel_Rate",                 
"VAR4","Age","Pay_Scale", "VAR7","Time_of_service", "VAR2","Compensation_and_Benefits","Relationship_Status"
            ,"Education_Level","VAR1"] 

X = data[features_4]
y = data["Attrition_rate"]

# Separating validation from training data
train_X, val_X, train_y, val_y = train_test_split(X,y,train_size=0.7, test_size=0.3,random_state = 0)


# ### We will test out various machine learning algortihms and finally select the best out of them.

# ### Model_1 : Random Forest

# In[ ]:


# Model selection and Training

random_forest_model = RandomForestRegressor(random_state = 1)
random_forest_model.fit(train_X,train_y)
model_rf_preds = random_forest_model.predict(val_X)
mae_score_rf = mean_absolute_error(val_y,model_rf_preds)
rmse_rf = mean_squared_error(val_y, model_rf_preds, squared=False)
print("Mean absolute error with Random Forest = ",mae_score_rf)
print("Root mean square error with Random Forest = ",rmse_rf)
print("Final Score for comp = ",100*(1-rmse_rf))


# * Mean absolute error with Random Forest with feature_1 =  0.13961009653174605
# * Mean absolute error with Random Forest with feature_2 =  0.13989972828117914
# * Mean absolute error with Random Forest with feature_3 =  0.1365071338095238

# ### Model_2 : XGBoost

# In[ ]:


# Model selection and training

def xgb_n_estimators_selection(x):
    xgb_model = XGBRegressor(n_estimators = x)
    xgb_model.fit(train_X,train_y)
    model_xgb_preds = xgb_model.predict(val_X)
    mae_score_xgb = mean_absolute_error(val_y,model_xgb_preds)
    return mae_score_xgb


# **Using Gradient Descent to find the n_estimators**

# In[ ]:


mae_list=[]
n_min = 0;
for i in range(1,50):
    mae_list.append(xgb_n_estimators_selection(i))
    plt.plot(i,mae_list[i-1],'bo')
    
plt.title("MAE vs n_estimators fo XGBoost")
plt.xlabel("n_estimators")
plt.ylabel("MAE")


# In[ ]:


print("The best score is {val} at n_estimator of {n}".format(val=min(mae_list),n= mae_list.index(min(mae_list))))


# **Therefore we will train our xgb model at n_estimators of 11. Another method will be to use `early_topping_rounds`**A small learning rate and large number of estimators will yield more accurate XGBoost models, though it will also take the model longer to train since it does more iterations through the cycle. As default, XGBoost sets learning_rate=0.1

# In[ ]:


xgbModel_updated = XGBRegressor(n_estimators = 500, learning_rate = 0.1,early_stopping_rounds = 20,
                  eval_set=[(val_X,val_y)], verbose = False)

xgbModel_updated.fit(train_X,train_y)


predictions_xgbModel_updated = xgbModel_updated.predict(val_X)
mae_xgbModel_updated = mean_absolute_error(predictions_xgbModel_updated,val_y)
print("MAE on updated XGB model wiht early stopping= ",mae_xgbModel_updated)
xgbModel_updated_rmse = mean_squared_error(predictions_xgbModel_updated,val_y,squared = False)
print("RMSE on xgbModel with early stopping= ",xgbModel_updated_rmse)
print("Final Score for comp = ",100*(1-xgbModel_updated_rmse))


# **But by the Gradient descent method we found out that the best mean_absolute_error is found out at n_estimators value of 11. So we will use that value to train out xgb_model.**

# > ## Model_3: Lasso Regression

# In[ ]:


from sklearn import linear_model
lr_model = linear_model.Lasso(alpha = 0.01)
lr_model.fit(train_X,train_y)
lr_model_prediction = lr_model.predict(val_X)
lr_model_mae = mean_absolute_error(lr_model_prediction,val_y)
print("MAE on Lasso regression model= ",lr_model_mae)
lr_model_rmse = mean_squared_error(lr_model_prediction,val_y,squared = False)
print("RMSE on lasso regression model= ",lr_model_rmse)
print("Final Score for comp = ",100*(1-lr_model_rmse))


# ## Model_4 : Elastic Net

# In[ ]:


from sklearn.linear_model import ElasticNet
en_model = ElasticNet(alpha = 0.01, l1_ratio = 0.8)       # l1_ratio is the ratio of ridge and lasso regression. Here 20:80
en_model.fit(train_X,train_y)
en_model_prediction = en_model.predict(val_X)
en_model_mae = mean_absolute_error(en_model_prediction,val_y)
print("MAE on Elastic Net regression model= ",en_model_mae)
en_model_rmse = mean_squared_error(en_model_prediction,val_y,squared = False)
print("RMSE on Elastic Net regression model= ",en_model_rmse)
print("Final Score for comp = ",100*(1-en_model_rmse))


# ## Model_5 : Support Vector Regressor

# In[ ]:


# Linear Support Vector Regression

from sklearn.svm import LinearSVR
linear_svr_model = LinearSVR(epsilon = 1.5)
linear_svr_model.fit(train_X,train_y)
linear_svr_model_pred = linear_svr_model.predict(val_X)
linear_svr_model_mae = mean_absolute_error(linear_svr_model_pred,val_y)
print("MAE on Linear Support Vector regression model= ",linear_svr_model_mae)
linear_svr_model_rmse = mean_squared_error(linear_svr_model_pred,val_y,squared = False)
print("RMSE on Linear Support Vector regression model= ",linear_svr_model_rmse)
print("Final Score for comp = ",100*(1-linear_svr_model_rmse))


# In[ ]:


# SVR with poly kernel

from sklearn.svm import SVR
svr_model = SVR(kernel = "poly",epsilon = 0.1)
svr_model.fit(train_X,train_y)
svr_model_pred = svr_model.predict(val_X)
svr_model_mae = mean_absolute_error(svr_model_pred,val_y)
print("MAE on Support Vector regression model with poly kernel= ",svr_model_mae)
svr_model_rmse = mean_squared_error(svr_model_pred,val_y,squared = False)
print("RMSE on Support Vector regression model with poly kernel= ",svr_model_rmse)
print("Final Score for comp = ",100*(max(0,1-svr_model_rmse)))


# ## Model_6 : Poission Regressor

# In[ ]:


from sklearn.linear_model import PoissonRegressor

pr_model = PoissonRegressor(max_iter=300)
pr_model.fit(train_X,train_y)
pr_model_pred = pr_model.predict(val_X)
pr_model_mae = mean_absolute_error(pr_model_pred,val_y)
print("MAE on Poisson regression model = ",pr_model_mae)
pr_model_rmse = mean_squared_error(pr_model_pred,val_y,squared = False)
print("RMSE on Poisson regression model = ",pr_model_rmse)
print("Final Score for comp = ",100*(max(0,1-pr_model_rmse)))


# In[ ]:



def alpha_tuning_pr(x):
    test_model  = PoissonRegressor(alpha = x,max_iter = 300)
    test_model.fit(train_X,train_y)
    test_pred = test_model.predict(val_X)
    test_rmse = mean_squared_error(pr_model_pred,val_y,squared = False)
    return test_rmse
    


# In[ ]:


rmse_list_pr= []
alpha_list_pr = []
a = 0;
for i in np.arange(1e-15,9e-15, 1e-15):
    rmse_list_pr.append(alpha_tuning_pr(i))
    alpha_list_pr.append(i)
    plt.plot(i,alpha_tuning_pr(i),'bo')
    
plt.title("RMSE vs alpha for Poisson Regression")
plt.xlabel("alpha")
plt.ylabel("RMSE")


# **Generally the RSME score doesnt change much with respect to the alpha value of the Poisson Regression model**

# ## Model_7 : Gradient Boosting Regression Trees for Poisson Regression

# In[ ]:


from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor

gbr_model =  HistGradientBoostingRegressor(loss="poisson",max_leaf_nodes=2)
gbr_model.fit(train_X,train_y)
gbr_model_pred = gbr_model.predict(val_X)
gbr_model_mae = mean_absolute_error(gbr_model_pred,val_y)
print("MAE on gbr_model = ",gbr_model_mae)
gbr_model_rmse = mean_squared_error(gbr_model_pred,val_y,squared = False)
print("RMSE on gbr_model = ",gbr_model_rmse)
print("Final Score for comp = ",100*(max(0,1-gbr_model_rmse)))


# In[ ]:


def maxLeafNodes_tuning_gbr(x):
    test_model  = HistGradientBoostingRegressor(loss="poisson",max_leaf_nodes=x)
    test_model.fit(train_X,train_y)
    test_pred = test_model.predict(val_X)
    test_rmse = mean_squared_error(test_pred,val_y,squared = False)
    return test_rmse


# In[ ]:


rmse_list_gbr= []
leaf_list_gbr = []
a = 0;
for i in range(10,200,10):
    rmse_list_gbr.append(maxLeafNodes_tuning_gbr(i))
    leaf_list_gbr.append(i)
    plt.plot(i,maxLeafNodes_tuning_gbr(i),'bo')
    
plt.title("RMSE vs Max Leaf nodes for GBR")
plt.xlabel("Max Leaf nodes for GBR")
plt.ylabel("RMSE")


# ## Model_8 : TweedieRegressor

# In[ ]:


from sklearn.linear_model import TweedieRegressor
tr_model = TweedieRegressor(power=0, alpha=1, link='log')
tr_model.fit(train_X,train_y)
tr_model_pred = tr_model.predict(val_X)
tr_model_mae = mean_absolute_error(tr_model_pred,val_y)
print("MAE on TweedieRegressor_model = ",tr_model_mae)
tr_model_rmse = mean_squared_error(tr_model_pred,val_y,squared = False)
print("RMSE on TweedieRegressor_model = ",tr_model_rmse)
print("Final Score for comp = ",100*(max(0,1-tr_model_rmse)))


# ## Model_9 : Pasive Agressive Regressor

# In[ ]:


from sklearn.linear_model import PassiveAggressiveRegressor
par_model = PassiveAggressiveRegressor(max_iter=100, random_state=0,tol=1e-3)
par_model.fit(train_X,train_y)
par_model_pred = par_model.predict(val_X)
par_model_mae = mean_absolute_error(par_model_pred,val_y)
print("MAE on Passive Agressive Regressor model = ",par_model_mae)
par_model_rmse = mean_squared_error(par_model_pred,val_y,squared = False)
print("RMSE on Passive Agressive Regressor model = ",par_model_rmse)
print("Final Score for comp = ",100*(max(0,1-par_model_rmse)))


# ## Model_10 : Orthogonal Matching Pursuit

# In[ ]:


from sklearn.linear_model import OrthogonalMatchingPursuit
omp_model = OrthogonalMatchingPursuit()
omp_model.fit(train_X,train_y)
omp_model_pred = omp_model.predict(val_X)
omp_model_mae = mean_absolute_error(omp_model_pred,val_y)
print("MAE on OMP model = ",omp_model_mae)
omp_model_rmse = mean_squared_error(omp_model_pred,val_y,squared = False)
print("RMSE on OMP model = ",omp_model_rmse)
print("Final Score for comp = ",100*(max(0,1-omp_model_rmse)))


# ## Model_11 : Bayesian Ridge

# In[ ]:


from sklearn import linear_model
br_model = linear_model.BayesianRidge()
br_model.fit(train_X,train_y)
br_model_pred = br_model.predict(val_X)
br_model_mae = mean_absolute_error(br_model_pred,val_y)
print("MAE on Bayesian Ridge model = ",br_model_mae)
br_model_rmse = mean_squared_error(br_model_pred,val_y,squared = False)
print("RMSE on BR model = ",br_model_rmse)
print("Final Score for comp = ",100*(max(0,1-br_model_rmse)))


# ## Model_12 : ARD Regressor

# In[ ]:


from sklearn import linear_model
ard_reg_model = linear_model.ARDRegression()
ard_reg_model.fit(train_X,train_y)
ard_reg_model_pred = ard_reg_model.predict(val_X)
ard_reg_model_mae = mean_absolute_error(ard_reg_model_pred,val_y)
print("MAE on ARD regresssor model = ",ard_reg_model_mae)
ard_reg_model_rmse = mean_squared_error(ard_reg_model_pred,val_y,squared = False)
print("RMSE on ARD Regressor model = ",ard_reg_model_rmse)
print("Final Score for comp = ",100*(max(0,1-ard_reg_model_rmse)))


# ### Cross validation on the final model

# In[ ]:


# final_model =  XGBRegressor(n_estimators = 11)
final_model   =  pr_model
cvScores = -1 * cross_val_score(final_model, X, y, cv = 5, scoring = "neg_root_mean_squared_error" )
print("RMSE scores = : ", cvScores)
comp_scores = 100 * (1 - cvScores)
print("Actual score =",comp_scores)


# In[ ]:


print("Worst score = ", comp_scores.min())
print("Best score = ", comp_scores.max())
print("Average score = ",comp_scores.mean())


# ### Let's predict the testing dataset on the final model and see the result
# The trained model is completely unaware of the testing dataframe. By observing the RMSE values of all the above models we are choosing the Poisson Regressor algortihm for the final model.

# In[ ]:


# final_model.fit(X,y)
pr_model.fit(X,y)
X_valid_for_submission = X_valid_full[features_4]

# X_valid_for_submission.head()

# Generating prediction on the testing data
new_X_valid = X_valid_for_submission.fillna(X_valid_for_submission.median())
full_prediction = pr_model.predict(new_X_valid)


# **Now we will save the data to a CSV file if asked so that we can successfully submit it to competitons**

# In[ ]:


output = pd.DataFrame({ 'Employee_ID':X_valid_full.index,
                       'Attrition_rate' : full_prediction
                      })
output.to_csv('submission21.csv', index = False)

