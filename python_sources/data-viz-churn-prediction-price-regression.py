#!/usr/bin/env python
# coding: utf-8

# # Telco Churn Modelling Dataset
# Link to dataset [here](https://www.kaggle.com/blastchar/telco-customer-churn)  
# Link to GitHub hosting [here](https://github.com/kartikay-bagla/Telco-Churn-Analysis)  
# Link to Kaggle hosting [here](https://www.kaggle.com/drvader/data-viz-churn-prediction-price-regression)
# Link to my blog post of this [here](https://kartikay-bagla.github.io/Telco-Churn-Analysis/)  

# A dataset provided by a telecommunications company regarding its customers and whether they stopped using their services or not (churn). I'm using this to explore the data, and practice feature engineering along with classification and regression both.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

pd.set_option("display.max_columns", 100)
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.simplefilter('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, roc_auc_score, confusion_matrix


# ## Initially we start with Data Visualization and Exploration

# In[ ]:


df = pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")


# Let's check out the top 5 rows of the data

# In[ ]:


df.head()


# Now lets see the datatypes of each column to make sure everything is correctly loaded

# In[ ]:


df.dtypes


# Senior citizen should be an object (it will be converted back to onehot in the end though)  
# Total charges should be a float/int instead of an object

# In[ ]:


df.SeniorCitizen.describe()


# In[ ]:


df.SeniorCitizen.replace([0, 1], ["No", "Yes"], inplace= True)


# Senior citizen problem fixed, now gotta fix the TotalCharges

# In[ ]:


df.TotalCharges.describe()


# In[ ]:


df.TotalCharges.unique()


# In[ ]:


for charge in df.TotalCharges:
    try:
        charge = float(charge)
    except:
        print("charge:", charge, "length", len(charge))


# It seems that some charges are blanks. Real smart, guys.

# In[ ]:


charges = [float(charge) if charge != " " else np.nan for charge in df.TotalCharges]


# In[ ]:


df.TotalCharges = charges


# Now its time to get a feel for the numerical variables

# In[ ]:


df.describe()


# Same for the categorical variables

# In[ ]:


df.describe(include=object)


# Time to plot **HISTOGRAMS**

# In[ ]:


df.hist(figsize=(15, 5), layout=(1, 3))
plt.show()


# But what about the categorical variables? We don't want them feeling left out. On to some **COUNTPLOTS!!!**

# In[ ]:


sns.countplot(y="MultipleLines", data= df)
plt.show()
sns.barplot(y="MultipleLines", x="TotalCharges", data= df)
plt.show()
sns.barplot(y="MultipleLines", x="MonthlyCharges", data= df)
plt.show()


# Those were all single variable distributions, but what happens when we bring more variables into the game?

# In[ ]:


sns.lmplot("MonthlyCharges", "TotalCharges", hue="InternetService", data= df, fit_reg= False)


# The graph above has Monthly Charges vs Total Charges where each datapoint is colored by the type of internet service they have

# In[ ]:


sns.lmplot("MonthlyCharges", "TotalCharges", hue="Contract", data= df, fit_reg= False)


# In[ ]:


sns.lmplot("MonthlyCharges", "MonthlyCharges", hue="InternetService", data= df, fit_reg= False)


# In[ ]:


sns.lmplot("tenure", "TotalCharges", data= df, hue="Churn", fit_reg= False)


# Now its time to create some **features**

# In[ ]:


for col in df.dtypes[df.dtypes == object].index:
    print(col, df[col].unique())


# A ProtectedCustomer is one who has both Online backups and security, similarly I create other features

# In[ ]:


df["ProtectedCustomer"] = ["Yes" if df.OnlineBackup[i]=="Yes" and df.OnlineSecurity[i]=="Yes" else "No" for i in range(len(df))]


# In[ ]:


df["StreamerCustomer"] = ["Yes" if df.StreamingMovies[i]=="Yes" and df.StreamingTV[i]=="Yes" else "No" for i in range(len(df))]


# In[ ]:


df["FamilyCustomer"] = ["Yes" if df.Partner[i]=="Yes" or df.Dependents[i]=="Yes" else "No" for i in range(len(df))]


# In[ ]:


df["OldFashioned"] = ["Yes" if df.PaperlessBilling[i]=="No" and df.PaymentMethod[i]=="Mailed check"                       else "No" for i in range(len(df))]


# In[ ]:


df["PowerUser"] = ["Yes" if df.ProtectedCustomer[i]=="Yes" and df.StreamerCustomer[i]=="Yes"                    and df.DeviceProtection[i]=="Yes" and df.TechSupport[i]=="Yes" else "No" for i in range(len(df))]


# In[ ]:


df["FamilyMultiple"] = ["Yes" if df.FamilyCustomer[i]=="Yes" and df.MultipleLines[i]=="Yes" else "No" for i in range(len(df))]


# In[ ]:


df.describe()


# I created another column - FullCharges which is the product of tenure and monthly charges and it closely resembles TotalCharges but differs by plus-minus 200 dollars at the max. Which may be due to excessive services or discounts applied which are not included in the monthly charges

# In[ ]:


df["FullCharges"] = df.tenure * df.MonthlyCharges


# In[ ]:


df["Discount"] = df.FullCharges - df.TotalCharges


# In[ ]:


df.head()


# Checking for missing values

# In[ ]:


df.isna().sum()


# Filling them with the medians

# In[ ]:


df.TotalCharges.fillna(df.TotalCharges.median(), inplace= True)
df.Discount.fillna(df.Discount.median(), inplace= True)


# In[ ]:


df.to_csv("cleaned.csv", index= False)


# ## Now we use the cleaned dataframe for predicting the churn of customers

# In[ ]:


df = pd.read_csv("cleaned.csv")


# In[ ]:


df.shape


# In[ ]:


df.head()


# Dropping the customerID as it is of no use to us currently

# In[ ]:


df.drop("customerID", axis= 1, inplace= True)


# Turning Churn into a binary categorical variable

# In[ ]:


df.Churn.replace(["Yes", "No"], [1, 0], inplace= True)


# Turning the database into a one-hot encoded database

# In[ ]:


df = pd.get_dummies(df)


# Initializing X and y

# In[ ]:


X = df.drop("Churn", axis= 1)
y = df.Churn


# ### Splitting the dataset

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 1234)


# In[ ]:


print( len(X_train), len(X_test), len(y_train), len(y_test) )


# In[ ]:


X_train.shape, y_train.shape


# ### Creating Model pipelines

# Creating pipelines through which we can feed data

# In[ ]:


pipelines = {
    "rf": make_pipeline(StandardScaler(), RandomForestClassifier(random_state=123)),
    "gb": make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=123)),
    "nb": make_pipeline(StandardScaler(), GaussianNB()),
    "kn": make_pipeline(StandardScaler(), KNeighborsClassifier())
}


# Getting the list of tunable parameters for the models

# In[ ]:


pipelines["nb"].get_params()


# ### Hyper parameter grid for all the models

# In[ ]:


rf_hyperparameters = {
    "randomforestclassifier__n_estimators": [100, 200],
    "randomforestclassifier__max_features": ["auto", "sqrt", 0.33]
}
gb_hyperparameters = {
    "gradientboostingclassifier__n_estimators": [100, 200],
    'gradientboostingclassifier__learning_rate': [0.05, 0.1, 0.2],
    'gradientboostingclassifier__max_depth': [1, 3, 5]
}
kn_hyperparameters = {
    'kneighborsclassifier__n_neighbors': [3, 5, 7, 10]
}
nb_hyperparameters = {
    'gaussiannb__priors': [None]
}
hyperparameters = {"rf": rf_hyperparameters,
                   "gb": gb_hyperparameters,
                   "nb": nb_hyperparameters,
                   "kn": kn_hyperparameters}


# ### Fitting the models to the training set

# In[ ]:


fitted_models = {}

for name, pipeline in pipelines.items():
    model = GridSearchCV(pipeline, hyperparameters[name], cv= 10, n_jobs= -1)
    model.fit(X_train, y_train)
    fitted_models[name] = model
    print(name, 'has been fitted.')


# #### Each model's best r2 score on the training set

# In[ ]:


for name, model in fitted_models.items():
    print(name, model.best_score_)


# #### Each model's scores on the test set

# In[ ]:


for name, model in fitted_models.items():
    print(name)
    print("-----------")
    pred = model.predict(X_test)
    print('ACC:', accuracy_score(y_test, pred))
    print("ROC:", roc_auc_score(y_test, pred))
    print("CoM:\n", confusion_matrix(y_test, pred))


# ### Saving the best model

# In[ ]:


with open('final_model_churn.pkl', 'wb') as f:
    pickle.dump(fitted_models['gb'].best_estimator_, f)


# ## Now we use regression to predict the total price each customer pays
# All of this follows a similar workflow as above!

# In[ ]:


df = pd.read_csv("cleaned.csv")


# In[ ]:


df.shape


# In[ ]:


df.head()


# In[ ]:


df.drop("customerID", axis= 1, inplace= True)


# In[ ]:


for col in df.dtypes[df.dtypes==object].index:
    if set(df[col].unique().tolist()) == set(["Yes", "No"]):
        df[col].replace(["Yes", "No"], [1, 0], inplace= True)


# In[ ]:


df.head()


# In[ ]:


df.gender.replace(["Male", "Female"], [1, 0], inplace=True)


# In[ ]:


df = pd.get_dummies(df)


# In[ ]:


X = df.drop(["TotalCharges", "FullCharges"], axis= 1)
y = df.TotalCharges


# ### Splitting the dataset

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 1234)


# In[ ]:


print( len(X_train), len(X_test), len(y_train), len(y_test) )


# In[ ]:


X_train.shape, y_train.shape


# ### Creating Model pipelines

# In[ ]:


pipelines = {
    "rf": make_pipeline(StandardScaler(), RandomForestRegressor(random_state=123)),
    "gb": make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state=123))
}


# ### Hyper parameter grid for all the models

# In[ ]:


rf_hyperparameters = {
    "randomforestregressor__n_estimators": [100, 200],
    "randomforestregressor__max_features": ["auto", "sqrt", 0.33]
}
gb_hyperparameters = {
    "gradientboostingregressor__n_estimators": [100, 200],
    'gradientboostingregressor__learning_rate': [0.05, 0.1, 0.2],
    'gradientboostingregressor__max_depth': [1, 3, 5]
}
hyperparameters = {"rf": rf_hyperparameters,
                   "gb": gb_hyperparameters}


# ### Fitting the models to the training set

# In[ ]:


fitted_models = {}

for name, pipeline in pipelines.items():
    model = GridSearchCV(pipeline, hyperparameters[name], cv= 10, n_jobs= -1)
    model.fit(X_train, y_train)
    fitted_models[name] = model
    print(name, 'has been fitted.')


# #### Each model's best r2 score on the training set

# In[ ]:


for name, model in fitted_models.items():
    print(name, model.best_score_)


# #### Each model's scores on the test set

# In[ ]:


for name, model in fitted_models.items():
    print(name)
    print("-----------")
    pred = model.predict(X_test)
    print('MAE:', mean_absolute_error(y_test, pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
    
print("\nMean:", np.mean(y_test))


# ### Plotting the results

# In[ ]:


plt.scatter(y, fitted_models["rf"].predict(X))
plt.xlabel("Actual")
plt.ylabel("Predicted")
x_lim = plt.xlim()
y_lim = plt.ylim()
plt.plot(x_lim, y_lim, "k--")
plt.show()


# ### Saving the best model

# In[ ]:


with open('final_model_total_price.pkl', 'wb') as f:
    pickle.dump(fitted_models['rf'].best_estimator_, f)


# ## Monthly bill regression

# In[ ]:


df = pd.read_csv("cleaned.csv")


# In[ ]:


df.shape


# In[ ]:


df.head()


# In[ ]:


df.drop(["customerID", "FullCharges", "Discount", "TotalCharges"], axis= 1, inplace= True)


# In[ ]:


for col in df.dtypes[df.dtypes==object].index:
    if set(df[col].unique().tolist()) == set(["Yes", "No"]):
        df[col].replace(["Yes", "No"], [1, 0], inplace= True)


# In[ ]:


df.head()


# In[ ]:


df.gender.replace(["Male", "Female"], [1, 0], inplace=True)


# In[ ]:


df = pd.get_dummies(df)


# In[ ]:


X = df.drop("MonthlyCharges", axis= 1)
y = df.MonthlyCharges


# ### Splitting the dataset

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 1234)


# In[ ]:


print( len(X_train), len(X_test), len(y_train), len(y_test) )


# In[ ]:


X_train.shape, y_train.shape


# ### Creating Model pipelines

# In[ ]:


pipelines = {
    "rf": make_pipeline(StandardScaler(), RandomForestRegressor(random_state=123)),
    "gb": make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state=123))
}


# ### Hyper parameter grid for all the models

# In[ ]:


rf_hyperparameters = {
    "randomforestregressor__n_estimators": [100, 200],
    "randomforestregressor__max_features": ["auto", "sqrt", 0.33]
}
gb_hyperparameters = {
    "gradientboostingregressor__n_estimators": [100, 200],
    'gradientboostingregressor__learning_rate': [0.05, 0.1, 0.2],
    'gradientboostingregressor__max_depth': [1, 3, 5]
}
hyperparameters = {"rf": rf_hyperparameters,
                   "gb": gb_hyperparameters}


# ### Fitting the models to the training set

# In[ ]:


fitted_models = {}

for name, pipeline in pipelines.items():
    model = GridSearchCV(pipeline, hyperparameters[name], cv= 10, n_jobs= -1)
    model.fit(X_train, y_train)
    fitted_models[name] = model
    print(name, 'has been fitted.')


# #### Each model's best r2 score on the training set

# In[ ]:


for name, model in fitted_models.items():
    print(name, model.best_score_)


# #### Each model's scores on the test set

# In[ ]:


for name, model in fitted_models.items():
    print(name)
    print("-----------")
    pred = model.predict(X_test)
    print('MAE:', mean_absolute_error(y_test, pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
    
print("\nMean:", np.mean(y_test))


# ### Plotting the results

# In[ ]:


plt.scatter(y, fitted_models["rf"].predict(X))
plt.xlabel("Actual")
plt.ylabel("Predicted")
x_lim = plt.xlim()
y_lim = plt.ylim()
plt.plot(x_lim, y_lim, "k--")
plt.show()


# ### Saving the best model

# In[ ]:


with open('final_model_monthly_price.pkl', 'wb') as f:
    pickle.dump(fitted_models['rf'].best_estimator_, f)


# # Finished!
