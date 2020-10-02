#!/usr/bin/env python
# coding: utf-8

# # Diamonds Dataset - Visualizing, Cleaning, Feature Engineering, Regression

# ### Imports

# In[ ]:


#Number manipulation
import numpy as np

#Data Manipulation
import pandas as pd

#Plotting Libraries
from matplotlib import pyplot as plt
import seaborn as sns


# In[ ]:


#Some configuration settings
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option("display.max_columns", 100)


# ### Getting the data

# In[ ]:


df = pd.read_csv("../input/diamonds.csv", index_col=0)


# ### Exploring the Data

# In[ ]:


df.shape


# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


df.describe(include=object)


# **Checking for any null values**

# In[ ]:


df.isna().any().any()


# **Plotting histograms**

# In[ ]:


df.hist(figsize=(20, 20))
plt.show()


# **Countplots for categorical variables**

# In[ ]:


for feature in df.dtypes[df.dtypes == object].index:
    sns.countplot(y= feature, data= df)
    plt.show()


# **Box and Violin Plots for numerical features**

# In[ ]:


for feature in df.dtypes[df.dtypes != object].index:
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.violinplot(x= feature, data= df)
    plt.subplot(1, 2, 2)
    sns.boxplot(x= feature, data= df)
    plt.show()


# The box plots show that x, y, z have values that are 0, which shouldn't be possible.
# 
# Exploring this

# In[ ]:


pd.concat([df[df["x"] == 0], df[df["y"] == 0], df[df["z"] == 0]]).drop_duplicates()


# In[ ]:


len(pd.concat([df[df["x"] == 0], df[df["y"] == 0], df[df["z"] == 0]]).drop_duplicates())


# Considering the size of the dataset, it is just better to drop these values

# In[ ]:


df = df[(df[['x','y','z']] != 0).all(axis=1)]


# In[ ]:


df[df["z"] == 0]


# **Creating a feature, volume**

# In[ ]:


df["volume"] = df["x"] * df["y"] * df["z"]


# **Creating a feature, density**

# In[ ]:


df["density"] = df["carat"]*0.2/df["volume"]


# In[ ]:


df.head()


# ### Grouping sparse classes

# In[ ]:


sns.countplot(y="clarity", data= df)


# In[ ]:


df.clarity.replace(["VVS1", "VVS2"], "VVS", inplace=True)
df.clarity.replace(["VS1", "VS2"], "VS", inplace= True)
df.clarity.replace(["SI1", "SI2"], "SI", inplace= True)
df.clarity.replace("I1", "I", inplace= True)


# In[ ]:


sns.countplot(y="clarity", data= df)


# In[ ]:


color_grades = {
    "Colorless": ["D", "E", "F"],
    "Near Colorless": ["G", "H", "I", "J"],
    "Faint Yellow": ["K", "L", "M"],
    "Very Light Yellow": ["N", "O", "P", "Q", "R"],
    "Light Yellow": ["S", "T", "U", "V", "W", "X", "Y", "Z"]
}


# In[ ]:


c_l = []
for color in df.color:
    for key, item in color_grades.items():
        if color in item:
            c_l.append(key)
            break
df["ColorGrade"] = c_l


# In[ ]:


sns.countplot(y="color", data= df)


# ### Exploring multivariate distributions

# In[ ]:


sns.lmplot(y="carat", x="price", hue="clarity", data= df, fit_reg= False)


# In[ ]:


sns.lmplot(y="carat", x="price", hue="clarity", data= df[df.clarity == "I"], fit_reg= False)


# In[ ]:


sns.lmplot(y="carat", x="price", hue="color", data= df, fit_reg= False)


# In[ ]:


sns.lmplot(y="carat", x="price", hue="cut", data= df, fit_reg= False)


# In[ ]:


df.head()


# ### Correlations Matrix

# In[ ]:


new_df = pd.get_dummies(df)


# In[ ]:


plt.figure(figsize=(20, 20))
corr = new_df.corr()
sns.heatmap(corr*100, cmap="YlGn", annot= True, fmt=".0f")


# ### Exporting the file

# In[ ]:


df.to_csv("cleaned.csv", index= False)


# ## Price Regression

# In[ ]:


df = pd.get_dummies(df)


# In[ ]:


df.head()


# In[ ]:


X = df.drop(["price"], axis= 1).astype(float)


# In[ ]:


y = df.price.astype(float)


# ### Importing tools for regression

# In[ ]:


#Models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

#Building everything
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

#Evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error

#Saving the model
import pickle


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
    print("MSE:", "\n", mean_squared_error(y_test, pred))
    
print(np.mean(y_test))


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


with open('final_model.pkl', 'wb') as f:
    pickle.dump(fitted_models['rf'].best_estimator_, f)


# # Finished!
