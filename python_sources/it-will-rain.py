#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import warnings
warnings.filterwarnings("ignore")
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


# In[ ]:


df = pd.read_csv("/kaggle/input/weather-dataset-rattle-package/weatherAUS.csv")


# In[ ]:


df.head()


# In[ ]:


df.info()


# * We can see that we have many missing values.
# * And we have 23 columns. (21 if we exclude target feature and RISK_MM)
# * We are going to drop RISK_MM because our target feature is based on this feature, like it says in dataset description.
# * Date feature needs to be converted to datetime type. Currently it's object.
# * And lastly in RainTomorrow column i'll replace No with 0 and Yes with 1. I'll do same to RainToday column too.

# In[ ]:


df["Date"] = pd.to_datetime(df["Date"])
df.drop("RISK_MM",axis=1,inplace = True)
df["RainTomorrow"] = [1 if each == "Yes" else 0 for each in df["RainTomorrow"]]
df["RainToday"] = [1 if each == "Yes" else 0 for each in df["RainToday"]]


# ### **Features and what do they mean**
# * Date: The date of observation
# * Location: The common name of the lcoation of the weather station
# * MinTemp: The minimum temperature in degrees celcius
# * MaxTemp: The maximum temperature in degrees celsius
# * Rainfall: The amount of rainfall recorded for the day in mm
# * Evaporation: The so-called Class A pan evaporation (mm) in the 24 hours to 9am
# * Sunshine: The number of hours of bright sunshine in the day.
# * WindGustDir: The direction of the strongest wind gust in the 24 hours to midnight
# * WindGustSpeed: The speed (km/h) of the strongest wind gust in the 24 hours to midnight
# * WindDir9am: Direction of the wind at 9am
# * WindDir3pm: Direction of the wind at 3pm
# * WindSpeed9am: Wind speed (km/hr) averaged over 10 minutes prior to 9am
# * WindSpeed3pm: Wind speed (km/hr) averaged over 10 minutes prior to 3pm
# * Humidity9am: Humidity (percent) at 9am
# * Humidity3pm: Humidity (percent) at 3pm
# * Pressure9am: Atmospheric pressure (hpa) reduced to mean sea level at 9am
# * Pressure3pm: Atmospheric pressure (hpa) reduced to mean sea level at 3pm
# * Cloud9am: Fraction of sky obscured by cloud at 9am. This is measured in "oktas", which are a unit of eigths. It records how many eigths of the sky are obscured by cloud. A 0 measure indicates completely clear sky whilst an 8 indicates that it is completely overcast.
# * Cloud3pm: Fraction of sky obscured by cloud (in "oktas": eighths) at 3pm. See Cload9am for a description of the values
# * Temp9am: Temperature (degrees C) at 9am
# * Temp3pm: Temperature (degrees C) at 3pm
# * RainToday: 1 if precipitation (mm) in the 24 hours to 9am exceeds 1mm, otherwise 0
# * RainTomorrow: The target variable. Did it rain tomorrow?

# In[ ]:


df.describe()


# * We have some extreme outliers in Ranfall and Evaporation column, we'll inspect these outliers later.

# # Exploratory Data Analysis (EDA)

# * First i'll define two functions to visualize categorical and numerical features to target feature. After that i'll look at relations between features.

# In[ ]:


cat_cols = []
num_cols = []
other_cols = []

for each in df.columns:
    if df[each].dtype == "object":
        cat_cols.append(each)
    elif df[each].dtype == "float64":
        num_cols.append(each)
    else:
        other_cols.append(each)
print("Categorical Columns: ",cat_cols)
print("Numerical Columns: ",num_cols)
print("Other Columns: ",other_cols)


# In[ ]:


def ctgplt(variable,to):
    
    "Function for visualization of categorical variables."
    
    var = df[variable]
    values=var.value_counts()
    
    f, ax = plt.subplots(figsize = (8,8))
    g = sns.barplot(x = variable, y = to, data = df)
    g.set_xticklabels(g.get_xticklabels(),rotation = 90)
    plt.show()
    
    print("{}:\n{}".format(variable,values))

def numplt(data,variable,to):
  
  "Function for visualization of numerical variables."

  c = sns.FacetGrid(data,col=to,height=6)
  c.map(sns.distplot,variable,bins=25)
  plt.show()


# In[ ]:


for i in cat_cols:
    ctgplt(i, "RainTomorrow")


# * Rain rates for some cities are very low, but most of them are around 0.2.
# * Wind is coming from Northwest if its going to rain, mostly.

# In[ ]:


for k in num_cols:
    numplt(df, k, "RainTomorrow")


# * Lets look at the graphs and comment the graps:
# * MinTemps are nicely distributed around 10 Degrees for RainTomorrow = 0. But for RainTomorrow = 1 its grouped around 15 degrees. And we have some more values around 25 - 30 degrees.
# * MaxTemps are stuck at 20 ish degrees for RainTomorrow = 1 and some in 30 - 35 degrees. But we can see that its warmer if tomorrow is a rainy day.
# * Like i said before Rainfall and Evaporation has some extreme outliers that makes impossible to commentate the graphs. I'll drop outliers and look at these graphs again.
# * Weather is mostly cloudy if its going to rain. (as expected) But no clouds if its not.
# * Wind is more distributed for RainTomorrow = 1 it's getting stronger if its going to rain.
# * 9 am wind speeds are almost same.
# * 3 pm wind speeds are a little bit strongter if its going to rain.
# * Humidity is more (as normal) if its going to rain. Especially in early hours of the day.
# * Atmospheric pressures distributions are almost same.
# * More clouds for rainy days.
# * Temperatures are not changed so much, like i said earlier most days grouped around 20 degrees for rainy days.

# In[ ]:


sns.boxplot(x = df["Rainfall"])
plt.show()


# * We cant even see the distribution. 

# In[ ]:


sns.boxplot(x= df["Evaporation"])
plt.show()


# * This one is not bad as the Rainfall column but still has some serious outliers.

# In[ ]:


corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool))
f, ax = plt.subplots(figsize=(12, 8))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, annot=True, mask=mask, cmap=cmap, ax=ax)
plt.show()


# * We have some highly correlated features. This collinear features wont do any good to our models. So we need to drop one of them.
# * Columns to drop: Temp3pm, Temp9am, Pressure9am. Lets drop these columns and look our correlation matrix again.

# In[ ]:


df.drop(columns = ["Temp3pm", "Temp9am", "Pressure9am"], axis=1, inplace = True)


# In[ ]:


corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool))
f, ax = plt.subplots(figsize=(12, 8))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, annot=True, mask=mask, cmap=cmap, ax=ax)
plt.show()


# * It's looking better now.

# In[ ]:


# I removed the columns that i just deleted from dataframe.
to_remove = ("Temp3pm", "Temp9am", "Pressure9am")
num_cols = [each for each in num_cols if each not in to_remove]


# ### Outlier Removal

# In[ ]:


Q3 = df["Rainfall"].quantile(0.75)
Q1 = df["Rainfall"].quantile(0.25)

IQR = Q3 - Q1
step = IQR * 3

maxm = Q3 + step
minm = Q1 - step

df = df[df["Rainfall"].fillna(1) < (maxm)]

Q3 = df["Evaporation"].quantile(0.75)
Q1 = df["Evaporation"].quantile(0.25)

IQR = Q3 - Q1
step = IQR * 3

maxm = Q3 + step
minm = Q1 - step

df = df[df["Evaporation"].fillna(1) < (maxm)]


# In[ ]:


sns.distplot(df["Evaporation"])
plt.show()


# * Distribution looks better without outliers.

# In[ ]:


df["RainTomorrow"].value_counts()


# In[ ]:


sns.countplot(x = "RainTomorrow", data=df, palette = "RdBu")
plt.show()


# * We have a imbalanced data.

# ### Missing Values

# In[ ]:


def missing_values_table(data):
        # Total missing values
        mis_val = data.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * data.isnull().sum() / len(data)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns


# In[ ]:


missing_values_table(df)


# * Most of the missing value percentages is between 0 - 10.
# * Cloud columns, Evaporation and Sunshine features have 40% missing data.
# * I will fill categorical variables with mode and numerical variables with median.

# In[ ]:


for i in cat_cols:
    df[i].fillna(value=df[i].mode()[0],inplace=True)

for k in num_cols:
    df[k].fillna(value=df[k].median(),inplace=True)


# In[ ]:


df.isnull().sum()


# * So now that we filled our missing values we can prepare out data for modeling.
# * i will drop date column after i seperated it to 3 pieces (year,month,day).
# * Label encode the categorical features.
# * Min max scale the numerical features.

# In[ ]:


df["Year"] = df["Date"].dt.year

df["Month"] = df["Date"].dt.month

df["Day"] = df["Date"].dt.day

df.drop("Date",axis=1,inplace=True)


# In[ ]:


df.head()


# In[ ]:


le = LabelEncoder()
mms = MinMaxScaler()

for each in cat_cols:
    df[each] = le.fit_transform(df[each])

df[df.columns] = mms.fit_transform(df[df.columns])


# In[ ]:


df.head()


# In[ ]:


df.describe()


# ## Modeling

# In[ ]:


from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score


# In[ ]:


X = df.drop("RainTomorrow",axis=1)
y = df["RainTomorrow"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)


# * First i will fit a baseline model with logistic regression, then i will use xgboost.

# In[ ]:


lr = LogisticRegression()

lr.fit(X_train, y_train)
preds = lr.predict(X_test)
print("train_score",lr.score(X_train, y_train))
print("test_score",lr.score(X_test,y_test))


# In[ ]:


cf_matrix = confusion_matrix(y_test, preds)


# In[ ]:


sns.heatmap(cf_matrix,annot = True, fmt="g",cmap="Greens")
plt.show()


# In[ ]:


xgb = XGBClassifier(objective = "binary:logistic")
xgb.fit(X_train,y_train)
pred = xgb.predict(X_test)
print(xgb.score(X_train,y_train))
print(xgb.score(X_test,y_test))


# * We get a better result with XGBoost but because we did not specify the parameters. Now lets apply Grid Search with xgb classifier.

# In[ ]:


# I will not use a huge parameter grid because it will took to long to train, so here few parameters that could be useful.
params = {
  'min_child_weight':[1,2],
  'max_depth': [3,5],
  'n_estimators':[200,300],
  'colsample_bytree':[0.7,0.8],
  'scale_pos_weight':[1.1,1.2]  
}

model = GridSearchCV(estimator=XGBClassifier(objective="binary:logistic"), param_grid=params, cv=StratifiedKFold(n_splits=5), scoring="f1_macro", n_jobs=-1, verbose=3)
model.fit(X_train, y_train)

print("Best Score: ",model.best_score_)
print("Best Estimator: ",model.best_estimator_)


# * So grid search is did not do better then default parameters. Let's look at the confusion matrix with default parameters.

# In[ ]:


mat = confusion_matrix(y_test,model.predict(X_test))
sns.heatmap(mat,annot=True,cmap="Greens", fmt="g")
plt.show()


# In[ ]:


print(classification_report(y_test,model.predict(X_test)))


# * Recall is pretty low because we missclassified most of the zeros as ones. This is caused because of the imbalance.
# * I will use over-sampling with imblearn library to overcome this problem.
# * To learn what is over-sampling you can read here: https://imbalanced-learn.readthedocs.io/en/stable/user_guide.html
# * Other then over-sampling what can we do?
# * We could do missing value imputation with KNN or MICE. These both are available in sklearn library. 
# * We could do more feature engineering.
# 

# In[ ]:


importances = pd.Series(data=xgb.feature_importances_,
                        index= X_train.columns)

importances_sorted = importances.sort_values()
plt.figure(figsize=(8,8))
importances_sorted.plot(kind='barh', color='lightgreen')
plt.title('Features Importances')
plt.show()


# * We can see that humidity and clouds are the most important features. But ratios are pretty close to each other.

# In[ ]:


from imblearn.over_sampling import SMOTE

method = SMOTE()

X_resampled, y_resampled = method.fit_sample(X_train, y_train)


# In[ ]:


xgb.fit(X_resampled, y_resampled)
pred1 = xgb.predict(X_test)
print("Train Score: ", xgb.score(X_resampled,y_resampled))
print("Test Score: ", xgb.score(X_test,y_test))


# In[ ]:


mat = confusion_matrix(y_test,pred1)
sns.heatmap(mat,annot=True,cmap="Greens", fmt="g")
plt.show()


# In[ ]:


print(classification_report(y_test,pred1))


# 
# ### UPDATE
# 
# * After some time i checked the notebook and i realized that i resampled in the all data, so because of that accuracy is increasing. But now i fixed it and accuracy went down a little with over sampling.

# ### That's all. Thank you for reading, i hope you like it. 
