#!/usr/bin/env python
# coding: utf-8

# # Import all necessary library for analysis and Predict Data

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt


# **Import sklearn for data model**

# In[ ]:


from sklearn.linear_model import (
    ElasticNet,
    ElasticNetCV,
    Lasso,
    LassoCV,
    LinearRegression,
    LogisticRegression,
    Ridge,
)

from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
)

from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)


# In[ ]:


import seaborn as sns
from sklearn.base import BaseEstimator
from xgboost import XGBClassifier, XGBRegressor
import hyperopt as hp
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe


# # Import all data and view Train and Test data shape 

# In[ ]:


#Import Date
xtrain = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/train.csv")
xtest = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/test.csv")
xsubmission = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/submission.csv")
# view shape of test and train data
print(xtrain.shape)
print(xtest.shape)


# ### View First 10 row of Training Data Set

# In[ ]:


# view head of train data
xtrain.head(10)


# ### View First 10 row of Test Data Set

# In[ ]:


# view head of test data
xtest.head(10)


# ### View First 10 row of Submission Data Set

# In[ ]:


# view head of submission data
xsubmission.head(10)


# ##### Count Date wise Total Data. In this Data Set we can see all Date there have same quantity of records.

# In[ ]:


# date wise value count
xtrain['Date'].value_counts()


# # ----- COVID19-Global-Forecasting-Week-2 --- Data Analysis-----

#  ##### Store Confirmed Cases in statevalue using Group by Province_State

# In[ ]:


# create ConfirmedCasesgroup by Province_State
statevalue = xtrain.groupby('Province_State').max().ConfirmedCases


# ##### View 10 Top States where most ConfirmedCases of CAVID19 in a barplot. This plot show thst most effected people in New York, secord Hunei and third New Jersey.

# In[ ]:


# view top state conformed cases in a barplot
top_states = statevalue.sort_values(ascending = False).head(10)
sns.barplot(x=top_states.index, y=top_states.values)
plt.xticks(rotation = 'vertical')


# ##### Made ConfirmedCases and Fatalities date as integer

# In[ ]:


# make data as integer
xtrain.ConfirmedCases = xtrain.ConfirmedCases.astype('int64')
xtrain.Fatalities = xtrain.Fatalities.astype('int64')


# ### View date wise Confirmedcases in a line plot. In this plow show that every day confirmed cases increase rapidy.

# In[ ]:


# Date wise confirm case view in an lineplot
plt.figure(figsize=(15,6))
sns.lineplot(x=xtrain.Date,y=xtrain.ConfirmedCases,markers=True,style=True)
plt.xticks(rotation = 'vertical')


# ### View date wise Fatalities in a line plot. In this plow show that every day Fatalities increased rapidy.

# In[ ]:


# Date wise Fatalities view in an lineplot
plt.figure(figsize=(15,6))
sns.lineplot(x=xtrain.Date,y=xtrain.Fatalities,markers=True,style=True)
plt.xticks(rotation = 'vertical')


# #### According to Country region groupby filtering Italy ConfirmedCases was 105792 and Fatalities was 12428, Spain ConfirmedCases 95923 and Fatalities 8464, US ConfirmedCases was 75833 and Fatalities was 1550, Germany ConfirmedCases was 71808 and Fatalities was 775, China ConfirmedCases was 67801 and Fatalities was 3187, France ConfirmedCases was 52128 and Fatalities was 3523, Iran ConfirmedCases was 44605 and Fatalities was 2898 in the Dataset. 

# In[ ]:


# ConfirmedCases and Fatalities column groupby Country Region
df_xtrain = xtrain.groupby(['Country_Region'])[['ConfirmedCases', 'Fatalities']].max()
print(df_xtrain.sort_values(by=['ConfirmedCases','Fatalities'],ascending=False).head(10))


# #### Belgium ConfirmedCases and Fatalities is very High, second Highest ConfirmedCases and Fatalities position for Austria and Third Highest ConfirmedCases and Fatalities for Brazil

# In[ ]:


# view countrywise ConfirmedCases and Fatalities in a plot
fig,ax = plt.subplots()
fig.set_figheight(10)
fig.set_figwidth(40)
ax.plot(df_xtrain[:29].index.values,df_xtrain[:29].ConfirmedCases, color="red", marker="o")
ax.set_xlabel("Countries",fontsize=24)
ax.set_ylabel("Confirmed Cases",color="red",fontsize=24)
ax.tick_params(axis = 'both', which = 'major', labelsize = 24,labelrotation=90)
ax2=ax.twinx()
ax2.plot(df_xtrain[:29].index.values,df_xtrain[:29].Fatalities,color="blue",marker="o")
ax2.set_ylabel("Fatalities",color="blue",fontsize=24)
ax2.tick_params(axis = 'both', which = 'major', labelsize = 24)
plt.show()


# ## ConfirmedCases and Fatalities data Analysis and Visualization Exclude China

# Since details of the initial breakthrough strongly interfere with the results, it's recomended to analyze China independently. Let's first see the results without China. Both ConfirmedCases and Fatalities are increase day by day.

# In[ ]:


# ConfirmedCases and Fatalities data Analysis Exclude China and view in two Plot
confirmed_total_date_noChina = xtrain[xtrain['Country_Region']!='China'].groupby(['Date']).agg({'ConfirmedCases':['sum']})
fatalities_total_date_noChina = xtrain[xtrain['Country_Region']!='China'].groupby(['Date']).agg({'Fatalities':['sum']})
total_date_noChina = confirmed_total_date_noChina.join(fatalities_total_date_noChina)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
total_date_noChina.plot(ax=ax1)
ax1.set_title("Global confirmed cases excluding China", size=13)
ax1.set_ylabel("Number of cases", size=13)
ax1.set_xlabel("Date", size=13)
fatalities_total_date_noChina.plot(ax=ax2, color='orange')
ax2.set_title("Global deceased cases excluding China", size=13)
ax2.set_ylabel("Number of cases", size=13)
ax2.set_xlabel("Date", size=13)


# ## ConfirmedCases and Fatalities data Analysis and Visualization for China

# #### Since China was the initial infected country, the COVID-19 behavior is different from the rest of the world. The medical system was not prepared for the pandemic, in fact no one was aware of the virus until several cases were reported. Moreover, China government took strong contention measures in a considerable short period of time and, while the virus is widely spread, they have been able to control the increasing of the infections.

# In[ ]:


#ConfirmedCases and Fatalities data Analysis and Visualization for China
confirmed_total_date_China = xtrain[xtrain['Country_Region']=='China'].groupby(['Date']).agg({'ConfirmedCases':['sum']})
fatalities_total_date_China = xtrain[xtrain['Country_Region']=='China'].groupby(['Date']).agg({'Fatalities':['sum']})
total_date_China = confirmed_total_date_China.join(fatalities_total_date_China)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
total_date_China.plot(ax=ax1)
ax1.set_title("China confirmed cases", size=13)
ax1.set_ylabel("Number of cases", size=13)
ax1.set_xlabel("Date", size=13)
fatalities_total_date_China.plot(ax=ax2, color='orange')
ax2.set_title("China Fatalities cases", size=13)
ax2.set_ylabel("Number of cases", size=13)
ax2.set_xlabel("Date", size=13)


# ## ConfirmedCases and Fatalities data Analysis and Visualization for Italy and Spain

# ****Both Italy and Spain are experiencing the larger increase in COVID-19 positives in Europe.****

# In[ ]:


#For Itally
confirmed_total_date_Italy = xtrain[xtrain['Country_Region']=='Italy'].groupby(['Date']).agg({'ConfirmedCases':['sum']})
fatalities_total_date_Italy = xtrain[xtrain['Country_Region']=='Italy'].groupby(['Date']).agg({'Fatalities':['sum']})
total_date_Italy = confirmed_total_date_Italy.join(fatalities_total_date_Italy)

#For Spain
confirmed_total_date_Spain = xtrain[xtrain['Country_Region']=='Spain'].groupby(['Date']).agg({'ConfirmedCases':['sum']})
fatalities_total_date_Spain = xtrain[xtrain['Country_Region']=='Spain'].groupby(['Date']).agg({'Fatalities':['sum']})
total_date_Spain = confirmed_total_date_Spain.join(fatalities_total_date_Spain)

plt.figure(figsize=(15,10))

plt.subplot(2, 2, 1)
total_date_Italy.plot(ax=plt.gca(), title='Italy')
plt.ylabel("Confirmed infection cases", size=13)

plt.subplot(2, 2, 2)
total_date_Spain.plot(ax=plt.gca(), title='Spain')
plt.ylabel("Confirmed infection cases", size=13)


# ## ConfirmedCases and Fatalities data Analysis and Visualization for UK and Singapore

# ** At the same time, UK is a unique case given that it's one of the most important countries in Europe but recently has left the European Union, which has create an effective barrier to human mobility from other countries. The fourth country we will study in this section is Singapore, since it's an asiatic island, is closer to China and its socio-economic conditions is different from the other three countries.**

# In[ ]:


#For UK
confirmed_total_date_UK = xtrain[xtrain['Country_Region']=='United Kingdom'].groupby(['Date']).agg({'ConfirmedCases':['sum']})
fatalities_total_date_UK = xtrain[xtrain['Country_Region']=='United Kingdom'].groupby(['Date']).agg({'Fatalities':['sum']})
total_date_UK = confirmed_total_date_UK.join(fatalities_total_date_UK)


#For Singapore
confirmed_total_date_Singapore = xtrain[xtrain['Country_Region']=='Singapore'].groupby(['Date']).agg({'ConfirmedCases':['sum']})
fatalities_total_date_Singapore = xtrain[xtrain['Country_Region']=='Singapore'].groupby(['Date']).agg({'Fatalities':['sum']})
total_date_Singapore = confirmed_total_date_Singapore.join(fatalities_total_date_Singapore)

plt.figure(figsize=(15,10))

plt.subplot(2, 2, 1)
total_date_UK.plot(ax=plt.gca(), title='United Kingdom')
plt.ylabel("Confirmed infection cases", size=13)

plt.subplot(2, 2, 2)
total_date_Singapore.plot(ax=plt.gca(), title='Singapore')
plt.ylabel("Confirmed infection cases", size=13)


# ## ConfirmedCases and Fatalities data Analysis and Visualization for Australia and Bangladesh

# **ConfirmedCases and Fatalities for Australia and Bangladesh are increasing day by day but not like other country. Australia ConfirmedCases  more than 4000 and Fatalities average but in Bangladesh ConfirmedCases is approximately 50  and Fatalities is approximately 10.**
# 

# In[ ]:


#For Australia
confirmed_total_date_Australia = xtrain[xtrain['Country_Region']=='Australia'].groupby(['Date']).agg({'ConfirmedCases':['sum']})
fatalities_total_date_Australia = xtrain[xtrain['Country_Region']=='Australia'].groupby(['Date']).agg({'Fatalities':['sum']})
total_date_Australia = confirmed_total_date_Australia.join(fatalities_total_date_Australia)

#For Bangladesh
confirmed_total_date_Bangladesh = xtrain[xtrain['Country_Region']=='Bangladesh'].groupby(['Date']).agg({'ConfirmedCases':['sum']})
fatalities_total_date_Bangladesh = xtrain[xtrain['Country_Region']=='Bangladesh'].groupby(['Date']).agg({'Fatalities':['sum']})
total_date_Bangladesh = confirmed_total_date_Bangladesh.join(fatalities_total_date_Bangladesh)

plt.figure(figsize=(15,10))

plt.subplot(2, 2, 1)
total_date_Australia.plot(ax=plt.gca(), title='Australia')
plt.ylabel("Confirmed infection cases", size=13)

plt.subplot(2, 2, 2)
total_date_Bangladesh.plot(ax=plt.gca(), title='Bangladesh')
plt.ylabel("Confirmed infection cases", size=13)


# #### Fractional Data Analysis acordint to total population of each country

# In[ ]:


pop_italy = 60486683.
pop_spain = 46749696.
pop_UK = 67784927.
pop_singapore = 5837230.

total_date_Italy.ConfirmedCases = total_date_Italy.ConfirmedCases/pop_italy*100.
total_date_Italy.Fatalities = total_date_Italy.ConfirmedCases/pop_italy*100.
total_date_Spain.ConfirmedCases = total_date_Spain.ConfirmedCases/pop_spain*100.
total_date_Spain.Fatalities = total_date_Spain.ConfirmedCases/pop_spain*100.
total_date_UK.ConfirmedCases = total_date_UK.ConfirmedCases/pop_UK*100.
total_date_UK.Fatalities = total_date_UK.ConfirmedCases/pop_UK*100.
total_date_Singapore.ConfirmedCases = total_date_Singapore.ConfirmedCases/pop_singapore*100.
total_date_Singapore.Fatalities = total_date_Singapore.ConfirmedCases/pop_singapore*100.

plt.figure(figsize=(15,10))
plt.subplot(2, 2, 1)
total_date_Italy.ConfirmedCases.plot(ax=plt.gca(), title='Italy')
plt.ylabel("Fraction of population infected")
plt.ylim(0, 0.06)

plt.subplot(2, 2, 2)
total_date_Spain.ConfirmedCases.plot(ax=plt.gca(), title='Spain')
plt.ylim(0, 0.06)

plt.subplot(2, 2, 3)
total_date_UK.ConfirmedCases.plot(ax=plt.gca(), title='United Kingdom')
plt.ylabel("Fraction of population infected")
plt.ylim(0, 0.005)

plt.subplot(2, 2, 4)
total_date_Singapore.ConfirmedCases.plot(ax=plt.gca(), title='Singapore')
plt.ylim(0, 0.005)


# #### In order to compare the 4 countries, it's also interesting to see the evolution of the infections from the first confirmed case:

# In[ ]:


# For Itally
confirmed_total_date_Italy = xtrain[(xtrain['Country_Region']=='Italy') & xtrain['ConfirmedCases']!=0].groupby(['Date']).agg({'ConfirmedCases':['sum']})
fatalities_total_date_Italy = xtrain[(xtrain['Country_Region']=='Italy') & xtrain['ConfirmedCases']!=0].groupby(['Date']).agg({'Fatalities':['sum']})
total_date_Italy = confirmed_total_date_Italy.join(fatalities_total_date_Italy)

# For Spain
confirmed_total_date_Spain = xtrain[(xtrain['Country_Region']=='Spain') & (xtrain['ConfirmedCases']!=0)].groupby(['Date']).agg({'ConfirmedCases':['sum']})
fatalities_total_date_Spain = xtrain[(xtrain['Country_Region']=='Spain') & (xtrain['ConfirmedCases']!=0)].groupby(['Date']).agg({'Fatalities':['sum']})
total_date_Spain = confirmed_total_date_Spain.join(fatalities_total_date_Spain)

# For UK
confirmed_total_date_UK = xtrain[(xtrain['Country_Region']=='United Kingdom') & (xtrain['ConfirmedCases']!=0)].groupby(['Date']).agg({'ConfirmedCases':['sum']})
fatalities_total_date_UK = xtrain[(xtrain['Country_Region']=='United Kingdom') & (xtrain['ConfirmedCases']!=0)].groupby(['Date']).agg({'Fatalities':['sum']})
total_date_UK = confirmed_total_date_UK.join(fatalities_total_date_UK)

# For Australia
confirmed_total_date_Australia = xtrain[(xtrain['Country_Region']=='Australia') & (xtrain['ConfirmedCases']!=0)].groupby(['Date']).agg({'ConfirmedCases':['sum']})
fatalities_total_date_Australia = xtrain[(xtrain['Country_Region']=='Australia') & (xtrain['ConfirmedCases']!=0)].groupby(['Date']).agg({'Fatalities':['sum']})
total_date_Australia = confirmed_total_date_Australia.join(fatalities_total_date_Australia)

# For Singapore
confirmed_total_date_Singapore = xtrain[(xtrain['Country_Region']=='Singapore') & (xtrain['ConfirmedCases']!=0)].groupby(['Date']).agg({'ConfirmedCases':['sum']})
fatalities_total_date_Singapore = xtrain[(xtrain['Country_Region']=='Singapore') & (xtrain['ConfirmedCases']!=0)].groupby(['Date']).agg({'Fatalities':['sum']})
total_date_Singapore = confirmed_total_date_Singapore.join(fatalities_total_date_Singapore)

italy = [i for i in total_date_Italy.ConfirmedCases['sum'].values]
italy_30 = italy[0:50] 
spain = [i for i in total_date_Spain.ConfirmedCases['sum'].values]
spain_30 = spain[0:50] 
UK = [i for i in total_date_UK.ConfirmedCases['sum'].values]
UK_30 = UK[0:50] 
singapore = [i for i in total_date_Singapore.ConfirmedCases['sum'].values]
singapore_30 = singapore[0:50] 


# In[ ]:


# Plots
plt.figure(figsize=(12,6))
plt.plot(italy_30)
plt.plot(spain_30)
plt.plot(UK_30)
plt.plot(singapore_30)
plt.legend(["Italy", "Spain", "UK", "Singapore"], loc='upper left')
plt.title("COVID-19 infections from the first confirmed case", size=15)
plt.xlabel("Days", size=13)
plt.ylabel("Infected cases", size=13)
plt.ylim(0, 60000)
plt.show()


# #### In Province_State cplumn there heve 11830 row null value

# In[ ]:


# Check if there have any null value
xtrain.isnull().sum()


# **Change Date column to Datetime**

# In[ ]:


# CHANGE TO PD.DATETIME
xtrain.Date = pd.to_datetime(xtrain.Date, infer_datetime_format=True)
xtest.Date = pd.to_datetime(xtest.Date, infer_datetime_format=True)


# **CONCISING THE TRAIN DATASET TO 18TH MARCH 2020**

# In[ ]:


# CONCISING THE TRAIN DATASET TO 18TH MARCH 2020.
MIN_TEST_DATE = xtest.Date.min()
xtrain = xtrain.loc[xtrain.Date < MIN_TEST_DATE, :]


# **Reset Indexing**

# In[ ]:


# RESETTING INDEX
xtrain.reset_index()


# ##### Fill up missing value in test and train dataset

# In[ ]:


# FILLING MISSING VALUES
xtrain.fillna("", inplace=True)
xtest.fillna("", inplace=True)


# #### Creating a column name "Region" for test and train dataser

# In[ ]:


# CREATING NEW REGION COLUMN
xtrain["Region"] = xtrain["Country_Region"] + xtrain["Province_State"]
xtest["Region"] = xtest["Country_Region"] + xtest["Province_State"]


# ##### DROPPING COUNTRY REGION AND PROVINCE STATE Column

# In[ ]:


# DROPPING COUNTRY REGION AND PROVINCE STATE
xtrain.drop(['Country_Region','Province_State'],axis=1,inplace=True)
xtest.drop(['Country_Region','Province_State'],axis=1,inplace=True)


# CONVERT The DATE COLUMN TO INTEGER COLUMN

# In[ ]:


# CONVERTING DATE COLUMN TO INTEGER
xtrain.loc[:, 'Date'] = xtrain.Date.dt.strftime("%m%d")
xtest.loc[:, 'Date'] = xtest.Date.dt.strftime("%m%d")


# ### View Region wise Confirmed Cases in a Line Plot.

# In[ ]:


# Region wise Confirmed Cases in LinePlot
sns.lineplot(data=xtrain, x="Date", y="ConfirmedCases", hue="Region")
plt.show()


# ### View Region wise Fatalities in a Line Plot.

# In[ ]:


#  Region wise Fatalities in Line Plot.
sns.lineplot(data=xtrain, x="Date", y="Fatalities", hue="Region")
plt.show()


# # Create Train and Test Dataset

# In[ ]:


# CREATING X AND Y for Train Dataset
X1 = xtrain.drop(["ConfirmedCases", "Fatalities"], axis=1)
X2 = xtrain.drop(["ConfirmedCases", "Fatalities"], axis=1)
y1 = xtrain["ConfirmedCases"]
y2 = xtrain["Fatalities"]


# In[ ]:


# Create TEST 1 AND TEST 2 for Test dataset
test_1 = xtest.copy()
test_2 = xtest.copy()


# **In Mean Encoding we take the number of labels into account along with the target variable to encode the labels into machine comprehensible values**

# In[ ]:


# FUNCTION FOR MEAN ENCODING
class MeanEncoding(BaseEstimator):

    def __init__(self, feature, C=0.1):
        self.C = C
        self.feature = feature

    def fit(self, X_train, y_train):

        df = pd.DataFrame(
            {"feature": X_train[self.feature], "target": y_train}
        ).dropna()

        self.global_mean = df.target.mean()
        mean = df.groupby("feature").target.mean()
        size = df.groupby("feature").target.size()

        self.encoding = (self.global_mean * self.C + mean * size) / (self.C + size)

    def transform(self, X_test):

        X_test[self.feature] = (
            X_test[self.feature].map(self.encoding).fillna(self.global_mean).values
        )

        return X_test

    def fit_transform(self, X_train, y_train):

        df = pd.DataFrame(
            {"feature": X_train[self.feature], "target": y_train}
        ).dropna()

        self.global_mean = df.target.mean()
        mean = df.groupby("feature").target.mean()
        size = df.groupby("feature").target.size()
        self.encoding = (self.global_mean * self.C + mean * size) / (self.C + size)

        X_train[self.feature] = (
            X_train[self.feature].map(self.encoding).fillna(self.global_mean).values
        )

        return X_train


# In[ ]:


for f2 in ["Region"]:
    me2 = MeanEncoding(f2, C=0.01 * len(X2[f2].unique()))
    me2.fit(X2, y2)
    X2 = me2.transform(X2)
    test_2 = me2.transform(test_2)


# In[ ]:


for f1 in ["Region"]:
    me1 = MeanEncoding(f1, C=0.01 * len(X1[f1].unique()))
    me1.fit(X1, y1)
    X1 = me1.transform(X1)
    test_1 = me1.transform(test_1)


# In[ ]:


# View Test_1
test_1


# In[ ]:


# View Test_2
test_2


# In[ ]:


# Load some Basic Library
import matplotlib.pyplot as plt
from sklearn import model_selection
import numpy as np


# **To compare metric of different algorithims Paramters-
#        algo_list : a list conataining algorithim models like random forest, decision trees etc.
#        X : dataframe without Target variable
#        y : dataframe with only Target variable
#        random_state : The seed of randomness. Default is 3
#        n_splits : Number of splits used. Default is 3
#        ( Default changes from organization to organization)
#        Returns-
#        median accuracy and the standard deviation accuracy.
#        Box Plot of Acuuracy**

# In[ ]:


# FUNCTION FOR COMPARING DIFFERENT REGRESSORS
def algorithim_boxplot_comparison(
    X, y, algo_list=[], random_state=3, scoring="r2", n_splits=10
):
    
    results = []
    names = []
    for algo_name, algo_model in algo_list:
        kfold = model_selection.KFold(
            shuffle=True, n_splits=n_splits, random_state=random_state
        )
        cv_results = model_selection.cross_val_score(
            algo_model, X, y, cv=kfold, scoring=scoring
        )
        results.append(cv_results)
        names.append(algo_name)
        msg = "%s: %s : (%f) %s : (%f) %s : (%f)" % (
            algo_name,
            "median",
            np.median(cv_results),
            "mean",
            np.mean(cv_results),
            "variance",
            cv_results.var(ddof=1),
        )
        print(msg)
    # boxplot algorithm comparison
    fig = plt.figure()
    fig.suptitle("Algorithm Comparison")
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()


# In[ ]:


# REGRESSORS
lr = LinearRegression(n_jobs=-1)
rfr = RandomForestRegressor(random_state=96, n_jobs=-1)
gbr = GradientBoostingRegressor(random_state=96)
xgbr = XGBRegressor()


# In[ ]:


# APPENDING THE REGRESSORS IN A LIST
models = []
models.append(('lr',lr))
models.append(('rfr',rfr))
models.append(('gbr',gbr))
models.append(('xgbr',xgbr))


# #### View X1 and Y1 in an BoxPlot

# In[ ]:


algorithim_boxplot_comparison(
    X1, y1, models, random_state=96, scoring="neg_root_mean_squared_error", n_splits=5
)


# ** TODO : USE MORE ADVANCED HYPERPARAMTER TUNING METHODS LIKE OPTUNA, KEARS-TUNER, HPBANDSTER,TUNE**

# In[ ]:


# HYPEROPT
def auc_model(params):
    params = {
        "n_estimators": int(params["n_estimators"]),
        "max_features": int(params["max_features"]),
        "min_samples_leaf": int(params["min_samples_leaf"]),
        "min_samples_split": int(params["min_samples_split"]),
    }
    clf = RandomForestRegressor(**params, random_state=96, n_jobs=-1)
    return cross_val_score(
        clf, X1, y1, cv=3, scoring="neg_mean_squared_log_error"
    ).mean()


params_space = {
    "n_estimators": hp.quniform("n_estimators", 0, 300, 50),
    "max_features": hp.quniform("max_features", 1, 3, 1),
    "min_samples_leaf": hp.quniform("min_samples_leaf", 1, 50, 1),
    "min_samples_split": hp.quniform("min_samples_split", 1, 50, 1),
}
best = 0


def f(params):
    global best
    auc = auc_model(params)
    if auc > best:
        print("New Best", best, params)
    return {"loss": -auc, "status": STATUS_OK}


trials = Trials()
best = fmin(f, params_space, algo=tpe.suggest, max_evals=200, trials=trials)
print("best:\n", best)


# ** TODO : USE MORE ADVANCED HYPERPARAMTER TUNING METHODS LIKE OPTUNA, KEARS-TUNER, HPBANDSTER,TUNE**

# In[ ]:


# HYPEROPT
def auc_model(params):
    params = {
        "n_estimators": int(params["n_estimators"]),
        "max_features": int(params["max_features"]),
        "min_samples_leaf": int(params["min_samples_leaf"]),
        "min_samples_split": int(params["min_samples_split"]),
    }
    clf = RandomForestRegressor(**params, random_state=96, n_jobs=-1)
    return cross_val_score(
        clf, X2, y2, cv=3, scoring="neg_mean_squared_log_error"
    ).mean()


params_space = {
    "n_estimators": hp.quniform("n_estimators", 0, 300, 50),
    "max_features": hp.quniform("max_features", 1, 3, 1),
    "min_samples_leaf": hp.quniform("min_samples_leaf", 1, 50, 1),
    "min_samples_split": hp.quniform("min_samples_split", 1, 50, 1),
}
best = 0


def f(params):
    global best
    auc = auc_model(params)
    if auc > best:
        print("New Best", best, params)
    return {"loss": -auc, "status": STATUS_OK}


trials = Trials()
best = fmin(f, params_space, algo=tpe.suggest, max_evals=200, trials=trials)
print("best:\n", best)


# ###  RANDOMFORESTREGRESSOR FOR CONFIRMEDCASUALTIES

# In[ ]:


# RANDOMFORESTREGRESSOR FOR CONFIRMEDCASUALTIES
rfr1 = RandomForestRegressor(
    max_features=3,
    min_samples_leaf=26,
    min_samples_split=31,
    n_estimators=200,
    random_state=96,
    n_jobs=-1,
)


# ### RANDOMFORESTREGRESSOR FOR FATALITIES

# In[ ]:


# RANDOMFORESTREGRESSOR FOR FATALITIES
rfr2 = RandomForestRegressor(
    max_features=3,
    min_samples_leaf=17,
    min_samples_split=17,
    n_estimators=100,
    random_state=96,
    n_jobs=-1,
)


# ### FITTING using RANDOM FOREST REGRESSOR algorithm FOR CONFIRMEDCASUALTIES

# In[ ]:


# FITTING RANDOMFORESTREGRESSOR FOR CONFIRMEDCASUALTIES
rfr1.fit(X1, y1)


# ### PREDICTING CONFIRMEDCASUALTIES using RANDOM FOREST REGRESSOR algorithm

# In[ ]:


# PREDICTING CONFIRMEDCASUALTIES using RANDOM FOREST REGRESSOR
y_n_1 = rfr1.predict(test_1)


# ### Fit CONFIRMEDCASUALTIES using K neareat neighbour algorithm Classifier

# In[ ]:


# Fit CONFIRMEDCASUALTIES using K neareat neighbour algorithm Classifier
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 4, metric = 'braycurtis', p = 1)
classifier.fit(X1, y1)


# ### Predict CONFIRMEDCASUALTIES using K neareat neighbour algorithm Classifier

# In[ ]:


### Predict CONFIRMEDCASUALTIES using K neareat neighbour algorithm Classifier
y_pred1 = classifier.predict(X1)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y1, y_pred1)
from sklearn.metrics import accuracy_score 
print( 'Accuracy Score confirmed cases :',accuracy_score(y1,y_pred1)*100)


# ### FITTING FATALITIES DATA USING RANDOMFORESTREGRESSOR ALGORITHM

# In[ ]:


# FITTING RANDOMFORESTREGRESSOR FOR FATALITIES
rfr2.fit(X2, y2)


# ### Predict FATALITIES DATA USING RANDOMFORESTREGRESSOR ALGORITHM

# In[ ]:


# PREDICTING FATALITIES
y_n_2 = rfr2.predict(test_2)


# ### Fit  Fatalities using K neareat neighbour algorithm Classifier

# In[ ]:


# ### Fit  Fatalities using K neareat neighbour algorithm Classifier
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 4, metric = 'braycurtis', p = 1)
classifier.fit(X2, y2)


# ### Predict Fatalities using K neareat neighbour algorithm Classifier. I have got 98% accuracy

# In[ ]:


### Predict Fatalities using K neareat neighbour algorithm Classifier
y_pred2 = classifier.predict(X2)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y2, y_pred2)
from sklearn.metrics import accuracy_score 
print( 'Accuracy Score confirmed cases :',accuracy_score(y2,y_pred2)*100)


# #### Add Confirmed Cases into sumbission data

# In[ ]:


# ADDING CONFIRMEDCASES
xsubmission.ConfirmedCases = round(pd.DataFrame(y_n_1))


# Add Fatalities into sumbission data

# In[ ]:



# ADDING FATALITIES
xsubmission.Fatalities = round(pd.DataFrame(y_n_2))


# ### Look at the final submission data

# In[ ]:


# View submission data
xsubmission


# ## Save Date to submission.csv file. 

# In[ ]:


# Save Date to submission file
xsubmission.to_csv("submission.csv", index=False)
print("Submission file create sucessfully")


# Also look at my third weed challenge [https://www.kaggle.com/mahmudds/covid19-global-forecasting-week-3]
