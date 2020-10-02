#!/usr/bin/env python
# coding: utf-8

# # Hotel Booking Analysis

# The aim is to create meaningful estimators from the data set we have and to select the model that predicts the cancellation best by comparing them with the accuracy scores of different ML models and ROC Curves.
# 
# ### **1- EDA** 
# 
# Content of exploratory data analysis.
# 
# * Repeated guest effect on cancellations.
# * Night spent at hotels.
# * Hotel type with more time spent.
# * Effects of deposit on cancellations by segments.
# * Relationship of lead time with cancellation.
# * Monthly customers and cancellations.
# 
# ### **2- Preprocessing**
# 
# This part is not much organized because I decided what to do some features with missing values after Correlation and The fact about 'reservation_status' part.
# 
# * Handling missing values
# * Handling features
# * Correlation
# * The fact about 'reservation status' (decision tree model)
# * Last arrangements before model comparisons.
# 
# ### **3- Models and ROC Curve Comparison**
# 
# Not all models have tuning part, the best two models tuned.
# 
# * Logistic Regression
# * Gaussian Naive Bayes
# * Support Vector Classification
# * Decision Tree Model
# * Random Forest
# * Model Tuning for Random Forest
# * XGBoost
# * Neural Network
# * Model Tuning for Neural Network

# In[ ]:


#Libraries

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, auc
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler 

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

from warnings import filterwarnings
filterwarnings('ignore')


# In[ ]:


df = pd.read_csv("../input/hotel-booking-demand/hotel_bookings.csv")


# In[ ]:


# Quick look 
df.head()


# In[ ]:


df.shape


# In[ ]:


print("# of NaN in each columns:", df.isnull().sum(), sep='\n')


# In[ ]:


# It is better to copy original dataset, it can be needed in some cases.
data = df.copy()


# # 1. EDA

# ### Cancellations by repeated guests

# In[ ]:


sns.set(style = "darkgrid")
plt.title("Canceled or not", fontdict = {'fontsize': 20})
ax = sns.countplot(x = "is_canceled", hue = 'is_repeated_guest', data = data)


# There is no surprise that repeated guests do not cancel their reservations. Of course there are some exceptions.
# Also most of the customers are not repeated guests.

# ### Boxplot Distribution of Nights Spent at Hotels by Market Segment and Hotel Type 

# In[ ]:


plt.figure(figsize = (15,10))
sns.boxplot(x = "market_segment", y = "stays_in_week_nights", data = data, hue = "hotel", palette = 'Set1');


# In[ ]:


plt.figure(figsize=(15,10))
sns.boxplot(x = "market_segment", y = "stays_in_weekend_nights", data = data, hue = "hotel", palette = 'Set1');


# It can be seen that most of the groups are normal distributed, some of them have high skewness.
# Looking at the distribution, most people do not seem to prefer to stay at the hotel for more than 1 week. But it seems normal to stay in resort hotels for up to 12-13 days. Although this changes according to the segments, staying longer than 15 days certainly creates outliers for each segment. If the total time feature was created by summing up the weekend and week nights, this would be clearer, but it can be clearly seen when looking at the two visualizations together.
# 
# As it turns out, customers from Aviation Segment do not seem to be staying at the resort hotels and have a relatively lower day average. Apart from that, the weekends and weekdays averages are roughly equal. Customers in the Aviation Segment are likely to arrive shortly due to business. Also probably most airports are a bit away from sea and its most likely to be closer to city hotels.
# 
# It is obvious that when people go to resort hotels, they prefer to stay more.

# ### Countplot Distribution of Market Segments

# In[ ]:


plt.figure(figsize = (13,10))
sns.set(style = "darkgrid")
plt.title("Countplot Distrubiton of Segment by Deposit Type", fontdict = {'fontsize':20})
ax = sns.countplot(x = "market_segment", hue = 'deposit_type', data = data)


# In[ ]:


plt.figure(figsize = (13,10))
sns.set(style = "darkgrid")
plt.title("Countplot Distributon of Segments by Cancellation", fontdict = {'fontsize':20})
ax = sns.countplot(x = "market_segment", hue = 'is_canceled', data = data)


# Looking at Offline TA/TO and Groups, the situations where the deposit was received were only in the scenarios where the groups came. It is quite logical to apply a deposit for a large number of customers who will fill important amount of the hotel capacity.
# 
# As a first thought, I expected the cancellation rate in the market segments where a deposit is applied to be lower than the other segments where no deposit applied. But when we look at the cancellations according to the segments in the other visualization, it seems that this is not the case. 
# 
# - Groups segment has cancellation rate more than 50%.
# - Offline TA/TO (Travel Agents/Tour Operators) and Online TA has cancellation rate more than 33%.
# - Direct segment has cancellation rate less than 20%.
# 
# It is surprising that the cancellation rate in these segments is high despite the application of a deposit. The fact that cancellations are made collectively like reservations may explain this situation a bit.
# 
# Cancellation rates for online reservations are as expected in a dynamic environment where the circulation is high.
# 
# Another situation that took my attention is that the cancellation rate in the direct segment is so low. At this point, I think that a mutual trust relationship has been established in case people are communicating one to one. I will not dwell on this much, but I think there is a psychological factor here.

# ### Density Curve of Lead Time by Cancelation

# In[ ]:


(sns.FacetGrid(data, hue = 'is_canceled',
             height = 6,
             xlim = (0,500))
    .map(sns.kdeplot, 'lead_time', shade = True)
    .add_legend());


# While lead time is more than roughly 60, people tend to cancel their reservations (cancellation rate is higher after this point). 
# Also people want their holiday or work plans resulted in 100 days which equals to half of the data.

# ### Monthly Cancellations and Customers by Hotel Types

# In[ ]:


plt.figure(figsize =(13,10))
sns.set(style="darkgrid")
plt.title("Total Customers - Monthly ", fontdict={'fontsize': 20})
ax = sns.countplot(x = "arrival_date_month", hue = 'hotel', data = data)


# In[ ]:


plt.figure(figsize = (13,10))
sns.barplot(x = 'arrival_date_month', y = 'is_canceled', data = data);


# In[ ]:


plt.figure(figsize = (20,10))
sns.barplot(x = 'arrival_date_month', y = 'is_canceled', hue = 'hotel', data = data);


# Looking at the first graph, it can be seen that the city hotels have more customers in all months. Considering proportionally, resort hotels seem to be a little closer to city hotels in summer.
# 
# An important interpretation can be made by examining three graphics together. Fewer customers come in the winter months, so when we look at the cancellation rates, it is quite normal that it appears less in the winter months. The point to be noted on these months is that the cancellation rates of city hotels are almost equal to other months even in winter. The fact that the total cancellation rates of the winter months are low is that the cancellation rates of the resort hotels are low in these months.
# In short, the possibility of cancellation of resort hotels in winter is very low. This information can be a very important factor when predicting 'is_canceled'.

# # Preprocessing 
# ### (Missing Values, Feature Engineering and Standardization)

# In[ ]:


print("# of NaN in each columns:", df.isnull().sum(), sep='\n')


# In[ ]:


def perc_mv(x, y):
    perc = y.isnull().sum() / len(x) * 100
    return perc

print('Missing value ratios:\nCompany: {}\nAgent: {}\nCountry: {}'.format(perc_mv(df, df['company']),
                                                                                   perc_mv(df, df['agent']),
                                                                                   perc_mv(df, df['country'])))


# In[ ]:


data["agent"].value_counts().count()


# As we can see 94.3% of company column are missing values. Therefore we do not have enough values to fill the rows of company column by predicting, filling by mean etc. It seems that the best option is dropping company column.
# 
# 13.68% of agent column are missing values, there is no need to drop agent column. But also we should not drop the rows because 13.68% of data is really huge amount and those rows have the chance to have crucial information. There are 333 unique agent, since there are too many agents they may not be predictable. 
# Also NA values can be the agents that are not listed in present 333 agents. We can't predict agents and since missing values are 13% of all data we can't drop them too. I will decide what to do about agent after correlation section.
# 
# It will not be a problem if we drop the rows that have missing values in country column. Still, I will wait for correlation.

# In[ ]:


# company is dropped
data = data.drop(['company'], axis = 1)


# In[ ]:


# We have also 4 missing values in children column. If there is no information about children, In my opinion those customers do not have any children.
data['children'] = data['children'].fillna(0)


# ### Handling Features

# We should check the features to create some more meaningful variables and reduce the number of features if it is possible.

# In[ ]:


data.dtypes


# In[ ]:


# I wanted to label them manually. I will do the rest with get.dummies or label_encoder.
data['hotel'] = data['hotel'].map({'Resort Hotel':0, 'City Hotel':1})

data['arrival_date_month'] = data['arrival_date_month'].map({'January':1, 'February': 2, 'March':3, 'April':4, 'May':5, 'June':6, 'July':7,
                                                            'August':8, 'September':9, 'October':10, 'November':11, 'December':12})


# In[ ]:


def family(data):
    if ((data['adults'] > 0) & (data['children'] > 0)):
        val = 1
    elif ((data['adults'] > 0) & (data['babies'] > 0)):
        val = 1
    else:
        val = 0
    return val

def deposit(data):
    if ((data['deposit_type'] == 'No Deposit') | (data['deposit_type'] == 'Refundable')):
        return 0
    else:
        return 1


# In[ ]:


def feature(data):
    data["is_family"] = data.apply(family, axis = 1)
    data["total_customer"] = data["adults"] + data["children"] + data["babies"]
    data["deposit_given"] = data.apply(deposit, axis=1)
    data["total_nights"] = data["stays_in_weekend_nights"]+ data["stays_in_week_nights"]
    return data

data = feature(data)


# In[ ]:


# Information of these columns is also inside of new features, so it is better to drop them.
# I did not drop stays_nights features, I can't decide which feature is more important there.
data = data.drop(columns = ['adults', 'babies', 'children', 'deposit_type', 'reservation_status_date'])


# After correlation we will decide what to do about country, agent and total_nights.

# ### Correlation

# In[ ]:


data.columns


# In[ ]:


# Lets copy data to check the correlation between variables. 
cor_data = data.copy()


# In[ ]:


le = LabelEncoder()


# In[ ]:


# This data will not be used while predicting cancellation. This is just for checking correlation.
cor_data['meal'] = le.fit_transform(cor_data['meal'])
cor_data['distribution_channel'] = le.fit_transform(cor_data['distribution_channel'])
cor_data['reserved_room_type'] = le.fit_transform(cor_data['reserved_room_type'])
cor_data['assigned_room_type'] = le.fit_transform(cor_data['assigned_room_type'])
cor_data['agent'] = le.fit_transform(cor_data['agent'])
cor_data['customer_type'] = le.fit_transform(cor_data['customer_type'])
cor_data['reservation_status'] = le.fit_transform(cor_data['reservation_status'])
cor_data['market_segment'] = le.fit_transform(cor_data['market_segment'])


# In[ ]:


cor_data.corr()


# In[ ]:


cor_data.corr()["is_canceled"].sort_values()


# As we can see in the sorted list, reservation_status seems to be most impactful feature. With that information accuracy rate should be really high. It can be better to drop reservation_status column to see how other features can predict. I am going to try both.
# 
# Impacts of three feature that are created:
# - deposit_given = 0,48131
# - is_family = -0,01327
# - total_customer = 0,04504
# 
# Apart from that, I will not use arrival_date_week_number, stays_in_weekend_nights and arrival_date_day_of_month since their importances are really low while predicting cancellations. 
# 
# Also, still we have some missing values in agent column. It has nice importance on predicting cancellation but since the missing values are equal to 13% of the total data it is better to drop that column. It has a lot of class inside of it otherwise we could try predicting missing values but they may misguide the predictions.

# In[ ]:


# It is highly correlated to total_nights and also there is no much difference impact, so I will not use total_nights.
# Week nights have higher impact.
"""
Actually, I tried some models by using different features as (only total_nights | weekend_nights and week_nights | only week_nights ...) 
and the models using only week nights seems to have a bit higher accuracy score. 
"""

cor_data.corr()['stays_in_week_nights']


# In[ ]:


cor_data = cor_data.drop(columns = ['total_nights', 'arrival_date_week_number', 'stays_in_weekend_nights', 'arrival_date_month', 'agent'], axis = 1)


# In[ ]:


#Lets delete the NA rows of country column
indices = cor_data.loc[pd.isna(cor_data["country"]), :].index 
cor_data = cor_data.drop(cor_data.index[indices])   
cor_data.isnull().sum()

#There is no missing value in the data


# **Since we have decided what to do with features and missing values, we can work on first data.**

# In[ ]:


indices = data.loc[pd.isna(data["country"]), :].index 
data = data.drop(data.index[indices])   
data = data.drop(columns = ['arrival_date_week_number', 'stays_in_weekend_nights', 'arrival_date_month', 'agent'], axis = 1)


# In[ ]:


data.columns


# In[ ]:


# I keep data in case of any changes on features, missing values etc.
df1 = data.copy()


# In[ ]:


#one-hot-encoding
df1 = pd.get_dummies(data = df1, columns = ['meal', 'market_segment', 'distribution_channel',
                                            'reserved_room_type', 'assigned_room_type', 'customer_type', 'reservation_status'])


# In[ ]:


df1['country'] = le.fit_transform(df1['country']) 
# There are more than 300 classes, so I wanted to use label encoder on this feature.


# ### Decision Tree Model (reservation_status included)

# In[ ]:


y = df1["is_canceled"]
X = df1.drop(["is_canceled"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)


# In[ ]:


cart = DecisionTreeClassifier(max_depth = 12)


# In[ ]:


cart_model = cart.fit(X_train, y_train)


# In[ ]:


y_pred = cart_model.predict(X_test)


# In[ ]:


print('Decision Tree Model')

print('Accuracy Score: {}\n\nConfusion Matrix:\n {}\n\nAUC Score: {}'
      .format(accuracy_score(y_test,y_pred), confusion_matrix(y_test,y_pred), roc_auc_score(y_test,y_pred)))


# In[ ]:


pd.DataFrame(data = cart_model.feature_importances_*100,
                   columns = ["Importances"],
                   index = X_train.columns).sort_values("Importances", ascending = False)[:20].plot(kind = "barh", color = "r")

plt.xlabel("Feature Importances (%)")


# In the correlation part, we have seen the impact of reservation status. Reservation status dominates other features totally. By keeping reservation_status in data, it is possible to achieve 100% accuracy rate because that feature is direct way to predict cancellations, its like cheating. For the sake of analysis I will drop reservation_status and continue analysis without it.

# ### Final Arrangements Before Comparing the Models

# In[ ]:


df2 = df1.drop(columns = ['reservation_status_Canceled', 'reservation_status_Check-Out', 'reservation_status_No-Show'], axis = 1)


# In[ ]:


y = df2["is_canceled"]
X = df2.drop(["is_canceled"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)


# In[ ]:


# We can use the functions to apply the models and roc curves to save space.
def model(algorithm, X_train, X_test, y_train, y_test):
    alg = algorithm
    alg_model = alg.fit(X_train, y_train)
    global y_prob, y_pred
    y_prob = alg.predict_proba(X_test)[:,1]
    y_pred = alg_model.predict(X_test)

    print('Accuracy Score: {}\n\nConfusion Matrix:\n {}'
      .format(accuracy_score(y_test,y_pred), confusion_matrix(y_test,y_pred)))
    

def ROC(y_test, y_prob):
    
    false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, y_prob)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    
    plt.figure(figsize = (10,10))
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, color = 'red', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1], linestyle = '--')
    plt.axis('tight')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')


# # Model and ROC Curve Comparison

# ### Logistic Regression Model

# In[ ]:


print('Model: Logistic Regression\n')
model(LogisticRegression(solver = "liblinear"), X_train, X_test, y_train, y_test)


# In[ ]:


LogR = LogisticRegression(solver = "liblinear")
cv_scores = cross_val_score(LogR, X, y, cv = 8, scoring = 'accuracy')
print('Mean Score of CV: ', cv_scores.mean())


# In[ ]:


ROC(y_test, y_prob)


# ### Gaussian Naive Bayes Model

# In[ ]:


print('Model: Gaussian Naive Bayes\n')
model(GaussianNB(), X_train, X_test, y_train, y_test)


# In[ ]:


NB = GaussianNB()
cv_scores = cross_val_score(NB, X, y, cv = 8, scoring = 'accuracy')
print('Mean Score of CV: ', cv_scores.mean())


# In[ ]:


ROC(y_test, y_prob)


# ### Support Vector Classification Model

# In[ ]:


#I excluded probability in the function for SVC, also I could not use other kernel methods because it takes really long and I don't think SVC as a good model for this dateset. 
print('Model: SVC\n')

def model1(algorithm, X_train, X_test, y_train, y_test):
    alg = algorithm
    alg_model = alg.fit(X_train, y_train)
    global y_pred
    y_pred = alg_model.predict(X_test)
    
    print('Accuracy Score: {}\n\nConfusion Matrix:\n {}'
      .format(accuracy_score(y_test,y_pred), confusion_matrix(y_test,y_pred)))
    
model1(SVC(kernel = 'linear'), X_train, X_test, y_train, y_test)


# ### Decision Tree Model (reservation_status excluded)

# In[ ]:


print('Model: Decision Tree\n')
model(DecisionTreeClassifier(max_depth = 12), X_train, X_test, y_train, y_test)


# In[ ]:


DTC = DecisionTreeClassifier(max_depth = 12)
cv_scores = cross_val_score(DTC, X, y, cv = 8, scoring = 'accuracy')
print('Mean Score of CV: ', cv_scores.mean())


# In[ ]:


ROC(y_test, y_prob)


# ### Random Forest

# In[ ]:


print('Model: Random Forest\n')
model(RandomForestClassifier(), X_train, X_test, y_train, y_test)


# In[ ]:


RFC = RandomForestClassifier()
cv_scores = cross_val_score(RFC, X, y, cv = 8, scoring = 'accuracy')
print('Mean Score of CV: ', cv_scores.mean())


# In[ ]:


ROC(y_test, y_prob)


# ### Random Forest Model Tuning

# In[ ]:


rf_parameters = {"max_depth": [10,13],
                 "n_estimators": [10,100,500],
                 "min_samples_split": [2,5]}


# In[ ]:


rf_model = RandomForestClassifier()


# In[ ]:


rf_cv_model = GridSearchCV(rf_model,
                           rf_parameters,
                           cv = 10,
                           n_jobs = -1,
                           verbose = 2)

rf_cv_model.fit(X_train, y_train)


# In[ ]:


print('Best parameters: ' + str(rf_cv_model.best_params_))


# In[ ]:


rf_tuned = RandomForestClassifier(max_depth = 13,
                                  min_samples_split = 2,
                                  n_estimators = 500)

print('Model: Random Forest Tuned\n')
model(rf_tuned, X_train, X_test, y_train, y_test)


# Tuned model has worse accuracy score than default one. In the default model there is no limit for max depth. Increasing max depth gives us better accuracy scores but may decrease generalization.

# ### XGBoost Model

# In[ ]:


print('Model: XGBoost\n')
model(XGBClassifier(), X_train, X_test, y_train, y_test)


# In[ ]:


XGB = XGBClassifier()
cv_scores = cross_val_score(XGB, X, y, cv = 8, scoring = 'accuracy')
print('Mean Score of CV: ', cv_scores.mean())


# In[ ]:


ROC(y_test, y_prob)


# ### Neural Network Model

# In[ ]:


scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[ ]:


print('Model: Neural Network\n')
model(MLPClassifier(), X_train_scaled, X_test_scaled, y_train, y_test)


# In[ ]:


ROC(y_test, y_prob)


# ### Neural Network Model Tuning

# In[ ]:


mlpc_parameters = {"alpha": [1, 0.1, 0.01, 0.001],
                   "hidden_layer_sizes": [(50,50,50),
                                          (100,100)],
                   "solver": ["adam", "sgd"],
                   "activation": ["logistic", "relu"]}


# In[ ]:


mlpc = MLPClassifier()
mlpc_cv_model = GridSearchCV(mlpc, mlpc_parameters,
                             cv = 10,
                             n_jobs = -1,
                             verbose = 2)

mlpc_cv_model.fit(X_train_scaled, y_train)


# In[ ]:


print('Best parameters: ' + str(mlpc_cv_model.best_params_))


# In[ ]:


mlpc_tuned = MLPClassifier(activation = 'relu',
                           alpha = 0.1,
                           hidden_layer_sizes = (100,100),
                           solver = 'adam')


# In[ ]:


print('Model: Neural Network Tuned\n')
model(mlpc_tuned, X_train_scaled, X_test_scaled, y_train, y_test)


# In[ ]:


ROC(y_test, y_prob)


# # Conclusion

# ### Feature Importances

# In[ ]:


randomf = RandomForestClassifier()
rf_model1 = randomf.fit(X_train, y_train)

pd.DataFrame(data = rf_model1.feature_importances_*100,
                   columns = ["Importances"],
                   index = X_train.columns).sort_values("Importances", ascending = False)[:15].plot(kind = "barh", color = "r")

plt.xlabel("Feature Importances (%)")


# ### Summary Table of the Models

# In[ ]:


table = pd.DataFrame({"Model": ["Decision Tree (reservation status included)", "Logistic Regression",
                                "Naive Bayes", "Support Vector", "Decision Tree", "Random Forest",
                                "Random Forest Tuned", "XGBoost", "Neural Network", "Neural Network Tuned"],
                     "Accuracy Scores": ["1", "0.804", "0.582", "0.794", "0.846",
                                         "0.883", "0.851", "0.869", "0.848", "0.859"],
                     "ROC | Auc": ["1", "0.88", "0.78", "0",
                                   "0.92", "0.95", "0", "0.94",
                                   "0.93", "0.94"]})


table["Model"] = table["Model"].astype("category")
table["Accuracy Scores"] = table["Accuracy Scores"].astype("float32")
table["ROC | Auc"] = table["ROC | Auc"].astype("float32")

pd.pivot_table(table, index = ["Model"]).sort_values(by = 'Accuracy Scores', ascending=False)


# - As we can see from the summary table, the best algorithm is random forest for this data set. 
# - 0 values are uncalculated ones.
# - We do not count decision tree with reservatiton status which is broken. All algorithms would give 100% accuracy scores while reservation status is included.
# - Tuning for XGBoost would be a good challenge too.

# ### Thanks for reading all of my analysis. If you have questions, do not hesitate to ask!
# ### I am open to all the feedbacks. 
