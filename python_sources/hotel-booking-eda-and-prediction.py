#!/usr/bin/env python
# coding: utf-8

# # This is my first Kaggle project. I will do some EDA on the dataset. I will then do some feature selection, apply different models, tune and select the best one and predict cancellation. I am new to this field and Kaggle so I appreciate if you can leave comments and suggestions. Thanks 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Ignore warnings
import warnings  
warnings.filterwarnings('ignore')


# In[ ]:


data = pd.read_csv('/kaggle/input/hotel-booking-demand/hotel_bookings.csv')


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


data.describe().T


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# ## Let's look at missing data

# In[ ]:


perc_missing_data = pd.DataFrame([data.isnull().sum(),data.isnull().sum()*100.0/data.shape[0]]).T
perc_missing_data.columns = ['No. of Missing Data', '% Missing Data']
perc_missing_data


# #### 0.003% of rows have missing information for children. Let's look at distribution of children to fill missing information

# In[ ]:


data['children'].value_counts()


# #### Most bookings had no children and hence we will fill the missing rows for children with value 0

# In[ ]:


data['children'].fillna(0,inplace=True)


# #### 0.4% of rows have missing information for country. Let's look at distribution of country to fill missing information

# In[ ]:


perc_country_data = pd.DataFrame([data['country'].value_counts(),data['country'].value_counts()*100/data.shape[0]]).T
perc_country_data.columns = ['Count', '% Distribution']
perc_country_data


# #### 40.7% of bookings are from Portugal. Only 0.4% is missing information. We will fill the missing rows of country as Portugal since the distribution will not change by much and we still get to preserve data and discard the row.

# In[ ]:


data['country'].fillna('PRT',inplace=True)


# #### 13% of agent ID and 94% of company ID is missing. It is possible to dive deep into the details of the dataset to find a correlation of missing information in agent ID and company ID vs other other features like market segment, distribution channel etc; For example most direct bookings may not have an agent ID or company ID and the information is probably null. It is therefore possible to fill these missing values based on other features however for simplicity, we will drop both the columns. 

# In[ ]:


data.drop(['agent','company'],axis=1,inplace=True)


# #### Let's confirm all missing data have been handled

# In[ ]:


perc_missing_data = pd.DataFrame([data.isnull().sum(),data.isnull().sum()*100.0/data.shape[0]]).T
perc_missing_data.columns = ['No. of Missing Data', '% Missing Data']
perc_missing_data


# # DATA VISUALIZATION

# In[ ]:


plt.style.use('fivethirtyeight')


# ### Let's look at distribution of hotel bookings and separate them by their cancellation status

# In[ ]:


plt.figure(figsize=(14,6))
sns.countplot(x='hotel',data=data,hue='is_canceled',palette='pastel')
plt.show()


# #### About 25% of resort hotel bookings have been cancelled and about 40% of city hotel bookings have been cancelled. These numbers are high and have potential implications in revenue for the hotels

# ### Let's look at deposit type vs cancellation status

# In[ ]:


plt.figure(figsize=(14,6))
sns.countplot(x='deposit_type',data=data,hue='is_canceled',palette='pastel')
plt.show()


# #### About 30k bookings of deposit type 'No Deposit' were cancelled. These numbers are huge if the hotels were not able to replace the cancelled bookings in time. It's a significant loss for the hotel. But in the next section, we will look at date of cancellation vs date of arrival to understand the impact of cancellation and how much time the hotel had to prepare for cancellations.
# 
# #### Also it is interesting to note that non-refundable deposits had more cancellation than refundable deposits. Logically one would have assumed that refundable deposits have more cancellation as hotel rates are usually higher for refundable deposit type rooms and customers pay more in anticipation of cancellation

# ### Date of Cancellation vs Date of Arrival

# #### We first need to create a new column called arrival_date that combines arrival date year, month and date. We then compare arrival_date to cancellation date to find out how the cancellation happens. Cancellation date can be identified from reservation_status_date for reservation_status = Cancelled

# In[ ]:


data['arrival_date'] = data['arrival_date_year'].astype(str) + '-' + data['arrival_date_month'] + '-' + data['arrival_date_day_of_month'].astype(str)
data['arrival_date'] = data['arrival_date'].apply(pd.to_datetime)
data['reservation_status_date'] = data['reservation_status_date'].apply(pd.to_datetime)


# #### Create a new dataframe for cancelled bookings called cancelled_data. Add a new column called canc_to_arrival_days that is the difference between cancellation date and arrival date

# In[ ]:


cancelled_data = data[data['reservation_status'] == 'Canceled']
cancelled_data['canc_to_arrival_days'] = cancelled_data['arrival_date'] - cancelled_data['reservation_status_date']
cancelled_data['canc_to_arrival_days'] = cancelled_data['canc_to_arrival_days'].dt.days


# ### Let's visualize distribution of days from cancellation to arrival

# In[ ]:


plt.figure(figsize=(14,6))
sns.distplot(cancelled_data['canc_to_arrival_days'])
plt.show()


# #### Assuming the hotel can sufficiently replace the cancelled reservation in a week, we are only interested in cancellations that happen less than a week to arrival date which bear a financial cost to the hotels

# In[ ]:


print('Percentage of cancellations that are within a week of arrival: ', 
      (cancelled_data[cancelled_data['canc_to_arrival_days']<=7]['canc_to_arrival_days'].count()*100/cancelled_data['canc_to_arrival_days'].count()).round(2), '%')


# #### 12% of cancellations happen less than a week. There is huge benefits to predicting if a customer will cancel a booking so the hotel can adequately prepare for it. 

# ### Let's visualize other features to have an idea about the dataset 

# #### Let's see at what times of the year do we have the highest bookings

# In[ ]:


month_sorted = ['January','February','March','April','May','June','July','August','September','October','November','December']
plt.figure(figsize=(14,6))
sns.countplot(data['arrival_date_month'], palette='pastel', order = month_sorted)
plt.xticks(rotation = 90)
plt.show()


# #### It looks like the summer months May-August have the highest demand. The winter months November-February have the lowest demand. Let's now see which months have the highest cancellations as our target is cancellations.

# In[ ]:


perc_monthly_canc = pd.DataFrame(data[data['is_canceled'] == 1]['arrival_date_month'].value_counts() * 100 / data['arrival_date_month'].value_counts())
perc_monthly_canc.reset_index()
plt.figure(figsize=(14,6))
sns.barplot(x=perc_monthly_canc.index,y='arrival_date_month',data=perc_monthly_canc, order=month_sorted, palette='pastel')
plt.xticks(rotation = 90)
plt.ylabel('% cancellation per month')
plt.show()


# #### There is not a significant difference in the percentage cancellations between months however the lowest demand months have the lowest % cancellations and the highest demand months have the highest % cancellations. The hotels will accept this trend as filling in cancelled rooms during peak season becomes easier. 

# ### Let's now look at market segment vs cancellation

# In[ ]:


plt.figure(figsize=(8,8))
explode = [0.005] * len(cancelled_data['market_segment'].unique())
colors = ['royalblue','orange','y','darkgreen','gray','purple','red','lightblue']
plt.pie(cancelled_data['market_segment'].value_counts(),
       autopct = '%.1f%%',
       explode = explode,
       colors = colors)
plt.legend(cancelled_data['market_segment'].unique(), bbox_to_anchor=(-0.1, 1.),
           fontsize=14)
plt.title('Market Segment vs Cancelled Bookings')
plt.tight_layout()
plt.show()


# #### About 65% of the cancelled bookings are by travel agents or tour operators

# #### So far we've looked at some features and plotted their general distribution and how they behave against cancellation. Now let's look at the entire feature set and see how they correlate with cancellation status. This step is going to help us select features for our model

# In[ ]:


plt.figure(figsize=(10,8))
data.corr()['is_canceled'].sort_values()[:-1].plot(kind='bar')
plt.show()


# #### The correlation happens only with numerical values. Let's look at distribution of some of categorical variables that we've not covered already in the previous sections to decide which ones out of those we want to carry over for the model.

# In[ ]:


plt.figure(figsize=(16,12))
plt.subplot(221)
sns.countplot(data['meal'], hue=data['is_canceled'])
plt.xlabel('Meal Type')
plt.subplot(222)
sns.countplot(data['customer_type'], hue=data['is_canceled'])
plt.xlabel('Customer Type')
plt.subplot(223)
sns.countplot(data['reserved_room_type'], hue=data['is_canceled'])
plt.xlabel('Reserved Room Type')
plt.subplot(224)
sns.countplot(data['reservation_status'], hue=data['is_canceled'])
plt.xlabel('Reservation Status')
plt.show()


# #### It's clear that meal type and reserved room type don't have bookings evenly distributed. In both the features, bookings heavily favour one category and hence we will drop both columns. We will drop deposit type(visualized previously) for the same reasons
# 
# #### We will keep customer type feature and convert to dummy variables.
# 
# #### The reservation status feature is basically the target variable. To avoid data leakage we will drop this column as well.
# 
# #### There are too many countries and will add a lot of dimension when converted to dummy variables. We will drop this column as well. (NOTE - we could have dropped this at the filling missing data stage itself)
# 
# #### We will also drop all the date features.
# 
# #### We will convert the other categorical features which we've visualized in the previous sections based on intuition.

# ## CONVERTING CATEGORICAL COLUMNS TO DUMMY VARIABLES AND DROPPING UNNECESSARY COLUMNS

# In[ ]:


data = data.drop(['meal','country','reserved_room_type','assigned_room_type','deposit_type','reservation_status','reservation_status_date','arrival_date'], axis=1)
data = pd.concat([data, 
                 pd.get_dummies(data['hotel'], drop_first=True), 
                 pd.get_dummies(data['arrival_date_month'], drop_first=True), 
                 pd.get_dummies(data['market_segment'], drop_first=True),
                 pd.get_dummies(data['distribution_channel'], drop_first=True),
                 pd.get_dummies(data['customer_type'], drop_first=True)
                 ], axis=1)
data = data.drop(['hotel','arrival_date_month','market_segment','distribution_channel','customer_type'], axis=1)


# #### Verifying no categorical variables exist in the dataset

# In[ ]:


data.info()


# #### Let us again look at the correlation of the target variable with rest of the selected features after dummy variable conversion

# In[ ]:


plt.figure(figsize=(16,8))
data.corr()['is_canceled'].sort_values()[:-1].plot(kind='bar')
plt.show()


# ## MODELING

# We will use all features left to build our model. The following are the steps we will take to build our model.
# 
# 1. Split into training and test sets
# 2. Apply feature scaling
# 3. Baseline model
# 4. Train and predict multiple models - LogisticRegression, KNearestNeighbors, SVM, RandomForest
# 5. Compare against baseline model and choose the best model using accuracy
# 6. Use grid search to tune hyperparameters
# 7. Retrain model using chosen hyperparameters
# 8. Predict 

# #### Split into training and test sets

# In[ ]:


X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# #### Feature Scaling

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# #### Functions and Variable Assignments

# In[ ]:


# Empty dictionary of model accuracy results
model_accuracy_results = {}

# Function for calculating accuracy from confusion matrix
from sklearn.metrics import confusion_matrix
def model_accuracy(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    accuracy = ((cm[0,0] + cm [1,1]) * 100 / len(y_test)).round(2)
    return accuracy


# #### Baseline Model - We will take the class that has most observations in the training set and applying it as the predicted result and compute accuracy

# In[ ]:


# Baseline model
(unique, counts) = np.unique(y_train, return_counts=True)
if counts[0]  > counts[1]:
    idx = 0
else:
    idx = 1

# Applying baseline results to y_pred
if idx == 0:
    y_pred = np.zeros(y_test.shape)
else:
    y_pred = np.ones(y_test.shape)

# Computing accuracy
model_accuracy_results['Baseline'] = model_accuracy(y_test, y_pred)


# #### Logistic Regression

# In[ ]:


# Fit and train
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0, max_iter=250)
classifier.fit(X_train, y_train)

# Predict
y_pred = classifier.predict(X_test)

# Computing accuracy
model_accuracy_results['LogisticRegression'] = model_accuracy(y_test, y_pred)


# ### K Nearest Neighbors

# In[ ]:


# Fit and train
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 10)
classifier.fit(X_train,y_train)

# Predict
y_pred = classifier.predict(X_test)

# Computing accuracy
model_accuracy_results['KNearestNeighbors'] = model_accuracy(y_test, y_pred)


# #### SVM

# In[ ]:


# Fit and train
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state=0)
classifier.fit(X_train,y_train)

# Predict
y_pred = classifier.predict(X_test)

# Computing accuracy
model_accuracy_results['SVM'] = model_accuracy(y_test, y_pred)


# #### RandomForest

# In[ ]:


# Fit and train
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
classifier.fit(X_train,y_train)

# Predict
y_pred = classifier.predict(X_test)

# Computing accuracy
model_accuracy_results['RandomForest'] = model_accuracy(y_test, y_pred)


# #### Let's visualize the accuracy results

# In[ ]:


df_model_accuracies = pd.DataFrame(list(model_accuracy_results.values()), index=model_accuracy_results.keys(), columns=['Accuracy'])
df_model_accuracies


# #### We will choose the RandomForest Model as it gives the best accuracy. Now we will tune the hyper parameters on the random forest model using grid search and retrain the model to see if performance improved

# In[ ]:


# Grid Search
from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators': [10,25,50,100,500] , 'criterion': ['entropy', 'gini']}]
randomforestclassifier = RandomForestClassifier()
grid_search = GridSearchCV(estimator = randomforestclassifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           n_jobs = -1)
grid_search.fit(X_train, y_train)


# #### Results of grid search

# In[ ]:


print('Best Score: ', grid_search.best_score_.round(2))
print('Best Parameters: ', grid_search.best_params_)


# #### We will choose the best parameters and re-train the model with the new parameters and compare against the original RandomForest classifier accuracy result

# #### RandomForest with new parameters

# In[ ]:


# Fit and train
optimized_classifier = RandomForestClassifier(n_estimators=500, criterion='entropy', random_state=0)
optimized_classifier.fit(X_train,y_train)

# Predict
y_pred = optimized_classifier.predict(X_test)

# Computing accuracy
model_accuracy_results['OptimizedRandomForest'] = model_accuracy(y_test, y_pred)


# #### Let's visualize the model accuracy results again with OptimizedRandomForest included

# In[ ]:


df_model_accuracies = pd.DataFrame(list(model_accuracy_results.values()), index=model_accuracy_results.keys(), columns=['Accuracy'])
df_model_accuracies


# #### Let's visualize the confusion matrix of the OptimizedRandomForest Model

# In[ ]:


orf_cm = confusion_matrix(y_test, optimized_classifier.predict(X_test))

names = ['True Neg','False Pos','False Neg','True Pos'] # list of descriptions for each group
values = [value for value in orf_cm.flatten()] # list of values for each group
percentages = [str(perc.round(2))+'%' for perc in orf_cm.flatten()*100/np.sum(orf_cm)] # list of percentages for each group
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(names,values,percentages)] # zip them into list of strings as labels
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(orf_cm, annot=labels, fmt='', cmap='binary')


# ### We've completed predicting cancellations with an accuracy of 86%. Please upvote and leave comments. Thanks for your time.

# In[ ]:




