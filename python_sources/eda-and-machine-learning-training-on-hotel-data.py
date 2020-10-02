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


# In[ ]:


data = pd.read_csv("../input/hotel-booking-demand/hotel_bookings.csv")


# # Hotel Bookings Analysis
# 
# The aim is to create a  variety of models that predicts the cancellation and compare them in order to see which one would work the best.Ultimately we want to pick the most efficient model.
# ### 1. Preprocessing
# - Handle missing values.
# - Prepare and transform the data if needed for the machine learning models.
# - Create a few more columns, namely:
# family - 1 if the booking is from a family and 0 otherwise
# n_people - total number of people the booking is for
# n_nights - total number of nights the booking is for
# 
# 
# ### 2. Exploratory Data Analysis
# 
# Questions for our EDA(exploratory data analysis)
# 1. Whats the most popular month for bookings for both resort hotel and and city hotel
# 2. Which are the major sources booking those 2 hotels.
# 3. Are there any major differences in lead time for both hotels.
# 4. Whats the ADR behaviour across different months
# 5. Whats the proportion between booked and cancelled(what's the cancellation rate for both hotels)
# 6. Whats the trend across multiple years.
# 7. Is there a link between number of adults and hotel booked
# 8. Are customers prefering to start their holiday during the weekend or weekday
# 9. Are cancellation spiking in a particular month or year
# 
# 
# ### 3. Models and comparison
# 
# - Logistic Regression
# - Naive Bayes Classifier
# - Support Vector Classification
# - Random Forest
# - K nearest neighbour

# ## Imports
# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

#LabelEncoder for transforming the categorical data.
from sklearn.preprocessing import LabelEncoder

#Train test split
from sklearn.model_selection import train_test_split
#Confusion matrix and ROC/AUC for comparing the models
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
#Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


#  # Preprocessing

# Shape of our DataFrame:

# In[ ]:


#Checking whats the shape of our data
data.shape


# Checking the general info of our data:

# In[ ]:


#Checking the general info of our data
data.info()


# Checking if we have missing data in the dataframe

# In[ ]:


# Checking if we have missing data in the dataframe
data.isnull().sum()


# [](http://)Seems like we have 4 problematic columns with missing data, namely:country,agent,company,children.
# We have a few options in order to tackle the incomplete data.
# 1. We can replace the missing entries with the most common occurances (the mode) in that particular column.
# 2. We can delete the missing data from the dataframe.
# 
# ### Decision about missing data
# We can take a mixed approach: We will drop the company and agent columns from the data frame as we have way too many missing entries there.
# 
# We can add the missing 488 country entries with the most frequest occurent country(mode).It would be reasonable to just drop these entries too as we have sufficiently large dataset.
# 
# Similarly we will add the 4 missing children entries with the mean of the column.
# 

# In[ ]:


#Dropping agent and company columns and filling missing data for children and country with the mode of those columns
data=data.drop(['agent','company'],axis=1)
data.country=data.country.fillna(data.country.mode()[0])
data.children=data.children.fillna(data.children.mean())


# Now we have fixed our missing data and we no longer have empty rows:

# In[ ]:


#Checking if we have more missing data:
data.isnull().sum()


# Checking how is our data looking:

# In[ ]:


#Getting a grasp of what the data contains and start planning what questions we want to answer in our exploratory data analysis
data.head()


# seeing all the categorical unique values for our categorical columns

# In[ ]:


# Finding out the unique values for the categorical variable in the dataset:
print('Unique values for hotel:\n', data.hotel.unique())
print('Unique values for arrival_date_month:\n', data.arrival_date_month.unique())
print('Unique values for customer_type:\n', data.customer_type.unique())
print('Unique values for reservation_status:\n', data.reservation_status.unique())
print('Unique values for deposit_type:\n',data.deposit_type.unique())
print('Unique values for reserved_room_type :\n',data.reserved_room_type .unique())
print('Unique values for assigned_room_type :\n',data.assigned_room_type .unique())
print('Unique values for distribution_channel :\n',data.distribution_channel .unique())
print('Unique values for market_segment :\n',data.market_segment.unique())
print('Unique values for meal :\n',data.meal.unique())


# Transforming arrival date month names to numbers 1-12 for easier use and visualisation. Also transforming hotels to Resort Hotel:0 ,City Hotel:1

# In[ ]:


# Transforming arrival date month names to numbers 1-12 for easier use and visualisation
# I did this manually originally and then I saw 

data.arrival_date_month=data.arrival_date_month.map({'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,'July':7,'August':8,'September':9,'October':10,'November':11,'December':12})

# transforming resort hotel and city hotel to 0 and 1 for easier use

data.hotel=data.hotel.map({'Resort Hotel':0 ,'City Hotel':1})


# In[ ]:


data.head()


# We are adding a few more columns that we will later check if they are relevant for our machine learning training.
# 
# The ones will be created are :
# 1. family - if the booking is made by a family then 1(if the booking includes adults and babies and children), else 0
# 2. n_people - that is the total number of adults babies and children
# 3. n_nights - thats the total number of nights the customers booked (weekend nights + week nights)

# In[ ]:


#creating a column that defines whether the customers are family or not using the adults/children/babies columns
def family(data):
    if (data.adults>0 and data.babies>0 and data.children>0):
        val=1
    else:
        val=0
    return val
data['family']=data.apply(family,axis=1)


# In[ ]:


#creating a column that calculates the tot number of people booking:
data['n_people']=data.adults + data.babies + data.children

#creating a column that includes the total number of nights the customer booked for:
data['n_nights']=data['stays_in_weekend_nights']+data['stays_in_week_nights']


# # EDA

# #### Findings:
# * We can see that the city hotel has bigger volume than the resort hotel. 
# * Appart from that given the fact that the dataset is for Portugal we can see that there is seasonality in summer (volume peaks in summer).
# * Customers tend to be more likely to cancel the city hotel than the resort hotel on average.

# In[ ]:


# 1. Whats the most popular month for bookings for both resort hotel and and city hotel
plt.figure(figsize=(16,10))
legend_data={'Resort Hotel':0 ,'City Hotel':1}
ax=sns.countplot(x='arrival_date_month',data=data,hue='hotel')
ax.set(xlabel='Months',ylabel='Number of bookings')
ax.legend(legend_data)
plt.show()


# In[ ]:


#How does cancellation looks like for both hotels.

ax=sns.barplot(x='hotel',y='lead_time',hue='is_canceled', data=data.groupby(["hotel","is_canceled"]).lead_time.count().reset_index())
#ax.legend(legend_data)
plt.show()


# A quick pivot method below to see if there is a significant difference in the means of some of the parameters for the canceled bookings vs the non-canceled bookings.
# * Firstly we can see again that cancellations tend to occur more frequently in the city breaks than the resort hotels(in the hotel column we can see that in the cancelled row the mean is 0.748 which means that cancellations are favoured towards the 1(city hotel rather than the 0(resort))
# * People who book much in advance tend to cancel their bookings more often (144.8 vs 79.98) which makes sense as short lead time bookings leave less time for the customers to change their mind.
# * Both arrival date months means are around 6.5-6.6 which means that somewhere between june and july volume for both peaks. Those numbers being so close together indicates that there wont be big correlation between which month customer books and cancellation though
# * There is no signigicant difference between weekdays and weekends which means that there shouldnt be a strong correlation between which days of the week the customers are booking and cancellation
# * The customers who have cancelled before are much more likely to cancel again.
# * The customers who amend their bookings more often tend to cancel much less.
# * The more customers are in the waiting list the more likely they will cancel.
# * Cheaper bookings get cancelled less on average than more expensive bookings.
# * The more special requests customers have the less likely they will cancel their booking.
# * There is no immediate inference we can make about the new metrics we created and cancellation (family,n_people,n_nights)

# In[ ]:


data.groupby(["is_canceled"]).mean().reset_index()


# #### Country data inference:
# *Biggest source is domestic with other major EU sources following.

# In[ ]:


country_data=data.country.value_counts()
# Adding the bookings for all countries outside of top 10 and naming that row other.
other_vals=pd.DataFrame({0:country_data[10:].sum()},index=['other'])
#Adding that new row to our data
country_data=country_data.append(other_vals)
# Sorting the country data in descending order
country_data=country_data.sort_values(by=[0],ascending=False)


# Creating the pie chart and displaying only the top 10 sources
plt.figure(figsize=(10,6))
plt.title('Top 10 sources per gross bookings volume')
fig=plt.pie(country_data[0:9],labels=country_data.index[0:9],autopct='%1.1f%%')
plt.show()


# In[ ]:


#Are there any major differences in lead time for both hotels.

data[['hotel','lead_time']].groupby(["hotel"]).mean().reset_index()


# In[ ]:


plt.figure(figsize=(10,6))
ax=sns.barplot(x='hotel',y='lead_time', data=data[['hotel','lead_time']].groupby(["hotel"]).mean().reset_index())
#ax.legend(legend_data)
plt.title('Booking lead times by hotel type')
plt.show()


# #### Pivoting the data in order to try and spot some trends in lead time over the years.
# * We can see that we have increasing lead time trend YOY thats consistent across all years in the dataset

# In[ ]:


#Whats the ADR behaviour across different months
data_line_plot=data[['arrival_date_year','arrival_date_month','lead_time']].groupby(['arrival_date_year',"arrival_date_month"]).mean().reset_index()
data_line_plot


# In[ ]:


#plotting a graph with lead time per year for all the data.
plt.figure(figsize=(10,6))
plt.title('Yearly Lead time trends for both hotels at the same time')
sns.lineplot(x='arrival_date_month',y='lead_time',data=data_line_plot,hue='arrival_date_year',marker='o',palette=['red','blue','green'])
plt.show()


# Whats the proportion between booked and cancelled(what's the cancellation rate for both hotels)
# * Cancelation rate for resort hotel :28%
# * Cancelation rate for city hotel: 42%

# In[ ]:


#Whats the proportion between booked and cancelled(what's the cancellation rate for both hotels)
lineplot2=data[['is_canceled','hotel','lead_time']].groupby(['is_canceled','hotel']).count().reset_index()
lineplot2


# In[ ]:


plt.figure(figsize=(10,6))
ax=sns.barplot(x='hotel',y='lead_time', data=lineplot2,hue='is_canceled')
#ax.legend(['Resort hotel is 0','City hotel is 1'])
plt.title('Booking lead times by hotel type')
plt.show()


# In[ ]:


# Calculating the cancelation rate for both hotels:
print('Calcelation rate for resort hotel:',round(lineplot2['lead_time'][2]/(lineplot2['lead_time'][0]+lineplot2['lead_time'][2]),2))
print('Calcelation rate for city hotel:',round(lineplot2['lead_time'][3]/(lineplot2['lead_time'][1]+lineplot2['lead_time'][3]),2))


# There is no big difference between number of people that have booked the hotel and cancelation which means that we shouldnt expect strong correlation between number of people and cancellation

# In[ ]:


#Is there a link between number of guests and hotel booked
#Pivoting the number of hotels and seeing what the mean of n_people is:
data[['hotel','n_people']].groupby(['hotel']).mean().reset_index()


# Cancellations for city hotel are consistently higher than the cancellations for the resort:

# In[ ]:


#Are cancellation spiking in a particular month or year
data_line_plot=data[['arrival_date_year','arrival_date_month','is_canceled','hotel']].groupby(['arrival_date_year',"arrival_date_month",'hotel']).mean().reset_index()
data_line_plot


# In[ ]:


#plotting a graph with lead time per year for all the data.
plt.figure(figsize=(10,6))
plt.title('Yearly cancelation trends for both hotels')
sns.lineplot(x='arrival_date_month',y='is_canceled',data=data_line_plot,hue='hotel',marker='o',palette=['red','blue'])
plt.show()


# # Label encoding the categorical columns 
# 
# In this section we are transforming the categorical data into numerical in order to run a correlation analysis and find which features are relevant for our machine learning model.

# In[ ]:


# Calling the LabelEncoder. Also I am duplicating data into data1 where I will encode all the categorical values.
# After that I will see the correlations of all the variables and particularly the correllation with is_cancelled.
#That way if some of the variables have very low correlation I can safely drop from the models.
le = LabelEncoder()
data1=data
data1.customer_type=le.fit_transform(data1.customer_type)
# changing the country strings to numerical data
data1.country=le.fit_transform(data1.country)
#Transforming all the data to numerical values so that we can use those variables for correlation analysis and the model building
data1.deposit_type=le.fit_transform(data1.deposit_type)
data1.reserved_room_type=le.fit_transform(data1.reserved_room_type)
data1.assigned_room_type=le.fit_transform(data1.assigned_room_type)
data1.distribution_channel=le.fit_transform(data1.distribution_channel)
data1.market_segment=le.fit_transform(data1.market_segment)
data1.meal=le.fit_transform(data1.meal)


# # Running correlation analysis to determine which variables are significant for our analysis.

# In[ ]:


data1.corr()


# #### From the chart below we can make a few decisions:
# 1. Drop arrival_date_week_number, stays_in_weekend_nights and arrival_date_day_of_month since their importances are really low while predicting cancellations.
# 2.  Also we need to drop babies and children and we have used them to create the n_people column.
# 3. We need to drop reservation_status as we have data there about which data is cancelled and which not, so we cannot use that as a X variable.
# 4. Adults has better explanation power than total number of people so drop that column that we previously created.
# 5. Drop n_nights ad the week_nights column has better explanaion power.

# In[ ]:


#Determining how how are all x values correlated to is_canceled
data1.corr()['is_canceled']


# In[ ]:


#Correlation heatmap
plt.figure(figsize=(40,25))
sns.heatmap(data1.corr(),annot=True,annot_kws={'size':18})
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.show()


# In[ ]:


#droping all the abovementioned columns
data1=data1.drop(['reservation_status','stays_in_weekend_nights','arrival_date_week_number','stays_in_weekend_nights','arrival_date_day_of_month','meal','babies','children','n_people'],axis=1)
data1=data1.drop(['n_nights'],axis=1)
data1=data1.drop(['reservation_status_date'],axis=1)


# Final visual check to see if our Label Encoder worked as intended:

# In[ ]:


data1.head()


# ### To prep the data for model building we need to:
# 1. move the is_canceled column in a var named y
# 2. drop the is_canceled column from data1 and name it x
# 3. split our data into train/test data.

# In[ ]:


# Firstly move the is canceled column to y as thats what we want to predict
y=data1['is_canceled']

#Then drop the is_canceled column from our features.
x=data1.drop(['is_canceled'],axis=1)

# Create the test/train split for our models. I have arbitrary chosen 80-20 split.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)


# # Model training and testing.
# Now we are ready to train our model and predict which customer will cancel their booking. There are many predictive algorithms to choose from. As this problem is a classification and regression problem I will choose:
# 1. Logistic Regression
# 2. Naive Bayes Classifier
# 3. SVC(Support Vector Classification)
# 4. Random Forest
# 5. KNN or k-Nearest Neighbors
# 
# I will then do 3 metrics:
# * Calculate the score of each model
# * Calculate the confusion matrix of each model
# * Calculate the ROC/AUC for each model.
# 
# #### Finally, I will combine the scoreas and AUC for each model and see which model performed best. The goal is too choose the model with the highest score/AUC.
# 

# ## Logistic Regression calculations

# In[ ]:


#logistic regression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
y.prob=logreg.decision_function(x_test)
acc_log = round(logreg.score(x_test, y_test) * 100, 2)
print('Score:',acc_log)
print('Confusion matrix:\n',confusion_matrix(y_test, y_pred))


# In[ ]:


#Creating roc and aoc for logreg
logreg_fpr,logreg_tpr,threshold=roc_curve(y_test,y_pred)
auc_logreg=auc(logreg_fpr,logreg_tpr)


# ## Naive Bayes Classifier calculations

# In[ ]:


#Naive bayes classifier
nbc = GaussianNB()
nbc.fit(x_train, y_train)
y_pred = nbc.predict(x_test)
acc_nbc = round(nbc.score(x_test, y_test) * 100, 2)
print('Score:',acc_nbc)
print('Confusion matrix:\n',confusion_matrix(y_test, y_pred))


# In[ ]:


#Creating roc and aoc for GaussianNB
nbc_fpr,nbc_tpr,threshold=roc_curve(y_test,y_pred)
auc_nbc=auc(nbc_fpr,nbc_tpr)


# ## SVC calculations

# In[ ]:


#SVC
svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
acc_svc = round(svc.score(x_test, y_test) * 100, 2)
print('Score:',acc_svc)
print('Confusion matrix:\n',confusion_matrix(y_test, y_pred))


# In[ ]:


#Creating roc and aoc for SVC
svc_fpr,svc_tpr,threshold=roc_curve(y_test,y_pred)
auc_svc=auc(svc_fpr,svc_tpr)


# ## Random Forest calculations

# In[ ]:


#Random Forest
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
acc_rf = round(rf.score(x_test, y_test) * 100, 2)
print('Score:',acc_rf)
print('Confusion matrix:\n',confusion_matrix(y_test, y_pred))


# In[ ]:


#Creating roc and aoc for Random Forest
rf_fpr,rf_tpr,threshold=roc_curve(y_test,y_pred)
auc_rf=auc(rf_fpr,rf_tpr)


# ## K-near neigbours Classifier calculations

# In[ ]:


#knn neghbours
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
acc_knn = round(knn.score(x_test, y_test) * 100, 2)
print('Score:',acc_knn)
print('Confusion matrix:\n',confusion_matrix(y_test, y_pred))


# In[ ]:


#Creating roc and aoc for knn neighbours
knn_fpr,knn_tpr,threshold=roc_curve(y_test,y_pred)
auc_knn=auc(knn_fpr,knn_tpr)


# ## Result summary

# #### We have put all results into a DataFrame to best see and comapare all 5 machine learning performance.
# #### We can see that the random forest model has the best accuracy (almost 89%) with the bes ROC/AUC measure.****

# In[ ]:


model_names=['Logistic Regression','Naive Bayes Classifier','SVC(Support Vector Classification)','Random Forest','k-Nearest Neighbors']
accuracy=[acc_log,acc_nbc,acc_svc,acc_rf,acc_knn]
auc=[auc_logreg,auc_nbc,auc_svc,auc_rf,auc_knn]
results=pd.DataFrame({'Model':model_names,'Accuracy':accuracy,'AUC':auc})
results


# # Visualising the Area under the curve with a line plot
# We can clearly see that the random forest perform best among the 5 machine learning models built and it is the furthest away from the diagonal of the graph.

# In[ ]:


#plotting the AUC curves to visualise that random forest model is the best in this situation
plt.figure(figsize=(10,10))
plt.plot(logreg_fpr,logreg_tpr,label='Logistic Regrassion (auc=%0.3f)' %auc_logreg)
plt.plot(nbc_fpr,nbc_tpr,label='Naive Bayes Classifier (auc=%0.3f)' %auc_nbc)
plt.plot(svc_fpr,svc_tpr, label='Support Vector Classifier (auc=%0.3f)' %auc_svc)
plt.plot(rf_fpr,rf_tpr, label='Random Forest (auc=%0.3f)' %auc_rf)
plt.plot(knn_fpr,knn_tpr, label='KNN neighbours (auc=%0.3f)' %auc_knn)
plt.title('AUC for all 5 models')
plt.xlabel('False Positives')
plt.ylabel('True Postives')
plt.legend(loc=4)
plt.show()


# ### Thanks for taking the time to read my analysis. If you have questions, do not hesitate to ask!
# ### I welcome all kind of feedback as I am still learning and trying to improve my data science skills.
