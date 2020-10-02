#!/usr/bin/env python
# coding: utf-8

# <h2> About The DataSet </h2>

# Since 2008, guests and hosts have used Airbnb to expand on traveling possibilities and present more unique, personalized way of experiencing the world. This dataset describes the listing activity and metrics in NYC, NY for 2019.
# 
# This data file includes all needed information to find out more about hosts, geographical availability, necessary metrics to make predictions and draw conclusions.
# 
# 
# Inspiration
# What can we learn about different hosts and areas?
# What can we learn from predictions? (ex: locations, prices, reviews, etc)
# Which hosts are the busiest and why?
# Is there any noticeable difference of traffic among different areas and what could be the reason for it?
# 

# ### Columns
# <HTML>
# <ol>
# <li>idlisting ID</li>
# <li>namename of the listing</li>
# <li>host_idhost ID</li>
# <li>host_namename of the host</li>
# <li>neighbourhood_grouplocation</li>
# <li>neighbourhoodarea</li>
# <li>latitudelatitude coordinates</li>
# <li>longitudelongitude coordinates</li>
# <li>room_typelisting space type</li>
# <li>priceprice in dollars</li>
# <li>minimum_nightsamount of nights minimum</li>
# <li>number_of_reviewsnumber of reviews</li>
# <li>last_reviewlatest review</li>
# <li>reviews_per_monthnumber of reviews per month</li>
# <li>calculated_host_listings_countamount of listing per host</li>
# <li>availability_365number of days when listing is available for booking</li>
# </ol>
# </HTML>

# ## Reading the Data

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import scipy


# In[ ]:


project_data=pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')


# In[ ]:


project_data.head(5)


# In[ ]:


# finding Number of rows and columns
print("Number of data points in resources data", project_data.shape)
print("="*100)


# In[ ]:


# columns/features in the dataset
print("Names of features are:- ")
print(project_data.columns.values)


# In[ ]:


#to check about the data type of the columns
project_data.info()


# In[ ]:


#to check for null values
np.sum(project_data.isna())


#  Firstly We will check weather  the features with null values are usefull or not then if they are usefull then we  will fill those null values else we will discard those

# In[ ]:


#Neighbourhood Group
project_data.neighbourhood_group.unique()


# In[ ]:


# to find the correlation between the features
project_data.corr()


# Note:                                                                                                                        
#  Positive Value Indicates Positive Relationship(if x increases then y increases)                                                               
#  Negative Value Indicates Negative Relationship (if x increases then y Decreases)

# In[ ]:


#to get the details like number of observations, min,max,25%,50%,75% ,mean,std
project_data.describe()


# ***Observations:-***            
# 1.The average price is 152 dollars                              
# 2.There is a 50% chance that the rooms will be available for atleast 45 days                   
# 3.75% of people spend 5 nights                                                  
# 4.Average Number of Host listings are 7       

# # EXPLORATORY DATA ANALYSIS (EDA) #

# ## Univariate Analysis

# Distributions plots Source Code:-                                            
#  https://www.kaggle.com/adikeshri/what-s-up-with-new-york

# ## Histograms for univariate Analysis

# In[ ]:


#distribution of price less than 2000
sns.distplot(project_data[project_data.price<2000].price)
plt.title('Distribution of price (only where price<2000)')
plt.show()


# Obseravtion:
# 
#  Price of Most of the Listings are 10-200
# 

# In[ ]:


#to check the distribution of minimum nights
sns.distplot(project_data[project_data.minimum_nights<20].minimum_nights)
plt.title('Distribution of Minimum nights ')
plt.show()


# Observations:
#     Most of the nights stayed are mostly 2-7 days

# In[ ]:


# Histograms for univariate Analysis
plt.hist(project_data['number_of_reviews'])
plt.show()


# In[ ]:


sns.distplot(project_data[project_data.number_of_reviews<50].number_of_reviews)
plt.title('Distribution of number_of_reviews ')
plt.show()


# Observations:                                             
#   There are More number listings with less Number of reviews(0-10) 

# In[ ]:


# Histograms for univariate Analysis
plt.hist(project_data['availability_365'])
plt.show()


# Observations:-                                                    
#     There are less than number of listings which has more number of days(<50) of availability

# In[ ]:


#https://www.kaggle.com/adikeshri/what-s-up-with-new-york
sns.countplot(project_data['neighbourhood_group'])
plt.title('boroughs wise listings in NYC')
plt.xlabel('boroughs name')
plt.ylabel('Count')
plt.show()


# Inspiration                                                                           
# 4.Is there any noticeable difference of traffic among different areas and what could be the reason for it?                                                                                                                                  
# Ans:-Yes, there is a huge traffic difference among (Brooklyn , Manhattan) and ( queens staten Island Bronx)
#      
# Manhattan has Highest number of listings because 
# 
# Manhattan is the center of theater, visual arts, television, museums, finance, advertising, and much more Which attracts Many Tourits 
# 
# Brooklyn has  more Number of listings after Manhattan because Brooklyn is Famous for its freak shows and rickety old roller coaster, the legendary beach is also home to the New York Aquarium.

# In[ ]:


sns.countplot(project_data.sort_values('room_type').room_type)
plt.title('Room type count')
plt.xlabel('Room type')
plt.ylabel('Count')
plt.show()


# 
# Observation:-                                           
# Most of the Listings are Entire Home/apartments and private rooms 

# In[ ]:


# Histograms for univariate Analysis
plt.hist(project_data['reviews_per_month'])
plt.show()


# Observations:
#     Almost 95% of them have reviews per month less than 10

# # Bi-variate Analysis

# ##  2-D Scatter Plots

# In[ ]:


project_data.plot(kind='scatter', x='price', y='minimum_nights') ;
plt.show()


# In[ ]:


project_data.plot(kind='scatter', y='price', x='calculated_host_listings_count') ;
plt.show()


# Obsevations:
#     Most(75%) of the host listing count are less than 50

# In[ ]:


project_data.plot(kind='scatter', y='price', x='number_of_reviews') ;
plt.show()


# In[ ]:


project_data.plot(kind='scatter', y='price', x='availability_365') ;
plt.show()


# In[ ]:


project_data.plot(kind='scatter', y='price', x='reviews_per_month') ;
plt.show()


# In[ ]:


# soure: previous project
sns.set_style("whitegrid");
sns.FacetGrid(project_data,hue='room_type',size=5).map(plt.scatter,'price','minimum_nights').add_legend()
plt.show()


# In[ ]:


# pairwise scatter plot: Pair-Plot.
plt.close();
sns.set_style("whitegrid");
sns.pairplot(project_data,hue='room_type',vars=['price','minimum_nights','availability_365'],size=6,diag_kind='kde');
plt.legend()
plt.show() 
# NOTE: the diagnol elements are PDFs for each feature. PDFs are expalined below.


# Observation:                 
# Price of Shared room is lower than  private room and entire house/ apartment                  
#    

# In[ ]:


# pairwise scatter plot: Pair-Plot.
plt.close();
sns.set_style("whitegrid");
sns.pairplot(project_data,hue='neighbourhood_group',vars=['price','minimum_nights','availability_365'],size=6,diag_kind='kde');
plt.legend()
plt.show() 
# NOTE: the diagnol elements are PDFs for each feature. PDFs are expalined below.


# Observations:
#     Bronx has less number of night stays as compared to other

# In[ ]:


counts,bin_edges=np.histogram(project_data['minimum_nights'],bins=10,density=True)
pdf=counts/(sum(counts))
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)
plt.show()
print(np.percentile(project_data['minimum_nights'],95))


# observations:-
#     95% of Minimum Number of nights are less than 30days

# In[ ]:


project_data['price_500']=project_data[project_data.price<500].price
sns.FacetGrid(project_data,hue='neighbourhood_group',size=5).map(sns.distplot,'price_500').add_legend()
plt.show()


# Observations:                                                   
# 
#     price of Rooms in Manhattan and Brooklyn boroughs are more than the rooms in boroughs

# In[ ]:


project_data.columns


# # Data Preprocessing

# ### from the above analysis it is clear that 'id', 'name', 'host_id', 'host_name', 'neighbourhood', 'last_review', 'reviews_per_month' has no impact or correlation to the price column  so  I am Ignoring the above columns 
# 

# In[ ]:


project_data.drop(['id', 'name', 'host_id', 'host_name', 'neighbourhood', 'last_review', 'reviews_per_month','price_500'], axis=1, inplace=True)


# In[ ]:


project_data.head(2)


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn import preprocessing

Y = project_data['price']
X = project_data[['neighbourhood_group', 'longitude', 'room_type', 'minimum_nights','availability_365','latitude', 'calculated_host_listings_count']]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)
X_cv,X_test,Y_cv,Y_test = train_test_split(X_train, Y_train, test_size = 0.3)
print("Train set shape:")
print(X_train.shape)
print(Y_train.shape)
print("="*50)
print("Test set shape:")
print(X_test.shape)
print(Y_test.shape)


# ## Encoding Categorical Values

# In[ ]:


from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()
lb_train_ng= lb.fit_transform(X_train['neighbourhood_group'])
lb_test_ng= lb.transform(X_test['neighbourhood_group'])

lb_train_ng = pd.DataFrame(lb_train_ng, columns=lb.classes_)
lb_test_ng = pd.DataFrame(lb_test_ng, columns=lb.classes_)

print("After vectorizations")
print(lb_train_ng.shape, Y_train.shape)
print(lb_test_ng.shape, Y_test.shape)


# In[ ]:


from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()
lb_train_rt= lb.fit_transform(X_train['room_type'])
lb_test_rt= lb.transform(X_test['room_type'])

lb_train_rt = pd.DataFrame(lb_train_rt, columns=lb.classes_)
lb_test_rt = pd.DataFrame(lb_test_rt, columns=lb.classes_)

print("After vectorizations")
print(lb_train_rt.shape, Y_train.shape)
print(lb_test_rt.shape, Y_test.shape)


# In[ ]:


#source Kaggle
from sklearn.preprocessing import StandardScaler
standard_vec = StandardScaler(with_mean = False)
# this will rise an error Expected 2D array, got 1D array instead: 
# array=[105.22 215.96  96.01 ... 368.98  80.53 709.67].
# Reshape your data either using 
# array.reshape(-1, 1) if your data has a single feature 
# array.reshape(1, -1)  if it contains a single sample.
standard_vec.fit(X_train['availability_365'].values.reshape(-1,1))

X_train_av_std = standard_vec.transform(X_train['availability_365'].values.reshape(-1,1))
X_test_av_std = standard_vec.transform(X_test['availability_365'].values.reshape(-1,1))

print("After vectorizations")
print(X_train_av_std.shape, Y_train.shape)
print(X_test_av_std.shape, Y_test.shape)


# In[ ]:


#source Kaggle
from sklearn.preprocessing import StandardScaler
standard_vec = StandardScaler(with_mean = False)

standard_vec.fit(X_train['calculated_host_listings_count'].values.reshape(-1,1))

X_train_chl_std = standard_vec.transform(X_train['calculated_host_listings_count'].values.reshape(-1,1))
X_test_chl_std = standard_vec.transform(X_test['calculated_host_listings_count'].values.reshape(-1,1))

print("After vectorizations")
print(X_train_chl_std.shape, Y_train.shape)
print(X_test_chl_std.shape, Y_test.shape)


# In[ ]:


#source Kaggle
from sklearn.preprocessing import StandardScaler
standard_vec = StandardScaler(with_mean = False)
standard_vec.fit(X_train['minimum_nights'].values.reshape(-1,1))

X_train_mn_std = standard_vec.transform(X_train['minimum_nights'].values.reshape(-1,1))

X_test_mn_std = standard_vec.transform(X_test['minimum_nights'].values.reshape(-1,1))

print("After vectorizations")
print(X_train_mn_std.shape, Y_train.shape)
print(X_test_mn_std.shape, Y_test.shape)


# In[ ]:


#source Kaggle
from sklearn.preprocessing import StandardScaler
standard_vec = StandardScaler(with_mean = False)

standard_vec.fit(X_train['latitude'].values.reshape(-1,1))

X_train_l_std = standard_vec.transform(X_train['latitude'].values.reshape(-1,1))
X_test_l_std = standard_vec.transform(X_test['latitude'].values.reshape(-1,1))

print("After vectorizations")
print(X_train_l_std.shape, Y_train.shape)
print(X_test_l_std.shape, Y_test.shape)


# In[ ]:


#source Kaggle
from sklearn.preprocessing import StandardScaler
standard_vec = StandardScaler(with_mean = False)
standard_vec.fit(X_train['longitude'].values.reshape(-1,1))

X_train_lo_std = standard_vec.transform(X_train['longitude'].values.reshape(-1,1))
X_test_lo_std = standard_vec.transform(X_test['longitude'].values.reshape(-1,1))

print("After vectorizations")
print(X_train_lo_std.shape, Y_train.shape)
print(X_test_lo_std.shape, Y_test.shape)


# ## Concatinating features

# In[ ]:


# merge two sparse matrices: https://stackoverflow.com/a/19710648/4084039
from scipy.sparse import hstack
X_tr = hstack((lb_train_ng,lb_train_rt,X_train_av_std,X_train_chl_std,X_train_mn_std,X_train_l_std,X_train_lo_std)).tocsr()
X_te = hstack((lb_test_ng,lb_test_rt,X_test_av_std,X_test_chl_std,X_test_mn_std,X_test_l_std,X_test_lo_std)).tocsr()

print("Final Data matrix")
print(X_tr.shape, Y_train.shape)
print(X_te.shape, Y_test.shape)
print("="*100)


# In[ ]:





# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

lin_model = LinearRegression().fit(X_tr, Y_train)
y_train_predict = lin_model.predict(X_tr)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
r2 = r2_score(Y_train, y_train_predict)


# In[ ]:



print("The model performance for training set")
print("--------------------------------------")
print('R2 score is {}'.format(r2*100))
print("\n")

# model evaluation for testing set
y_test_predict = lin_model.predict(X_te)
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
r2 = r2_score(Y_test, y_test_predict)

print("The model performance for testing set")
print("--------------------------------------")
print('R2 score is {}'.format(r2*100))


# In[ ]:


error_frame = pd.DataFrame({'Actual': np.array(Y_test).flatten(), 'Predicted': y_test_predict.flatten()})
error_frame.head(10)


# In[ ]:



print("Actul Vs Predicted")
plt.scatter(Y_test, y_test_predict)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")
plt.show()


# # Summary

# 
# 1.   Price of Most of the Listings are 10-200
# 2.   Price of Shared room is lower than private room and entire house/ apartment
# 3.   price of Rooms in Manhattan and Brooklyn boroughs are more than the rooms in boroughs
# 4.   75% of people spend 5 nights
# 5.   There is a 50% chance that the rooms will be available for atleast 45 days
# 6.Manhattan has Highest number of listings
# 7.Brooklyn has more Number of listings after Manhattan
# 8.Most of the Listings are Entire Home/apartments and private rooms
# 9.95% of them have reviews per month less than 10
# 10.Most(75%) of the host listing count are less than 50
# 11.Bronx has less number of night stays as compared to other

# In[ ]:




