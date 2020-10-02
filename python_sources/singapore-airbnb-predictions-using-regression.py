#!/usr/bin/env python
# coding: utf-8

# ***Importing the required libraries***

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# for visualisation
import matplotlib.pyplot as plt
import seaborn as sns
#for data preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
#for selecting the best feature effecting the target variable
from sklearn.feature_selection import SelectKBest,chi2
# for splitting the data into train and test dataset
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
# regression metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ***Load the dataset***

# In[ ]:


dataset=pd.read_csv('../input/singapore-airbnb/listings.csv')


# ***Taking a look at the dataset ***

# In[ ]:


dataset.head()


# ***Checking the data types of all the features***

# In[ ]:


dataset.info()


# ***Dropping the unnecessary columns***

# In[ ]:


dataset.drop(['id','name','host_name','host_id','last_review'],axis=1,inplace=True)


# In[ ]:


dataset.describe()


# In[ ]:


correlation_matrix=dataset.corr()
sns.heatmap(correlation_matrix)


# ***Here we can see that 'number_of_reviews' and 'reviews_per_month' are correlated with each other and hence one of them must be dropped in order to get better results from our model.***

# In[ ]:


dataset.drop(['reviews_per_month'],axis=1,inplace=True)


# In[ ]:


dataset.shape


# ***Checking for missing values***

# In[ ]:


dataset.isnull().sum()


# ***Checking for  univariate outliers using boxplot***

# In[ ]:


names=['latitude','longitude','price','minimum_nights','number_of_reviews','calculated_host_listings_count','availability_365']
plt.figure(figsize=(10,9))
for i in range(1,8):
    
    plt.subplot(2,4,i)
    fig=dataset.boxplot(column=names[i-1])


# ***Checking the distribution of each numerical feature to decide the method of detecting outliers for each of them ***

# In[ ]:


plt.figure(figsize=(12,10))
for j in range(1,8):
    plt.subplot(2,4,j)
    sns.distplot(dataset[names[j-1]])


# ***For all those who following a normal distribution they can use 'Z-Score' method for detecting outliers***

# In[ ]:


#for latitude
std=np.std(dataset['latitude'])
mean=np.mean(dataset['latitude'])
median=np.median(dataset['latitude'])
outliers=[]
for x in dataset['latitude']:
    zscore=(x-mean)/std
    if zscore>abs(3):
        outliers.append(x)


# In[ ]:


len(outliers)


# ***Removing the outliers by imputing them with median value***

# In[ ]:


dataset_new=dataset.replace(outliers,median)


# In[ ]:


plt.figure(figsize=(7,5))
plt.subplot(1,2,1)
fig=sns.distplot(dataset['latitude'])
plt.title('before removing outliers')
plt.subplot(1,2,2)
fig2=sns.distplot(dataset_new['latitude'])
plt.title('after removing outliers')


# In[ ]:


#for longitude
std=np.std(dataset['longitude'])
mean=np.mean(dataset['longitude'])
median=np.median(dataset['longitude'])
outliers=[]
for x in dataset['longitude']:
    zscore=(x-mean)/std
    if -3<zscore>3:
        outliers.append(x)


# In[ ]:


len(outliers)


# In[ ]:


#for minimum_nights
q1=dataset['minimum_nights'].quantile(0.25)
q3=dataset['minimum_nights'].quantile(0.75)
outlier=[]
iqr=q3-q1
lower_bound=q1-(1.5*iqr)
upper_bound=q3+(1.5*iqr)
for i in dataset['minimum_nights']:
    if i<lower_bound or i>upper_bound:
            outlier.append(i)        


# In[ ]:


len(outlier)


# In[ ]:


plt.figure(figsize=(20,8))
sns.countplot(outlier)


# * Here we can see that after '365' there is a sudden increase in number of minimum nights which is not desirable as the hosts at air-bnb provide a maximum of one year stay in the form of rent to the visitors i.e of 365 days.
# * So all the values above 365 are considered as outliers. For eg : 1000 number of minimum nights is next to impossible.
# * Such values are supposed to get filtered out.

# In[ ]:


dataset_new=dataset[dataset['minimum_nights']<=365]


# In[ ]:


plt.figure(figsize=(7,5))
plt.subplot(1,2,1)
sns.boxplot(dataset['minimum_nights'])
plt.title('before removing outliers')
plt.subplot(1,2,2)
sns.boxplot(dataset_new['minimum_nights'])
plt.title('after removing outliers')


# ***Now the scale of values has been changed and after filtering out the outlier values we get a maximum of 365 number of nights***

# In[ ]:


#for calculated_host_listings_count
q1=dataset['calculated_host_listings_count'].quantile(0.25)
q3=dataset['calculated_host_listings_count'].quantile(0.75)
outlier=[]
iqr=q3-q1
lower_bound=q1-(1.5*iqr)
upper_bound=q3+(1.5*iqr)
for i in dataset['calculated_host_listings_count']:
    if i<lower_bound or i>upper_bound:
            outlier.append(i)        


# In[ ]:


len(outlier)


# In[ ]:


sns.countplot(outlier)


# ***These values of detected as outliers according to upper and lower bound rule are not outliers as values like '274','203','157' and '141' can be a count of host listings on air-bnb. So these values are left untouched.***

# ***Now looking for the categorical features in the dataset ***

# In[ ]:


dataset_new['room_type'].unique()


# ***As there are only three levels of categories in 'room_type' feature we can map them with certain values***

# In[ ]:


mappings={'Entire home/apt':1,'Private room':2,'Shared room':3}
dataset_new['room_type']=dataset_new['room_type'].map(mappings)


# In[ ]:


dataset_new.head()


# In[ ]:


dataset_new['neighbourhood'].unique()


# In[ ]:


len(dataset_new['neighbourhood'].unique())


# ***As there are total 43 levels in 'neighbourhood' feature we can go for Binary Encoder so as to prevent dimensionality reduction by using one hot encoding***

# In[ ]:


import category_encoders as ce
binary=ce.BinaryEncoder(cols=['neighbourhood'])
dataset_new=binary.fit_transform(dataset_new)


# In[ ]:


dataset_new.head()


# ***Now splitting the dataset into dependent and independent features ***

# In[ ]:


x=dataset_new.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,13,14,15]].values
y=dataset_new.iloc[:,12:13].values


# In[ ]:


dataset['neighbourhood_group'].unique()


# ***As there are 5 levels and the values are nominal type we can perform label encoding***

# In[ ]:


# In the ndarray of independent features 'neighbourhood_group' is at 0th position
label=LabelEncoder()
x[:,0]=label.fit_transform(x[:,0])


# ***After converting all the features into numeric form we can check which indenpendent features are best effecting the target feature i.e. 'price'***

# In[ ]:


best_features=SelectKBest(score_func=chi2,k=5)
fit=best_features.fit(x,y)


# In[ ]:


dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(['neighbourhood_group', 'neighbourhood_0', 'neighbourhood_1',
       'neighbourhood_2', 'neighbourhood_3', 'neighbourhood_4',
       'neighbourhood_5', 'neighbourhood_6', 'latitude', 'longitude',
       'room_type','minimum_nights', 'number_of_reviews',
       'calculated_host_listings_count', 'availability_365'])
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']
result=featureScores.nlargest(5,'Score')
print(result)


# In[ ]:


plt.figure(figsize=(12,5))
sns.barplot(x=result['Specs'],y=result['Score'])


# ***Here we can see that 'calculated_host_listings_count' and 'minimum_nights' are the features that mostly affect our target variable (feature) i.e. 'price'***

# In[ ]:


sns.pairplot(dataset)


# ***As the features are not having linear relationship with each other we cant use linear regression model.Instead we can go for SVR.***

# In[ ]:


# scaling the features is necessary to implement svr 
sc_x=StandardScaler()
sc_y=StandardScaler()
x=sc_x.fit_transform(x)
y=sc_y.fit_transform(y)


# ***Splitting into train and test data***

# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# ***Fitting the train data to the model and making out the predictions ***

# In[ ]:


regressor=SVR(kernel='rbf')
regressor.fit(x_train,y_train)
predictions=regressor.predict(x_test)


# In[ ]:


predict=pd.DataFrame(predictions)
ytest=pd.DataFrame(y_test)
resultant=pd.concat([predict,ytest],axis=1)


# In[ ]:


resultant.columns=['Predicted_values','Actual_values']


# In[ ]:


resultant.head()


# In[ ]:


mae=mean_absolute_error(y_test,predictions)
rmse=sqrt(mean_squared_error(y_test,predictions))


# In[ ]:


mae


# * Usually mae ranges from 0 to infinity and lower the value better are the predictions.
# * So a value of 0.33 indicates that the model can give good predictions 

# In[ ]:


rmse


# ***RMSE is always much higher than the mean absolute error(mae) as they are the squared values of error.***
