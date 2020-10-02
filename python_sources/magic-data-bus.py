#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

#Import Library 


# In[ ]:


df =  pd.read_csv('../input/bus-breakdown-and-delays.csv')
 #Import dataset 


# In[ ]:


df.head() #target variable isn't a number, will need to do regex to extract relevant information 


# In[ ]:


df.info() #See quick summary of data- 328k total rows, but some columns have missingg values 


# In[ ]:


df.isna().sum()/df['School_Year'].count()


# In[ ]:


sns.heatmap(df.isnull()) #See distribution of missing data 
plt.figsize = (5,2.5)
plt.tight_layout()
plt.title('Distribution of Missing Data by Variable ')

#Incident number has high number of NAs 


# In[ ]:


df = df.drop(['Incident_Number'], axis = 1) #Drop incident number, most of column is missing
df.info()


# In[ ]:


df_clean = df.dropna() #Drop remainaing NAs
df_clean.info()


# In[ ]:


df_clean.head(100)


# In[ ]:


df_clean['Delay'] = df_clean['How_Long_Delayed'].str.extract('(\d+)') #Extract digits from string column 
df_clean.head() #Check if regex worked- Yes!


# In[ ]:


df_clean[df_clean['Delay'].isnull()] #Check if data is null
#We see that there's question marks or other irregularaties- lets drop this data 


# In[ ]:


df_clean = df_clean.dropna() #Drop new NAs 
df_clean.isnull().sum() #Check that no NAs are left 


# In[ ]:


df_clean['Delay'] =  pd.to_numeric(df_clean['Delay']) #Convert string to integer 


# In[ ]:


df_clean = df_clean.drop(['How_Long_Delayed'], axis = 1) #Drop original column 


# Before we do more data wrangling, let's do some EDA! 

# In[ ]:


reasons = pd.pivot_table(df_clean, index = 'Reason', values = 'Delay', aggfunc = [np.mean, np.max,np.size]).sort_values(by = 
                                                                                                    ('mean', 'Delay'), 
                                                                                        ascending = False)
plt.figure(figsize = (5,50))
reasons.plot(kind = 'bar', y = ('mean','Delay'), color = 'lightblue')
plt.title('Average Delay in Minutes')
plt.legend().remove()
plt.xticks(rotation = 80)

#See size distribution by reason
plt.figure(figsize = (5,50))
reasons.plot(kind = 'bar', y = ('size','Delay'), color = 'lightblue')
plt.title('Number of Delays')
plt.legend().remove()
plt.xticks(rotation = 80)


# In[ ]:


sns.boxplot( x = df_clean['Delay'])
#Looks like two clear outliers- 1 around 50,000 and the other around 200,000 Let's remove 


# In[ ]:


df_exoutliers = df_clean[df_clean['Delay'] < 50000]
sns.boxplot(x = df_exoutliers['Delay']) #Check if we need to remove further outliers 


# In[ ]:


df_clean.head()


# In[ ]:


df_clean['Route_Number'].value_counts()


# In[ ]:


pd.pivot_table(df_clean, index = 'Route_Number', values = 'Delay', aggfunc = [np.mean,np.size]).sort_values(by = 
                                                                                                           ('size','Delay'), 
                                                                                                           ascending = False).head(6)


# In[ ]:


routes = ['1','2','3','5','4','6']
top_routes = df_clean[df_clean['Route_Number'].isin(routes)] #Filter to see cases where route is top 6 in # of delays


# In[ ]:


routes_pivot = pd.pivot_table(top_routes, index = 'Route_Number', values = 'Delay', aggfunc = [np.mean,np.size])
routes_pivot.head(6)


# In[ ]:


df_clean.head()


# In[ ]:


df_clean['Bus_Company_Name'].value_counts()


# **Machine Learning Model Prep **

# In[ ]:


#First Let's remove unnecessary features, checking 1 by 1 

#School Year- is it relevant? 
df_clean['School_Year'].value_counts().plot(kind = 'bar')
plt.xticks(rotation = 75) #Make Data cleaner to read 

#See an increasing trend year on year in quantity-let's investigate if there's any significant deviations in delay by year


# In[ ]:


#Let's first see average delay, across the dataset 
df_clean['Delay'].mean() #Around 29 mins is the average delay time 


# In[ ]:


pd.pivot_table(df_clean, index = 'School_Year', values = 'Delay', aggfunc = np.mean).plot(kind = 'bar')
plt.legend().remove() #Get rid of legend 
plt.title('Average Delay by Year')
plt.xticks(rotation = 75) #Make easier to read 

#Doesn't look any year is terribly far off from another but also not congruent- will keep for now 


# In[ ]:


df_clean.head() #Let's check what the data looked like again 


# In[ ]:


df_clean['Busbreakdown_ID'].value_counts() #Data seems like no noise, we'll drop 


# In[ ]:


df_clean = df_clean.drop(['Busbreakdown_ID'], axis = 1)
df_clean.head()


# In[ ]:


bus_num = pd.pivot_table(df_clean, index = 'Bus_No', values = 'Delay',aggfunc = np.size).sort_values(by = 'Delay', 
                                                                                                    ascending = False)
bus_num

#Create pivot to see number of delays by bus number- we see that a lot have only have 1. 
#Instead of one hot encoding, let's just convert to digits 


# In[ ]:


df_clean['Bus_Number'] = df_clean['Bus_No'].str.extract('(\d+)') #Extract digits from string column 

df_clean['Bus_Number'] =  pd.to_numeric(df_clean['Bus_Number']) #Convert string to integer 
df_clean.isnull().sum() #We now have some more NAs- let's do a quick investigation 


# In[ ]:


df_clean[df_clean['Bus_Number'].isnull()] #Looks like noisy data, will drop 
df_clean = df_clean.dropna()
df_clean = df_clean.drop(['Bus_No'], axis = 1) #Drop original column 


# In[ ]:


df_clean.head() #Look familiar? 


# In[ ]:


df_clean.corr() #Let's look at the current correlation across features


# In[ ]:


df_clean['Run_Type'].value_counts().plot(kind = 'bar') 
plt.title('Trip distribution ')
plt.xticks(rotation = 75) #Data heavily weighted towards Special Ed AM in terms of quantity 


# In[ ]:


df_clean.head()


# In[ ]:


df_clean.nunique() #See number of unique values per feature 


# In[ ]:


from sklearn.model_selection import train_test_split

y = df_clean['Delay'] #store target variable
df_model = df_clean.drop(['Delay'], axis = 1)
X = df_model[['Run_Type','Reason','Boro','Number_Of_Students_On_The_Bus','Breakdown_or_Running_Late',
             'School_Age_or_PreK']] #store some basic features


# In[ ]:


dummy_df = pd.get_dummies(X) #Convert data to dummies to enable modeling


# In[ ]:


print(dummy_df.shape) 
print(y.shape)

#Check that number of rows match number of labels (target) 


# In[ ]:


#Split Data into test, train 
from sklearn.model_selection import train_test_split

X_train, X_test,y_train, y_test = train_test_split(dummy_df,y,test_size = .2, random_state = 40) #Split into 20% test data


# In[ ]:


from sklearn.ensemble import RandomForestRegressor 
model = RandomForestRegressor(n_estimators = 100) #create model 
model.fit(X_train,y_train) #Run model on training set


# In[ ]:


predictions = model.predict(X_test) #Predict on testing set


# In[ ]:


from sklearn import metrics 

print('MAE:', metrics.mean_absolute_error(y_test,predictions)) #We see that average error is 12.92 mins- compared to the naive guess of 28 mins


# Let's change the model up by adding more features and check out the impact 

# In[ ]:


df_clean.head()


# In[ ]:


from sklearn.model_selection import train_test_split

y = df_clean['Delay'] #store target variable
X = df_model[['School_Year','Run_Type','Reason','Boro','Bus_Company_Name','Number_Of_Students_On_The_Bus','Breakdown_or_Running_Late',
             'School_Age_or_PreK']] #Added bus company name/school year features
dummy_df = pd.get_dummies(X) #Convert data to dummies to enable modeling
print(dummy_df.shape)
print(y.shape)

#Shape of both datasets is matching, ok to proceed to next step 


# In[ ]:


#Split Data into test, train 
from sklearn.model_selection import train_test_split

X_train, X_test,y_train, y_test = train_test_split(dummy_df,y,test_size = .2, random_state = 40) #Split into 20% test data


# In[ ]:


#Run model again 

from sklearn.ensemble import RandomForestRegressor 
model = RandomForestRegressor(n_estimators = 150) #create model 
model.fit(X_train,y_train) #Run model on training set


# In[ ]:


from sklearn import metrics 

predictions = model.predict(X_test) #Predict on testing set
print('MAE:', metrics.mean_absolute_error(y_test,predictions)) #We see that average error is 11.13 mins- down from the last model!


# Let's take a look at the importance of the features. 

# In[ ]:


feature_values = pd.DataFrame(model.feature_importances_,
                              index = X_train.columns,
                              columns = ['importance']).sort_values('importance',
                                                                    ascending=False)
feature_values.head(8).plot(kind = 'bar', color = 'lightgreen')
plt.xticks(rotation = 85)
plt.title('Feature Importance')


# Let's work with the Schools Serviced Feature 

# In[ ]:


df_clean['Size'] = df_clean.groupby('Schools_Serviced')['Delay'].transform(len) #Create a column to see count of Bus
df_clean.head()


# In[ ]:


df_clean['Size'].nunique() #365 unique values for Schools Serviced


# In[ ]:


df_clean[df_clean['Size'] > 500]['Size'].nunique() #Cut down number of unique schools serviced to 58 by filtering value count


# In[ ]:


df_clean['Schools_Serviced2'] = np.where(df_clean['Size'] > 500,df_clean['Schools_Serviced'], 'other')


# In[ ]:



y = df_clean['Delay'] #store target variable
X = df_clean[['School_Year','Run_Type','Reason','Boro','Bus_Company_Name','Number_Of_Students_On_The_Bus','Breakdown_or_Running_Late',
             'School_Age_or_PreK','Schools_Serviced2']] #added additional feature
dummy_df = pd.get_dummies(X) #look familiar? 
X_train, X_test,y_train, y_test = train_test_split(dummy_df,y,test_size = .2, random_state = 40) #Split into 20% test data


# Re-run Model 

# In[ ]:


model = RandomForestRegressor(n_estimators = 150) #create model 
model.fit(X_train,y_train) #Run model on training set


# In[ ]:


predictions = model.predict(X_test) #Predict on testing set
print('MAE:', metrics.mean_absolute_error(y_test,predictions)) #Down to 10.99- can we get to single digits? 


# Let's dive into the contractor notification features 

# In[ ]:


pd.pivot_table(df_clean, index = 'Has_Contractor_Notified_Schools', values = 'Delay', aggfunc = np.mean)


# In[ ]:


pd.pivot_table(df_clean, index = 'Has_Contractor_Notified_Parents', values = 'Delay', aggfunc = np.mean)


# In[ ]:


y = df_clean['Delay'] #store target variable
X = df_clean[['School_Year','Run_Type','Reason','Boro','Bus_Company_Name','Number_Of_Students_On_The_Bus','Breakdown_or_Running_Late',
             'School_Age_or_PreK','Schools_Serviced2', 'Has_Contractor_Notified_Parents','Has_Contractor_Notified_Schools']] 
    #added additional features related to contractors 
dummy_df = pd.get_dummies(X) #look familiar? 
X_train, X_test,y_train, y_test = train_test_split(dummy_df,y,test_size = .2, random_state = 40) #Split into 20% test data


# In[ ]:


model = RandomForestRegressor(n_estimators = 150) #create model 
model.fit(X_train,y_train) #Run model on training set


# In[ ]:


predictions = model.predict(X_test) #Predict on testing set
print('MAE:', metrics.mean_absolute_error(y_test,predictions)) 

