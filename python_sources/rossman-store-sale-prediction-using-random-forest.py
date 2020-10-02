#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Loading packages 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # data visualization
import datetime as dt # for date and time manipulation

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot # data visualization
init_notebook_mode(connected=True)

import cufflinks as cf #  to call plots directly off of a pandas dataframe
cf.go_offline()

# Input data files are available in the "../input/" directory.
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # 1. Exploratory Data Analysis and Visualization

# In[ ]:


# Loading dataset train, test and store
train=pd.read_csv('/kaggle/input/rossmann-store-sales/train.csv')
test=pd.read_csv('/kaggle/input/rossmann-store-sales/test.csv')
store=pd.read_csv('/kaggle/input/rossmann-store-sales/store.csv')


# In[ ]:


# Looking into each dataset size
print('training dataset shape',train.shape)
print('testing dataset shape',test.shape)
print('store dataset shape',store.shape)


# In[ ]:


# train dataset
train.head(3)


# In[ ]:


# test dataset
test.head(3)


# In[ ]:


# store dataset
store.head(3)


# In[ ]:


# Samples having 0 sales 
print('Sample having 0 sales :',train[train['Sales']<=0].shape,'\n')
print('Store Open : 1,','Store Closed : 0')
print(train['Open'].value_counts(),'\n')
print('Sample having closed store and sale is 0 :',train[(train['Sales']<=0)&(train['Open']==0)].shape)


# * The train dataset have total 172817 samples having 0 sale and the store is closed. Model is building to predict the sale, therefore getting rid of irrelevant samples 

# In[ ]:


# making new train dataset having sales greater than zero
train=train[train['Sales']>0]
print('New training dataset shape',train.shape)


# In[ ]:


# basic information about columns of train dataset
train.info()


# * Date column having date information but is of object type, so have to convert into datetime format 
# * Dataset does not cantain any missing values

# In[ ]:


# Converting into datetime format 
train['Date']=pd.to_datetime(train['Date'])
train.sort_values(by='Date',ascending=True,inplace=True)
train.reset_index(inplace=True)
train.drop('index',axis=1,inplace=True)

# Start date record in dataset
start_date=train['Date'][0]
print('Start date :',start_date)

# End date record in dataset
end_date=train['Date'][844337]
print('End date :',end_date)


# In[ ]:


# unique values of categorical columns 
for i in ['DayOfWeek','Open','Promo','StateHoliday','SchoolHoliday']:
    print( i+':',train[i].unique())


# * Open column having only 1 value so no need of this column
# * StateHoliday column having two identical values but in different format so need to deal with this 

# In[ ]:


# Droping 'Open' column and making correction in 'StateHoliday' column
index=train[train['StateHoliday']==0].index
train['StateHoliday'][index]='0'
train['StateHoliday'].value_counts()
train.drop('Open',axis=1,inplace=True)

# Note : I have converted numeric into string format in StateHoliday column because majority of variables are in string format


# In[ ]:


# Checking for negative values
pd.options.display.float_format = '{:.0f}'.format # in order to round up the numbers 
train.describe()


# * We don't have negative values in dataset, therefore dataset is very clean

# ### 1.1 Visualization based on Days of Week

# In[ ]:


# Scatter plot "Sales Vs Customers"
plt.figure(figsize=(15,8))
sns.scatterplot(x='Sales',y='Customers',data=train,hue='DayOfWeek',palette='coolwarm')
plt.title('Sales vs Customers',fontdict={'fontsize':20})
plt.show()

# Count of Weekdays 
plt.figure(figsize=(15,4))
print(train['DayOfWeek'].value_counts())
sns.countplot('DayOfWeek',data=train,palette='coolwarm')
plt.title('Count of weekdays',fontdict={'fontsize':20})
plt.show()

# Weekdays wise scatter plot between Sales and Customers
g=sns.FacetGrid(row='DayOfWeek',data=train,height=3,aspect=4)
g.map(plt.scatter,'Sales','Customers',color='green',alpha=0.4)
plt.show()

# Boxplot - "Sales"
plt.figure(figsize=(15,8))
sns.boxplot(y='Sales',x='DayOfWeek',data=train,palette='Set3')
plt.title('Sales Statistics',fontdict={'fontsize':20})
plt.show()


# Boxplot - "Customers"
plt.figure(figsize=(15,8))
sns.boxplot(y='Customers',x='DayOfWeek',data=train,palette='Set3')
plt.title('Customers Statistics',fontdict={'fontsize':20})
plt.show()


# * The 7th day of week has very less variability as compare to other days of week
# * The count of 7th days is very less as compare to other days but the average sales and average number of customers are pretty much high. One possible reason could be on sunday customers comes for a specific commodity as an essential need for survival

# ### 1.2 Visualization based on running Promo

# In[ ]:


# Scatter plot "Sales Vs Customers"
plt.figure(figsize=(15,8))
sns.scatterplot(x='Sales',y='Customers',data=train,hue='Promo',palette='plasma')
plt.title('Sales vs Customers',fontdict={'fontsize':20})
plt.show()

# Promo wise scatter plot between Sales and Customers
g=sns.FacetGrid(row='Promo',data=train,height=3,aspect=4)
g.map(plt.scatter,'Sales','Customers',color='blue',alpha=0.4)
plt.show()

# Boxplot - "Sales"
plt.figure(figsize=(10,8))
sns.boxplot(y='Sales',x='Promo',data=train,palette='Set3')
plt.title('Sales Statistics',fontdict={'fontsize':20})
plt.show()

# Boxplot - "Customers"
plt.figure(figsize=(10,8))
sns.boxplot(y='Customers',x='Promo',data=train,palette='Set3')
plt.title('Customers Statistics',fontdict={'fontsize':20})
plt.show()


# * There is much difference in sale before and after running Promo. It indicates that promo have done a great job in increasing the sale
# * Not much noticable difference is seen in number of customers visit to store. Promo idea was not capable to attract new customers but the buying quantity of existing old customers have increased, therefore overall the running of promo worked  

# ### 1.3 Visualization based on State Holidays 

# In[ ]:


# Scatter plot "Sales Vs Customers"
plt.figure(figsize=(15,8))
sns.scatterplot(x='Sales',y='Customers',data=train,hue='StateHoliday',palette='autumn',hue_order=['a','b','c','0'])
plt.title('Sales vs Customers',fontdict={'fontsize':20})
plt.show()

# State Holiday wise scatter plot between Sales and Customers
g=sns.FacetGrid(row='StateHoliday',data=train,height=3,aspect=4)
g.map(plt.scatter,'Sales','Customers',color='purple',alpha=0.4)
plt.show()

# Boxplot - "Sales"
plt.figure(figsize=(10,8))
sns.boxplot(y='Sales',x='StateHoliday',data=train,order=['a','b','c','0'],palette='Set3')
plt.title('Sales Statistics',fontdict={'fontsize':20})
plt.show()

# Boxplot - "Customers"
plt.figure(figsize=(10,8))
sns.boxplot(y='Customers',x='StateHoliday',data=train,order=['a','b','c','0'],palette='Set3')
plt.title('Customers Statistics',fontdict={'fontsize':20})
plt.show()


# * People are more often to buy more on Christmas and Easter festival, therefore sales and count of customers visit are more on this seasons
# * But it is clearly seen the variation in public choice is less in these festival as people tend to buy particular range and type of product,where as the opposite behavior is been observed when there is no holiday

# ### 1.4 Visualization based on School Holidays 

# In[ ]:


# Scatter plot "Sales Vs Customers"
plt.figure(figsize=(15,8))
sns.scatterplot(x='Sales',y='Customers',data=train,hue='SchoolHoliday',palette='viridis')
plt.title('Sales vs Customers',fontdict={'fontsize':20})
plt.show()

# State Holiday wise scatter plot between Sales and Customers
g=sns.FacetGrid(row='SchoolHoliday',data=train,height=3,aspect=4)
g.map(plt.scatter,'Sales','Customers',color='orange',alpha=0.4)
plt.show()

# Boxplot - "Sales"
plt.figure(figsize=(10,8))
sns.boxplot(y='Sales',x='SchoolHoliday',data=train,palette='Set3')
plt.title('Sales Statistics',fontdict={'fontsize':20})
plt.show()

# Boxplot - "Customers"
plt.figure(figsize=(10,8))
sns.boxplot(y='Customers',x='SchoolHoliday',data=train,palette='Set3')
plt.title('Customers Statistics',fontdict={'fontsize':20})
plt.show()


# * Not much difference is observed by the closure of public schools

# # 2. Feature Engineering

# In[ ]:


# Making new column of sales per customers 
train['SalesPerCustomer']=train['Sales']/train['Customers']
train['SalesPerCustomer']


# ## 2.1 Table_1 : Average sales, average number of customers and average sales per customers based on weekdays and whether the promo is running on that day. 

# In[ ]:


Table_1=pd.pivot_table(data=train,index=['DayOfWeek','Promo'],values=['Sales','Customers','SalesPerCustomer'],aggfunc='mean').round(0)
Table_1.rename(columns=lambda x : 'Avg_' + x, inplace=True)

# Visualization
Table_1.iplot(kind='bar',y=['Avg_Sales','Avg_Customers'],title='Average Sales and Average Customers',xTitle='(DayOfWeek,Promo)')
Table_1.iplot(y='Avg_SalesPerCustomer',title='Average Sales per Customers',xTitle='(DayOfWeek,Promo)')

Table_1


# * No promo was runned on weekends.
# * The average sales and average number of customers on weekdays while running promo is high than not running promo.
# * The average sale on sunday is quite high than some of the weekdays and the average number of customers is the highest on sunday.
# * The average sale per customer is also high on days when promo was runned but the weekends is showing low sales rate. This means people are not buying even though they are capable of as it is evident in the weekdays.
# * Should capture the profit potential seen on weekends by running Promos on weekends 

# ## 2.2 Table_2 : Average sales, average number of customers and average sales per customer based on State Holidays and whether the promo is running on that day. 

# In[ ]:


Table_2=pd.pivot_table(data=train,index=['Promo','StateHoliday'],values=['Sales','Customers','SalesPerCustomer'],aggfunc='mean').round(0)
Table_2.rename(columns=lambda x : 'Avg_' + x, inplace=True)

# Visualization
Table_2.iplot(kind='bar',y=['Avg_Sales','Avg_Customers'],title='Average Sales and Average Customers',xTitle='(Promo,StateHoliday)')
Table_2.iplot(y='Avg_SalesPerCustomer',title='Average Sales per Customers',xTitle='(Promo,StateHoliday)')

Table_2


# * As early said Christman and Easter festival season shows high rate of sales 

# In[ ]:


# Checking the columns of store dataset
store.info()


# * There are lots of column having null values
# * Some columns are not having right format data, so need to take care of it

# In[ ]:


# Converting Promo2 column data from integer to category type
store['Promo2']=store['Promo2'].astype(object)
store.dtypes


# In[ ]:


# Checking the unique values of category columns
for i in store.columns[store.dtypes=='object']:
    print(i,':',store[i].unique(),'\n')


# * Dataset is pretty much clear

# ## 2.3 Creating Dataframe - "avg_store" and merging with store dataset for further analysis

# In[ ]:


pd.options.display.float_format='{:.3f}'.format # in order to show number upto 3 decimal place

# avg_store Dataframe containing columns : 'Average Sales','Average Customers','Average Sales Per Customer'
avg_store=train.groupby('Store')[['Sales','Customers','SalesPerCustomer']].mean()
avg_store.rename(columns=lambda x : 'Avg_' + x,inplace=True)
avg_store.reset_index(inplace=True)

# Adding column Max_Customers(containing maximum value of customers) to avg_store Dataframe 
Max_customer=train.groupby('Store')['Customers'].max()
avg_store=pd.merge(avg_store,Max_customer,how='inner',on='Store')
avg_store.rename(columns={'Customers':'Max_Customers'},inplace=True)

# Adding column Min_Customers(containing mimimum value of customers) to avg_store Dataframe 
Min_customer=train.groupby('Store')['Customers'].min()
avg_store=pd.merge(avg_store,Min_customer,how='inner',on='Store')
avg_store.rename(columns={'Customers':'Min_Customers'},inplace=True)

# Adding column Std_Customers(containing Standard Deviation value of customers) to avg_store Dataframe 
Std_customer=train.groupby('Store')['Customers'].std()
avg_store=pd.merge(avg_store,Std_customer,how='inner',on='Store')
avg_store.rename(columns={'Customers':'Std_Customers'},inplace=True)

# Adding column Med_Customers(containing Median value of customers) to avg_store Dataframe 
Med_customer=train.groupby('Store')['Customers'].median()
avg_store=pd.merge(avg_store,Med_customer,how='inner',on='Store')
avg_store.rename(columns={'Customers':'Med_Customers'},inplace=True)

avg_store.head()

# In order to capture all the variability of customer columns, these much columns are made 


# In[ ]:


# Merging avg_store with store
store=pd.merge(store,avg_store,how='inner',on='Store')
store.head()


# In[ ]:


# Removing missing values in CompetitionDistance column
index=store[store['CompetitionDistance'].isnull()].index
store.loc[index,'CompetitionDistance']=0
store['CompetitionDistance'].isnull().any() # for checking


# In[ ]:


# Scatter plot - Average sale against Competition Distance
plt.figure(figsize=(15,6))
sns.set_style('darkgrid')
sns.scatterplot(x='Avg_Sales',y='CompetitionDistance',data=store)
plt.title('Average sale against Competition Distance',fontdict={'fontsize':20})
plt.show()

# Visualization of Competition Distance data
plt.figure(figsize=(15,6))
sns.distplot(store['CompetitionDistance'])
plt.title('Competition Distance distribution',fontdict={'fontsize':20})
plt.xlim(0,80000)
plt.show()


# * It seens there are less competitor's store near to Rossman's store
# * Even though many competitor's are near to Rossman's store, the sale is far good enough to beat the compoetition. May be this is because good service quality, better behaviour towards customers and last but not least is the quality of product.

# In[ ]:


# unique value check 

print('CompetitionOpenSinceMonth :',store['CompetitionOpenSinceMonth'].unique(),'\n')

print('CompetitionOpenSinceYear :',store['CompetitionOpenSinceYear'].unique(),'\n')

print('Promo2SinceWeek :',store['Promo2SinceWeek'].unique(),'\n')

print('Promo2SinceYear :',store['Promo2SinceYear'].unique())


# In[ ]:


# Getting free from missing values

index=store[(store['CompetitionOpenSinceMonth'].isnull())&(store['CompetitionOpenSinceYear'].isnull())].index
store.loc[index,['CompetitionOpenSinceMonth','CompetitionOpenSinceYear']]=0

index=store[(store['Promo2SinceWeek'].isnull())&(store['Promo2SinceYear'].isnull())&(store['Promo2']==0)].index
store.loc[index,['Promo2SinceWeek','Promo2SinceYear']]=0

store[['CompetitionOpenSinceMonth','CompetitionOpenSinceYear','Promo2SinceWeek','Promo2SinceYear']].isnull().any() # To check


# In[ ]:


# Converting from float into integer type
store[['CompetitionOpenSinceMonth',
       'CompetitionOpenSinceYear',
       'Promo2SinceWeek',
       'Promo2SinceYear']]=store[['CompetitionOpenSinceMonth',
                                  'CompetitionOpenSinceYear',
                                  'Promo2SinceWeek',
                                  'Promo2SinceYear']].astype(int)

store[['CompetitionOpenSinceMonth','CompetitionOpenSinceYear','Promo2SinceWeek','Promo2SinceYear']].dtypes # To check


# In[ ]:


# Setting Promo Interval equal to zero for those who are not continuing Promo and for missing values
index=store[(store['Promo2']==0)&(store['PromoInterval'].isnull().any())].index
store.loc[index,'PromoInterval']=0

store['PromoInterval'].isnull().any() # To check


# In[ ]:


# Last check in columns of store
store.info()


# * The data is clean and free from missing values

# ## 2.4 Visualization based on Store type

# In[ ]:


# scatter plot - Average Customers against Average Sales
plt.figure(figsize=(15,8))
sns.lmplot(x='Avg_Customers',y='Avg_Sales',hue='StoreType',data=store,hue_order=['a','b','c','d'],height=6,aspect=2.5)
plt.title('Average Customers Vs Average Sales', fontdict={'fontsize':20})
plt.show()

# boxplot - Average sale per customers 
plt.figure(figsize=(15,5))
sns.boxplot(x='StoreType',y='Avg_SalesPerCustomer',data=store,order=['a','b','c','d'],palette='Set3')
plt.title('Average sale per customers Statistics', fontdict={'fontsize':20})
plt.show()


# * Store type d is having much higher sale but the variability is very less as people are buying one particular set of products 
# * Store type a and c are same sales but both are less than d type
# * The least one is b type store and its shows large variability. May be this type of store is located in rural or suburban area 
# * As expected the sales rate of d type is higher than a and c type and b type have very less sales rate 

# ## 2.5 Visualization based on Assortment

# In[ ]:


# scatter plot - Average Customers against Average Sales
plt.figure(figsize=(15,8))
sns.lmplot(x='Avg_Customers',y='Avg_Sales',hue='Assortment',data=store,hue_order=['a','b','c'],height=6,aspect=2.5)
plt.title('Average Customers Vs Average Sales', fontdict={'fontsize':20})
plt.show()

# boxplot - Average sale per customers
plt.figure(figsize=(15,5))
sns.boxplot(x='Assortment',y='Avg_SalesPerCustomer',data=store,order=['a','b','c'],palette='Set3')
plt.title('Average sale per customers Statistics', fontdict={'fontsize':20})
plt.show()


# * It is quite interesting to observe that the line of assortment type a and c are almost parallel, ie. both are having the same average number of customers but the sales are higher in type c
# * Also the sales rate is highest in type c

# ## 2.6 Visualization based on whether the Promo is continuing 

# In[ ]:


# scatter plot - Average Customers against Average Sales
sns.lmplot(x='Avg_Customers',y='Avg_Sales',hue='Promo2',data=store,height=6,aspect=2.5)
plt.title('Average Customers Vs Average Sales', fontdict={'fontsize':20})
plt.show()

# boxplot - Average sale per customers
plt.figure(figsize=(15,5))
sns.boxplot(x='Promo2',y='Avg_SalesPerCustomer',data=store,palette='Set3')
plt.title('Average sale per customers Statistics', fontdict={'fontsize':20})
plt.show()

# scatter plot - Maximum Average Customers against Minimum Average Customers
sns.lmplot(x='Max_Customers',y='Min_Customers',hue='Promo2',data=store,height=6,aspect=2.5)
plt.title('Maximum Average Customers Vs Minimum Average Customers', fontdict={'fontsize':20})
plt.show()


# * In the first figure : the line at the begining overlaps each other but as its moves further it diverge from each other with 0 assigned line showing more sales for the same number of customers 
# * The box plot shows the sales rate is high for those store who are continuing the promo but the difference is narrow 
# * In the last figure : there is still divergence in line and the mimimum number of customers is less for the 1 assigned line as compare to 0 assigned one 
# * this indicates the continuing of advertisements and promo is not a good idea as people are getting bored with the same promo.

# ## 2.7 Visualization based on Promo Interval

# In[ ]:


# scatter plot - Average Customers against Average Sales
plt.figure(figsize=(15,8))
sns.lmplot(x='Avg_Customers',y='Avg_Sales',hue='PromoInterval',data=store,height=6,aspect=2.5)
plt.title('Average Customers Vs Average Sales', fontdict={'fontsize':20})
plt.show()

# boxplot - Average sale per customers
plt.figure(figsize=(15,5))
sns.boxplot(x='PromoInterval',y='Avg_SalesPerCustomer',data=store,palette='Set3')
plt.title('Average sale per customers Statistics', fontdict={'fontsize':20})
plt.show()


# * The sales rate are almost same 

# ## 2.8 Table_3 : Average sales, average number of customers and average sales per customer based on StoreType, Assortment and Promo Interval.

# In[ ]:


Table_3=pd.pivot_table(data=store,index=['StoreType','Assortment','PromoInterval'],
               values=['Avg_Sales','Avg_Customers','Avg_SalesPerCustomer'],aggfunc='mean').round(0)

Table_3.iplot(kind='bar',y=['Avg_Sales','Avg_Customers'],title='Average Sales and Average Customers',
              xTitle='(StoreType,Assortment,Assortment)')

Table_3.iplot(y='Avg_SalesPerCustomer',title='Average Sales per customers',xTitle='(StoreType,Assortment,Assortment)')

Table_3


# * From the above table as well as figures it is clear that continuing promo is not helping the business, it does not matter on which month the promo was runned .
# * The b type store is most effected as sales rate is lowest

# ## 2.9 Merging train and store dataset to form new train dataset containing all the attributes 

# In[ ]:


# Merging
new_train=pd.merge(train,store,how='left',on='Store')
print('New training dataset shape :',new_train.shape)
new_train.head()


# In[ ]:


# Making new columns to show Date information separately 
new_train['Year']=new_train['Date'].dt.year
new_train['Month']=new_train['Date'].dt.month
new_train['Day']=new_train['Date'].dt.day
new_train['Week']=new_train['Date'].dt.week
new_train.head()


# * 4 new columns were added

# In[ ]:


# Making column "MonthCompetitionOpen" which contains date information in months since the competition was opened 
new_train['MonthCompetitionOpen']=12*(new_train['Year']-new_train['CompetitionOpenSinceYear'])+new_train['Month']-new_train['CompetitionOpenSinceMonth']

new_train.loc[(new_train['CompetitionOpenSinceYear']==0),'MonthCompetitionOpen']=0
# Negative values indcate that the competitor's store was opened after the Rossman's store opening date.


# In[ ]:


# Making column "WeekPromoOpen" which contains date information in weeks since the promo is running
new_train['WeekPromoOpen']=52.14298*(new_train['Year']-new_train['Promo2SinceYear'])+new_train['Week']-new_train['Promo2SinceWeek']

new_train.loc[(new_train['Promo2SinceYear']==0),'WeekPromoOpen']=0


# ## 2.10 Visualization based on Months 

# In[ ]:


# scatter plot - Sales against Customers
g=sns.FacetGrid(row='Month',data=new_train,height=3,aspect=4)
g.map(plt.scatter,'Sales','Customers',color='red',alpha=0.4)
plt.show()

# Boxplot - "Sales"
plt.figure(figsize=(10,8))
sns.boxplot(y='Sales',x='Month',data=new_train,palette='Set3')
plt.title('Sales Statistics',fontdict={'fontsize':20})
plt.show()

# Boxplot - "Customers"
plt.figure(figsize=(10,8))
sns.boxplot(y='Customers',x='Month',data=new_train,palette='Set3')
plt.title('Customers Statistics',fontdict={'fontsize':20})
plt.show()


# * Now it is obvious that the month of November and December the sale and number of customers would be high
# * Sales are also high in the summer season

# ## 2.11 Table_4 : Average sales, average number of customers and average sales per customer based on Months.

# In[ ]:


Table_4=pd.pivot_table(data=new_train,index=['Month','Promo'],
                      values=['Avg_Sales','Avg_Customers','Avg_SalesPerCustomer'],aggfunc='mean').round(0)

# Visualization
Table_4.iplot(kind='bar',y=['Avg_Sales','Avg_Customers'],title='Average Sale and Average Customers',xTitle='(Months,Promo)')
Table_4.iplot(y='Avg_SalesPerCustomer',title='Average sales per customer',xTitle='(Months,Promo)')

del(Table_4)


# In[ ]:


new_train.info()


# In[ ]:


# converting into integer type
new_train['Promo2']=new_train['Promo2'].astype(int)


# In[ ]:


"""Droping column customer because we are performing sales prediction and knowing the number of customers on particular store before actual
   sales happen is not possible"""
new_train.drop('Customers',axis=1,inplace=True)


# # 3. Building and Implementing Machine Learning Models

# ## 3.1 Preparing dataset and Normalizing certain column data

# In[ ]:


# Making a new data set for model building
trainS=new_train[['Store', 'DayOfWeek','Sales','Promo',
       'StateHoliday', 'SchoolHoliday','StoreType',
       'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth',
       'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek',
       'Promo2SinceYear', 'PromoInterval','Avg_Customers',
       'Max_Customers', 'Min_Customers',
       'Std_Customers', 'Med_Customers', 'Year', 'Month', 'Day', 'Week',
       'MonthCompetitionOpen', 'WeekPromoOpen']]
trainS.shape


# In[ ]:


# Visualization to chech whether the data is distributed normally
trainS.hist(figsize=(25,25))
plt.show()


# * Some of the columns data are positively skewed, therefore have to deal with it.
# * This is not a necessary step but normally distributed data yields better result

# In[ ]:


# Taking log transformation 
trainS['Log_Sales']=np.log(trainS['Sales'])

index=trainS[trainS['CompetitionDistance']==0].index
trainS['CompetitionDistance'][index]=1
trainS['Log_CompetitionDistance']=np.log(trainS['CompetitionDistance'])

trainS[['Log_Sales','Log_CompetitionDistance',]].hist(figsize=(8,5))
plt.show()

trainS.drop(['Sales','CompetitionDistance'],axis=1,inplace=True)

# Note : This is not a necessary step but doing this will redistribute the data in normal curve which is better for good prediction 


# * Almost normally distributed 

# In[ ]:


# Getting dummies columns for categorical columns 
final_train=pd.get_dummies(data=trainS,columns=['StoreType','StateHoliday','Assortment','PromoInterval'])
final_train.shape


# ## 3.2 Model Building : Random Forest Regressor 

# ### 3.2.1 Spliting the dataset

# In[ ]:


# Spliting dataset into X and y 
X=final_train.drop('Log_Sales',axis=1)
y=final_train['Log_Sales']

# Spliting dataset into test and train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=101) 


# ### 3.2.2 Creating and Training the model 

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

rfr=RandomForestRegressor(n_estimators=200,
                          criterion='mse',
                          max_features='sqrt',
                          oob_score=True,
                          n_jobs=32,
                          verbose=1,
                          random_state=101)

rfr.fit(X_train,y_train)


# In[ ]:


# Prediction
predict=rfr.predict(X_test)
predict


# ### 3.2.3 Model Evaluation 

# In[ ]:


# Out of Bag score
print('oob score :',rfr.oob_score_)


# In[ ]:


# Root mean square error
from sklearn.metrics import mean_squared_error
from math import sqrt
mse=mean_squared_error(np.exp(y_test),np.exp(predict))
print('Root Mean Square Error {}'.format(sqrt(mse)))


# In[ ]:


# Import attributes according to model
pd.options.display.float_format='{:.5f}'.format
important_features=pd.DataFrame(rfr.feature_importances_,index=X_train.columns)
important_features.sort_values(by=0,ascending=False)


# In[ ]:


# Visualization of important features
important_features.sort_values(by=0,ascending=False).iplot(mode='markers',title='Important Attributes')


# ### 3.2.4 Predictions from Model

# In[ ]:


prediction=pd.DataFrame(np.exp(y_test))
prediction['Sales_prediction']=np.exp(predict).round()
prediction.rename(columns={'Log_Sales':'Sales'},inplace=True)

# Visualization
plt.figure(figsize=(8,8))
sns.scatterplot(x='Sales',y='Sales_prediction',data=prediction)
plt.title('Actual vs Prediction of Sales',fontdict={'fontsize':20})
plt.show()

prediction.reset_index(inplace=True)
prediction.drop('index',axis=1,inplace=True)
prediction


# In[ ]:


# Residual Histogram
plt.figure(figsize=(15,8))
sns.distplot((np.exp(y_test)-np.exp(predict)),bins=100)
plt.title('Residual Histogram',fontdict={'fontsize':15})
plt.show()


# #### Further model can be enhanced by
# * Droping some of the very low assigned weightage columns 
# * Standard normalize scaling the whole dataset
# * Hyper tuning the parameters 
# 
# #### Some of the code for hyper tuning the parameters using GridSearchCv
# 
#       from sklearn.model_selection import GridSearchCV
#       from sklearn.ensemble import RandomForestRegressor
# 
#       rfr=RandomForestRegressor(random_state=101)
# 
#       param_grid=dict(n_estimators=np.arange(200,500,50),max_depth=np.arange(2,6,1),
#                 max_features=['auto','sqrt'],min_samples_leaf=[3,4,5],min_samples_split=[3,4,5])
# 
#       gridSearch=GridSearchCV(estimator = rfr, param_grid = param_grid, cv = 4, n_jobs = 32,
#                         verbose=1,scoring='neg_mean_squared_error')
# 
#       gridSearch.fit(X_train,y_train)
#       gridSearch.best_param
# 
# * Using best parameter obtain by GridSearch to build again random forest model and making the prediction and evaluating the model based on root mean square error.
# 
# Note : I have not performed hypertuning because i have used maximum of provided ram by kaggle.

# # 4. Test dataset prediction for submission

# In[ ]:


test.info()


# In[ ]:


# Merging the store set
new_test=pd.merge(test,store,how='inner',on='Store')
new_test.drop(['Avg_Sales','Avg_SalesPerCustomer','Open'],axis=1,inplace=True)
new_test.info()


# In[ ]:


# Making new columns to show Date information separately 
new_test['Year']=new_train['Date'].dt.year
new_test['Month']=new_train['Date'].dt.month
new_test['Day']=new_train['Date'].dt.day
new_test['Week']=new_train['Date'].dt.week
new_test.drop('Date',axis=1,inplace=True)
new_test.info()


# In[ ]:


# Making column "MonthCompetitionOpen" which contains date information in months since the competition was opened 
new_test['MonthCompetitionOpen']=12*(new_test['Year']-new_test['CompetitionOpenSinceYear'])+new_test['Month']-new_test['CompetitionOpenSinceMonth']

new_test.loc[(new_test['CompetitionOpenSinceYear']==0),'MonthCompetitionOpen']=0
# Negative values indcate that the competitor's store was opened after the Rossman's store opening date.

# Making column "WeekPromoOpen" which contains date information in weeks since the promo is running
new_test['WeekPromoOpen']=52.14298*(new_test['Year']-new_test['Promo2SinceYear'])+new_test['Week']-new_test['Promo2SinceWeek']

new_test.loc[(new_test['Promo2SinceYear']==0),'WeekPromoOpen']=0


# In[ ]:


# Checking the categorical variable columns unique values
for i in new_test.columns[new_test.dtypes=='object']:
    print(i+':',new_test[i].unique())


# In[ ]:


# Converting into integer format
new_test['Promo2']=new_test['Promo2'].astype(int)

# Deleting Id column from new_test dataset
Id=new_test['Id']
new_test.drop('Id',axis=1,inplace=True)

# Getting dummy columns
new_test=pd.get_dummies(new_test)
new_test.info()


# In[ ]:


# Making new columns to match the numbers with the X_train columns
new_test['StateHoliday_b']=0
new_test['StateHoliday_c']=0
new_test=pd.get_dummies(new_test,columns=['StateHoliday_b','StateHoliday_c'])
new_test.rename(columns={'StateHoliday_b_0':'StateHoliday_b','StateHoliday_c_0':'StateHoliday_c'},inplace=True)
new_test.info()


# In[ ]:


# Final prediction for submission

Sales=np.exp(rfr.predict(new_test))

pd.options.display.float_format='{:.0f}'.format
Final_submission=pd.DataFrame(Sales)
Final_submission['Id']=test['Id']
Final_submission.rename(columns={0:'Sales'},inplace=True)
Final_submission=Final_submission[['Id','Sales']]
Final_submission


# In[ ]:


Final_submission.to_csv('Random_Forest_Regression_submission.csv',index=False)


# In[ ]:




