#!/usr/bin/env python
# coding: utf-8

# # Problem statement
# 
# #### -A large U.S. Electrical appliance's retailer has many branches. There is no Fixed Price for a product(for various reasons),the SalesPerson has the freedom to choose the price at which they sell the product. There is no cap on the minimum and maximum quantity of sales on the Salesperson.Due to these reasons the average sale size and average quantity for a transaction varies.The company wants to do a 'Sales and Productivity' analysis.
# #### -It is for this reason the company wants to implement a system to classify the reports into one of the Three categories, Suspicious/Not Suspicious/Indeterminate.
# #### -The company also wants the Salespersons to be grouped into HighRisk or MediumRisk or LowRisk categories based on the report info provided by them.

# # Overview of this notebook
# ### -Univariate Analysis
# ### -Multivariate Analysis

#     

#    

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


import os
#os.chdir("C:\\Users\\deeks\\Documents\\PhD\\PHD_TrainData_and_ProblemStatement-1558070454703\\PHD_TrainData_and_ProblemStatement")


# In[ ]:


train_data=pd.read_csv("../input/Train.csv",header=0)


# In[ ]:


train_data.head()


# In[ ]:


train_data.isnull().sum()
print(train_data.shape)
train_data.dtypes


# In[ ]:


train_data.describe(include='all')


# In[ ]:


train_data['ReportID']=train_data['ReportID'].astype('category')
train_data['SalesPersonID']=train_data['SalesPersonID'].astype('category')
train_data['ProductID']=train_data['ProductID'].astype('category')
train_data['Suspicious']=train_data['Suspicious'].astype('category')


# In[ ]:


train_data.dtypes


# In[ ]:


print(train_data.ProductID.unique().shape)
print(train_data.SalesPersonID.unique().shape)
print(train_data.ReportID.unique().shape)


# #### The above numericals indicates that there are:-
# #### 1) 593 Products
# #### 2) 992 Sales Persons
# #### 3) 42582 transactions had done

# ## Univariate Analysis

# In[ ]:


100*train_data.Suspicious.value_counts()/len(train_data.Suspicious)


# #### This shows that the transactions made by the sales persons are 0.4% Fraud; 6% are Not Fraud; 93% are not able to determine

# In[ ]:


plt.figure(figsize=(20,10))
count_classes = pd.value_counts(train_data['Suspicious'], sort = True)
count_classes.plot(kind = 'bar', rot=0,color='#819980')
plt.title("Frequency of each level")
plt.xlabel("Class")
plt.ylabel("Number of Observations");


# ## Bivariate Analysis

# In[ ]:


sns.factorplot(x="Suspicious", y="TotalSalesValue", data=train_data,size=6)
plt.xlabel("Suspicious", fontsize=15)
plt.ylabel("TotalSalesValue", fontsize=15)


# In[ ]:


sns.factorplot(x="Suspicious", y="Quantity", data=train_data,size=6)
plt.xlabel("Suspicious", fontsize=15)
plt.ylabel("Quantity of products", fontsize=15)


# In[ ]:


train_data.groupby('Suspicious')['Quantity','TotalSalesValue'].sum().plot.bar()


# In[ ]:


train_data['SalesPersonID'].value_counts().sort_index().sort_values(ascending=False)[:10].plot.bar(color='#819980')


# In[ ]:


train_data['ProductID'].value_counts().sort_index().sort_values(ascending=False)[:10].plot.bar(color='#819980')


# In[ ]:


train_data['Price_of_oneProduct'] = train_data.apply(lambda row: row.TotalSalesValue / row.Quantity, axis = 1)


# In[ ]:


train_data.head()


# In[ ]:


train_data[train_data.Suspicious=='No'].SalesPersonID.count()


# In[ ]:


train_Yes=train_data[train_data.Suspicious=='Yes']
train_Indeterminate=train_data[train_data.Suspicious=='indeterminate']
train_No=train_data[train_data.Suspicious=='No']


# In[ ]:


plt.figure(figsize=(20,10))
bins = np.linspace(1000, 5000, 100)
plt.hist(train_Yes.TotalSalesValue, alpha=1,bins=bins,color="#a5177d", normed=True, label='Fraud')
plt.hist(train_No.TotalSalesValue, alpha=1,bins=bins,color="#1769a5", normed=True, label='Not Fraud')
plt.legend(loc='upper right')
plt.title("Amount by percentage of totalsales")
plt.xlabel("Total sales Value")
plt.ylabel("Percentage of Sales value (%)");
plt.show()


# In[ ]:


sns.boxplot(train_No.TotalSalesValue,orient='v')


# In[ ]:


sns.boxplot(train_Yes.TotalSalesValue,orient='v')


# In[ ]:


train_data.groupby('Suspicious').apply(lambda x: x.count())


# In[ ]:


train_Indeterminate.head()


# In[ ]:


train_data.groupby(train_data['ProductID'])['Price_of_oneProduct'].mean().sort_index().sort_values(ascending=False)[:10].plot.bar(color='#819980')


# ### Above products are  top 10 high valued products

#     

# In[ ]:


train_data.groupby(train_data['ProductID'])['Price_of_oneProduct'].mean().sort_index().sort_values(ascending=True)[:10].plot.bar(color='#819980')


# ### Above are low valued Products

#     

# In[ ]:


train_Yes['SalesPersonID'].value_counts().sort_values(ascending=False)[:10].plot.bar(color='#819980')


# ### These salesmen made high number of 'Fraud' transactions

#    

# In[ ]:


total_quantity=train_data['Quantity'].sum()
total_totalsalesvalue=train_data['TotalSalesValue'].sum()


# In[ ]:


(((train_data.groupby(train_data['ProductID'])['Quantity'].sum())/(total_quantity))*100).sort_index().sort_values(ascending=False)[:10].plot.bar(color='#819980')


# #### Top 10 products got saled in abundance amount compared to other products

#     

# In[ ]:


(((train_data.groupby(train_data['SalesPersonID'])['Quantity'].sum())/(total_quantity))*100).sort_index().sort_values(ascending=False)[:10].plot.bar(color='#819980')


# #### Top 10 salesmen who sold high amount of quantity

#       

# In[ ]:


high_sales_person=train_data.groupby(['SalesPersonID']).sum().sort_values(by='TotalSalesValue',ascending=False).head(10)
low_sales_person=train_data.groupby(['SalesPersonID']).sum().sort_values(by='TotalSalesValue',ascending=False).tail(10)


# In[ ]:


((high_sales_person['TotalSalesValue']/total_totalsalesvalue)*100).plot.bar(color='#819980')


# #### Top 10 salesmen having high sales value

#                    

# In[ ]:


((low_sales_person['TotalSalesValue']/total_totalsalesvalue)*100).plot.bar(color='#819980')


# #### top 10 Salesmen having low sales value

#       

# In[ ]:


outliers=[]
def detect_outlier(data_1):
    
    threshold=3
    mean_1 = np.mean(data_1)
    std_1 =np.std(data_1)
    
    
    for y in data_1:
        z_score= (y - mean_1)/std_1 
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers


# In[ ]:


outlier_datapoints = detect_outlier(train_data['Price_of_oneProduct'])
print(len(outlier_datapoints))


# In[ ]:


train_df=train_data.groupby(["ProductID"]).median()


# In[ ]:


train_df.describe(include='all')


# In[ ]:


train_df=train_df.drop(['TotalSalesValue','Quantity'],axis = 1)


# In[ ]:


type(train_df)


# In[ ]:


train_df=train_df.rename(columns={'Price_of_oneProduct': 'Avg_of_eachProduct'})


# In[ ]:


train_updated = train_data.join(train_df, on='ProductID')
train_updated.describe(include='all')


# In[ ]:


sns.boxplot(x=train_updated['TotalSalesValue'],orient='v')


# In[ ]:


Q1 = train_updated.quantile(0.25)
Q3 = train_updated.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[ ]:


data1 = train_updated.groupby(['SalesPersonID'])[['TotalSalesValue']].median()
data1 = data1.rename(columns={'TotalSalesValue': 'Average sales per person'})
train_updated = train_updated.join(data1,on = ['SalesPersonID'])


# In[ ]:


train_updated.head()


# In[ ]:


outlier_datapoints = detect_outlier(train_data['TotalSalesValue'])
print(len(outlier_datapoints))


# In[ ]:


data2 = train_updated.groupby(['SalesPersonID'])[['Quantity']].mean()
data2 = data2.rename(columns={'Quantity': 'Average amount Quantity per Person'})
train_updated = train_updated.join(data2,on=['SalesPersonID'])


# In[ ]:


data3 = train_updated.groupby(['SalesPersonID'])[['ProductID']].count()
data3 = data3.rename(columns={'ProductID': 'No of products per person'})
train_updated = train_updated.join(data3,on=['SalesPersonID'])


# In[ ]:


data4 = train_updated.groupby(['SalesPersonID','ProductID'])[['Quantity']].sum()
data4=data4.rename(columns={'Quantity': 'sum of each product quuantity per person'})
train_updated = train_updated.join(data4,on=['SalesPersonID','ProductID'])


# In[ ]:


data5 = train_updated.groupby(['SalesPersonID','ProductID'])[['TotalSalesValue']].sum()
data5 =data5.rename(columns={'TotalSalesValue': 'sum of each product sales per person'})
train_updated = train_updated.join(data5,on=['SalesPersonID','ProductID'])


# In[ ]:


data6 = train_updated.groupby(['SalesPersonID','ProductID'])[['TotalSalesValue']].median()
data6 =data6.rename(columns={'TotalSalesValue': 'Avg of each product sales per person'})
train_updated = train_updated.join(data6,on=['SalesPersonID','ProductID'])


# In[ ]:


data7 = train_updated.groupby(['SalesPersonID','ProductID'])[['Quantity']].mean()
data7 =data7.rename(columns={'Quantity': 'Avg of each product quuantity per person'})
train_updated = train_updated.join(data7,on=['SalesPersonID','ProductID'])


# In[ ]:


data8=train_updated.groupby(['ProductID'])[['Quantity']].mean()
data8 = data8.rename(columns={'Quantity': 'Average amount Quantity per Product'})
train_updated = train_updated.join(data8,on=['ProductID'])


# In[ ]:


data9=train_updated.groupby(['ProductID'])[['TotalSalesValue']].median()
data9 = data9.rename(columns={'TotalSalesValue': 'Average sales per Product'})
train_updated = train_updated.join(data9,on=['ProductID'])


# In[ ]:


train_updated.shape


# train_updated.to_csv('feature_eng.csv',index=False)
