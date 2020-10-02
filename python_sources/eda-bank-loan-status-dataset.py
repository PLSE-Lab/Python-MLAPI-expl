#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
print(os.listdir("../input"))


# In[ ]:


#Read the Train and Test Dataset for EDA .
Train_csv=pd.read_csv('../input/credit_train.csv')
Test_csv=pd.read_csv('../input/credit_test.csv')


# In[ ]:


#Check the Sample of Data understand the Values contain by different Columns.
Train_csv.head()


# In[ ]:


Train_csv.shape


# In[ ]:


#Generalised Function which can give the Percentage of missing values present in DataSet.
def Train_missing_values(training_dataset):
    Missing_Data_Percent=pd.DataFrame(training_dataset.isna().sum())
    Missing_Data_Percent.reset_index(inplace=True)
    Missing_Data_Percent.columns=['Feild_Name','Missing_value_count']
    Missing_Data_Percent['Percent_missing_values']=Missing_Data_Percent['Missing_value_count'].                                                apply(lambda Missing_value_count:(Missing_value_count/len(training_dataset))*100)
    return Missing_Data_Percent.sort_values(['Percent_missing_values'],ascending=False)


Train_missing_values(Train_csv)


# ### Results After Analysis of Missing Values
# * We have very high percentage of missing value in column "Months since last delinquent" 
# * So its better to remove that column from Train set instead of going for imputations.
# * Also we have 2 Unique Ids for Customers which are of no use in Modelling So remove them also
# * "Customer ID" and "Loan ID"
# * Now we have total 16 Feilds out of which 1 is Target variable.

# In[ ]:


#Dropping the column from Train Dataset
Train_csv.drop(['Months since last delinquent','Loan ID','Customer ID'],inplace=True,axis=1)


# In[ ]:


Train_csv.head()


# ### Handling NA vs Changing the type of Feilds
# * Always Make sure that ,You should handle Null values in Data on first priority and the typecase them into some categories or in some other form.
# * Lets Find out the Row level Duplicate and  Remove them ,Because they are of no Use to us.

# In[ ]:


#Check for all Row level NULL(It means all column values for that row are null) from Dataframe because they are not carring any information.
Train_csv=Train_csv[Train_csv.isna().all(axis=1)==False]


# In[ ]:


#Check whether Row level NULL are hablded or Not
Train_csv.shape


# In[ ]:


#Focus on Remaining the NULL values which are needs to  handled.
Train_csv.isna().sum()


# ### Handling the Data which is incorrect .ex(Credit Score neven be above 900 but in Our Data we have Credit Score till 9000 .So this is Human error) So Lets Handle this type of error and then go for Imputation.

# In[ ]:


import numpy as np
Train_csv['Credit Score'] = np.where(Train_csv['Credit Score']>900, Train_csv['Credit Score']/10, Train_csv['Credit Score'])


# In[ ]:


#Remove the NaN records for Column 'Tax Liens' and 'Maximum Open Credit' because these missing values are very less in terms of complete population.ta

Train_csv.drop(Train_csv[Train_csv['Tax Liens'].isnull()].index,inplace=True)
Train_csv.drop(Train_csv[Train_csv['Maximum Open Credit'].isnull()].index,inplace=True)


# In[ ]:


Train_csv[['Credit Score','Annual Income','Bankruptcies']].describe()


# ### How to replace the Missing Values.
# * So we will prefer here to replace missing values by Random Number fron the Respective column.That will help us to keep the Distribution in original form.So lets Check Summary for these column before and after Imputation.
# * We tried interpolation here because we dont want to replace missing values by mean or median .Reason is that if you do that it will affect your distribution of data.So we want to handle data as well as maintain the distribution in line for modelling purporse.

# In[ ]:


Train_csv.interpolate(inplace=True)


# ### Check distribtion of data is Not Changed.It means that sense is still remain in data after our imputations .

# In[ ]:


Train_csv[['Credit Score','Annual Income','Bankruptcies']].describe()


# In[ ]:


Train_csv['Years in current job'].fillna('10+ years',inplace=True)
#vals_to_replace = {'Small':'1', 'Medium':'5', 'High':'15'}
#df['a'] = df['a'].map(vals_to_replace)


# In[ ]:


Test_data=pd.read_csv('../input/credit_test.csv')
Test_data.shape


# ## Learning From this EDA
# * Prefer Data correctness before doing any imputations on  any feild.
# * Always Consider the distribution of Data before imputation.
# * Do the imputation such way that original distribution of Random variable should not be affected.
# * Always Remove the Data which contains high level of  missing Values (Like More tahtn 50 percent)
# * Remove missing values records which are very neglisible in term of Training and Test Dataset.
# 
# 
