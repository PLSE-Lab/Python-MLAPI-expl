#!/usr/bin/env python
# coding: utf-8

# # Importing the required libraries
# 
# Import all the required libraries. %matplotlib inline is to show the charts inline.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

get_ipython().run_line_magic('matplotlib', 'inline')


# setting the chart style

# In[ ]:


sns.set(style='white', context='notebook', palette='deep')
sns.set_style('white')


# ## A brief about the data
# 
# Importing the Test 1 data into the customers dataframe. We would be using this data to train and test our model.
# The steps that I am following for this analysis are :
#     1. Explatory analysis.
#     2. Visualization of the data
#     3. Normalization of the data fields.
#     4. Building of the model

# In[ ]:


customers = pd.read_csv('../input/Test 1.csv')


# Looking at the imported data. 

# In[ ]:


customers.head()


# Include = "all", so that I can see the metrics on string fields as well. From the data below, we can see that demographic_slice has 4 unique values, country_reg has 2 and ad_exp has 2. We could use this information to our benefit by converting these string fields into numerical fields for us to process. 

# In[ ]:


customers.describe(include="all")


# By looking at the information of the columns and their datatype, we see that the demographic_slice,country_reg and ad_exp are of object datatype. We have to convert them string.

# In[ ]:


customers.info()


# In[ ]:


customers.columns


# # Visualization
# 
# We plot the data as per our understanding of the business as well as relations between columns

# In[ ]:


sns.countplot(x='card_offer',data=customers)


# In[ ]:


plt.figure(figsize=(14,8))
sns.heatmap(data=customers.corr(),cmap='plasma',annot=True)


# In[ ]:


plt.figure(figsize=(14,8))
sns.relplot(x='est_income',y='pref_cust_prob',data=customers,hue='card_offer')


# In[ ]:


customers[customers.dtypes[(customers.dtypes=="float64")|(customers.dtypes=="int64")]
                       .index.values].hist(figsize=[11,11])


# After looking at the graphs above, we can see that, the values for RiskScore and imp_score range between 0 to 800, and 
# for est_income, the value go till 15000. Compared to the other features, these three have different scales. The est_income values may dominate the outcome of the result.
# 

# # Data preprocssing.
# 
# Create a function to convert the categorical columns into encoded vectors. I have created this function, so that I can repeat the same my test dataset rather than doing it altogether.

# In[ ]:


def clean_data(df):
    
    # encoding the categorical columns    
    demographic_slice_df = pd.get_dummies(data=df['demographic_slice'],drop_first=True,prefix='ds')
    country_reg_df = pd.get_dummies(data=df['country_reg'],prefix='cr',drop_first=True)
    ad_exp_df = pd.get_dummies(data=df['ad_exp'],prefix='ae',drop_first=True)
    
    
    # scaling the features
    minMax = MinMaxScaler()
    customers_ent_income = pd.DataFrame(minMax.fit_transform(df[['est_income','RiskScore','imp_cscore']])
                                    ,columns=['scaled_est_income','scales_riskScore','scaled_imp_csore'])
    
    # creating the transfored dataframe
    df = pd.concat([df,demographic_slice_df,country_reg_df,ad_exp_df,customers_ent_income],axis=1)
    df.drop(['demographic_slice','country_reg','ad_exp','est_income','RiskScore','imp_cscore'],axis=1,inplace=True)
    
    return(df)
    


# In[ ]:


customers = clean_data(customers)
customers['card_offer'] = customers['card_offer'].apply(np.int32)
customers.head()


# # Model Creation
# 
# Based on the data, we run it for Logistic regression model for the test and train data that we have

# In[ ]:


X = customers.drop(['customer_id','card_offer'],axis=1)
y = customers['card_offer']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[ ]:


print('Shapes of the dataframes')

print('X_train',X_train.shape)
print('X_test',X_test.shape)

print('y_train',y_train.shape)
print('y_test',y_test.shape)


# In[ ]:


lr = LogisticRegression()
lr.fit(X_train,y_train)


# In[ ]:


predictions = lr.predict(X_test)


# In[ ]:


print(accuracy_score(y_test,predictions))


# In[ ]:


print(classification_report(y_test,predictions))


# In[ ]:


print(confusion_matrix(y_test,predictions))


# In[ ]:


Test2Data = pd.read_csv('../input/Test 2.csv')


# # Getting the Test 2.csv file
# 
# Now that we have the model trained, and with a precision of 87%, we can start with the predecting of the data.
# 
# The Test 2 file, goes through the same cleansing/preprocesing process like the Test 1.csv

# In[ ]:


Test2Data.head()


# In[ ]:


Test2DataOrig = Test2Data
Test2DataDf = clean_data(Test2Data)


# In[ ]:


Test2DataDf.head()


# In[ ]:


Test2DatePredict = lr.predict(Test2DataDf.drop(['customer_id','card_offer'],axis=1))


# In[ ]:


Test2DatePredictDf = pd.DataFrame(data=Test2DatePredict)


# In[ ]:


finalDf = pd.concat([Test2DataOrig,Test2DatePredictDf],axis=1)


# In[ ]:


finalDf.drop('card_offer',axis=1,inplace=True)


# In[ ]:


finalDf.rename(columns={'0':'card_offer'},inplace=True)


# Inserting the results into the file. The card offer column is renamed to "0" in the dataset

# In[ ]:





# In[ ]:




