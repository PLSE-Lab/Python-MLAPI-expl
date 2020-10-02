#!/usr/bin/env python
# coding: utf-8

# Importing important dependencies

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import warnings    #warnings to ignore any kind of warnings that we may recieve.
warnings.filterwarnings('ignore')


# In[ ]:


def display_all(df):
    '''
    input: dataframe
    description: it takes a dataframe and allows use to show a mentioned no. of rows and columns in the screen
    '''
    with pd.option_context("display.max_rows",10,"display.max_columns",9):  #you might want to change these numbers.
        display(df)


# In[ ]:


df=pd.read_csv('../input/diabetes.csv')
df.shape


# In[ ]:


display_all(df)


# Making a function for showing us a well defind table regarding the no. of missing values in each rows of the dataframe

# In[ ]:


def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        
        return mis_val_table_ren_columns


# **Checking Missing values**

# In[ ]:


missing_values_table(df)


# We found that no missing values are there in our dataset. But one thing we forgot to analyse that few features that are mentioned in the dataset like **BMI**  ,**Insulin** ,**BloodPressure**,**SkinThickness**,**Glucose** cannot have a value of zero. So there is a strong possibility that the rows in which these features are termed as zero is due to unavailability of data and hence can be termed as missing values 

# In[ ]:


features_with_missing_values=['BMI','SkinThickness','BloodPressure','Insulin','Glucose']


# **It is worth mentioning that why we used median and not mean to replace the value of 0 in these mentioned columns?**
# It is due to the fact that there can be some outliers (more spread out data points) that may have a strong effect on mean and mean can be more biased towards these outliers. So a good thing is to use median since median is not affected by outliers. To study more on this topic : [https://medium.com/@pswaldia1/statistics-for-data-science-why-it-is-important-e30c60c5018d](http://)

# In[ ]:


for i in features_with_missing_values:
    df[i]=df[i].replace(0,np.median(df[i].values))


# **Making target column different from the dataset**

# In[ ]:


target=df['Outcome'].values
df.drop(['Outcome'],inplace=True,axis=1)


# **Now we need to standardise the dataset because data is not well spread and is varied in magnitude that may make training harder**

# In[ ]:


#from sklearn importing standard scalar that will convert the provided dataframe into standardised one.
from sklearn.preprocessing import StandardScaler                                              
sta=StandardScaler()
input=sta.fit_transform(df)    #will give numpy array as output


# **Splitting the dataset into train and test set**

# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(input,target,test_size=0.1,random_state=0)


# **Using Knearest classifier**

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


knn=KNeighborsClassifier(n_neighbors=7)


# **Training model on train set**

# In[ ]:


knn.fit(X_train,y_train)


# **Checking accuracy on test set**

# In[ ]:


knn.score(X_test,y_test)


# In[ ]:




