#!/usr/bin/env python
# coding: utf-8

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

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
df = pd.read_csv("/kaggle/input/black-friday/train.csv")


#limiting the amount of data we use to keep computation fast 
df = df.iloc[:]
df.shape


# # FIRST QUICK LOOK AT THE DATA 

# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


print("Data columns")
print("--------------------------------------------")
print(pd.DataFrame(df.info()))


# ## we can see that these colums have some null values
#     Product_Category_2           
#     Product_Category_3            

# In[ ]:


print(df.isnull().sum())


# # HANDLING MISSING DATA 

# ## percentage of missing data:

# In[ ]:



df.isnull().sum()/df.isnull().count()*100


# ## we can delete Product_Category_3 column  because missing data in this columns more than 60% of the observations

# In[ ]:


df = df.drop('Product_Category_3', axis=1)

## also dropping user id category 
df = df.drop('User_ID', axis=1)
df = df.drop('Product_ID', axis=1)


# In[ ]:


df['Marital_Status'].unique()


# 

# # IMPUTING 
# 
# ## for Product_Category_2 we can impute the 31.5 % of missing data 
# ## lets use simple imputer for now, later we can take care of these missing data with other methods as well, to perhaps boost our accuracy 

# In[ ]:


df['Product_Category_2'].unique()


# In[ ]:


len(df['Product_Category_2'].unique())


# In[ ]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer = imputer.fit(pd.DataFrame(df['Product_Category_2']))


df['Product_Category_2'] = imputer.transform(pd.DataFrame(df['Product_Category_2']))
#data_train['Product_Category_2'] = np.round(data_train['Product_Category_2'])


# In[ ]:


imputer = imputer.fit(pd.DataFrame(df['Marital_Status']))
df['Marital_Status'] = imputer.transform(pd.DataFrame(df['Marital_Status']))


# In[ ]:


len(df['Product_Category_2'].unique())


# ## confirming no missing values 

# In[ ]:


df.isnull().sum()/df.isnull().count()*100


# # Mean purchase for men Vs women
# ###  males buying higher value purchases than females, But the difference not large.

# In[ ]:


print("average purchase for male purchasers = ",
np.mean(df['Purchase'].loc[df['Gender'] == 'M']),'\n')
print("-"*115,'\n')
print("average purchase for female purchasers = ",
np.mean(df['Purchase'].loc[df['Gender'] == 'F']))


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
fig= plt.figure(figsize=(12,7))


sns.set(style="darkgrid")


x = pd.DataFrame({"male average purchase": [9437], "Female average purchase": [8734]})

sns.barplot(data=x)


# # number of observations of female individuals vs male 

# In[ ]:


print('Number of Female purchasers = ',df['Gender'][df['Gender'] == 'F'].count())
print('Number of male purchasers   = ',df['Gender'][df['Gender'] == 'M'].count())


# # FEATURE ENGINEERING FOR BUILDING MODELS
# 

# In[ ]:


#change gender from 'm' and 'f' to binary 
df.loc[:, 'Gender'] = np.where(df['Gender'] == 'M', 1, 0)

#renaming some columns 
df = df.rename(columns={
                #'Product_ID': 'ProductClass',
                'Product_Category_1': 'Category1',
                'Product_Category_2': 'Category2',
                'City_Category': 'City',
                'Stay_In_Current_City_Years': 'City_Stay'
})
#y = train.pop('Purchase')


# In[ ]:


df.head()


# ## ENCODING CATEGORICAL VARIABLES 
# 
# 

# In[ ]:


#len(df['ProductClass'].unique())


# # LABEL ENCODING FOR PRODUCT CLASS

# In[ ]:


# from sklearn.preprocessing import LabelEncoder
# L_encoder =  LabelEncoder()
# for col in ['ProductClass']:    
#     df.loc[:, col] =L_encoder.fit_transform(df[col])
# df[['ProductClass']]


# # OneHotEncoder for other Classes

# In[ ]:


from sklearn.preprocessing import  OneHotEncoder
cats = ['Occupation', 'Age', 'City', 'Category1','Category2','City_Stay']

#creating the encoder, fit it to our data 
encoder = OneHotEncoder().fit(df[cats])


# In[ ]:


#generating feature names for our encoded data
encoder.get_feature_names(cats)


# In[ ]:


#building dataframe with encoded catgegoricals 

## we use index values from our original data 
## we GENERATE feature names using our encoder

endcoded_data = pd.DataFrame(encoder.transform(df[cats]).toarray(),index=df.index, columns=encoder.get_feature_names(cats))
endcoded_data.head()


# ## dropping categorical data, adding the encoded data
# 

# In[ ]:


df = pd.concat([df, endcoded_data],sort=False,axis=1)

df=df.drop(cats, axis=1)


# ## REPLACING NANS(after OH encoding) with 0

# In[ ]:


df = df.fillna(0)
df.head(15)


# In[ ]:





# # DEFINING FEATURES AND LABELS 

# In[ ]:


X = df.drop('Gender',axis=1)
y = df.pop('Gender')


# In[ ]:





# In[ ]:


#X=np.nan_to_num(X)


# # SCALING

# In[ ]:


from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)


# In[ ]:





# # TRAIN TEST SPLIT 

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size= 0.25)


# # FITTING OUR MODEL
# 

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)


# ## FIRST SCORE

# In[ ]:


knn.score(X_test,y_test)


# # HYPER PARAMETER TUNING
# # WITH GridSearchCV 

# In[ ]:


# parameters = { 'algorithm' : ['ball_tree', 'kd_tree', 'brute'],
#                'leaf_size' : [18,20,25,27,30,32,34],
#                'n_neighbors' : [3,5,7,9,10,11,12,13]
#               }

# from sklearn.model_selection import GridSearchCV
# gridsearch = GridSearchCV(knn, parameters,verbose=3)
# gridsearch.fit(X_train,y_train)
# gridsearch.best_params_


# # CREATING the TUNED THE MODEL AGAIN

# In[ ]:


knn = KNeighborsClassifier(algorithm = 'auto', leaf_size =35, n_neighbors =5)
knn.fit(X_train,y_train)
knn.score(X_test,y_test)


# # MAKING A PREDICTION 

# ## Using, and transforming available data to make a prediction, user input will need to be transformed also 

# # OUR INPUT DATA FOR PREDICTION NEEDS TO LOOK LIKE OUR FEATURES FOR TRAINING
# 
# ## this means, we have to : 
#     drop the same caolumns 
#     encode the categorical features in the same way, 
#     etc. 
#     
# ## basically our input data needs to go through all the things we did to our training data, 
# ## this exercise will also help us find what all inputs to take from the user 

# > ## drop columns

# In[ ]:


dft = pd.read_csv("/kaggle/input/black-friday/test.csv")

dft = dft.drop('Product_Category_3', axis=1)

## also dropping user id category 
dft = dft.drop('User_ID', axis=1)
dft = dft.drop('Product_ID', axis=1)

#Product_ID
dft.head()


# In[ ]:


dft = dft.iloc[:1]
dft


# > ## Rename similarly

# In[ ]:


dft = dft.rename(columns={
                #'Product_ID': 'ProductClass',
                'Product_Category_1': 'Category1',
                'Product_Category_2': 'Category2',
                'City_Category': 'City',
                'Stay_In_Current_City_Years': 'City_Stay'
})
dft


# In[ ]:


# dft['ProductClass']='P00248942'
# dft


# # THESE ARE THE INPUTS THAT WE REQUIRE FROM THE USER, AFTER TAKING THE INPUT, WE NEED TO DO OUR TRANSFORMATIONS TO GET A PREDICTION
# 
# Gender 	Age 	Occupation 	City 	City_Stay 	Marital_Status 	Category1 	Category2
# 
# 
# ## SINCE THESE ARE CATEGORICAL WE NEED TO PUT A LEGEND SO THAT THE USER CAN PUT THE CORRECT INPUTS

# In[ ]:


#change gender from 'm' and 'f' to binary 
dft['Gender'] = 9851
dft


# ## IMP*
# > ## encode all required fequres with the exact same encoder

# In[ ]:


p =pd.DataFrame(encoder.transform(dft[cats]).toarray(),columns=encoder.get_feature_names(cats))
p


# In[ ]:


dft=dft.drop(cats, axis=1)
dft


# # CREATE THE FINAL INPUT FOR PREDICTION 

# In[ ]:


dft = pd.concat([dft, p],sort=False,axis=1)


# In[ ]:


df.head(1)


# In[ ]:


dft


# In[ ]:


# dft['ProductClass'] =L_encoder.transform(dft['ProductClass'])
# p


# # get prediction 

# In[ ]:


knn.predict(dft)


# In[ ]:





# In[ ]:




