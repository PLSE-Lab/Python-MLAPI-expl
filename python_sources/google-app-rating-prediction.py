#!/usr/bin/env python
# coding: utf-8

# Import all the libraries to handle the Data processing.

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


gapp = pd.read_csv("../input/google-play-store-apps/googleplaystore.csv")
gapp.head()


# It is good idea to clean column names especially the special charecters to avoid confusion/errors while coding.

# In[ ]:


print("Column names before clean-up:: ", gapp.columns)
gapp.columns = [colName.replace(" ","_") if(len(colName.split())>1) else colName for colName in gapp.columns]
print("======================================================================================================")
print("Column names after clean-up:: ", gapp.columns)


# Now let us look at the gaps in the data frame. 

# In[ ]:


# Find the columns wise NaN counts
gapp.isnull().sum(axis = 0)


# So there are 1474 rows (Apps) with-out rating and one without Content_Rating, 8 without current version numbers and 3 without Android version. Though there are several methods for dealing this depending on the question under investigation, for now let us drop those rows with NaN and store this data in gappC (google Apps clean)

# In[ ]:


gappC = gapp.dropna()
print("Number of rows after cleaning:: ", gappC.shape[0])
gappC.isnull().sum(axis = 0)


# From above it is clear now that there are no NaN (Gaps) values in any columns. Now let us begin our analysis. Let us start our analysis with App Ratings. 
# 
# **App Ratings**

# In[ ]:


#sns.distplot(gapp["Rating"]) # this will through an error because there are NaN
print("Unique values of ratings across apps:: ",gapp["Rating"].unique())
print("==================================================")
print("Number of rows before cleaning:: ", gapp.shape[0])


# Couple of observations on Ratings columns. There are rows (apps) without rating (nan), which is quite understandable because not apps will be rated. But what is funny is though the rating scale is 1-5 there is a rating of 19. Let us see how to fix them. 

# In[ ]:


gappC = gappC[gappC.Rating != 19]
gappC["Rating"].unique()


# In[ ]:


#So apparently there are no NaN values 
#From now on we will use the cleaned data frame for our anlysis
#Let us c how the vlaues are distributed in ratings
plt.figure(figsize=(26, 10))
sns.distplot(gappC["Rating"])


# Let us explore the Price of apps. Zero indicates Free apps. 

# In[ ]:


print("Unique values in Price column :: ", gappC["Price"].unique()) # There are  $ signs so we can not plot them So let us remove them 
gappC["Price"] = gappC["Price"].replace({'\$':''}, regex = True).astype(float) # remove the $ sign and make it a numeric column
print("=============================================================================")
print("Unique values in price column post processing::", gappC["Price"].unique())


# In[ ]:


gappC["Price"].describe()


# A quick check on the Price column indicates that more than 75% of the apps are free. But the maximum price paid for an app is 400 \$.
# So let us see what are the apps that are Priced over 100$

# In[ ]:


gappC.loc[gappC['Price'] >=100, ['App', "Price"]]


# In[ ]:


gappC["Type"].unique()


# In[ ]:


plt.figure(figsize=(26, 10))
sns.scatterplot(x = gappC.Price, y = gappC.Rating, s = 80)


# In[ ]:


cvrF = gappC[gappC.Price == 0] # Category Vs rating for free apps
plt.figure(figsize=(26, 10))
plt.xticks(rotation=90, horizontalalignment='right')
sns.scatterplot(x = cvrF.Category, y = cvrF.Rating, hue=cvrF.Content_Rating, s = 80)


# In[ ]:


cvrP = gappC[gappC.Price != 0] # Category Vs rating for free apps
plt.figure(figsize=(26, 10))
plt.xticks(rotation=90, horizontalalignment='right')
sns.scatterplot(x = cvrP.Category, y = cvrP.Rating, hue=cvrP.Content_Rating, s = 80)


# In[ ]:


plsf = gappC[(gappC.Price > 0) & (gappC.Price < 75) ] # Paid apps less than 75
plt.figure(figsize=(26, 10))
plt.xticks(rotation=90, horizontalalignment='right')
sns.scatterplot(x = plsf.Category, y = plsf.Rating, hue=plsf.Content_Rating, s = 80)


# In[ ]:


plt.figure(figsize=(26, 10))
plt.xticks(rotation=90, horizontalalignment='right')
sns.boxplot(x = plsf.Category, y = plsf.Rating, hue=plsf.Content_Rating)


# In[ ]:


pgtf = gappC[gappC.Price > 250] # Paid apps greater than 250
plt.figure(figsize=(26, 10))
plt.xticks(rotation=90, horizontalalignment='right')
sns.scatterplot(x = pgtf.Category, y = pgtf.Rating, hue=pgtf.Content_Rating, s = 80)


# In[ ]:


gappC[gappC.Price >100].shape[0]
# so there are only 15 Applications above 100$.


# In[ ]:


fig, axs = plt.subplots(2,2,figsize=(15, 10))
sns.distplot(gappC[(gappC.Price >= 0.0)&(gappC.Price <= 0.99)].Price, ax=axs[0,0])
sns.distplot(gappC[(gappC.Price >= 1.00)&(gappC.Price <= 10)].Price, ax=axs[0,1])
sns.distplot(gappC[(gappC.Price >= 11.00)&(gappC.Price <= 100)].Price, ax=axs[1,0])
sns.distplot(gappC[gappC.Price >= 101.00].Price, ax=axs[1,1])


fig.suptitle("App distribution by Price segments", fontsize=16)
axs[0, 0].set_title('Free apps and below 1$')
axs[0, 1].set_title('Apps Priced below 10 $ excluding free')
axs[1, 0].set_title('Apps Priced below 100$ and above 11')
axs[1, 1].set_title('Apps Priced above 100$')


# In[ ]:


gappC.groupby(['Category'])['Category'].count().sort_values(ascending = False).head(10)


# In[ ]:


plt.figure(figsize=(26, 10))
plt.xticks(rotation=90, horizontalalignment='right')
plt.title('Count of app in each category',size = 20)
g = sns.countplot(x="Category",data=gappC, palette = "Set1")
g 


# In[ ]:


gappC.groupby(['Category'])['Price'].max().sort_values(ascending = False).head(10)


# In[ ]:


#Ratings by Category
plt.figure(figsize=(26, 10))
plt.xticks(rotation=90, horizontalalignment='right')
sns.boxplot(x="Category", y="Rating", data=gappC)


# 

# In[ ]:


gappC.info()


# In[ ]:


gappC.head(5)


# Fix Size column so that it can become number

# In[ ]:


gappC['Size'] = gappC['Size'].map(lambda x: x.rstrip('M'))
gappC['Size'] = gappC['Size'].map(lambda x: str(round((float(x.rstrip('k'))/1024), 1)) if x[-1]=='k' else x)
gappC['Size'] = gappC['Size'].map(lambda x: np.nan if x.startswith('Varies') else x)


# Fix installs by eliminating the + sign

# In[ ]:


gappC['Installs'] = gappC['Installs'].map(lambda x: x.rstrip('+'))
gappC['Installs'] = gappC['Installs'].map(lambda x: ''.join(x.split(',')))


# In[ ]:


gappC.info()


# Convert some of the numeric columns to int/float

# In[ ]:


gappC['Reviews'] = gappC['Reviews'].apply(lambda x: float(x))
gappC['Size'] = gappC['Size'].apply(lambda x: float(x))
gappC['Installs'] = gappC['Installs'].apply(lambda x: float(x))


# In[ ]:


gappC.info()


# In[ ]:


gappC.isnull().sum(axis = 0)


# In[ ]:


gappDF = gappC.dropna()


# In[ ]:


gappDFM = gappDF.drop(['App', 'Category', 'Type', 'Content_Rating', 'Genres', 'Last_Updated', 'Current_Ver', 'Android_Ver'], axis = 1)
gappDFM.head()


# In[ ]:


plt.figure(figsize=(15, 5))
sns.regplot(x = gappDFM.Rating, y = gappDFM.Reviews)


# In[ ]:


plt.figure(figsize=(15, 5))
sns.regplot(x = gappDFM.Rating, y = gappDFM.Installs)


# In[ ]:


plt.figure(figsize=(15, 5))
sns.regplot(x = gappDFM.Rating, y =gappDFM.Size )


# Now let us begin the prediction of App ratings by the availbe data using regression techniques.

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split


# In[ ]:


#for evaluation of error term  
def evalMat(y_true, y_predict):
    print ('Mean Squared Error: '+ str(metrics.mean_squared_error(y_true,y_predict)))
    print ('Mean absolute Error: '+ str(metrics.mean_absolute_error(y_true,y_predict)))
    print ('Mean squared Log Error: '+ str(metrics.mean_squared_log_error(y_true,y_predict)))


# In[ ]:


def evalMat_dict(y_true, y_predict, name = 'Linear Regression'):
    dict_matrix = {}
    dict_matrix['Regression Method'] = name
    dict_matrix['Mean Squared Error'] = metrics.mean_squared_error(y_true,y_predict)
    dict_matrix['Mean Absolute Error'] = metrics.mean_absolute_error(y_true,y_predict)
    dict_matrix['Mean Squared Log Error'] = metrics.mean_squared_log_error(y_true,y_predict)
    return dict_matrix


# In[ ]:


x = gappDFM.drop(["Rating"], axis = 1)
y = gappDFM.Rating


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)


# In[ ]:


linMod = LinearRegression()
linMod.fit(x_train, y_train)
linRes = linMod.predict(x_test)


# In[ ]:


resDF = pd.DataFrame()
resDF = resDF.from_dict(evalMat_dict(y_test,linRes),orient = 'index')
resDF = resDF.transpose()


# In[ ]:


resDF.head()


# In[ ]:


print('Actual mean of Population:: ', y.mean())
print('Predicted mean:: ', linRes.mean())


# In[ ]:


plt.figure(figsize = (12, 6))
sns.regplot(linRes, y_test, marker = 'x')
plt.title('Linear model')
plt.xlabel('Predicted Ratings')
plt.ylabel('Actual Ratings')
plt.show()


# In[ ]:


linMod.coef_


# In[ ]:


linMod.intercept_


# So in summary through the MSE is ~0.2 Most ofthe predicted ratings are centered between 4.0 and 4.5. This needs to be improved. Now let us try Support Vector Regression

# Now let us try support vector regression

# In[ ]:


from sklearn import svm

svrMod = svm.SVR(gamma='auto')
svrMod.fit(x_train, y_train)
svrRes = svrMod.predict(x_test)


# In[ ]:


print('Actual mean of Population:: ', y.mean())
print('Predicted mean:: ', svrRes.mean())


# In[ ]:


resDF = resDF.append(evalMat_dict(y_test, svrRes, name = "SVR"), ignore_index = True)


# In[ ]:


resDF


# In[ ]:


plt.figure(figsize = (12, 6))
sns.regplot(svrRes, y_test, marker = 'x')
plt.title('SVR model')
plt.xlabel('Predicted Ratings')
plt.ylabel('Actual Ratings')
plt.show()


# Now let us do the Random forest regressor. 

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rfrMod = RandomForestRegressor()
rfrMod.fit(x_train, y_train)
rfrRes = rfrMod.predict(x_test)


# In[ ]:


print('Actual mean of Population:: ', y.mean())
print('Predicted mean:: ', rfrRes.mean())


# In[ ]:


resDF = resDF.append(evalMat_dict(y_test, rfrRes, name = "RFR"), ignore_index = True)
resDF


# In[ ]:


plt.figure(figsize = (12, 6))
sns.regplot(rfrRes, y_test, marker = 'x')
plt.title('RFR model')
plt.xlabel('Predicted Ratings')
plt.ylabel('Actual Ratings')
plt.show()

