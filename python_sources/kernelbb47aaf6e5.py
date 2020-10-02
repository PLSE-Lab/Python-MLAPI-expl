#!/usr/bin/env python
# coding: utf-8

# # House Prices: Advanced Regression Techniques (Project2)

# ### Team Name on Kaggle: HBE
# **Members:**
# Bader Abanmi, Husain Al-Amer , Ebrahim Balghunaim
# 
# <span style="color:blue"> **Link of Kaggle:** </span>   
# https://www.kaggle.com/hma2022/kernelbb47aaf6e5?scriptVersionId=22926066

# ## Problem Statement

# When we speak about home most people have an idea of what they want in a place called home.There are many features that contribute to the value of a house. In order to estimate the value of a home we will try and analyze the top features that has an effect on the house value.
# We are aiming to create a prediction model that will analyze the features and give an overall prediction of the house value.This will give us the ability to predict a house value with only the features.
# 
# **This solution will contribute to solving many problems:**
# - Any home buyer can have a value approximation of their dream house.
# - The Real-estate agent will have an expected value of a house given it's features.
# - Real-estate investors can have an idea about the most desired features that contibute to the properties value.
# 
# The data obtained has 79 explanatory variables describing (almost) every aspect of residential homes. The data was gathered in Ames, Iowa.We will analize the data,clean it,visulize it then fit it in a model to get a prediction of the expected `SalePrice` of the house.

# ### Imports

# In[ ]:


#Imports
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.style as style
style.use('fivethirtyeight')
# style.use('default')

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")



# sns.set(font_scale=1.5)
# %config InlineBackend.figure_format = 'retina'
get_ipython().run_line_magic('matplotlib', 'inline')

# set display max rows and columns
pd.set_option('display.max_rows',1500)
pd.set_option('display.max_columns',85)


# ### Load Data

# In[ ]:


house_train=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv',sep=',') # load the data
house_test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv',sep=',') #load the data


# In[ ]:


y=house_train.SalePrice #set the target


# ### Merge Data
# - **We need to concat the two datasets to make data cleaning on all of the two datasets.**

# In[ ]:


X = pd.concat([house_train,house_test],join='inner',ignore_index=True) #Merge the data for easier cleaning


# ### Data Exploration

# In[ ]:


X.head(2) #Get the head 


# In[ ]:


X.shape #Get The Shape of Data


# In[ ]:


X.columns #Get The Data Columns


# In[ ]:


X.info() #Get the info of data


# In[ ]:


X.describe() #Give basic stat discription of data


# In[ ]:


X.isnull().sum().sort_values(ascending=False)#Get the null values


# ## Checking Missing Values and Do Cleaning

# In[ ]:


# Doing a function for heatmap of missing variables to be called many times in this notebook
def missing_heat_map(DataFrame):#Plot a heat map of the missing varibles
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (18, 6))
    sns.heatmap(DataFrame.isnull(), yticklabels=False, ax = ax, cbar=False, cmap='viridis')
    ax.set_title('Train & Test Concatenated DataSet')
    plt.show()
missing_heat_map(X)


# **A lot of the data has null values which need to be dealt with**
# **After inspecting the data and the Data Discription document we relized that there are many features that have NA values which is intepreted by python as `Nan(Null)`. We are going to convert them back to `string:NA`**
# 
# |Feature|Feature|Feature|
# |---|---|---|
# |Alley|BsmtQual|BsmtCond|
# |BsmtExposure|BsmtFinType1|BsmtFinType2|
# |FireplaceQu|GarageType|GarageFinish|
# |GarageQual|GarageCond|PoolQC|
# |Fence|MiscFeature||
# 

# In[ ]:


#Specify the features that contain NaN to replace it by the string NA
features_with_na=['Alley','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu',                  'GarageType','GarageFinish','GarageQual','GarageCond','PoolQC','Fence','MiscFeature']

for i in features_with_na:
    X[i]=X[i].map(lambda x: 'NA' if pd.isnull(x) else x) # Convert to NA insted of null (Just like the data discription)


# In[ ]:


# Call the heatmap function again to check what is still missing
missing_heat_map(X)


# ##### Cleaning GarageYrBlt

# In[ ]:


X.GarageYrBlt.isnull().sum() # Get the sum of of null values


# In[ ]:


#Grabbing all the features of Garage to try and understand why the year is missing values
Gara_temp=X[['GarageYrBlt','GarageFinish','GarageType','GarageCars','GarageArea','GarageQual','GarageCond','YearBuilt']]
Gara_temp=Gara_temp[pd.isnull(Gara_temp['GarageYrBlt'])]
Gara_temp.shape
Gara_temp.head()


# In[ ]:


X['GarageYrBlt']=X['GarageYrBlt'].fillna(X['YearBuilt']) #Replace the null values of garage with year built
X.GarageYrBlt.isnull().sum()# Check the sum of null values 


# **`GarageYrBlt` with NaN value actually means that a garage was never built as shown in the above table.`GarageYrBlt` will not affect the sale price but to make the data heatmap clean and to make sure that we did not miss any missing values, we are going to fill the empty data with the YearBuilt Value.**

# ##### Cleaning LotFrontage

# In[ ]:


X.LotFrontage.isnull().sum() #Check null values


# In[ ]:


X.LotFrontage=X.LotFrontage.fillna(X.LotFrontage.median()) #Replace the null values with the median of LotFrontage
X.LotFrontage.isnull().sum() #Check the null values after change 
## Change after to mean and look at the difference


# ##### Cleaning KitchenQual

# In[ ]:


X.KitchenQual.isnull().sum() #Getting the null values


# In[ ]:


X.MasVnrType.value_counts() #Getting the values in MasVnrType


# In[ ]:


X.KitchenQual=X.KitchenQual.fillna(X.KitchenQual.mode()[0]) #Filling the missing value with the mode
# X.KitchenQual.mode()[0]


# In[ ]:


X.KitchenQual.isnull().sum() #Getting the null values


# In[ ]:


X.KitchenQual.value_counts() #Getting the values of KitchenQual


# ##### Cleaning MasVnrType

# In[ ]:


X.MasVnrType.isnull().sum() #Getting the sum of null values


# In[ ]:


X.MasVnrType.value_counts() #Getting values of MasVnrType


# In[ ]:


# We are going to take the mode for the missing items. Mode is 'None'
X.MasVnrType=X.MasVnrType.fillna(X.MasVnrType.mode()[0])
#X.MasVnrType=X.MasVnrType.fillna('None')


# In[ ]:


X.MasVnrType.isnull().sum() #Getting the null value


# In[ ]:


X.MasVnrType.value_counts() #Getting the values of MasVnrType


# **MasVnrType is catagorical data which means that the removel of outliers is unnessary**

# ##### Cleaning MasVnrArea

# In[ ]:


X.MasVnrArea.isnull().sum() #Getting the null values 


# In[ ]:


X.MasVnrArea=X.MasVnrArea.fillna(X.MasVnrArea.mean()) #Replacing the null values with the mean


# In[ ]:


X.MasVnrArea.isnull().sum()#Getting the null values


# ##### Cleaning MSZoning

# In[ ]:


X.MSZoning.isnull().sum() #Getting the null values


# In[ ]:


X.MSZoning.value_counts() #Getting the values of MSZoning


# In[ ]:


X.MSZoning = X.MSZoning.fillna(X.MSZoning.mode()[0]) #replacing the null values with the mode
#X.MSZoning=X.MSZoning.fillna('RL')


# In[ ]:


X.MSZoning.isnull().sum()#Getting the null values


# In[ ]:


X.MSZoning.value_counts()#Getting the values of MSZoning


# ##### Cleaning the rest of data

# In[ ]:


#these columns don't have effective numbers of null values therefore we applied this function to filter a fill the null values
#Choosing the desired columns
rest_of_columns = ['BsmtHalfBath','Functional','Utilities','KitchenQual','Exterior1st','Exterior2nd',                  'GarageCars','GarageArea','BsmtFinSF1','Electrical','SaleType','TotalBsmtSF','BsmtUnfSF',                  'BsmtFinSF2','BsmtFullBath']
def column_null_cleaner(list_of_columns):
    for i in list_of_columns: #Iterate over the selected columns
        
        if X[i].dtype == 'float64': #If the value of the column is a float replace the null values with the median
            #print(X[i].dtype)
            #print(X[i].median())
            X[i]=X[i].fillna(X[i].median())
            
        elif X[i].dtype == 'object': #If the values of the column is object replace the null values with mode
            #print(i)
            #print(X[i].dtype)
            #print((X[i].mode()[0]))
            X[i]=X[i].fillna(X[i].mode()[0])

            
                  
        elif X[i].dtype == 'int64': #If the values of the column is int then replace the null values with median
            #print(X[i].dtype)
            #print((X[i].median()))
            X[i]=X[i].fillna(X[i].median())

column_null_cleaner(rest_of_columns)


# In[ ]:


#X.BsmtHalfBath = X.BsmtHalfBath.fillna(X.BsmtHalfBath.mode()[0])
#X.Functional  = X.Functional.fillna('Typ')
#X.Utilities  = X.Utilities.fillna('AllPub')
#X.KitchenQual = X.KitchenQual.fillna('TA')
#X.Exterior1st = X.Exterior1st.fillna('VinylSd')
#X.Exterior2nd = X.Exterior2nd.fillna('VinylSd')
#X.GarageCars = X.GarageCars.fillna(X.GarageCars.mode()[0])
#X.GarageArea = X.GarageArea.fillna(X['GarageArea'].median())
#X.BsmtFinSF1 = X.BsmtFinSF1.fillna(X['BsmtFinSF1'].median())
#X.Electrical = X.Electrical.fillna('SBrkr')
#X.SaleType = X.SaleType.fillna('WD')
#X.TotalBsmtSF = X.TotalBsmtSF.fillna(X['TotalBsmtSF'].median())
#X.BsmtUnfSF = X.BsmtUnfSF.fillna(X['BsmtUnfSF'].median())
#X.BsmtFinSF2 = X.BsmtFinSF2.fillna(X.BsmtFinSF2.mode()[0])
#X.BsmtFullBath = X.BsmtFullBath.fillna(X.BsmtFullBath.mode()[0])


# In[ ]:


histo_grams = X.hist(bins=20, figsize=(20, 15)) #Histogram of columns


# In[ ]:


X.isnull().sum().sort_values(ascending=False).head(35) #Check the null values in all columns


# In[ ]:


missing_heat_map(X)


# **Done with cleaning data**

# ## Exploratory Data Analysis EDA

# ### Checking Outliers

# In[ ]:


# This function will check for outliers and plot the distibutions of data. The input takes the column as string an the dataframe
def no_outlier(Data_column,data_set): 
    import math #Import math
    X = data_set[Data_column] #Set the target column to a new varible
    no_outlier = [] #Make a list of the features values without outliers
    confidence = [] #Make a list to get the 90% confidance intervals
    
    q1 = float(X.describe()['25%']) #Get the first quartile (Using the describe function)
    q3 = float(X.describe()['75%']) #Get the third quartile (Using the describe function)
    iqr = (q3 - q1)*1.5 #Calculate the IQR
    std = float(X.describe()['std']) #Get the standered deviation
    mean = float(X.describe()['mean']) #Get the mean
    lower_limit = mean-(1.645*(std/math.sqrt(len(X)))) #Compute the upper limit of the 90% interval
    higher_limit = mean+(1.645*(std/math.sqrt(len(X)))) #Compute the upper limit of the 90% interval
    
    for total in X: #Loop over the target
        if lower_limit < total < higher_limit: #Get the points within the 90% interval
            confidence.append(total) #Append it to the list
        
        if (q1 - iqr) < (total) < (q3 + iqr): #Get points without the outliers 
            no_outlier.append(total) # Append it to the no_outlier list
        else:
            pass
    print('Tukeys method number of outliers is {}'.format((len(X)-len(sorted(no_outlier)))))
    print('90% confidence interval has {} values between {} and {}'.format(len(sorted(confidence)),round(lower_limit),round(higher_limit)))
    
    #Plot representaions
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(12, 12))
    sns.distplot(X, ax=ax[0,0])
    sns.distplot(no_outlier,color='red', ax=ax[0,1])
    sns.boxplot(X,notch=True,orient='v',ax=ax[1,0])
    sns.boxplot(no_outlier,notch=True,orient='v',color='red',ax=ax[1,1])
    
    fig.suptitle('{}'.format(Data_column), fontsize=24)
    ax[0,0].set_title('Distribution of {}'.format(Data_column), fontsize=12)
    ax[0,1].set_title('Distribution of {} after removing outliers'.format(Data_column), fontsize=10)
    ax[1,0].set_title('Boxplot of {}'.format(Data_column), fontsize=10)
    ax[1,1].set_title('Boxplot of {} after removing outliers'.format(Data_column), fontsize=10)
    


# In[ ]:


no_outlier('LotFrontage',X)


# **After analyzing the outlier distribution we relized that we can't use Tukey's method for outline detection because after applying it on the 'LotFrontage' column we got 275 outliers which is alot of data being lost. Therefore we will keep the distribution of data the way it is.**

# In[ ]:


no_outlier('MasVnrArea',X) #Applying the function to understand the distribution


# **After analyzing the outlier distribution we relized that we can't use Tukey's method for outline detection because after applying it on the 'MasVnrArea' column we got 202 outliers which is alot of data being lost. Therefore we will keep the distribution of data the way it is.**

# ### Checking and Fixing Data Types

# In[ ]:


house_train.dtypes


# In[ ]:


#Converting MSSubClass, OverallQual,OverallCond to objects (because they are classifications represented as floats)
X.MSSubClass=X.MSSubClass.map(lambda x : str(x))
X.OverallQual=X.OverallQual.map(lambda x : str(x))
X.OverallCond=X.OverallCond.map(lambda x : str(x))


# ##### Data Engineering

# In[ ]:


X.shape


# In[ ]:


# Adding the Building Age when it was sold. This can be done by subtracting YrSold-YearBuilt and add month sold
X['BuildingAge']=X.YrSold-X.YearBuilt 


# In[ ]:


# Adding Age Since Remodel 
X['RemodelAge']=X.YrSold-X.YearRemodAdd 
X[['RemodelAge']].head(5)


# In[ ]:


X.shape


# ### Creating Dummy 

# In[ ]:


z = pd.get_dummies(X,drop_first=True) #Converting the data into dummies to apply to models
z.head()


# In[ ]:


z.shape


# ### Train-Test Split Data back

# In[ ]:


#Split based on index

X_train=z.iloc[0:1460,:] #Spliting the train 
X_test=z.iloc[1460:,:] #Spliting the test

y_train=y


# In[ ]:


# Drop the ID column from the X_train and X_test
X_train.drop('Id',axis=1,inplace=True) 
X_test.drop('Id',axis=1,inplace=True)


# In[ ]:


X_train_corr=X_train.copy()
X_train_corr['SalesPrice']=y #Set the target to the Salesprice in X_train_corr


# In[ ]:


X_train_corr.head()


# ### Doing Correlation

# In[ ]:


#Grab a list of 20 best positive features based on pairwise correlation
best_feature_corr=X_train_corr.corr()['SalesPrice'].sort_values(ascending=False).index[1:20].tolist()
print('list of 20 best positive features based on pairwise correlation:\n',best_feature_corr)


# In[ ]:


'SalesPrice' in best_feature_corr


# In[ ]:


best_feature_corr.append('SalesPrice')


# In[ ]:


'SalesPrice' in best_feature_corr


# In[ ]:


#Make a heat map to visulize the corrilations
plt.figure(figsize=(18,8))
corr=X_train_corr[best_feature_corr].corr()
sns.heatmap(corr,annot=True, vmin=0, vmax=1, cmap = 'rainbow')


# **From the heat map we concluded that the top corrilated features with `SalesPrice` are ,`GrlivArea`,`GarageCars`,`GarageArea`,`TotalBsmtSF`**

# In[ ]:


X_train_corr[['TotalBsmtSF','1stFlrSF']].head(10) #Investigate 'TotalBsmtSF' and '1stFlrSF'


# **After checking `TotalBsmtSF` and `1stFlrSF` we concluded that these features are extremly similar. The only difference is wether the house has a basment or not**

# In[ ]:


X_train_corr[['GarageArea','GarageCars']] #Investigate 'GarageArea' and 'GarageCars'
grouped_cars = X_train_corr.groupby(['GarageCars'])
grouped_cars['GarageArea'].mean()


# **Analyzing the relationship between `GarageArea`,`GarageCars` It seems that one car takes around 300 square meters The number of cars that fit into the garage is a consequence of the garage area. Therefore we don't need both features since the both convay the same thing.**

# In[ ]:


# Investigate the Correlation with of the features to SalesPrice

cmap=sns.diverging_palette(5, 250, as_cmap=True)
corr = X_train_corr.corr()[['SalesPrice']].head(15)
def magnify():
    
    return [dict(selector="th",
                 props=[("font-size", "7pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "12pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '200px'),
                        ('font-size', '12pt')])
]
corr.style.background_gradient(cmap, axis=1)    .set_properties(**{'max-width': '80px', 'font-size': '10pt'})    .set_caption("Hover to magify")    .set_precision(2)    .set_table_styles(magnify())


# In[ ]:


correlation = X_train_corr.corr(method='pearson')
columns_larg = correlation.nlargest(15,'SalesPrice')[['SalesPrice']].head(15)
columns_larg #Grabbing the largest corrilations (Positive)


# In[ ]:


correlation = X_train_corr.corr(method='pearson')
columns_smal = correlation.nsmallest(15,'SalesPrice')[['SalesPrice']].head(15)
columns_smal #Grabbing the largest corrilations (Neagitive)


# In[ ]:


#Making scatter plots for viewing corrilations
sns.set()
cols = ['SalesPrice', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(X_train_corr[cols], size = 2.5)
plt.show();


# In[ ]:


#Positive corr(red) , Negitive Corr(blue) 
fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(12, 12)) #Intoduce a figure that includes 4 graphs

#plot the graph and set to axs
sns.regplot('GrLivArea', 'SalesPrice', data=X_train_corr, fit_reg=False,color='red', ax=ax[0,0]) 
sns.regplot('GarageArea', 'SalesPrice', data=X_train_corr, fit_reg=False,color='red', ax=ax[0,1])
sns.regplot('YearBuilt', 'SalesPrice', data=X_train_corr, fit_reg=False,color='red', ax=ax[1,0])
sns.regplot('BuildingAge', 'SalesPrice', data=X_train_corr, fit_reg=False,color='blue', ax=ax[1,1])
sns.regplot('YearRemodAdd', 'SalesPrice', data=X_train_corr, fit_reg=False,color='red', ax=ax[2,0])
sns.regplot('RemodelAge', 'SalesPrice', data=X_train_corr, fit_reg=False,color='blue', ax=ax[2,1])



plt.show()


# ### Rescale the variables

# In[ ]:


scaler = StandardScaler() #Call the StandardScaler
scaler.fit(X_train) #Fit x_train to the scaler


# In[ ]:


import joblib # import joblib 

# Saving the transformation, a good ML practice
joblib.dump(scaler, 'scaling_transformation.pkl')
print('transformation saved as scaling_transformation.pkl')


# In[ ]:


# Loading saved transformation 
scaler = joblib.load('scaling_transformation.pkl') 
print('Saved transformation in loaded.')


# In[ ]:


# transforming features 
X_train_ss = pd.DataFrame(scaler.transform(X_train),columns=X_train.columns)
X_test_ss = pd.DataFrame(scaler.transform(X_test),columns=X_test.columns)
print("scaled features are in 'X_train_ss' and 'X_test_ss'")

# X_train_ss = scaler.transform(X_train)
# X_test_ss = scaler.transform(X_test)


# In[ ]:


(len(X_train_ss),len(X_test_ss)) #Compairing the shape of test and train


# In[ ]:


#Make month map to try and understand the data
month_map = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 
             7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'} 
X_train_corr['Month_sold'] = X_train_corr['MoSold'].map(month_map)


# In[ ]:


fig = plt.figure(figsize=(14,8)) #introduce a figure and set the figure size
ax = fig.gca() # create the axis
sns.barplot(x="Month_sold", y="SalesPrice",hue='YrSold',data=X_train_corr,ax=ax,ci=None) #plot the graph
ax.set_title('Month_sold/SalesPrice')# Set the title


# ## Fit the model

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression,RidgeCV, LassoCV, ElasticNetCV
from sklearn.neighbors import KNeighborsRegressor


# #### Linear Regression

# In[ ]:


lm = LinearRegression()
lm.fit(X_train_ss,y_train)

print('Training Score :' , lm.score(X_train_ss,y_train))


# #### Ridge CV

# In[ ]:


ridge_cv = RidgeCV(cv=5)
ridge_cv.fit(X_train_ss, y_train)

print('Training Score :' , ridge_cv.score(X_train_ss , y_train))


# #### Lasso

# In[ ]:


lasso_alphas = np.arange(1,200, 0.05)
lasso_cv = LassoCV(cv=5,alphas=lasso_alphas)
lasso_cv.fit(X_train_ss, y_train)

print('Training Score :' , lasso_cv.score(X_train_ss , y_train))


# #### Elastic

# In[ ]:


elastic_cv = ElasticNetCV(cv=5)
elastic_cv.fit(X_train_ss, y_train)

print('Training Score :' , elastic_cv.score(X_train_ss , y_train))


# #### Random Forest

# In[ ]:


forest_cv = RandomForestRegressor(n_jobs=-1)
forest_cv.fit(X_train_ss,y_train)

print('Train Score :' , forest_cv.score(X_train_ss , y_train))


# #### KNN Regressor

# In[ ]:


param_grid = {'n_neighbors' : [3,4,5,6,7,8,9,10,15]}

grid_knn = GridSearchCV(KNeighborsRegressor(), param_grid, cv=5, refit=True, verbose=1)
grid_knn.fit(X_train_ss, y_train)
print('Training Score :' , grid_knn.score(X_train_ss , y_train))


# ### The best prediction

# ### After doing all the models above, we got the best result with Random Forest (0.9708189495134608). We are submitting this result to the kaggle site.

# In[ ]:


y_test=forest_cv.predict(X_test_ss)
Sub1 = [x for x in range (1461,2920)]
Submission1 = {'Id':Sub1,
               'SalePrice':y_test}
df_submission = pd.DataFrame(Submission1)
df_submission.to_csv('submission',index=False)

