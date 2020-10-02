#!/usr/bin/env python
# coding: utf-8

# **work in progress. will remove this line when it is ready**
# 
# Thank you for opening this script!
# 
# I have made all efforts to document each and every step involved in the prediction process so that this notebook acts as a good starting point for new Kagglers and new machine learning enthusiasts.
# 
# Please **upvote** this kernel so that it reaches the top of the chart and is easily locatable by new users. Your comments on how we can improve this kernel is welcome. Thanks.
# 
# My other exploratory studies can be accessed here :
# https://www.kaggle.com/sharmasanthosh/kernels
# ***
# ## Data statistics
# * Shape
# * Peek
# * Description
# * Skew
# 
# ## Transformation
# * Correction of skew
# 
# ## Data Interaction
# * Correlation
# * Scatter plot
# 
# ## Data Visualization
# * Box and density plots
# * Grouping of one hot encoded attributes
# 
# ## Data Preparation
# * One hot encoding of categorical data
# * Test-train split
# 
# ## Evaluation, prediction, and analysis
# * Linear Regression (Linear algo)
# * Ridge Regression (Linear algo)
# * LASSO Linear Regression (Linear algo)
# * Elastic Net Regression (Linear algo)
# * KNN (non-linear algo)
# * CART (non-linear algo)
# * SVM (Non-linear algo)
# * Bagged Decision Trees (Bagging)
# * Random Forest (Bagging)
# * Extra Trees (Bagging)
# * AdaBoost (Boosting)
# * Stochastic Gradient Boosting (Boosting)
# * MLP (Deep Learning)
# * XGBoost
# 
# ## Make Predictions
# ***

# ## Load raw data:
# 
# Information about all the attributes can be found here:
# 
# https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
# 
# Open data_description.txt for full details about all the attributes
# 
# Learning: 
# We need to predict the 'SalePrice' which is a continuous attribute. Hence, this is a regression problem.

# In[ ]:


# Supress unnecessary warnings so that the presentation looks clean
import warnings
warnings.filterwarnings('ignore')

# Read raw data from the file

import pandas #provides data structures to quickly analyze data
import numpy
#Since this code runs on Kaggle server, data can be accessed directly in the 'input' folder
#Read the train dataset
dataset_train = pandas.read_csv("../input/train.csv") 
dataset_test = pandas.read_csv("../input/test.csv") 

#Print all rows and columns. Dont hide any
pandas.set_option('display.max_rows', None)
pandas.set_option('display.max_columns', None)

#Display the first five rows to get a feel of the data
print(dataset_train.head(5))

#Learning:
#Data has a mix of continuous, categorical, ordinal, date, and discrete attributes
#NaN means that the data is missing


# ## Data statistics
# * Shape

# In[ ]:


# Size of the dataframe

print(dataset_train.shape)
print(dataset_test.shape)
# We can see that there are 1460 instances having 81 attributes in train and 1459 instances in test


# ## Data statistics
# * Skew

# In[ ]:


print(dataset_train.skew())
#Attribute LotArea is highly skewed. It could be corrected using log transform


# ## Feature types
# * Create separate list of column names for continuous and categorical attributes

# In[ ]:


#Constants
MED = 0  #median
ENC = 1  #encoded
SIZE = [MED, ENC] #list of types
SIZE_STR = ['Med','Enc']
DUMMY_STR = 'Dummy'

#Stores both list of median cols and list of encoded cols
dataset_train_list = [pandas.DataFrame(),pandas.DataFrame()]
dataset_test_list = [pandas.DataFrame(),pandas.DataFrame()]


# ## Feature description and feature engineering
# * Function definition (general function used to analyze all attributes)

# In[ ]:


#import plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt

#import for Imputing missing values
from sklearn.preprocessing import Imputer

#import for one hot encoding
from sklearn.feature_extraction import DictVectorizer

#Column name of target
target = 'SalePrice'

def analyse(var,cat):
    
    if (cat == "categorical"):        

        # Imputer
        # Convert NA to the word 'DUMMY_STR' so that the median on 'target' is used
        dataset_train[var].fillna(inplace=True,value=DUMMY_STR)
        dataset_test[var].fillna(inplace=True,value=DUMMY_STR)
        
        #Obtain the labels
        labels = numpy.array(dataset_train[var].dropna().unique())
        
        #Rotate alignment of labels if there are too many to prevent overlap
        if(len(labels) < 12):
            # Plot the attribute against target
            sns.boxplot(y=target, x=var, data=dataset_train)    
        
        else:
            fig, ax = plt.subplots()
            #plot
            sns.boxplot(y=target, x=var, data=dataset_train)    
            #Sort as the plot is already sorted
            labels = sorted(labels)
            #Rotate vertical if there are too many labels which result in overlap
            ax.set_xticklabels(labels,rotation='vertical')
            plt.show()

        # Hypothesis
        # Nominal values must be transformed to numerical values. We can transform the nominal value using 
        # one hot encoding or we can replace the nominal value with the median value of the target in the boxplot
        
        # Word of caution
        # Ideally, the median must be taken only for the 90% of the train  dataset as it may potentially 
        # cause some data leakage into the validation set. 
        # But to maintain simiplicity of the presentation, this is overlooked

        # Obtain the median value of target for each class
        medians = dataset_train.groupby(var)[target].median()
        #print("\n\nMedians for")
        #print(medians)

        # Add median column name
        var_median = var+"_median"

        # Map from label to median
        dataset_train_list[MED][var_median] = dataset_train[var].map(medians)
        dataset_test_list[MED][var_median] = dataset_test[var].map(medians)
        #print(dataset_train_list[MED][var_median])
        #print(dataset_test_list[MED][var_median])

        # One-hot encoding using DictVectorizer

        # Obtain column name, value pairs (covert numerical into string)
        vals_train = dataset_train[var].apply(lambda x : {var : "%s" % x} )
        vals_test = dataset_test[var].apply(lambda x : {var : "%s" % x} )

        # Create Dict Vectorizer class
        dv = DictVectorizer(sparse=False)
        # Concatenate is used to ensure all labels in both test and train are used
        dv.fit(numpy.concatenate((vals_train,vals_test),axis=0))

        # Perform one-hot encoding
        new_data_train = dv.transform(vals_train)
        new_data_test = dv.transform(vals_test)

        # Obtain column names
        new_cols = dv.get_feature_names()

        # Add new columns
        for i, col in enumerate(new_cols):
            dataset_train_list[ENC][col] = new_data_train[:,i]
            dataset_test_list[ENC][col] = new_data_test[:,i]
            #print(dataset_train_list[ENC][col])
            #print(dataset_test_list[ENC][col])
    
    else:
        if(cat == "target"):
            for s in SIZE:
                dataset_train_list[s][var] = dataset_train[var]
                #print(dataset_train_list[s][var])
            
            sns.violinplot(data=dataset_train, size=7, y=var)    
        else:    
            if(cat == "date"):
                var_orig = var
                var = var+"_time"
                # Convert year into time until 'YrSold' and fill NaN with median
                for s in SIZE:
                    dataset_train_list[s][var] = Imputer(strategy='median',axis=1).fit_transform(dataset_train["YrSold"] - dataset_train[var_orig]).transpose()
                    dataset_test_list[s][var] = Imputer(strategy='median',axis=1).fit_transform(dataset_test["YrSold"] - dataset_test[var_orig]).transpose()
                    #print(dataset_train_list[s][var])
                    #print(dataset_test_list[s][var])
                
            elif(cat == "skewed"):
                var_orig = var
                var = var+"_log"
                # Log transform to correct skew , if NaN fill with median
                for s in SIZE:
                    dataset_train_list[s][var] = Imputer(strategy='median',axis=1).fit_transform(numpy.log1p(dataset_train[var_orig])).transpose()
                    dataset_test_list[s][var] = Imputer(strategy='median',axis=1).fit_transform(numpy.log1p(dataset_test[var_orig])).transpose()
                    #print(dataset_train_list[s][var])
                    #print(dataset_test_list[s][var])
                
            elif(cat == "continuous"):
                #If NaN, fill with median
                for s in SIZE:
                    dataset_train_list[s][var] = Imputer(strategy='median',axis=1).fit_transform(dataset_train[var]).transpose()
                    dataset_test_list[s][var] = Imputer(strategy='median',axis=1).fit_transform(dataset_test[var]).transpose()
                    #print(dataset_train_list[s][var])
                    #print(dataset_test_list[s][var])
                
            #Obtain the correlation for the attribute against target
            col_list = [target,var]
            d_corr = dataset_train_list[MED][col_list].corr().iloc[0,1]
            #KDE plot if good correlation else scatter plot
            if(d_corr > 0.15):
                sns.jointplot(data=dataset_train_list[MED], size=7, x=var,y=target , kind="kde" )
            else:
                sns.jointplot(data=dataset_train_list[MED], size=7, x=var,y=target  )

    #print("Train : Median : ")
    #print(dataset_train_list[MED].shape)
    #print(dataset_train_list[MED].columns)
    #print("Train : Encoded : ")
    #print(dataset_train_list[ENC].shape)
    #print(dataset_train_list[ENC].columns)
    #print("Test : Median : ")
    #print(dataset_test_list[MED].shape)
    #print(dataset_test_list[MED].columns)
    #print("Test : Encoded : ")
    #print(dataset_test_list[ENC].shape)
    #print(dataset_test_list[ENC].columns)


# ## Feature description and feature engineering
# * Id and SalePrice

# In[ ]:


#Id is ignored as it is not useful
analyse(var="SalePrice",cat="target")


# ## Feature description and feature engineering
# * MSSubClass

# In[ ]:


analyse(var="MSSubClass",cat="categorical")


# ## Feature description and feature engineering
# * MSZoning

# In[ ]:


analyse(var="MSZoning",cat="categorical")


# ## Feature description and feature engineering
# * LotFrontage

# In[ ]:


analyse(var="LotFrontage",cat="continuous")


# ## Feature description and feature engineering
# * LotArea

# In[ ]:


analyse(var="LotArea",cat="skewed")


# ## Feature description and feature engineering
# * Street

# In[ ]:


analyse(var="Street",cat="categorical")


# ## Feature description and feature engineering
# * Alley

# In[ ]:


analyse(var="Alley",cat="categorical")


# ## Feature description and feature engineering
# * LotShape

# In[ ]:


analyse(var="LotShape",cat="categorical")


# ## Feature description and feature engineering
# * LandContour

# In[ ]:


analyse(var="LandContour",cat="categorical")


# ## Feature description and feature engineering
# * Utilities

# In[ ]:


analyse(var="Utilities",cat="categorical")


# ## Feature description and feature engineering
# * LotConfig

# In[ ]:


analyse(var="LotConfig",cat="categorical")


# ## Feature description and feature engineering
# * LandSlope

# In[ ]:


analyse(var="LandSlope",cat="categorical")


# ## Feature description and feature engineering
# * Neighborhood

# In[ ]:


analyse(var="Neighborhood",cat="categorical")


# ## Feature description and feature engineering
# * Condition1

# In[ ]:


analyse(var="Condition1",cat="categorical")


# ## Feature description and feature engineering
# * Condition2

# In[ ]:


analyse(var="Condition2",cat="categorical")


# ## Feature description and feature engineering
# * BldgType

# In[ ]:


analyse(var="BldgType",cat="categorical")


# ## Feature description and feature engineering
# * HouseStyle

# In[ ]:


analyse(var="HouseStyle",cat="categorical")


# ## Feature description and feature engineering
# * OverallQual

# In[ ]:


analyse(var="OverallQual",cat="continuous")


# ## Feature description and feature engineering
# * OverallCond

# In[ ]:


analyse(var="OverallCond",cat="continuous")


# ## Feature description and feature engineering
# * YearBuilt

# In[ ]:


analyse(var="YearBuilt",cat="date")


# ## Feature description and feature engineering
# * YearRemodAdd

# In[ ]:


analyse(var="YearRemodAdd",cat="date")


# ## Feature description and feature engineering
# * RoofStyle

# In[ ]:


analyse(var="RoofStyle",cat="categorical")


# ## Feature description and feature engineering
# * RoofMatl

# In[ ]:


analyse(var="RoofMatl",cat="categorical")


# ## Feature description and feature engineering
# * Exterior1st

# In[ ]:


analyse(var="Exterior1st",cat="categorical")


# ## Feature description and feature engineering
# * Exterior2nd

# In[ ]:


analyse(var="Exterior2nd",cat="categorical")


# ## Feature description and feature engineering
# * MasVnrType

# In[ ]:


analyse(var="MasVnrType",cat="categorical")


# ## Feature description and feature engineering
# * MasVnrArea

# In[ ]:


analyse(var="MasVnrArea",cat="continuous")


# ## Feature description and feature engineering
# * ExterQual

# In[ ]:


analyse(var="ExterQual",cat="categorical")


# ## Feature description and feature engineering
# * ExterCond

# In[ ]:


analyse(var="ExterCond",cat="categorical")


# ## Feature description and feature engineering
# * Foundation

# In[ ]:


analyse(var="Foundation",cat="categorical")


# ## Feature description and feature engineering
# * BsmtQual

# In[ ]:


analyse(var="BsmtQual",cat="categorical")


# ## Feature description and feature engineering
# * BsmtCond

# In[ ]:


analyse(var="BsmtCond",cat="categorical")


# ## Feature description and feature engineering
# * BsmtExposure

# In[ ]:


analyse(var="BsmtExposure",cat="categorical")


# ## Feature description and feature engineering
# * BsmtFinType1

# In[ ]:


analyse(var="BsmtFinType1",cat="categorical")


# ## Feature description and feature engineering
# * BsmtFinSF1

# In[ ]:


analyse(var="BsmtFinSF1",cat="continuous")


# ## Feature description and feature engineering
# * BsmtFinType2

# In[ ]:


analyse(var="BsmtFinType2",cat="categorical")


# ## Feature description and feature engineering
# * BsmtFinSF2

# In[ ]:


analyse(var="BsmtFinSF2",cat="continuous")


# ## Feature description and feature engineering
# * BsmtUnfSF

# In[ ]:


analyse(var="BsmtUnfSF",cat="continuous")


# ## Feature description and feature engineering
# * TotalBsmtSF

# In[ ]:


analyse(var="TotalBsmtSF",cat="continuous")


# ## Feature description and feature engineering
# * Heating

# In[ ]:


analyse(var="Heating",cat="categorical")


# ## Feature description and feature engineering
# * HeatingQC

# In[ ]:


analyse(var="HeatingQC",cat="categorical")


# ## Feature description and feature engineering
# * CentralAir

# In[ ]:


analyse(var="CentralAir",cat="categorical")


# ## Feature description and feature engineering
# * Electrical

# In[ ]:


analyse(var="Electrical",cat="categorical")


# ## Feature description and feature engineering
# * 1stFlrSF

# In[ ]:


analyse(var="1stFlrSF",cat="continuous")


# ## Feature description and feature engineering
# * 2ndFlrSF

# In[ ]:


analyse(var="2ndFlrSF",cat="continuous")


# ## Feature description and feature engineering
# * LowQualFinSF

# In[ ]:


analyse(var="LowQualFinSF",cat="continuous")


# ## Feature description and feature engineering
# * GrLivArea

# In[ ]:


analyse(var="GrLivArea",cat="continuous")


# ## Feature description and feature engineering
# * BsmtFullBath

# In[ ]:


analyse(var="BsmtFullBath",cat="continuous")


# ## Feature description and feature engineering
# * BsmtHalfBath

# In[ ]:


analyse(var="BsmtHalfBath",cat="continuous")


# ## Feature description and feature engineering
# * FullBath

# In[ ]:


analyse(var="FullBath",cat="continuous")


# ## Feature description and feature engineering
# * HalfBath

# In[ ]:


analyse(var="HalfBath",cat="continuous")


# ## Feature description and feature engineering
# * BedroomAbvGr

# In[ ]:


analyse(var="BedroomAbvGr",cat="continuous")


# ## Feature description and feature engineering
# * KitchenAbvGr

# In[ ]:


analyse(var="KitchenAbvGr",cat="continuous")


# ## Feature description and feature engineering
# * KitchenQual

# In[ ]:


analyse(var="KitchenQual",cat="categorical")


# ## Feature description and feature engineering
# * TotRmsAbvGrd

# In[ ]:


analyse(var="TotRmsAbvGrd",cat="continuous")


# ## Feature description and feature engineering
# * Functional

# In[ ]:


analyse(var="Functional",cat="categorical")


# ## Feature description and feature engineering
# * Fireplaces

# In[ ]:


analyse(var="Fireplaces",cat="continuous")


# ## Feature description and feature engineering
# * FireplaceQu

# In[ ]:


analyse(var="FireplaceQu",cat="categorical")


# ## Feature description and feature engineering
# * GarageType

# In[ ]:


analyse(var="GarageType",cat="categorical")


# ## Feature description and feature engineering
# * GarageYrBlt

# In[ ]:


analyse(var="GarageYrBlt",cat="date")


# ## Feature description and feature engineering
# * GarageFinish

# In[ ]:


analyse(var="GarageFinish",cat="categorical")


# ## Feature description and feature engineering
# * GarageCars

# In[ ]:


analyse(var="GarageCars",cat="continuous")


# ## Feature description and feature engineering
# * GarageArea

# In[ ]:


analyse(var="GarageArea",cat="continuous")


# ## Feature description and feature engineering
# * GarageQual

# In[ ]:


analyse(var="GarageQual",cat="categorical")


# ## Feature description and feature engineering
# * GarageCond

# In[ ]:


analyse(var="GarageCond",cat="categorical")


# ## Feature description and feature engineering
# * PavedDrive

# In[ ]:


analyse(var="PavedDrive",cat="categorical")


# ## Feature description and feature engineering
# * WoodDeckSF

# In[ ]:


analyse(var="WoodDeckSF",cat="continuous")


# ## Feature description and feature engineering
# * OpenPorchSF

# In[ ]:


analyse(var="OpenPorchSF",cat="continuous")


# ## Feature description and feature engineering
# * EnclosedPorch

# In[ ]:


analyse(var="EnclosedPorch",cat="continuous")


# ## Feature description and feature engineering
# * 3SsnPorch

# In[ ]:


analyse(var="3SsnPorch",cat="continuous")


# ## Feature description and feature engineering
# * ScreenPorch

# In[ ]:


analyse(var="ScreenPorch",cat="continuous")


# ## Feature description and feature engineering
# * PoolArea

# In[ ]:


analyse(var="PoolArea",cat="continuous")


# ## Feature description and feature engineering
# * PoolQC

# In[ ]:


analyse(var="PoolQC",cat="categorical")


# ## Feature description and feature engineering
# * Fence

# In[ ]:


analyse(var="Fence",cat="categorical")


# ## Feature description and feature engineering
# * MiscFeature

# In[ ]:


analyse(var="MiscFeature",cat="categorical")


# ## Feature description and feature engineering
# * MiscVal

# In[ ]:


analyse(var="MiscVal",cat="continuous")


# ## Feature description and feature engineering
# * MoSold

# In[ ]:


analyse(var="MoSold",cat="categorical")


# ## Feature description and feature engineering
# * YrSold

# In[ ]:


analyse(var="YrSold",cat="categorical")


# ## Feature description and feature engineering
# * SaleType

# In[ ]:


analyse(var="SaleType",cat="categorical")


# ## Feature description and feature engineering
# * SaleCondition

# In[ ]:


analyse(var="SaleCondition",cat="categorical")


# ## Data Interaction
# * Correlation

# In[ ]:


# Correlation tells relation between two attributes.

# Calculates pearson co-efficient for all combinations
data_corr = dataset_train_list[MED].corr()

# Set the threshold to select only highly correlated attributes
threshold = 0.68

# List of pairs along with correlation above threshold
corr_list = []

#Length of the list
med_cols = dataset_train_list[MED].columns
size = len(med_cols)

#Search for the highly correlated pairs
for i in range(0,size): #for 'size' features
    for j in range(i+1,size): #avoid repetition
        if (data_corr.iloc[i,j] >= threshold and data_corr.iloc[i,j] < 1) or (data_corr.iloc[i,j] < 0 and data_corr.iloc[i,j] <= -threshold):
            corr_list.append([data_corr.iloc[i,j],i,j]) #store correlation and columns index

#Sort to show higher ones first            
s_corr_list = sorted(corr_list,key=lambda x: -abs(x[0]))

#Print correlations and column names
for v,i,j in s_corr_list:
    print ("%s and %s = %.2f" % (med_cols[i],med_cols[j],v))

# Strong correlation is observed between the following pairs


# ## Data Interaction
# * Scatter plot

# In[ ]:


# Scatter plot of only the highly correlated pairs
for v,i,j in s_corr_list:
    sns.jointplot(data=dataset_train_list[MED], size=7, x=med_cols[i],y=med_cols[j] , kind="kde" )
    plt.show()


# ## Data Preparation
# * Split train dataset for estimating performance
# * Original
# * StandardScaler
# * MinMaxScaler
# * Normalizer

# In[ ]:


#Import libraries for data transformations
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
    
#All features
X_all = [[],[]]
X_all_add = [[],[]]

#List of combinations
comb = [[],[]]

#Split the data into two chunks
from sklearn import cross_validation
    
#Validation chunk size
val_size = 0.1

#Use a common seed in all experiments so that same chunk is used for validation
seed = 0

for s in SIZE:
    #Extract only the values 
    array = dataset_train_list[s].values

    #Shape of the dataset
    r , c = dataset_train_list[s].shape
    print(dataset_train_list[s].shape)
    
    #Y is the target column, X has the rest
    X = array[:,1:]
    Y = array[:,0]

    #Split X and Y
    X_train, X_val, Y_train, Y_val = cross_validation.train_test_split(X, Y, test_size=val_size, random_state=seed)
    
    #create an array which has indexes of columns for X
    i_cols = []
    for i in range(0,c-1):
        i_cols.append(i)
    #print("i_cols")
    #print(i_cols)
    
    #create an array which has names of columns for X
    cols = dataset_train_list[s].columns[1:]    
    #print(cols)
    
    #Add the original version of X which is not transformed to the list
    n = 'All'
    i_bin =[]
    X_all[s].append(['Orig',n, X_train,X_val,Y_train,Y_val,cols,i_cols,i_cols,i_bin])

    #create separate array for binary and continuous columns for X
    i_cols_binary = []
    i_cols_conti = []
    cols_binary = []
    cols_conti = []

    for i,col in enumerate(cols):
        #Presence of '=' means that the column is binary
        if('=' in col):
            i_cols_binary.append(i)
            cols_binary.append(col)
        else:
            i_cols_conti.append(i)
            cols_conti.append(col)

    #print('i_cols_conti')
    #print(i_cols_conti)
    #print('cols_conti')
    #print(cols_conti)
    #print('i_cols_binary')
    #print(i_cols_binary)
    #print('cols_binary')
    #print(cols_binary)
        
    #Preprocessing list
    prep_list = [('StdSca',StandardScaler()),('MinMax',MinMaxScaler()),('Norm',Normalizer())]
    for name, prep in prep_list:
        #Prevent data leakage by applying the transforms separately for X_train and X_val
        #Apply transform only for non-categorical data
        X_temp = prep.fit_transform(X_train[:,i_cols_conti])
        X_val_temp = prep.fit_transform(X_val[:,i_cols_conti])
        #Concatenate non-categorical data and categorical
        X_con = numpy.concatenate((X_temp,X_train[:,i_cols_binary]),axis=1)
        X_val_con = numpy.concatenate((X_val_temp,X_val[:,i_cols_binary]),axis=1)
        #Column name location would have changed. Hence overwrite 
        cols = numpy.concatenate((cols_conti,cols_binary),axis=0)
        #pandas.DataFrame(data=X_con,columns=cols).to_csv("trans%sType%sTrain.csv" % (name,s))
        #pandas.DataFrame(data=X_val_con,columns=cols).to_csv("trans%sType%sVal.csv" % (name,s))
        #Add this version of X to the list 
        X_all[s].append([name,n, X_con,X_val_con,Y_train,Y_val,cols,i_cols,i_cols_conti,i_cols_binary])

#print(X_all)        


# ## Feature importance
# * ExtraTreesClassifier

# In[ ]:


# % of features to select for median and enc
ratio_list = [[0.25],[]]


# In[ ]:


#Feature selection only for median
for s in range(1):
    #List of feature selection models
    feat = []

    #List of names of feature selection models
    feat_list =[]

    #Import the libraries
    from sklearn.ensemble import ExtraTreesClassifier

    #Add ExtraTreeClassifiers to the list
    n = 'ExTree'
    feat_list.append(n)
    for val in ratio_list[s]:
        feat.append([n,val,ExtraTreesClassifier(n_estimators=100,max_features=val,n_jobs=-1,random_state=seed)])      

    #For all transformations of X
    for trans,n, X, X_val,Y_train,Y_val, cols, i_cols, conti,binr in X_all[s]:
        #For all feature selection models
        for name,v, model in feat:
            #Train the model against Y
            model.fit(X,Y_train)
            #Combine importance and index of the column in the array joined
            joined = []
            for i, pred in enumerate(list(model.feature_importances_)):
                joined.append([i,cols[i],pred])
            #Sort in descending order    
            joined_sorted = sorted(joined, key=lambda x: -x[2])
            #Starting point of the columns to be dropped
            rem_start = int(v*(len(cols)))
            #List of names of columns selected
            cols_list = []
            #Indexes of columns selected
            i_cols_list = []
            #Ranking of all the columns
            rank_list =[]
            #Split the array. Store selected columns in cols_list and removed in rem_list
            for j, (i, col, x) in enumerate(list(joined_sorted)):
                #Store the rank
                rank_list.append([i,j])
                #Store selected columns in cols_list and indexes in i_cols_list
                if(j < rem_start):
                    cols_list.append(col)
                    i_cols_list.append(i)
            #Sort the rank_list and store only the ranks. Drop the index 
            #Append model name, array, columns selected to the additional list        
            X_all_add[s].append([trans,name,X,X_val,Y_train,Y_val,cols_list,[x[1] for x in sorted(rank_list,key=lambda x:x[0])],i_cols_list,cols])    

    #Set figure size
    plt.rc("figure", figsize=(12, 10))

    #Plot a graph for different feature selectors        
    for f_name in feat_list:
        #Array to store the list of combinations
        leg=[]
        fig, ax = plt.subplots()
        #Plot each combination
        for trans,name,X,X_val,Y,Y_val,cols_list,rank_list,i_cols_list,cols in X_all_add[s]:
            if(name==f_name):
                plt.plot(rank_list)
                leg.append(trans+"+"+name)
        #Set the tick names to names of columns
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols,rotation='vertical')
        #Display the plot
        plt.legend(leg,loc='upper left')    
        #Plot the rankings of all the features for all combinations
        plt.show()


# ## Evaluation, prediction, and analysis
# * Setup

# In[ ]:


import math

#Dictionary to store the RMSE for all algorithms 
mse = [[],[]]

#Scoring parameter
from sklearn.metrics import mean_squared_error


# ## Evaluation, prediction, and analysis
# * Ridge Regression (Linear algo)

# In[ ]:


#Evaluation of various combinations of LinearRegression

#Import the library
from sklearn.linear_model import Ridge

algo = "Ridge"

#Add the alpha value to the below list if you want to run the algo
a_list = numpy.array([0.1])


for s in SIZE:
    for alpha in a_list:
        #Accuracy of the model using all features
        for name,fe, X_train, X_val,Y_train,Y_val, cols, i_cols_list, i_con, i_bin in X_all[s]:
            if(name == 'MinMax'):
                continue
            print(name)
            #Set the base model
            model = Ridge(random_state=seed)
            model.fit(X_train,Y_train)
            result = math.sqrt(mean_squared_error(numpy.log1p(Y_val), numpy.log1p(model.predict(X_val))))
            mse[s].append(result)
            print(name + " %s" % result)
            comb[s].append(name+" "+algo + " %s" % alpha )
##Plot the MSE of all combinations for both types in the same figure
#fig, ax = plt.subplots()
#for s in SIZE:
#    plt.plot(mse[s])
##Set the tick names to names of combinations
#ax.set_xticks(range(len(comb[s])))
#ax.set_xticklabels(comb[s],rotation='vertical')
##Plot the accuracy for all combinations
#plt.legend(SIZE_STR,loc='best')    
#plt.show()    


# ## Evaluation, prediction, and analysis
# * LASSO Linear Regression (Linear algo)

# In[ ]:


#Evaluation of various combinations of LinearRegression

#Import the library
from sklearn.linear_model import Lasso

algo = "Lasso"

#Add the alpha value to the below list if you want to run the algo
a_list = numpy.array([0.1])

for s in SIZE:
    for alpha in a_list:
        #Accuracy of the model using all features
        for name,fe, X_train, X_val,Y_train,Y_val, cols, i_cols_list, i_con, i_bin in X_all[s]:
            if(name == 'MinMax'):
                continue
            print(name)
            #Set the base model
            model = Lasso(alpha=alpha,random_state=seed)
            model.fit(X_train,Y_train)
            result = math.sqrt(mean_squared_error(numpy.log1p(Y_val), numpy.log1p(model.predict(X_val))))
            mse[s].append(result)
            print(name + " %s" % result)
            comb[s].append(name+" "+algo + " %s" % alpha )
##Plot the MSE of all combinations for both types in the same figure
#fig, ax = plt.subplots()
#for s in SIZE:
#    plt.plot(mse[s])
##Set the tick names to names of combinations
#ax.set_xticks(range(len(comb[s])))
#ax.set_xticklabels(comb[s],rotation='vertical')
##Plot the accuracy for all combinations
#plt.legend(SIZE_STR,loc='best')    
#plt.show()    


# ## Evaluation, prediction, and analysis
# * Elastic Net Regression (Linear algo)

# In[ ]:


#Evaluation of various combinations of ElasticNet

#Import the library
from sklearn.linear_model import ElasticNet

algo = "Elastic"

#Add the alpha value to the below list if you want to run the algo
a_list = numpy.array([0.01])

for s in SIZE:
    for alpha in a_list:
        #Accuracy of the model using all features
        for name,fe, X_train, X_val,Y_train,Y_val, cols, i_cols_list, i_con, i_bin in X_all[s]:
            if(name == 'MinMax'):
                continue
            print(name)
            #Set the base model
            model = ElasticNet(alpha=alpha,random_state=seed)
            model.fit(X_train,Y_train)
            result = math.sqrt(mean_squared_error(numpy.log1p(Y_val), numpy.log1p(model.predict(X_val))))
            mse[s].append(result)
            print(name + " %s" % result)
            comb[s].append(name+" "+algo + " %s" % alpha )
##Plot the MSE of all combinations for both types in the same figure
#fig, ax = plt.subplots()
#for s in SIZE:
#    plt.plot(mse[s])
##Set the tick names to names of combinations
#ax.set_xticks(range(len(comb[s])))
#ax.set_xticklabels(comb[s],rotation='vertical')
##Plot the accuracy for all combinations
#plt.legend(SIZE_STR,loc='best')    
#plt.show()    


# ## Evaluation, prediction, and analysis
# * KNN (non-linear algo)

# In[ ]:


#Evaluation of various combinations of KNN

#Import the library
from sklearn.neighbors import KNeighborsRegressor

algo = "KNN"

#Add the N value to the below list if you want to run the algo
n_list = numpy.array([9])

for s in SIZE:
    for n_neighbors in n_list:

        #Accuracy of the model using all features
        for name,fe, X_train, X_val,Y_train,Y_val, cols, i_cols_list, i_con, i_bin in X_all[s]:
            print(name)
            #Set the base model
            model = KNeighborsRegressor(n_neighbors=n_neighbors,n_jobs=-1)
            model.fit(X_train,Y_train)
            result = math.sqrt(mean_squared_error(numpy.log1p(Y_val), numpy.log1p(model.predict(X_val))))
            mse[s].append(result)
            print(name + " %s" % result)
            comb[s].append(name+" "+algo + " %s" % n_neighbors )
##Plot the MSE of all combinations for both types in the same figure
#fig, ax = plt.subplots()
#for s in SIZE:
#    plt.plot(mse[s])
##Set the tick names to names of combinations
#ax.set_xticks(range(len(comb[s])))
#ax.set_xticklabels(comb[s],rotation='vertical')
##Plot the accuracy for all combinations
#plt.legend(SIZE_STR,loc='best')    
#plt.show()    


# ## Evaluation, prediction, and analysis
# * CART (non-linear algo)

# In[ ]:


#Evaluation of various combinations of CART

#Import the library
from sklearn.tree import DecisionTreeRegressor

algo = "CART"

#Add the max_depth value to the below list if you want to run the algo
d_list = numpy.array([13])

for s in SIZE:
    for max_depth in d_list:
        #Accuracy of the model using all features
        for name,fe, X_train, X_val,Y_train,Y_val, cols, i_cols_list, i_con, i_bin in X_all[s]:
            print(name)
            #Set the base model
            model = DecisionTreeRegressor(max_depth=max_depth,random_state=seed)
            model.fit(X_train,Y_train)
            result = math.sqrt(mean_squared_error(numpy.log1p(Y_val), numpy.log1p(model.predict(X_val))))
            mse[s].append(result)
            print(name + " %s" % result)
            comb[s].append(name+" "+algo + " %s" % max_depth )
##Plot the MSE of all combinations for both types in the same figure
#fig, ax = plt.subplots()
#for s in SIZE:
#    plt.plot(mse[s])
##Set the tick names to names of combinations
#ax.set_xticks(range(len(comb[s])))
#ax.set_xticklabels(comb[s],rotation='vertical')
##Plot the accuracy for all combinations
#plt.legend(SIZE_STR,loc='best')    
#plt.show()    


# ## Evaluation, prediction, and analysis
# * SVM (Non-linear algo)

# In[ ]:


#Evaluation of various combinations of SVM

#Import the library
from sklearn.svm import SVR

algo = "SVM"

#Add the C value to the below list if you want to run the algo
c_list = numpy.array([10000])

for s in SIZE:
    for C in c_list:
        #Accuracy of the model using all features
        for name,fe, X_train, X_val,Y_train,Y_val, cols, i_cols_list, i_con, i_bin in X_all[s]:
            if(name == 'Orig' or name =='Norm'):  #very poor results, spoils the graph
                continue
            print(name)
            #Set the base model
            model = SVR(C=C)
            model.fit(X_train,Y_train)
            result = math.sqrt(mean_squared_error(numpy.log1p(Y_val), numpy.log1p(model.predict(X_val))))
            mse[s].append(result)
            print(name + " %s" % result)
            comb[s].append(name+" "+algo + " %s" % C )
##Plot the MSE of all combinations for both types in the same figure
#fig, ax = plt.subplots()
#for s in SIZE:
#    plt.plot(mse[s])
##Set the tick names to names of combinations
#ax.set_xticks(range(len(comb[s])))
#ax.set_xticklabels(comb[s],rotation='vertical')
##Plot the accuracy for all combinations
#plt.legend(SIZE_STR,loc='best')    
#plt.show()    


# ## Evaluation, prediction, and analysis
# * Bagged Decision Trees (Bagging)

# In[ ]:


#Evaluation of various combinations of Bagged Decision Trees

#Import the library
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

algo = "Bag"

#Add the n_estimators value to the below list if you want to run the algo
n_list = numpy.array([200])

for s in SIZE:
    for n_estimators in n_list:
        #Accuracy of the model using all features
        for name,fe, X_train, X_val,Y_train,Y_val, cols, i_cols_list, i_con, i_bin in X_all[s]:
            print(name)
            #Set the base model
            model = BaggingRegressor(n_jobs=-1,n_estimators=n_estimators)
            model.fit(X_train,Y_train)
            result = math.sqrt(mean_squared_error(numpy.log1p(Y_val), numpy.log1p(model.predict(X_val))))
            mse[s].append(result)
            print(name + " %s" % result)
            comb[s].append(name+" "+algo + " %s" % n_estimators )
##Plot the MSE of all combinations for both types in the same figure
#fig, ax = plt.subplots()
#for s in SIZE:
#    plt.plot(mse[s])
##Set the tick names to names of combinations
#ax.set_xticks(range(len(comb[s])))
#ax.set_xticklabels(comb[s],rotation='vertical')
##Plot the accuracy for all combinations
#plt.legend(SIZE_STR,loc='best')    
#plt.show()    


# ## Evaluation, prediction, and analysis
# * Random Forest (Bagging)

# In[ ]:


#Evaluation of various combinations of RandomForest
    
#Import the library
from sklearn.ensemble import RandomForestRegressor

algo = "RF"

#Add the n_estimators value to the below list if you want to run the algo
n_list = numpy.array([300])

for s in SIZE:
    for n_estimators in n_list:
        #Accuracy of the model using all features
        for name,fe, X_train, X_val,Y_train,Y_val, cols, i_cols_list, i_con, i_bin in X_all[s]:
            print(name)
            #Set the base model
            model = RandomForestRegressor(n_jobs=-1,n_estimators=n_estimators,random_state=seed)
            model.fit(X_train,Y_train)
            result = math.sqrt(mean_squared_error(numpy.log1p(Y_val), numpy.log1p(model.predict(X_val))))
            mse[s].append(result)
            print(name + " %s" % result)
            comb[s].append(name+" "+algo + " %s" % n_estimators )
##Plot the MSE of all combinations for both types in the same figure
#fig, ax = plt.subplots()
#for s in SIZE:
#    plt.plot(mse[s])
##Set the tick names to names of combinations
#ax.set_xticks(range(len(comb[s])))
#ax.set_xticklabels(comb[s],rotation='vertical')
##Plot the accuracy for all combinations
#plt.legend(SIZE_STR,loc='best')    
#plt.show()    


# ## Evaluation, prediction, and analysis
# * Extra Trees (Bagging)

# In[ ]:


#Evaluation of various combinations of ExtraTrees

#Import the library
from sklearn.ensemble import ExtraTreesRegressor

algo = "ET"

#Add the n_estimators value to the below list if you want to run the algo
n_list = numpy.array([300])

for s in SIZE:
    for n_estimators in n_list:
        #Accuracy of the model using all features
        for name,fe, X_train, X_val,Y_train,Y_val, cols, i_cols_list, i_con, i_bin in X_all[s]:
            print(name)
            #Set the base model
            model = ExtraTreesRegressor(n_jobs=-1,n_estimators=n_estimators,random_state=seed)
            model.fit(X_train,Y_train)
            result = math.sqrt(mean_squared_error(numpy.log1p(Y_val), numpy.log1p(model.predict(X_val))))
            mse[s].append(result)
            print(name + " %s" % result)
            comb[s].append(name+" "+algo + " %s" % n_estimators )
##Plot the MSE of all combinations for both types in the same figure
#fig, ax = plt.subplots()
#for s in SIZE:
#    plt.plot(mse[s])
##Set the tick names to names of combinations
#ax.set_xticks(range(len(comb[s])))
#ax.set_xticklabels(comb[s],rotation='vertical')
##Plot the accuracy for all combinations
#plt.legend(SIZE_STR,loc='best')    
#plt.show()    


# ## Evaluation, prediction, and analysis
# * AdaBoost (Boosting)

# In[ ]:


#Evaluation of various combinations of ExtraTrees

#Import the library
from sklearn.ensemble import AdaBoostRegressor

algo = "Ada"

#Add the n_estimators value to the below list if you want to run the algo
n_list = numpy.array([300])

for s in SIZE:
    for n_estimators in n_list:
        #Accuracy of the model using all features
        for name,fe, X_train, X_val,Y_train,Y_val, cols, i_cols_list, i_con, i_bin in X_all[s]:
            print(name)
            #Set the base model
            model = AdaBoostRegressor(n_estimators=n_estimators,random_state=seed)
            model.fit(X_train,Y_train)
            result = math.sqrt(mean_squared_error(numpy.log1p(Y_val), numpy.log1p(model.predict(X_val))))
            mse[s].append(result)
            print(name + " %s" % result)
            comb[s].append(name+" "+algo + " %s" % n_estimators )
##Plot the MSE of all combinations for both types in the same figure
#fig, ax = plt.subplots()
#for s in SIZE:
#    plt.plot(mse[s])
##Set the tick names to names of combinations
#ax.set_xticks(range(len(comb[s])))
#ax.set_xticklabels(comb[s],rotation='vertical')
##Plot the accuracy for all combinations
#plt.legend(SIZE_STR,loc='best')    
#plt.show()    


# ## Evaluation, prediction, and analysis
# * Stochastic Gradient Boosting (Boosting)

# In[ ]:


#Evaluation of various combinations of SGB

#Import the library
from sklearn.ensemble import GradientBoostingRegressor

algo = "SGB"

#Add the n_estimators value to the below list if you want to run the algo
n_list = numpy.array([300])

for s in SIZE:
    for n_estimators in n_list:
        #Accuracy of the model using all features
        for name,fe, X_train, X_val,Y_train,Y_val, cols, i_cols_list, i_con, i_bin in X_all[s]:
            print(name)
            #Set the base model
            model = GradientBoostingRegressor(n_estimators=n_estimators,random_state=seed)
            model.fit(X_train,Y_train)
            result = math.sqrt(mean_squared_error(numpy.log1p(Y_val), numpy.log1p(model.predict(X_val))))
            mse[s].append(result)
            print(name + " %s" % result)
            comb[s].append(name+" "+algo + " %s" % n_estimators )
##Plot the MSE of all combinations for both types in the same figure
#fig, ax = plt.subplots()
#for s in SIZE:
#    plt.plot(mse[s])
##Set the tick names to names of combinations
#ax.set_xticks(range(len(comb[s])))
#ax.set_xticklabels(comb[s],rotation='vertical')
##Plot the accuracy for all combinations
#plt.legend(SIZE_STR,loc='best')    
#plt.show()    


# ## Evaluation, prediction, and analysis
# * XGBoost

# In[ ]:


#Evaluation of various combinations of XGB

#Import the library
from xgboost import XGBRegressor

algo = "XGB"

#Add the n_estimators value to the below list if you want to run the algo
n_list = numpy.array([300])

for s in SIZE:
    for n_estimators in n_list:
        #Accuracy of the model using all features
        for name,fe, X_train, X_val,Y_train,Y_val, cols, i_cols_list, i_con, i_bin in X_all[s]:
            print(name)
            #Set the base model
            model = XGBRegressor(n_estimators=n_estimators,seed=seed)
            model.fit(X_train,Y_train)
            result = math.sqrt(mean_squared_error(numpy.log1p(Y_val), numpy.log1p(model.predict(X_val))))
            mse[s].append(result)
            print(name + " %s" % result)
            comb[s].append(name+" "+algo + " %s" % n_estimators )
##Plot the MSE of all combinations for both types in the same figure
#fig, ax = plt.subplots()
#for s in SIZE:
#    plt.plot(mse[s])
##Set the tick names to names of combinations
#ax.set_xticks(range(len(comb[s])))
#ax.set_xticklabels(comb[s],rotation='vertical')
##Plot the accuracy for all combinations
#plt.legend(SIZE_STR,loc='best')    
#plt.show()    


# ## Evaluation, prediction, and analysis
# * MLP (Deep Learning)

# In[ ]:


#Evaluation of various combinations of multi-layer perceptrons

#Import libraries for deep learning
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense

# define baseline model
def baseline(v):
     # create model
     model = Sequential()
     model.add(Dense(v, input_dim=v, init='normal', activation='relu'))
     model.add(Dense(1, init='normal'))
     # Compile model
     model.compile(loss='mean_squared_error', optimizer='adam')
     return model

# define smaller model
def smaller(v):
     # create model
     model = Sequential()
     model.add(Dense(v/2, input_dim=v, init='normal', activation='relu'))
     model.add(Dense(1, init='normal', activation='relu'))
     # Compile model
     model.compile(loss='mean_squared_error', optimizer='adam')
     return model

# define deeper model
def deeper(v):
 # create model
 model = Sequential()
 model.add(Dense(v, input_dim=v, init='normal', activation='relu'))
 model.add(Dense(v/2, init='normal', activation='relu'))
 model.add(Dense(1, init='normal', activation='relu'))
 # Compile model
 model.compile(loss='mean_squared_error', optimizer='adam')
 return model

# Optimize using dropout and decay
from keras.optimizers import SGD
from keras.layers import Dropout
from keras.constraints import maxnorm

def dropout(v):
    #create model
    model = Sequential()
    model.add(Dense(v, input_dim=v, init='normal', activation='relu',W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(v/2, init='normal', activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(1, init='normal', activation='relu'))
    # Compile model
    sgd = SGD(lr=0.1,momentum=0.9,decay=0.0,nesterov=False)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    return model

# define decay model
def decay(v):
    # create model
    model = Sequential()
    model.add(Dense(v, input_dim=v, init='normal', activation='relu'))
    model.add(Dense(1, init='normal', activation='relu'))
    # Compile model
    sgd = SGD(lr=0.1,momentum=0.8,decay=0.01,nesterov=False)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    return model

est_list = [('MLP',baseline),('smaller',smaller),('deeper',deeper),('dropout',dropout),('decay',decay)]

for s in SIZE:
    for mod, est in est_list:
        algo = mod
        #Accuracy of the model using all features
        for name,fe, X_train, X_val,Y_train,Y_val, cols, i_cols_list, i_con, i_bin in X_all[s]:
            if(name != 'Orig' or mod != 'MLP'):
               continue
            print(name+" "+mod)
            #Set the base model
            model = KerasRegressor(build_fn=est, v=len(cols), nb_epoch=10, verbose=0)
            model.fit(X_train,Y_train)
            result = math.sqrt(mean_squared_error(numpy.log1p(Y_val), numpy.log1p(model.predict(X_val))))
            mse[s].append(result)
            print(name + " %s" % result)
            comb[s].append(name+" "+algo )

#Plot the MSE of all combinations for both types in the same figure
fig, ax = plt.subplots()
for s in SIZE:
    plt.plot(mse[s])
#Set the tick names to names of combinations
ax.set_xticks(range(len(comb[s])))
ax.set_xticklabels(comb[s],rotation='vertical')
#Plot the accuracy for all combinations
plt.legend(SIZE_STR,loc='best')    
plt.show()    


# ## Make Predictions

# In[ ]:


# Make predictions using one-hot encoding of the Original version of the dataset along 
# with RandomForest algo (300 estimators) as it gave the best estimated performance        

#Extract only the values 
array = dataset_train_list[ENC].values
    
#Y is the target column, X has the rest
X = array[:,1:]
Y = array[:,0]

n_estimators = 300

#Best model definition
best_model = RandomForestRegressor(n_jobs=-1,n_estimators=n_estimators,random_state=seed)
best_model.fit(X,Y)

#Extract the ID for submission file
ID = dataset_test['Id']

#Use only values
X_test = dataset_test_list[ENC].values

#Make predictions using the best model
predictions = best_model.predict(X_test)
# Write submissions to output file in the correct format
with open("submission.csv", "w") as subfile:
    subfile.write("Id,SalePrice\n")
    #print("Id,SalePrice\n")
    for i, pred in enumerate(list(predictions)):
        #print("%s,%s\n"%(ID[i],pred))
        subfile.write("%s,%s\n"%(ID[i],pred))


# In[ ]:




