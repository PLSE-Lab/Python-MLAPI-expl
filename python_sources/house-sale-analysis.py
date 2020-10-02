#!/usr/bin/env python
# coding: utf-8

# This is my first attempt at any data science problem. I have tried to use as basic techniques as possible. 
# 
# **Nature of this problem:**
# In supervised machine learning there are two types of problems. Classification and regression. In classification problem the variable to be predicted is discrete like an item is red, blue or green color. In case of regression problems the variable to be predicted is a continious variable.  As house price can be any positive real number (finite ofcouse) so this is a case of regression problem. 
# First we will import all the libraries we will need during analysis.

# In[1]:


# data analysis 
import pandas as pd
import numpy as np
import random as rnd
print("Data analysis imported")
# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
print("Visualization libraries imported")
from sklearn.linear_model import LinearRegression
print('Linear Regression Library imported')


# Now we will obtain training and testing data set

# In[2]:


train_df=pd.read_csv('../input/train.csv')
test_df=pd.read_csv('../input/test.csv')
print("Training and testing set imported")


# Now we will see how many variable are there 
# 

# In[3]:


train_df.shape,test_df.shape


# **Data Prepration:
# 
# First of all we will work on preparing our data for analysis. In order to run any algorithm for analysis, all variables should be in numeric form. But first we need to identify nulls from dataset
# For example if a variable has very few not null values then we can drop it from analysis. So first lets identify which variables are null

# In[4]:


TotalRecs=train_df['Id'].count()
for column in train_df:
    if(train_df[column].isnull().any()):
        print(column,",",train_df[column].dtype,",",train_df[column].count(),' | Non null values Percent:',round(train_df[column].count()/TotalRecs,3)*100)


# As we can here that Alley column has very few not null values so we can drop it right away. We don't need to fill null values for this column. However for other cases, there can be different ways to fill null values. Depending upon the column we can select appropriate method. For example null values in case of feature "Electrical" means that data is not available for that particular house so we can use the most frequent value and fill it up. However for variable PoolQC null means that house has no pool so in this case we will fill it with 'None'.

# In[6]:


#Null means no Linear feet of street connected to property
train_df['LotFrontage']=train_df['LotFrontage'].fillna(0)
#Masonry veneer type is non structual external layer :MasVnrType none means no such layer
train_df['MasVnrType']=train_df['MasVnrType'].fillna('None')
#Masonry veneer area : null means zero area
train_df['MasVnrArea']=train_df['MasVnrArea'].fillna(0)
#For basement related features null means no basment 
train_df['BsmtQual']=train_df['BsmtQual'].fillna('None')
train_df['BsmtCond']=train_df['BsmtCond'].fillna('None')
train_df['BsmtExposure']=train_df['BsmtExposure'].fillna('None')
train_df['BsmtFinType1']=train_df['BsmtFinType1'].fillna('None')
train_df['BsmtFinType2']=train_df['BsmtFinType2'].fillna('None')
#For electrical, in data description its not mentioned that null means no electricity so we will take mean value for this
ElectVal=train_df.Electrical.dropna().mode()[0]
ElectVal
train_df['Electrical']=train_df['Electrical'].fillna(ElectVal)
#FireplaceQu : fire place quality, null means no fireplace
train_df['FireplaceQu']=train_df['FireplaceQu'].fillna('None')
#For all garage related fields, null means no garage 
train_df['GarageType']=train_df['GarageType'].fillna('None')
train_df['GarageYrBlt']=train_df['GarageYrBlt'].fillna(0)
train_df['GarageFinish']=train_df['GarageFinish'].fillna('None')
train_df['GarageQual']=train_df['GarageQual'].fillna('None')
train_df['GarageCond']=train_df['GarageCond'].fillna('None')
#PoolQC: NA means no pool
train_df['PoolQC']=train_df['PoolQC'].fillna('None')
#Fence : Fence quality, NA means no fence, since a large number has no fence so we can drop this attribute from both train and test data
train_df=train_df.drop(['Fence'],axis=1)
test_df=test_df.drop(['Fence'],axis=1)
#MiscFeature: since only 54 houses have miscellenous features so we can drop this assuming that it doesnt have much effect on price
#We will also drop MiscVal column
train_df=train_df.drop(['MiscFeature'],axis=1)
test_df=test_df.drop(['MiscFeature'],axis=1)
train_df=train_df.drop(['MiscVal'],axis=1)
test_df=test_df.drop(['MiscVal'],axis=1)
train_df=train_df.drop(['Alley'],axis=1)
test_df=test_df.drop(['Alley'],axis=1)
train_df.shape,test_df.shape


# Now we will run the null checks again on training data set to see if all nulls are gone or not. After that we will repeat same null checks process on test data set as well. Since test data is used for prediction so it is important that we process both datasets simultaneously.

# In[7]:


#print columns with null data in training dataset, print data type as well, run this again to see if all nulls are gone

for column in train_df:
    if(train_df[column].isnull().any()):
       #TrainDFNullInfo= TrainDFNullInfo.append({'ColumnName',column},{'DataType',train_df[column].dtype},{'NullCounts',train_df[column].count()})
        print(column,",",train_df[column].dtype,",",train_df[column].count(),' | Non null values Percent:',round(train_df[column].count()/TotalRecs,3)*100)


# In[8]:


#Now doing same on testing data
TestRecs=test_df['Id'].count()
TestRecs
#print columns with null data in test dataset, print data type as well, run this again to see if all nulls are gone

for column in test_df:
    if(test_df[column].isnull().any()):
       #TrainDFNullInfo= TrainDFNullInfo.append({'ColumnName',column},{'DataType',train_df[column].dtype},{'NullCounts',train_df[column].count()})
        print(column,",",test_df[column].dtype,",",test_df[column].count(),' | Non null values Percent:',round(test_df[column].count()/TestRecs,3)*100)
   # print(train_df[column].isnull().any(),column)


# In[9]:


#MSZoning : identifies the zoning class, since in data description null is not mentioned, we will fill with most frequent value
test_df['MSZoning']=test_df['MSZoning'].fillna(test_df.MSZoning.dropna().mode()[0])
#Null means no Linear feet of street connected to property
test_df['LotFrontage']=test_df['LotFrontage'].fillna(0)
#Utlities: type of utilities available, here also we need to fill most frequent value
test_df['Utilities']=test_df['Utilities'].fillna(test_df.Utilities.dropna().mode()[0])
#Exterior1st and Exterior2nd 
test_df['Exterior1st']=test_df['Exterior1st'].fillna(test_df.Exterior1st.dropna().mode()[0])
test_df['Exterior2nd']=test_df['Exterior2nd'].fillna(test_df.Exterior2nd.dropna().mode()[0])
#Masonry veneer type is non structual external layer :MasVnrType none means no such layer
test_df['MasVnrType']=test_df['MasVnrType'].fillna('None')
#Masonry veneer area : null means zero area
test_df['MasVnrArea']=test_df['MasVnrArea'].fillna(0)
#For basement related features null means no basment 
test_df['BsmtQual']=test_df['BsmtQual'].fillna('None')
test_df['BsmtCond']=test_df['BsmtCond'].fillna('None')
test_df['BsmtExposure']=test_df['BsmtExposure'].fillna('None')
test_df['BsmtFinType1']=test_df['BsmtFinType1'].fillna('None')
test_df['BsmtFinType2']=test_df['BsmtFinType2'].fillna('None')
test_df['BsmtFinSF1']=test_df['BsmtFinSF1'].fillna(0)
test_df['BsmtFinSF2']=test_df['BsmtFinSF2'].fillna(0)
test_df['BsmtUnfSF']=test_df['BsmtUnfSF'].fillna(0)
test_df['TotalBsmtSF']=test_df['TotalBsmtSF'].fillna(0)
test_df['BsmtFullBath']=test_df['BsmtFullBath'].fillna('None')
test_df['BsmtHalfBath']=test_df['BsmtHalfBath'].fillna('None')
#KitchenQual : fill most frequent one
test_df['KitchenQual']=test_df['KitchenQual'].fillna(test_df.KitchenQual.dropna().mode()[0])
#Functional: home functionality, fill most frequent one
test_df['Functional']=test_df['Functional'].fillna(test_df.Functional.dropna().mode()[0])
#FireplaceQu : Null meansno fire place
test_df['FireplaceQu']=test_df['FireplaceQu'].fillna('None')
#For all garage related fields, null means no garage 
test_df['GarageType']=test_df['GarageType'].fillna('None')
test_df['GarageYrBlt']=test_df['GarageYrBlt'].fillna(0)
test_df['GarageFinish']=test_df['GarageFinish'].fillna('None')
test_df['GarageCars']=test_df['GarageCars'].fillna(0)
test_df['GarageArea']=test_df['GarageArea'].fillna(0)
test_df['GarageQual']=test_df['GarageQual'].fillna('None')
test_df['GarageCond']=test_df['GarageCond'].fillna('None')
#PoolQC: NA means no pool
test_df['PoolQC']=test_df['PoolQC'].fillna('None')
#SaleType : fill the most frequent value
test_df['SaleType']=test_df['SaleType'].fillna(test_df.SaleType.dropna().mode()[0])
train_df.shape,test_df.shape


# Now again we will see if there are any null columns left in test data set

# In[10]:


#Now check again for nulls in test_df
#print columns with null data in test dataset, print data type as well, run this again to see if all nulls are gone

for column in test_df:
    if(test_df[column].isnull().any()):
       #TrainDFNullInfo= TrainDFNullInfo.append({'ColumnName',column},{'DataType',train_df[column].dtype},{'NullCounts',train_df[column].count()})
        print(column,",",test_df[column].dtype,",",test_df[column].count(),' | Non null values Percent:',round(test_df[column].count()/TestRecs,3)*100)
   # print(train_df[column].isnull().any(),column)


# There are various types of variables present in dataset and we will process them accordingly.
# 
# 1. Categorical variables:
# These are variables which can have a finite set of values. There is no order involved, like you cannot tell which one is superior. Example of such variable is color, color can be red, blue, green, white or black and so on but there are limited number of colors available and there is no order involve. Like you cannot decide if red is superior than black. It is better not to map these variables numerically, you can assign red as 1 and blue as 2 and so on based on your choice, but algorithim might think that red has lower order than blue. In order to avoid that we will use encoding vector technique. In such cases we will use get_dummies function, what it does is take an input and convert it into a vector. For example if possible values for color column are red, blue and green and a row has color=red then it will be converted into three columns color_red,color_blue and color_green with values 1,0 and 0 respectively.
# 
# 2. Ordinal variables:
# These are variables which are categorical i.e. they can have a finite set of values but there is an order involved. For example if there is column about "Quality", it can have values Good,Average and Poor, although these values are finite and categorical still there is an order involved. We can map Poor to 0, Average to 1 and Good to 2. 
# 
# 3. Continious variables:
# These are variables which can take an infinite possible values. For example area of a play ground, it can have any value. For such variables we convert them to discrete variables via defining bands, for example area 0 to 500 is band 0 and area > 500 and <=1000 is band 2 and so on. 
# 

# Now before proceeding to further processing we will define some functions which will help us in making analysis smooth.
# 
# 1. FindMinMaxRange:
# This function will take a numeric column and a dataframe as input and will return minimum and maximum value for that column. It will help us define bands for continious columns. As mentioned earlier, all features must be in numeric format but for cases where the variable is numeric but a continious one, like for example "area" feature. Now it will have continious values so it will be converted into bands so that we have a discrete set of values for processing such columns.
# 
# 2. EncodeColumn:
# This function will convert categorical variables into vectors as explained earlier.
# 
# 3. ProcessColumn:
# It will take two column names as input one is dependant Iin this case SalePrice) and other is independant. It will evaluate average sale price against each value from independant variable and plot a graph. If there is some relation observed between price and the variable under observation then we will keep it else it will be dropped. 
# 

# First we will just analyze if there is any relation of variable with sale price, once that analysis is done then we will convert them into numeric variables. Because there is no point in processing a column which has no relationship with price.
# 
# Note:
# dataframe.shape tells number of rows and columns in dataframe, 1st value is rows and 2nd is number of columns. It is important to check train_df.shape and test_df.shape after every processing to make sure that both have same dimensions. In our data, test_df doesnt have SalePrice column so test_df will always be one column less so keep this in mind.

# In[11]:


def FindMinMaxRange(df,ColName):
    return (df[ColName].min(),'is minimum: ',df[ColName].max(),'is maximum',df[ColName].max()-df[ColName].min(),' is the range')


# In[12]:


def EncodeColumn(cats,ColName,df):
    dummiesTrain=pd.get_dummies(df[ColName],prefix=ColName,prefix_sep='_')
    dummiesTrain=dummiesTrain.T.reindex(cats).T.fillna(0).astype(int)
    df=pd.concat([df,dummiesTrain],axis=1)
    df.drop([ColName],axis=1,inplace=True)
    return df


# In[13]:


def ProcessColumn(XColName,YColName):
    dfColName=pd.DataFrame(train_df[[XColName,YColName]].groupby([XColName],as_index=False).mean().sort_values(by=YColName,ascending=False))
    dfColName.plot(kind='bar',x=XColName,y=YColName,legend=True)


# First we will analyze Lot related variables. LotFrontage is linear feet of street connected to property, as its a continious variable so we will first analyze range and will then divide it into bands for further processing, I could have used pd.qcut as well but that divides data into equal ranges and since record count is different in test and training data which may result in different bands for same value. For example in training dataset 1000 is in band 5 but since test doesnt have much variety then 1000 might end up in band 2.

# In[14]:


FindMinMaxRange(train_df,'LotFrontage')


# In[15]:


#For LotFrontage 0 means null so we cannot include 0 in band creation range that is why we have followed a different approach
LotFrontage_Band = []
for row in train_df['LotFrontage']:
    if (row==0):
        LotFrontage_Band.append(0)
    elif(row>0 and row<=50):
        LotFrontage_Band.append(1)
    elif(row>50 and row<=100):
        LotFrontage_Band.append(2)
    elif(row>100 and row<=150):
        LotFrontage_Band.append(3)
    elif(row>150 and row<=200):
        LotFrontage_Band.append(4)
    elif(row>200 and row<=250):
        LotFrontage_Band.append(5)
    else:
        LotFrontage_Band.append(5)
                                    
train_df['LotFrontage_Band']=LotFrontage_Band
#Now all Lot related variables are converted into categorical ones so we will now see how much each variable is related with sale price
ProcessColumn('LotFrontage_Band','SalePrice'),ProcessColumn('LotShape','SalePrice'),ProcessColumn('LotConfig','SalePrice')


# In[16]:


#There is no direct relation of LotFrontage i.e. linear feet of street connected to property and price so we will drop LotFrontage and LotFrontage_Band from datasets
train_df=train_df.drop(['LotFrontage'],axis=1)
test_df=test_df.drop(['LotFrontage'],axis=1)
train_df=train_df.drop(['LotFrontage_Band'],axis=1)
train_df.shape,test_df.shape


# In[17]:


#MSSubClass analyzes type of dwelling so lets see if it has any relationship with price
ProcessColumn('MSSubClass','SalePrice')


# As per data description and graph shown above newer houses or houses with more than 1 story have more prices so we will keep this column. Now we will analyze MSZoning column, which identifies general zonning of the property. Consult data_description.txt for further details

# In[18]:


#Now lets analyze MS Zoning field
ProcessColumn('MSZoning','SalePrice')


# As we can see a pattern here that less populated areas have more price and commercial area has less price so we will keep this for processing. Now lets analyze if Street type has any impact on price

# In[19]:


ProcessColumn('Street','SalePrice')


# As we can see, paved street houses have more price than non paved one so we will keep this column. 
# Now we will analyze LandContour and LandSlope columns
# 

# In[21]:


ProcessColumn('LandContour','SalePrice'),ProcessColumn('LandSlope','SalePrice')


# As we can see that LandSlope doesn't affect price much so we will drop this column. 

# In[22]:


train_df=train_df.drop(['LandSlope'],axis=1)
test_df=test_df.drop(['LandSlope'],axis=1)
train_df.shape,test_df.shape


# Now lets see utilities and Neighborhood columns

# In[24]:


#now analyze utilities column
ProcessColumn('Utilities','SalePrice'),ProcessColumn('Neighborhood','SalePrice')


# We will keep both columns as some neighborhoods are expensive than others. Also houses with all utilities have higher prices.
# Now analyzing Condition1 and Condition2 columns, which means how adjacent is property to various facilities or conditions
# 

# In[25]:


ProcessColumn('Condition1','SalePrice'),ProcessColumn('Condition2','SalePrice')


# As there are two columns with condition, so we will see how many rows have same values for both columns, if same count is high then we can drop column condition2 else we can use both columns to construct a new feature
# 

# In[27]:


train_df['ConditionF']=np.where(train_df['Condition1']==train_df['Condition2'],'Same','Different')
len(train_df.groupby(['ConditionF']).groups['Same']),len(train_df.groupby(['ConditionF']).groups['Different'])


# In[28]:


#Since 1265 have same values so we can drop condition2 column
train_df=train_df.drop(['Condition2'],axis=1)
test_df=test_df.drop(['Condition2'],axis=1)
train_df=train_df.drop(['ConditionF'],axis=1)
train_df.shape,test_df.shape


# Now we will analyze BldgType(type of dwelling),HouseStyle(Style of house),OverallQual and OverallCond columns.

# In[29]:


ProcessColumn('BldgType','SalePrice'),ProcessColumn('HouseStyle','SalePrice'),ProcessColumn('OverallQual','SalePrice'),ProcessColumn('OverallCond','SalePrice')


# In[30]:


#OverallCond isnt directly related to price so we will drop this
#OverallQual has a direct relationship with price so we will keep it
#For HouseStyle finished stories or double stories have more prices
#For BldgType houses build for single family are more expensive and other three catrogies have same price so this doesnt offer too much variety so we can drop this column too
train_df=train_df.drop(['BldgType'],axis=1)
test_df=test_df.drop(['BldgType'],axis=1)
train_df=train_df.drop(['OverallCond'],axis=1)
test_df=test_df.drop(['OverallCond'],axis=1)
train_df.shape,test_df.shape


# Now next two columns are YearRemodAdd and YearBuilt. Value is same for both when house was built and not remodeled afterwards. We will see count of such cases i.e. how many houses are built and not remodeled afterwards. If count is high like condition column then we will drop YearRemodAdd column.
# 

# In[32]:


#Lets analyze now YearBuilt and YearRemodAdd
FindMinMaxRange(train_df,'YearRemodAdd')


# In[33]:


FindMinMaxRange(train_df,'YearBuilt')


# In[34]:


train_df['RemodFlag'] = np.where(train_df['YearBuilt']==train_df['YearRemodAdd'], 'Yes', 'No')
len(train_df.groupby(['RemodFlag']).groups['Yes']),len(train_df.groupby(['RemodFlag']).groups['No'])


# As we can see that 52% houses were remodeled soo we cannot drop this column. Converting both columns to bands

# In[35]:


#As it shows almost 50% houses were remodeled so we will keep this column
YearBuilt_Band = []
for row in train_df['YearBuilt']:
    if (row<=1950):
        YearBuilt_Band.append(0)
    elif(row>1950 and row<=1960):
        YearBuilt_Band.append(1)
    elif(row>1960 and row<=1970):
        YearBuilt_Band.append(2)
    elif(row>1970 and row<=1980):
        YearBuilt_Band.append(3)
    elif(row>1980 and row<=1990):
        YearBuilt_Band.append(4)
    elif(row>1990 and row<=2000):
        YearBuilt_Band.append(5)
    else:
        YearBuilt_Band.append(6)

#As it shows almost 50% houses were remodeled so we will keep this column
YearBuilt_Band2 = []
for row in test_df['YearBuilt']:
    if (row<=1950):
        YearBuilt_Band2.append(0)
    elif(row>1950 and row<=1960):
        YearBuilt_Band2.append(1)
    elif(row>1960 and row<=1970):
        YearBuilt_Band2.append(2)
    elif(row>1970 and row<=1980):
        YearBuilt_Band2.append(3)
    elif(row>1980 and row<=1990):
        YearBuilt_Band2.append(4)
    elif(row>1990 and row<=2000):
        YearBuilt_Band2.append(5)
    else:
        YearBuilt_Band2.append(6)
train_df['YearBuilt_Band']=YearBuilt_Band
test_df['YearBuilt_Band']=YearBuilt_Band2

YearRemod_Band = []
for row in train_df['YearRemodAdd']:
    if (row<=1950):
        YearRemod_Band.append(0)
    elif(row>1950 and row<=1960):
        YearRemod_Band.append(1)
    elif(row>1960 and row<=1970):
        YearRemod_Band.append(2)
    elif(row>1970 and row<=1980):
        YearRemod_Band.append(3)
    elif(row>1980 and row<=1990):
        YearRemod_Band.append(4)
    elif(row>1990 and row<=2000):
        YearRemod_Band.append(5)
    else:
        YearRemod_Band.append(6)

train_df['YearRemod_Band']=YearRemod_Band
YearRemod_Band2 = []
for row in test_df['YearRemodAdd']:
    if (row<=1950):
        YearRemod_Band2.append(0)
    elif(row>1950 and row<=1960):
        YearRemod_Band2.append(1)
    elif(row>1960 and row<=1970):
        YearRemod_Band2.append(2)
    elif(row>1970 and row<=1980):
        YearRemod_Band2.append(3)
    elif(row>1980 and row<=1990):
        YearRemod_Band2.append(4)
    elif(row>1990 and row<=2000):
        YearRemod_Band2.append(5)
    else:
        YearRemod_Band2.append(6)

test_df['YearRemod_Band']=YearRemod_Band2
ProcessColumn('YearRemod_Band','SalePrice'),ProcessColumn('YearBuilt_Band','SalePrice')


# As we can see that even if a house was built previously but as it was remodeled recently so it increased the price so we don't need yearbuild column. We will only keep YearRemod_Band column

# In[36]:


# keept band columsn
train_df=train_df.drop(['RemodFlag'],axis=1)
train_df=train_df.drop(['YearBuilt'],axis=1)
train_df=train_df.drop(['YearBuilt_Band'],axis=1)
train_df=train_df.drop(['YearRemodAdd'],axis=1)
test_df=test_df.drop(['YearBuilt'],axis=1)
test_df=test_df.drop(['YearBuilt_Band'],axis=1)
test_df=test_df.drop(['YearRemodAdd'],axis=1)
train_df.shape,test_df.shape


# Now we will analyze if roof material and roof style has any impact on price

# In[37]:


ProcessColumn('RoofStyle','SalePrice'),ProcessColumn('RoofMatl','SalePrice')


# We will keep both roof related columns. Lets now analyze Exterior1st and Exterior2nd column. Both identifies the exterior covering on house. Here we will also see how many rows have same value. If count is high then we can drop one column.

# In[39]:


#for exterior column, check how many columns have same value in both
train_df['ExterMat']=np.where(train_df['Exterior1st']==train_df['Exterior2nd'],'Same','Different')
len(train_df.groupby(['ExterMat']).groups['Same']),len(train_df.groupby(['ExterMat']).groups['Different'])
#as 1245 have same valaue so drop Exterior2nd
train_df=train_df.drop(['Exterior2nd'],axis=1)
train_df=train_df.drop(['ExterMat'],axis=1)
test_df=test_df.drop(['Exterior2nd'],axis=1)
train_df.shape,test_df.shape


# In[40]:


ProcessColumn('Exterior1st','SalePrice')


# Now we will analyze column MasVnrType i.e. Masonry veneer type . This is basically an external non structural layer on walls. 

# In[41]:


ProcessColumn('MasVnrType','SalePrice')


# Here None means no such wall, we will check its count and see how many houses have no such wall

# In[42]:



len(train_df.groupby(['MasVnrType']).groups['None'])


# In[43]:


#60% houses doesnt have external structure wall so drop two columns
train_df=train_df.drop(['MasVnrType'],axis=1)
train_df=train_df.drop(['MasVnrArea'],axis=1)

test_df=test_df.drop(['MasVnrType'],axis=1)
test_df=test_df.drop(['MasVnrArea'],axis=1)

train_df.shape,test_df.shape


# In[44]:


#Now analyzing exterior qual and coond
ProcessColumn('ExterQual','SalePrice'),ProcessColumn('ExterCond','SalePrice')


# Exterior quality and condition are ordinal variables but we can see here that for ExterCond column, prices for average houses exceed good ones so we will drop this column and will only keep ExterQual since it gives us a direct relation

# In[45]:


#Droping condititon column
train_df=train_df.drop(['ExterCond'],axis=1)
test_df=test_df.drop(['ExterCond'],axis=1)
train_df.shape,test_df.shape


# Now we will analyze Foundation column

# In[46]:


ProcessColumn('Foundation','SalePrice')


# We will keep Foundation column.
# Now lets analyze categorical variables related to basement 

# In[47]:


#Now we will analyze basement related variables, first lets see categorical ones
ProcessColumn('BsmtQual','SalePrice'),ProcessColumn('BsmtCond','SalePrice'),ProcessColumn('BsmtExposure','SalePrice'),ProcessColumn('BsmtFinType1','SalePrice'),ProcessColumn('BsmtFinType2','SalePrice')


# In[48]:


#We will see number of houses having no basements, if count is too high then we can drop these features 
len(train_df.groupby(['BsmtCond']).groups['None'])


# As only 37 houses have no basement so we will keep these variables
# As we can see basement finish type doesnt really matter so dropping columns related to that.
# 

# In[49]:


train_df=train_df.drop(['BsmtFinType1'],axis=1)
train_df=train_df.drop(['BsmtFinType2'],axis=1)
test_df=test_df.drop(['BsmtFinType1'],axis=1)
test_df=test_df.drop(['BsmtFinType2'],axis=1)
train_df=train_df.drop(['BsmtFinSF2'],axis=1)
train_df=train_df.drop(['BsmtFinSF1'],axis=1)

test_df=test_df.drop(['BsmtFinSF2'],axis=1)
test_df=test_df.drop(['BsmtFinSF1'],axis=1)
train_df.shape,test_df.shape


# Now we will analyze if finished and unfinished basement area has any impact on price, i.e. more the finished area of basement higher the price

# In[50]:


#Create coloumn for finished area
train_df['BsmtFA']=train_df['TotalBsmtSF'].astype(float)-train_df['BsmtUnfSF'].astype(float)
test_df['BsmtFA']=test_df['TotalBsmtSF'].astype(float)-test_df['BsmtUnfSF'].astype(float)
train_df.shape,test_df.shape


# In[51]:


FindMinMaxRange(train_df,'BsmtFA')


# In[52]:


FindMinMaxRange(train_df,'BsmtUnfSF')


# In[53]:


FindMinMaxRange(train_df,'TotalBsmtSF')


# In[54]:


#Total basement area, converting it into bands
TotalBsmtSF_Band = []
for row in train_df['TotalBsmtSF']:
    if (row==0):
        TotalBsmtSF_Band.append(0)
    elif(row>0 and row<=1000):
        TotalBsmtSF_Band.append(1)
    elif(row>1000 and row<=2000):
        TotalBsmtSF_Band.append(2)
    elif(row>2000 and row<=3000):
        TotalBsmtSF_Band.append(3)
    elif(row>3000 and row<=4000):
        TotalBsmtSF_Band.append(4)
    elif(row>4000 and row<=5000):
        TotalBsmtSF_Band.append(5)
    else:
        TotalBsmtSF_Band.append(6)
train_df['TotalBsmtSF_Band']=TotalBsmtSF_Band

#We dont need to analyze unfinshed area so lets drop that as well
train_df=train_df.drop(['BsmtUnfSF'],axis=1)
test_df=test_df.drop(['BsmtUnfSF'],axis=1)
train_df.shape,test_df.shape


# In[55]:


#Total finished area
BsmtFA_Band = []
for row in train_df['BsmtFA']:
    if (row==0):
        BsmtFA_Band.append(0)
    elif(row>0 and row<=1000):
        BsmtFA_Band.append(1)
    elif(row>1000 and row<=2000):
        BsmtFA_Band.append(2)
    elif(row>2000 and row<=3000):
        BsmtFA_Band.append(3)
    elif(row>3000 and row<=4000):
        BsmtFA_Band.append(4)
    elif(row>4000 and row<=5000):
        BsmtFA_Band.append(5)
    else:
        BsmtFA_Band.append(6)
train_df['BsmtFA_Band']=BsmtFA_Band


# In[57]:


ProcessColumn('BsmtFA_Band','SalePrice'),ProcessColumn('TotalBsmtSF_Band','SalePrice')


# In[58]:


#As we can see here finished area or overall area has no significant impact so dropping area related columns
train_df=train_df.drop(['TotalBsmtSF_Band'],axis=1)
train_df=train_df.drop(['BsmtFA_Band'],axis=1)
train_df=train_df.drop(['TotalBsmtSF'],axis=1)
test_df=test_df.drop(['TotalBsmtSF'],axis=1)
train_df=train_df.drop(['BsmtFA'],axis=1)
test_df=test_df.drop(['BsmtFA'],axis=1)
train_df.shape,test_df.shape


# In[59]:


#Now analyzing basement full bath and half baths
ProcessColumn('BsmtFullBath','SalePrice'),ProcessColumn('BsmtHalfBath','SalePrice')


# In[60]:


#No relation with price so lets drop this as well
train_df=train_df.drop(['BsmtFullBath'],axis=1)
train_df=train_df.drop(['BsmtHalfBath'],axis=1)
test_df=test_df.drop(['BsmtFullBath'],axis=1)
test_df=test_df.drop(['BsmtHalfBath'],axis=1)
train_df.shape,test_df.shape


# Now lets analyze heating condition and heating type columns

# In[61]:



ProcessColumn('Heating','SalePrice'),ProcessColumn('HeatingQC','SalePrice')


# For heating column wall, gravl and floor heating types have almost same price values. Heating quality column seems more directly related so we will drop heating column.

# In[62]:


train_df=train_df.drop(['Heating'],axis=1)
test_df=test_df.drop(['Heating'],axis=1)
train_df.shape,test_df.shape


# CentralAir column indicates if house is centrally air conditioned or not

# In[63]:



ProcessColumn('CentralAir','SalePrice')


# As we can see, houses with central air conditioning are more expensive so we will keep this column.
# Now lets analyze column 'electrical' which identifies the type of electrical system installed in house

# In[65]:


ProcessColumn('Electrical','SalePrice')


# We will now analyze 1stFlrSF and 2ndFlrSF columns, we will see if 2nd floor finished area has any significant impact on price or not. So will create a new column in training set i.e. 2ndFloorFlag, and see if it has any impact on price

# In[66]:


train_df['2ndFloorFlag']=np.where(train_df['2ndFlrSF']>0,'Yes','No')
ProcessColumn('2ndFloorFlag','SalePrice')


# In[67]:


#as we can see there is not much difference so lets drop 2nd floor column
train_df=train_df.drop(['2ndFloorFlag'],axis=1)
train_df=train_df.drop(['2ndFlrSF'],axis=1)
test_df=test_df.drop(['2ndFlrSF'],axis=1)
train_df.shape,test_df.shape


# There wasnt much difference observed in sale price for houses where there was a second floor so we have dropped that column. Now we will analyze if finished area of 1st floor impacts price or not

# In[68]:


FindMinMaxRange(train_df,'1stFlrSF')


# In[69]:


FstFlrSF_Band=[]
for row in train_df['1stFlrSF']:
    if(row>=0 and row<700):
        FstFlrSF_Band.append(0)
    elif(row>=700 and row<1400):
        FstFlrSF_Band.append(1)
    elif(row>=1400 and row<2100):
        FstFlrSF_Band.append(2)
    elif(row>=2100 and row<2800):
        FstFlrSF_Band.append(3)
    elif(row>=2800 and row<3500):
        FstFlrSF_Band.append(4)
    elif(row>=3500 and row<4200):
        FstFlrSF_Band.append(5)
    else:
        FstFlrSF_Band.append(6)
train_df['1stFlrSF_Band']=FstFlrSF_Band
FstFlrSF_Band2=[]
for row in test_df['1stFlrSF']:
    if(row>=0 and row<700):
        FstFlrSF_Band2.append(0)
    elif(row>=700 and row<1400):
        FstFlrSF_Band2.append(1)
    elif(row>=1400 and row<2100):
        FstFlrSF_Band2.append(2)
    elif(row>=2100 and row<2800):
        FstFlrSF_Band2.append(3)
    elif(row>=2800 and row<3500):
        FstFlrSF_Band2.append(4)
    elif(row>=3500 and row<4200):
        FstFlrSF_Band2.append(5)
    else:
        FstFlrSF_Band2.append(6)
test_df['1stFlrSF_Band']=FstFlrSF_Band2
ProcessColumn('1stFlrSF_Band','SalePrice')


# As we can see area has no impact on price, so we will drop this column. There is another LowQualFinSF, it means low quality finish present on all floors. We will see how many houses have 0 values for this and if count is significant then we will drop this along with 1stFlrSF_Band column

# In[70]:


len(train_df.groupby(['LowQualFinSF']).groups[0])


# In[71]:


#Dropping 1stFlrSF and LowQualFinSF column
train_df=train_df.drop(['1stFlrSF'],axis=1)
train_df=train_df.drop(['1stFlrSF_Band'],axis=1)
train_df=train_df.drop(['LowQualFinSF'],axis=1)

test_df=test_df.drop(['1stFlrSF'],axis=1)
test_df=test_df.drop(['1stFlrSF_Band'],axis=1)
test_df=test_df.drop(['LowQualFinSF'],axis=1)

train_df.shape,test_df.shape


# Now lets see if number of bathrooms , rooms and kitchen count has any impact on price

# In[72]:


ProcessColumn('FullBath','SalePrice'),ProcessColumn('HalfBath','SalePrice'),ProcessColumn('BedroomAbvGr','SalePrice'),ProcessColumn('KitchenAbvGr','SalePrice'),ProcessColumn('KitchenQual','SalePrice'),ProcessColumn('TotRmsAbvGrd','SalePrice')


# There is no impact of these variables so we will drop these

# In[73]:


train_df=train_df.drop(['FullBath'],axis=1)
train_df=train_df.drop(['HalfBath'],axis=1)
train_df=train_df.drop(['BedroomAbvGr'],axis=1)
train_df=train_df.drop(['KitchenAbvGr'],axis=1)
train_df=train_df.drop(['TotRmsAbvGrd'],axis=1)

test_df=test_df.drop(['FullBath'],axis=1)
test_df=test_df.drop(['HalfBath'],axis=1)
test_df=test_df.drop(['BedroomAbvGr'],axis=1)
test_df=test_df.drop(['KitchenAbvGr'],axis=1)
test_df=test_df.drop(['TotRmsAbvGrd'],axis=1)

train_df.shape,test_df.shape


# Now we will analyze Functional column, which identifies any deductions on home or not

# In[74]:


ProcessColumn('Functional','SalePrice')


# Fire place related columns will be abalyzed now. We will see if number of fire places and quality matters in terms of price or not

# In[75]:


ProcessColumn('Fireplaces','SalePrice'),ProcessColumn('FireplaceQu','SalePrice')


# As we can see that fire place count and quality both contribute to higher sale prise so these columns will be kept for processing.
# After this lets move on to categorical and ordinal variables related to Garage

# In[76]:


ProcessColumn('GarageType','SalePrice'),ProcessColumn('GarageFinish','SalePrice'),ProcessColumn('GarageCars','SalePrice'),ProcessColumn('GarageQual','SalePrice'),ProcessColumn('GarageCond','SalePrice')


# As we can see, garagecond which is an ordinal variable but price doesnt increase with order so we will drop it. Also number of cars garage can accomodate has no impact on price

# In[77]:


train_df=train_df.drop(['GarageCars'],axis=1)
test_df=test_df.drop(['GarageCars'],axis=1)
train_df=train_df.drop(['GarageCond'],axis=1)
test_df=test_df.drop(['GarageCond'],axis=1)
train_df.shape,test_df.shape


# GarageYrBlt is equal to year in which garage was built. We will see if newness of garage has any impact on price

# In[78]:


FindMinMaxRange(train_df,'GarageYrBlt')


# Here 0 means Garage is absent so we need some additional processing to see minimum value for year

# In[79]:


train_df['GarageYrBlt'].loc[train_df['GarageYrBlt'] > 0].sort_values().head(1)


# In[80]:


#as we can see here first non zero value is 1900 so we will divid this into bands of 20 years
GarageYrBlt_Band=[]
for row in train_df['GarageYrBlt']:
    if(row==0):
        GarageYrBlt_Band.append(0)
    elif(row>=1900 and row <1920):
        GarageYrBlt_Band.append(1)
    elif(row>=1920 and row<1940):
        GarageYrBlt_Band.append(2)
    elif(row>=1940 and row<1960):
        GarageYrBlt_Band.append(3)
    elif(row>=1960 and row<1980):
        GarageYrBlt_Band.append(4)
    elif(row>=1980 and row<2000):
        GarageYrBlt_Band.append(5)
    else:
        GarageYrBlt_Band.append(6)

train_df['GarageYrBlt_Band']=GarageYrBlt_Band
ProcessColumn('GarageYrBlt_Band','SalePrice')


# As no direct relation is observed so lets drop this column

# In[81]:


#No direct relation so lets drop this
train_df=train_df.drop(['GarageYrBlt_Band'],axis=1)
train_df=train_df.drop(['GarageYrBlt'],axis=1)
test_df=test_df.drop(['GarageYrBlt'],axis=1)
train_df.shape,test_df.shape


# DriveWay column indicates that if driveway of house is paved or not, so lets see how it affects price

# In[82]:


ProcessColumn('PavedDrive','SalePrice')


# Clearly paved or partially paved driveways have higher prices. 
# Now moving on to PoolQC column, this column shows quality of pool but as seen in train.csv , too many values are null for this column so we will count these values and drop this column in case null value count is too high

# In[83]:


len(train_df.groupby(['PoolQC']).groups['None'])


# In[84]:


#Only 7 houses have pools so we will drop these columns
train_df=train_df.drop(['PoolQC'],axis=1)
train_df=train_df.drop(['PoolArea'],axis=1)

test_df=test_df.drop(['PoolQC'],axis=1)
test_df=test_df.drop(['PoolArea'],axis=1)

train_df.shape, test_df.shape


# In[85]:


#Month and year sold might not be related so dropping these
train_df=train_df.drop(['MoSold'],axis=1)
train_df=train_df.drop(['YrSold'],axis=1)


test_df=test_df.drop(['MoSold'],axis=1)
test_df=test_df.drop(['YrSold'],axis=1)

test_df.shape,train_df.shape


# Now analyzing sales type and sales condition columns

# In[86]:


ProcessColumn('SaleType','SalePrice'),ProcessColumn('SaleCondition','SalePrice')


# Now lets compare lotarea with price of house

# In[87]:


FindMinMaxRange(train_df,'LotArea')


# In[88]:


#Divinding into 7 intervals
LotArea_Band=[]
for row in train_df['LotArea']:
    if(row>=0 and row<3000):
        LotArea_Band.append(0)
    elif(row>=3000 and row<6000):
        LotArea_Band.append(1)
    elif(row>=6000 and row<9000):
        LotArea_Band.append(2)
    elif(row>=9000 and row<12000):
        LotArea_Band.append(3)
    elif(row>=12000 and row <15000):
        LotArea_Band.append(4)
    elif(row>=15000 and row<18000):
        LotArea_Band.append(5)
    else:
        LotArea_Band.append(6)
train_df['LotArea_Band']=LotArea_Band
ProcessColumn('LotArea_Band','SalePrice')


# In[89]:


#We will keep this column, now same processing for test df as we want to keep bands same
#Divinding into 7 intervals
LotArea_Band2=[]
for row in test_df['LotArea']:
    if(row>=0 and row<3000):
        LotArea_Band2.append(0)
    elif(row>=3000 and row<6000):
        LotArea_Band2.append(1)
    elif(row>=6000 and row<9000):
        LotArea_Band2.append(2)
    elif(row>=9000 and row<12000):
        LotArea_Band2.append(3)
    elif(row>=12000 and row <15000):
        LotArea_Band2.append(4)
    elif(row>=15000 and row<18000):
        LotArea_Band2.append(5)
    else:
        LotArea_Band2.append(6)
test_df['LotArea_Band']=LotArea_Band2


# In[90]:


#dropping LotArea column
train_df=train_df.drop(['LotArea'],axis=1)
test_df=test_df.drop(['LotArea'],axis=1)
train_df.shape,test_df.shape


# Now analyzing if GarageArea has impact on price or not

# In[91]:


FindMinMaxRange(train_df,'GarageArea')


# In[92]:


GarageArea_Band=[]
for row in train_df['GarageArea']:
    if(row==0):
        GarageArea_Band.append(0)
    elif(row>0 and row<=300):
        GarageArea_Band.append(1)
    elif(row>300 and row<=600):
        GarageArea_Band.append(2)
    elif(row>600 and row<=900):
        GarageArea_Band.append(3)
    elif(row>900 and row<=1200):
        GarageArea_Band.append(4)
    else:
        GarageArea_Band.append(5)
train_df['GarageArea_Band']=GarageArea_Band        
ProcessColumn('GarageArea_Band','SalePrice')


# Dropping area column

# In[93]:


train_df=train_df.drop(['GarageArea_Band'],axis=1)
train_df=train_df.drop(['GarageArea'],axis=1)
test_df=test_df.drop(['GarageArea'],axis=1)
train_df.shape,test_df.shape


# GrLivArea has number of square feet of living area above the ground.

# In[94]:


FindMinMaxRange(train_df,'GrLivArea')


# In[95]:


GrLivArea_Band=[]
for row in train_df['GrLivArea']:
    if(row>=0 and row<500):
        GrLivArea_Band.append(0)
    elif(row>=500 and row<1000):
        GrLivArea_Band.append(1)
    elif (row>=1000 and row <1500):
        GrLivArea_Band.append(2)
    elif(row>=1500 and row<2000):
        GrLivArea_Band.append(3)
    elif(row>=2000 and row<2500):
        GrLivArea_Band.append(4)
    elif(row>=2500 and row<3000):
        GrLivArea_Band.append(5)
    elif(row>=3000 and row<3500):
        GrLivArea_Band.append(6)
    elif(row>=3500 and row<4000):
        GrLivArea_Band.append(7)
    else:
        GrLivArea_Band.append(8)
train_df['GrLivArea_Band']=GrLivArea_Band


# In[96]:


ProcessColumn('GrLivArea_Band','SalePrice')


# We will keep this column as other than band 7 and 8, all bands follow direct relation

# In[97]:


GrLivArea_Band2=[]
for row in test_df['GrLivArea']:
    if(row>=0 and row<500):
        GrLivArea_Band2.append(0)
    elif(row>=500 and row<1000):
        GrLivArea_Band2.append(1)
    elif (row>=1000 and row <1500):
        GrLivArea_Band2.append(2)
    elif(row>=1500 and row<2000):
        GrLivArea_Band2.append(3)
    elif(row>=2000 and row<2500):
        GrLivArea_Band2.append(4)
    elif(row>=2500 and row<3000):
        GrLivArea_Band2.append(5)
    elif(row>=3000 and row<3500):
        GrLivArea_Band2.append(6)
    elif(row>=3500 and row<4000):
        GrLivArea_Band2.append(7)
    else:
        GrLivArea_Band2.append(8)
test_df['GrLivArea_Band']=GrLivArea_Band2
train_df=train_df.drop(['GrLivArea'],axis=1)
test_df=test_df.drop(['GrLivArea'],axis=1)

train_df.shape,test_df.shape


# Now analyzing WoodDeckSF which is square feet area of wood deck present in house

# In[98]:


FindMinMaxRange(train_df,'WoodDeckSF')


# In[99]:


#Lets divide in intervals of 300
WoodDeckSF_Band=[]
for row in train_df['WoodDeckSF']:
    if(row==0):
        WoodDeckSF_Band.append(0)
    elif(row>0 and row<=300):
        WoodDeckSF_Band.append(1)
    elif(row>300 and row<=600):
        WoodDeckSF_Band.append(2)
    else:
        WoodDeckSF_Band.append(3)
train_df['WoodDeckSF_Band']=WoodDeckSF_Band
ProcessColumn('WoodDeckSF_Band','SalePrice')


# In[100]:


#More area more price so we will keep this column
WoodDeckSF_Band2=[]
for row in test_df['WoodDeckSF']:
    if(row==0):
        WoodDeckSF_Band2.append(0)
    elif(row>0 and row<=300):
        WoodDeckSF_Band2.append(1)
    elif(row>300 and row<=600):
        WoodDeckSF_Band2.append(2)
    else:
        WoodDeckSF_Band2.append(3)
test_df['WoodDeckSF_Band']=WoodDeckSF_Band2


# In[101]:


#Dropping wooddecksf
train_df=train_df.drop(['WoodDeckSF'],axis=1)
test_df=test_df.drop(['WoodDeckSF'],axis=1)
train_df.shape,test_df.shape


# Wood deck area has direct relation with price. Greater the band higher the price so this will be part of our analysis.
# OpenPorchSF is open porch area in square feet, lets see if it has any relation with price.

# In[102]:


FindMinMaxRange(train_df,'OpenPorchSF')


# In[103]:


OpenPorchSF_Band=[]
for row in train_df['OpenPorchSF']:
    if(row==0):
        OpenPorchSF_Band.append(0)
    elif(row>0 and row<=100):
        OpenPorchSF_Band.append(1)
    elif(row>100 and row<=200):
        OpenPorchSF_Band.append(2)
    elif(row>200 and row<=300):
        OpenPorchSF_Band.append(3)
    elif(row>300 and row<=400):
        OpenPorchSF_Band.append(4)
    else:
        OpenPorchSF_Band.append(5)
train_df['OpenPorchSF_Band']=OpenPorchSF_Band
ProcessColumn('OpenPorchSF_Band','SalePrice')


# In[104]:


# as we can see there is no relation so lets drop this column
train_df=train_df.drop(['OpenPorchSF_Band'],axis=1)
train_df=train_df.drop(['OpenPorchSF'],axis=1)
test_df=test_df.drop(['OpenPorchSF'],axis=1)
train_df.shape,test_df.shape


# Now analyzing EclosedPorch,3SsnPorchArea and ScreenPorch, for these columns, null counts looks high from csv file so lets count null values and then decide if its worth keeping or not

# In[107]:


len(train_df.groupby(['EnclosedPorch']).groups[0]),len(train_df.groupby(['3SsnPorch']).groups[0]),len(train_df.groupby(['ScreenPorch']).groups[0])


# In[108]:


#as 1252 values equal to zero so we will ignore it before further processing
train_df=train_df.drop(['EnclosedPorch'],axis=1)
test_df=test_df.drop(['EnclosedPorch'],axis=1)
train_df=train_df.drop(['3SsnPorch'],axis=1)
test_df=test_df.drop(['3SsnPorch'],axis=1)
train_df=train_df.drop(['ScreenPorch'],axis=1)
test_df=test_df.drop(['ScreenPorch'],axis=1)
test_df.shape,train_df.shape


# As mentioned earlier, all columns should be in numeric format so that we can run algorithm. Now that we have processed all columns, we will see which columns are not in number format and covert them to numeric.

# In[109]:


train_df.dtypes


# We will use encoding on categorical columns and mapping on ordinal ones

# In[110]:


#using encoding on this one
MSZoning_cats=['MSZoning_A', 'MSZoning_C', 'MSZoning_FV', 'MSZoning_I', 'MSZoning_RH', 'MSZoning_RL', 'MSZoning_RP', 'MSZoning_RM']
train_df=EncodeColumn(MSZoning_cats,'MSZoning',train_df)
test_df=EncodeColumn(MSZoning_cats,'MSZoning',test_df)
#For this we will not encode since Pave streets rank higher than Grvl ones
train_df['Street']=train_df['Street'].map({'Grvl':0, 'Pave':1}).astype(int)
test_df['Street']=test_df['Street'].map({'Grvl':0, 'Pave':1}).astype(int)
train_df.shape,test_df.shape
# we will encode LotShape column
LotShape_cats=['LotShape_Reg', 'LotShape_IR1', 'LotShape_IR2', 'LotShape_IR3']
train_df=EncodeColumn(LotShape_cats,'LotShape',train_df)
test_df=EncodeColumn(LotShape_cats,'LotShape',test_df)
#now LandContour..its categorical
LandContour_cats=['LandContour_Lvl', 'LandContour_Bnk', 'LandContour_HLS', 'LandContour_Low']
train_df=EncodeColumn(LandContour_cats,'LandContour',train_df)
test_df=EncodeColumn(LandContour_cats,'LandContour',test_df)
train_df.shape,test_df.shape
#Mapping utilities column. I have considered this as ordinal because presence of all utlities is  we will rank AllPub as higher
train_df['Utilities']=train_df['Utilities'].map({'NoSeWa':0, 'AllPub':1}).astype(int)
test_df['Utilities']=test_df['Utilities'].map({'NoSeWa':0, 'AllPub':1}).astype(int)
train_df.shape,test_df.shape


# In[111]:


LotConfig_cats=['LotConfig_Inside', 'LotConfig_Corner', 'LotConfig_CulDSac', 'LotConfig_FR2', 'LotConfig_FR3']
train_df=EncodeColumn(LotConfig_cats,'LotConfig',train_df)
test_df=EncodeColumn(LotConfig_cats,'LotConfig',test_df)
#Now neighboorhood
NB_cats=['NB_Blmngtn', 'NB_Blueste', 'NB_BrDale', 'NB_BrkSide', 'NB_ClearCr', 'NB_CollgCr', 'NB_Crawfor', 'NB_Edwards', 'NB_Gilbert', 'NB_IDOTRR', 'NB_MeadowV', 'NB_Mitchel', 'NB_Names', 'NB_NoRidge', 'NB_NPkVill', 'NB_NridgHt', 'NB_NWAmes', 'NB_OldTown', 'NB_SWISU', 'NB_Sawyer', 'NB_SawyerW', 'NB_Somerst', 'NB_StoneBr', 'NB_Timber', 'NB_Veenker']
train_df=EncodeColumn(NB_cats,'Neighborhood',train_df)
test_df=EncodeColumn(NB_cats,'Neighborhood',test_df)

HouseStyle_cats=['HouseStyle_1Story', 'HouseStyle_1_5Fin', 'HouseStyle_1_5Unf', 'HouseStyle_2Story', 'HouseStyle_2_5Fin', 'HouseStyle_2_5Unf', 'HouseStyle_SFoyer', 'HouseStyle_SLvl']
train_df=EncodeColumn(HouseStyle_cats,'HouseStyle',train_df)
test_df=EncodeColumn(HouseStyle_cats,'HouseStyle',test_df)

RoofStyle_cats=['RoofStyle_Flat', 'RoofStyle_Gable', 'RoofStyle_Gambrel', 'RoofStyle_Hip', 'RoofStyle_Mansard', 'RoofStyle_Shed']
train_df=EncodeColumn(RoofStyle_cats,'RoofStyle',train_df)
test_df=EncodeColumn(RoofStyle_cats,'RoofStyle',test_df)

RoofMatl_cats=['RoofMatl_ClyTile', 'RoofMatl_CompShg', 'RoofMatl_Membran', 'RoofMatl_Metal', 'RoofMatl_Roll', 'RoofMatl_Tar&Grv', 'RoofMatl_WdShake', 'RoofMatl_WdShngl']
train_df=EncodeColumn(RoofMatl_cats,'RoofMatl',train_df)
test_df=EncodeColumn(RoofMatl_cats,'RoofMatl',test_df)

Exterior1st_cats=['Exterior1st_AsbShng', 'Exterior1st_AsphShn', 'Exterior1st_BrkComm', 'Exterior1st_BrkFace', 'Exterior1st_CBlock', 'Exterior1st_CemntBd', 'Exterior1st_HdBoard', 'Exterior1st_ImStucc', 'Exterior1st_MetalSd', 'Exterior1st_Other', 'Exterior1st_Plywood', 'Exterior1st_PreCast', 'Exterior1st_Stone', 'Exterior1st_Stucco', 'Exterior1st_VinylSd', 'Exterior1st_Wd Sdng', 'Exterior1st_WdShing']
train_df=EncodeColumn(Exterior1st_cats,'Exterior1st',train_df)
test_df=EncodeColumn(Exterior1st_cats,'Exterior1st',test_df)
train_df.shape,test_df.shape


# In[112]:


#We will ExterQual column as its an ordinal column it
train_df['ExterQual']=train_df['ExterQual'].map({'Po':0, 'Fa':1,'TA':2,'Gd':3,'Ex':4}).astype(int)
test_df['ExterQual']=test_df['ExterQual'].map({'Po':0, 'Fa':1,'TA':2,'Gd':3,'Ex':4}).astype(int)

Foundation_cats=['Foundation_BrkTil', 'Foundation_CBlock', 'Foundation_PConc', 'Foundation_Slab', 'Foundation_Stone', 'Foundation_Wood']
train_df=EncodeColumn(Foundation_cats,'Foundation',train_df)
test_df=EncodeColumn(Foundation_cats,'Foundation',test_df)

#BsmtQuality will be mapped
train_df['BsmtQual']=train_df['BsmtQual'].map({'None':0, 'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}).astype(int)
test_df['BsmtQual']=test_df['BsmtQual'].map({'None':0, 'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}).astype(int)

train_df['BsmtCond']=train_df['BsmtCond'].map({'None':0, 'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}).astype(int)
test_df['BsmtCond']=test_df['BsmtCond'].map({'None':0, 'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}).astype(int)

train_df['BsmtExposure']=train_df['BsmtExposure'].map({'None':0, 'No':1,'Mn':2,'Av':3,'Gd':4}).astype(int)
test_df['BsmtExposure']=test_df['BsmtExposure'].map({'None':0, 'No':1,'Mn':2,'Av':3,'Gd':4}).astype(int)

train_df['HeatingQC']=train_df['HeatingQC'].map({'Po':0, 'Fa':1,'TA':2,'Gd':3,'Ex':4}).astype(int)
test_df['HeatingQC']=test_df['HeatingQC'].map({'Po':0, 'Fa':1,'TA':2,'Gd':3,'Ex':4}).astype(int)

train_df['CentralAir']=train_df['CentralAir'].map({'N':0, 'Y':1}).astype(int)
test_df['CentralAir']=test_df['CentralAir'].map({'N':0, 'Y':1}).astype(int)

#We will treat this as ordinal, there is not much difference in FuseA,FuseP and FuseF price so we are mapping it to 1
train_df['Electrical']=train_df['Electrical'].map({'Mix':0, 'FuseA':1,'FuseP':1,'FuseF':1,'SBrkr':2}).astype(int)
test_df['Electrical']=test_df['Electrical'].map({'Mix':0, 'FuseA':1,'FuseP':1,'FuseF':1,'SBrkr':2}).astype(int)

train_df['KitchenQual']=train_df['KitchenQual'].map({'Po':0, 'Fa':1,'TA':2,'Gd':3,'Ex':4}).astype(int)
test_df['KitchenQual']=test_df['KitchenQual'].map({'Po':0, 'Fa':1,'TA':2,'Gd':3,'Ex':4}).astype(int)

Func_cats=['Functional_Typ', 'Functional_Min1', 'Functional_Min2', 'Functional_Mod', 'Functional_Maj1', 'Functional_Maj2', 'Functional_Sev', 'Functional_Sal']
train_df=EncodeColumn(Func_cats,'Functional',train_df)
test_df=EncodeColumn(Func_cats,'Functional',test_df)

train_df['FireplaceQu']=train_df['FireplaceQu'].map({'None':0, 'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}).astype(int)
test_df['FireplaceQu']=test_df['FireplaceQu'].map({'None':0, 'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}).astype(int)

GarageType_cats=['GarageType_2Types', 'GarageType_Attchd', 'GarageType_Basment', 'GarageType_BuiltIn', 'GarageType_CarPort', 'GarageType_Detchd', 'GarageType_NA']
train_df=EncodeColumn(GarageType_cats,'GarageType',train_df)
test_df=EncodeColumn(GarageType_cats,'GarageType',test_df)

train_df['GarageFinish']=train_df['GarageFinish'].map({'None':0, 'Unf':1,'RFn':2,'Fin':3}).astype(int)
test_df['GarageFinish']=test_df['GarageFinish'].map({'None':0, 'Unf':1,'RFn':2,'Fin':3}).astype(int)

train_df['GarageQual']=train_df['GarageQual'].map({'None':0, 'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}).astype(int)
test_df['GarageQual']=test_df['GarageQual'].map({'None':0, 'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}).astype(int)

train_df['PavedDrive']=train_df['PavedDrive'].map({'N':0, 'P':1,'Y':2}).astype(int)
test_df['PavedDrive']=test_df['PavedDrive'].map({'N':0, 'P':1,'Y':2}).astype(int)

saletype_cats=['SaleType_WD ', 'SaleType_CWD', 'SaleType_VWD', 'SaleType_New', 'SaleType_COD', 'SaleType_Con', 'SaleType_ConLw', 'SaleType_ConLI', 'SaleType_ConLD', 'SaleType_Oth']
train_df=EncodeColumn(saletype_cats,'SaleType',train_df)
test_df=EncodeColumn(saletype_cats,'SaleType',test_df)

SaleConds_cats=['SaleCond_Normal', 'SaleCond_Abnorml', 'SaleCond_AdjLand', 'SaleCond_Alloca', 'SaleCond_Family', 'SaleCond_Partial']
train_df=EncodeColumn(SaleConds_cats,'SaleCondition',train_df)
test_df=EncodeColumn(SaleConds_cats,'SaleCondition',test_df)
train_df.shape,test_df.shape






# In[113]:


cond_cats=['Cond_Artery', 'Cond_Feedr', 'Cond_Norm', 'Cond_RRNn', 'Cond_RRAn', 'Cond_PosN', 'Cond_PosA', 'Cond_RRNe', 'Cond_RRAe']
train_df=EncodeColumn(cond_cats,'Condition1',train_df)
test_df=EncodeColumn(cond_cats,'Condition1',test_df)
train_df.shape,test_df.shape


# We will use simple linear regression classifier for analyzing this dataset. Linear regression determines relationship between single or multiple independent variables with a dependant variable.
# For this, data needs to be divided into training and testing set. Training set will not have column SalePrice.
# We will train our model on train_df dataframe (excluding SalePrice column) 
# After that we will use predict function on test_df to predict sale price of houses present in test dataframe

# In[114]:


#ID column isnt required for training 
train_df=train_df.drop(['Id'],axis=1)
y_train=train_df['SalePrice']
x_train=train_df.drop(['SalePrice'],axis=1)
x_test  = test_df.drop("Id", axis=1).copy()
regressor=LinearRegression()
regressor.fit(x_train, y_train) 
y_pred = regressor.predict(x_test)
submission = pd.DataFrame({
        "Id": test_df["Id"],
        "SalePrice": y_pred
    })
submission=submission.round(2)
#submission.to_csv('submission_house.csv', index=False)


# In[115]:





# In[ ]:





# In[ ]:




