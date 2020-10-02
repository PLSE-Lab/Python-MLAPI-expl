#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import random
import os
print(os.listdir("../input"))


# In[ ]:


DataTrain = pd.read_csv("../input/train.csv")
DataTest = pd.read_csv("../input/test.csv")


# **Separation of the DataTrain and the DataTest into two Data Frame each, one with numerical values (QuantTrain & QuantTest) and one with categorical values (QualTrain & QualTest)**

# In[ ]:


Y = DataTrain["SalePrice"].copy()
QuantTrain = DataTrain.select_dtypes(include=['float','integer']).copy()
QuantTrain.drop("SalePrice", inplace=True, axis=1)
QualTrain = DataTrain.select_dtypes(include=['object']).copy()

QuantTest = DataTest.select_dtypes(include=['float','integer']).copy()
QualTest = DataTest.select_dtypes(include=['object']).copy()


# **Visualize the QuantTrain DataFrame to see the empty boxes**

# In[ ]:


msno.matrix(df= QuantTrain, figsize=(20,14), color=(0.5,0,0))


# **Fill the NaN boxes in the "LotFrontage" & "MasVnrArea" columns with the median and the NaN boxes in "GarageYrBlt" with 0 because there is no garage**

# In[ ]:


QuantTrain = QuantTrain.fillna({"LotFrontage": QuantTrain['LotFrontage'].median()})
QuantTrain = QuantTrain.fillna({"GarageYrBlt": 0})
QuantTrain = QuantTrain.fillna({"MasVnrArea": QuantTrain['MasVnrArea'].median()})


# In[ ]:


QuantTrain.isnull().values.any()


# **=> All the QuantTrain's boxes are full now**

# **Visualize the QuantTest DataFrame to see the empty boxes**

# In[ ]:


msno.matrix(df= QuantTest, figsize=(20,14), color=(0.5,0,0))


# **Fill the all the NaN boxes with the median in the QuantTest DataFrame except the the NaN boxes in "GarageYrBlt" with 0 because there is no garage**

# In[ ]:


QuantTest = QuantTest.fillna({"LotFrontage": QuantTest['LotFrontage'].median()})
QuantTest = QuantTest.fillna({"MasVnrArea": QuantTest['MasVnrArea'].median()})
QuantTest = QuantTest.fillna({"GarageYrBlt": 0})
QuantTest = QuantTest.fillna({"BsmtFinSF1": QuantTest['BsmtFinSF1'].median()})
QuantTest = QuantTest.fillna({"BsmtFinSF2": QuantTest['BsmtFinSF2'].median()})
QuantTest = QuantTest.fillna({"BsmtUnfSF": QuantTest['BsmtUnfSF'].median()})
QuantTest = QuantTest.fillna({"TotalBsmtSF": QuantTest['TotalBsmtSF'].median()})
QuantTest = QuantTest.fillna({"BsmtFullBath": QuantTest['BsmtFullBath'].median()})
QuantTest = QuantTest.fillna({"BsmtHalfBath": QuantTest['BsmtHalfBath'].median()})
QuantTest = QuantTest.fillna({"GarageCars": QuantTest['GarageCars'].median()})
QuantTest = QuantTest.fillna({"GarageArea": QuantTest['GarageArea'].median()})


# In[ ]:


QuantTest.isnull().values.any()


# **=> all the values in the Quantitative DataFrames are defined**

# **Visualize the QualTrain DataFrame to see the empty boxes**

# In[ ]:


msno.matrix(df= QualTrain, figsize=(20,14), color=(0.5,0,0))


# **Drop the "Alley" and "PoolQc" columns beacause they are almost empty. Besides, for the "PoolQc", we have an other feature which is "PoolArea" in the Quantitative DataFrame that can replace it.**

# In[ ]:


ToDrop=["PoolQC","Alley"]
QualTrain.drop(ToDrop, inplace=True, axis=1)
QualTest.drop(ToDrop, inplace=True, axis=1)


# **Fill the empty boxes in the QualTrain DataFrame with significant strings because NaN means that there is no that equipement except for the feature "Electrical", we fill them by the most frequent value because there is a lack of information **

# In[ ]:


QualTrain = QualTrain.fillna({"BsmtQual": "NoB"})
QualTrain = QualTrain.fillna({"BsmtCond": "NoB"})
QualTrain = QualTrain.fillna({"BsmtExposure": "NoB"})
QualTrain = QualTrain.fillna({"BsmtFinType1": "NoB"})
QualTrain = QualTrain.fillna({"BsmtFinType2": "NoB"})

QualTrain = QualTrain.fillna({"FireplaceQu": "NoF"})

QualTrain = QualTrain.fillna({"GarageType": "NoG"})
QualTrain = QualTrain.fillna({"GarageFinish": "NoG"})
QualTrain = QualTrain.fillna({"GarageQual": "NoG"})
QualTrain = QualTrain.fillna({"GarageCond": "NoG"})

QualTrain = QualTrain.fillna({"MiscFeature": "NoGMisc"})

QualTrain = QualTrain.fillna({"Fence": "NoF"})

QualTrain = QualTrain.fillna({"Electrical": QualTrain['Electrical'].value_counts().idxmax()})


# **See if there are other empty boxes in the DataFrame**

# In[ ]:


nan_rows = QualTrain[QualTrain.isnull().T.any().T]
nan_rows


# **Verify what these NaN values in "MasVnrArea" column mean by using the QuantTrain DataFrame**

# In[ ]:


for i in [234,529,650,936,973,977,1243,1278]:
    print(QuantTrain.loc[i,"MasVnrArea"])


# **All of them are 0 => There is no  Masonry veneer area => we replace the NaN values by "None"**

# In[ ]:


QualTrain = QualTrain.fillna({"MasVnrType": "None"})


# In[ ]:


QualTrain.isnull().values.any()


# **=> All the QualtTrain's boxes are full now**

# **Visualize the QualTest DataFrame to see the empty boxes**

# In[ ]:


msno.matrix(df= QualTest, figsize=(20,14), color=(0.5,0,0))


# **Fill the empty boxes in the QualTest DataFrame with significant strings because NaN means that there is no that equipement**

# In[ ]:


QualTest = QualTest.fillna({"BsmtQual": "NoB"})
QualTest = QualTest.fillna({"BsmtCond": "NoB"})
QualTest = QualTest.fillna({"BsmtExposure": "NoB"})
QualTest = QualTest.fillna({"BsmtFinType1": "NoB"})
QualTest = QualTest.fillna({"BsmtFinType2": "NoB"})

QualTest = QualTest.fillna({"FireplaceQu": "NoF"})

QualTest = QualTest.fillna({"GarageType": "NoG"})
QualTest = QualTest.fillna({"GarageFinish": "NoG"})
QualTest = QualTest.fillna({"GarageQual": "NoG"})
QualTest = QualTest.fillna({"GarageCond": "NoG"})

QualTest = QualTest.fillna({"MiscFeature": "NoGMisc"})

QualTest = QualTest.fillna({"Fence": "NoF"})


# **Search the indexes of the NaN values in the feature "MasVnrType"**

# In[ ]:


QualTest[QualTest["MasVnrType"].isnull()]


# **Verify what these NaN values in "MasVnrArea" column mean by using the QuantTest DataFrame**

# In[ ]:


for i in [231,246,422,532,544,581,851,865,880,889,908,1132,1150,1197,1226,1402]:
    print(QuantTest.loc[i,"MasVnrArea"])


# **All of them refer that there is no  Masonry veneer area so we have to fill them by "None", except one boxe so we fill by the most frequent value**

# **=> See the most frequent value**

# In[ ]:


QualTest['MasVnrType'].value_counts().idxmax()


# **=> Fill all the NaN boxes by "None"**

# In[ ]:


QualTest = QualTest.fillna({"MasVnrType": QualTest['MasVnrType'].value_counts().idxmax()})
#or QualTest = QualTest.fillna({"MasVnrType": "None"})


# **Fill the other missing values by the most frequent one because we have a lack of information**

# In[ ]:


QualTest = QualTest.fillna({"KitchenQual": QualTest['KitchenQual'].value_counts().idxmax()})
QualTest = QualTest.fillna({"MSZoning": QualTest['MSZoning'].value_counts().idxmax()})
QualTest = QualTest.fillna({"Exterior1st": QualTest['Exterior1st'].value_counts().idxmax()})
QualTest = QualTest.fillna({"Functional": QualTest['Functional'].value_counts().idxmax()})
QualTest = QualTest.fillna({"Utilities": QualTest['Utilities'].value_counts().idxmax()})
QualTest = QualTest.fillna({"SaleType": QualTest['SaleType'].value_counts().idxmax()})
QualTest = QualTest.fillna({"Exterior2nd": QualTest['Exterior2nd'].value_counts().idxmax()})


# In[ ]:


nan_rows = QualTest[QualTest.isnull().T.any().T]
nan_rows


# In[ ]:


QualTest.isnull().values.any()


# **=> Both of Qualitative Data are well defined**

# **Now we create Dummy variables (Binary variables) for categorical features so that we can use them in our linear model**

# In[ ]:


QualTrain = pd.get_dummies(QualTrain)
QualTest = pd.get_dummies(QualTest)


# **Now we scale the numerical values so they will be in the same interval**

# In[ ]:


cols_to_scale = ['MSSubClass','LotFrontage','LotArea', 'OverallQual',
       'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
       'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
       'MiscVal', 'MoSold', 'YrSold']
scaler = preprocessing.StandardScaler()
for col in cols_to_scale:
    QuantTrain[col] = pd.DataFrame(scaler.fit_transform(pd.DataFrame(QuantTrain[col])),columns=[col])
    QuantTest[col] = pd.DataFrame(scaler.fit_transform(pd.DataFrame(QuantTest[col])),columns=[col])


# **Reassembly of the Data Frames**

# In[ ]:


Xtrain = pd.concat([QuantTrain,QualTrain,Y], axis=1, sort=False)
Xtest = pd.concat([QuantTest,QualTest], axis=1, sort=False)


# **Find the correlation between all the features and the "SalePrice" so we can choose the most correlated features and use them then in our model**

# In[ ]:


corrmat = Xtrain.corr()
corrmat['SalePrice'].sort_values(ascending = False)[:100]


# **We take the features that are correlated with the "SalePrice" with a coefficient above than 0.05**

# In[ ]:


corr = ["OverallQual","GrLivArea","GarageCars","GarageArea","TotalBsmtSF","1stFlrSF","FullBath","BsmtQual_Ex",
        "TotRmsAbvGrd","YearBuilt","YearRemodAdd","KitchenQual_Ex","Foundation_PConc","MasVnrArea","Fireplaces",
        "ExterQual_Gd","ExterQual_Ex","BsmtFinType1_GLQ","HeatingQC_Ex","GarageFinish_Fin","Neighborhood_NridgHt",
        "BsmtFinSF1","SaleType_New","SaleCondition_Partial","FireplaceQu_Gd","GarageType_Attchd",
        "LotFrontage","MasVnrType_Stone","Neighborhood_NoRidge","WoodDeckSF","KitchenQual_Gd","2ndFlrSF",
        "OpenPorchSF","BsmtExposure_Gd","Exterior2nd_VinylSd","Exterior1st_VinylSd","HalfBath","GarageCond_TA",
        "LotArea","GarageYrBlt","FireplaceQu_Ex","CentralAir_Y","GarageQual_TA","MSZoning_RL","HouseStyle_2Story",
        "Electrical_SBrkr","RoofStyle_Hip","GarageType_BuiltIn","BsmtQual_Gd","PavedDrive_Y","BsmtFullBath",
        "LotShape_IR1","Neighborhood_StoneBr","BsmtUnfSF","MasVnrType_BrkFace","Fence_NoF","GarageFinish_RFn",
        "RoofMatl_WdShngl","BedroomAbvGr","FireplaceQu_TA","LotConfig_CulDSac","Neighborhood_Somerst",
        "BldgType_1Fam","BsmtExposure_Av","Exterior1st_CemntBd","Exterior2nd_CmentBd","Neighborhood_Timber",
        "LotShape_IR2","LandContour_HLS","BsmtFinType2_Unf","Functional_Typ","Condition1_Norm","ScreenPorch",
        "ExterCond_TA","BsmtCond_TA","Heating_GasA","PoolArea","MSZoning_FV","BsmtCond_Gd","Exterior2nd_ImStucc",
        "Neighborhood_CollgCr","MiscFeature_NoGMisc","Neighborhood_Crawfor","Neighborhood_Veenker",
        "Neighborhood_ClearCr"]
       
XTrain = Xtrain[corr]
XTest = Xtest[corr]


# **Drow the correlation Matrix between the most correlated features with the "SalePrice", to see if there is a correlation between 2 features so we might eliminate one of them**

# In[ ]:


corr1 = ["OverallQual","GrLivArea","GarageCars","GarageArea","TotalBsmtSF","1stFlrSF","FullBath","BsmtQual_Ex",
        "TotRmsAbvGrd","YearBuilt","YearRemodAdd","KitchenQual_Ex","Foundation_PConc","MasVnrArea","Fireplaces",
        "ExterQual_Gd","ExterQual_Ex","BsmtFinType1_GLQ","HeatingQC_Ex","GarageFinish_Fin","Neighborhood_NridgHt",
         "GarageYrBlt","SalePrice"]
Visualisation = Xtrain[corr1]
plt.figure(figsize =(15,8))
sns.heatmap(Visualisation.corr(),annot=True,cmap='coolwarm')
plt.show()


# **"GarageCars" and "GarageArea" are so correlated together and boh of them give us the same information so we can get rid of one of them. But when I let both of them the prediction error has gone down so I preferred to not drop one of them**

# In[ ]:


#ToDrop = ["GarageArea"]
#XTrain.drop(ToDrop, inplace=True, axis=1)
#XTest.drop(ToDrop, inplace=True, axis=1)


# **Here we drow the "SalePrice" in terms of the best 6 features (most correlated) to see if there are some outliers that we can eliminate**

# In[ ]:


fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(16, 10))
axes = np.ravel(axes)
col_name = ["OverallQual","GrLivArea","GarageCars","TotalBsmtSF","1stFlrSF","FullBath"]
for i, c in zip(range(6), col_name):
    Visualisation.plot.scatter(ax=axes[i], x=c, y='SalePrice', sharey=True, colorbar=False, c='orange')


# **=> Find the indexes of the outliers**

# In[ ]:


index= [(Visualisation[Visualisation['GrLivArea'] > 5]).index & (Visualisation[Visualisation['SalePrice'] < 300000]).index ,
         (Visualisation[Visualisation['TotalBsmtSF'] > 10]).index & (Visualisation[Visualisation['SalePrice'] < 300000]).index ,
         (Visualisation[Visualisation['1stFlrSF'] > 8]).index & (Visualisation[Visualisation['SalePrice'] < 300000]).index]
index


# **In fact we will not delete them because when I deleted them the model didn't get improved**

# **Linear Regression Model**

# In[ ]:


# Create linear regression object
LM = LinearRegression()
# Train the model using the training DataDrame
LM.fit(XTrain,Y)
#Prediction of the Sale Prices from the testing DataDrame
predictions = LM.predict(XTest)


# **Submission**

# In[ ]:


submit = pd.DataFrame({'Id': Xtest.loc[:,"Id"], 'SalePrice': predictions})
submit.to_csv('submission.csv', index=False)

