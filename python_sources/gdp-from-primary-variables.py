#!/usr/bin/env python
# coding: utf-8

# MODEL - STEP FUNCTION 1 
# 
# What this model tries to do is predict the GDP per Capita of a country based on certain indicators. The indicators (independent variables) are below :-
#     
#     1. Women making informed choices to reproductive health care (Units - % Women Aged 15-49) [DONE#]
#     2. Rural Population as % of Total Population (Units - % of Total Population) [DONE#] 
#     3. Public Education as a share of GDP (Units - expenditure as % of GDP) [CURRENTLY IGNORE THIS DATA SET]
#     4. Legal Rights Strengths Index (Units - Index from 0 to 12) [DONE#]
#     5. Domestic Credit to Private Sector (Units - % of GDP) [DONE#]
#     6. Births attended by skilled health staff (Units - % of total births) [DONE#]
#     7. ATM machines per 100,000 adults (Units - number of machines per 100,000 adults) [DONE#]
#     8. Agricultural Machinery (Units - number of tractors per 100 sq Km of arable land) [DONE#]
#     9. Literacy rates in the adult population (Units - % of male population educated above 15 years of age) [DONE#]
#     10. Accounts at Financial Institutions (Units - % of male population above 15 years of age having accounts) [DONE]
#     
# The dependant variables are below :-
#     1. [DONE] GDP by country (Units - Current US $) 
#     2. [DONE] Population per country (Units - in decimal numbers)

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# In[ ]:


import seaborn as sns
sns.set(color_codes=True)


# In[ ]:


population_df = pd.read_csv('../input/PopulationPerCountry.csv', skiprows = range(0,4))
population_df.head()


# In[ ]:


GDP_df = pd.read_csv('../input/GDP by Country.csv', skiprows = range(0,4))
GDP_df.head()


# In[ ]:


GDPData_df = pd.merge(GDP_df, population_df, on= ['Country Code','Country Name'], how='inner')
GDPData_df.head()


# In[ ]:


GDPDataCurated_df = GDPData_df.drop(['Indicator Name_x','Indicator Code_x','Indicator Name_y','Indicator Code_y','Unnamed: 62_y','Unnamed: 62_x'], axis = 1)
GDPDataCurated_df.head()
# GDP_df is the X column, Population_df is the Y column.


# In[ ]:


GDPperCapita_df = pd.DataFrame()
for col in GDPDataCurated_df.columns:
    if col.endswith("Name"):
        country = col[:]
        GDPperCapita_df[country] = GDPDataCurated_df[country]
    if col.endswith("_x"):
        year = col[:4]
        GDPperCapita_df[year] = GDPDataCurated_df[year + '_x']/GDPDataCurated_df[year + '_y']
    if col.endswith("Code"):
        code = col[:]
        GDPperCapita_df['Units:- US$/person' + code] = GDPDataCurated_df[code]
        
GDPDataCurated_df.head()


# In[ ]:


GDPperCapita_df.head()      


# In[ ]:


GDP_Stacked_df = pd.melt(GDPperCapita_df,id_vars=['Country Name','Units:- US$/personCountry Code'])
GDP_Stacked_df.head()


# In[ ]:


WomenMakingInformedChoices_df = pd.read_csv('../input/WomenMakingInformedChoicestoReproductiveHealthCare.csv', skiprows = range(0,4))
WomenMakingInformedChoices_df = pd.melt(WomenMakingInformedChoices_df,id_vars = ['Country Name','Country Code','Indicator Name','Indicator Code'])
WomenMakingInformedChoices_df


# In[ ]:


RuralPopulationPerCent_df = pd.read_csv('../input/RuralPopulationofTotalPopulation.csv', skiprows = range(0,4))
RuralPopulationPerCent_df = RuralPopulationPerCent_df.drop(['Unnamed: 62'],axis = 1)
RuralPopulationPerCent_df = pd.melt(RuralPopulationPerCent_df,id_vars = ['Country Name','Country Code','Indicator Name','Indicator Code'])
RuralPopulationPerCent_df.head()


# In[ ]:


PublicEduRatioGDP_df = pd.read_csv('../input/public-education-expenditure-as-share-of-gdp.csv')
#PublicEduRatioGDP_df = pd.melt(PublicEduRatioGDP_df,id_vars = ['Country Name','Country Code','Indicator Name','Indicator Code'])
#PublicEduRatioGDP_df

#IGNORE THIS DATA SET IN THE CURRENT SET OF RESULTS.


# In[ ]:


LegalRightsStrength_df = pd.read_csv('../input/LegalRightsStrengthIndex.csv', skiprows = range(0,4))
LegalRightsStrength_df = pd.melt(LegalRightsStrength_df,id_vars = ['Country Name','Country Code','Indicator Name','Indicator Code'])
LegalRightsStrength_df.head()


# In[ ]:


CreditToPrivateSector_df = pd.read_csv('../input/DomesticCreditToPrivateSector.csv', skiprows = range(0,4))
CreditToPrivateSector_df = CreditToPrivateSector_df.drop(['Unnamed: 62'],axis = 1)
CreditToPrivateSector_df = pd.melt(CreditToPrivateSector_df,id_vars = ['Country Name','Country Code','Indicator Name','Indicator Code'])
CreditToPrivateSector_df.head()


# In[ ]:


BirthsAttendedbySkilledStaff_df = pd.read_csv('../input/BirthsAttendedbySkilledHealthStaffofTotal.csv', skiprows = range(0,4))
BirthsAttendedbySkilledStaff_df = pd.melt(BirthsAttendedbySkilledStaff_df,id_vars = ['Country Name','Country Code','Indicator Name','Indicator Code'])
BirthsAttendedbySkilledStaff_df.head()


# In[ ]:


ATMMachinesRatio_df = pd.read_csv('../input/ATMMachines_Per100000Adults.csv', skiprows = range(0,4))
ATMMachinesRatio_df = pd.melt(ATMMachinesRatio_df,id_vars = ['Country Name','Country Code','Indicator Name','Indicator Code'])
ATMMachinesRatio_df.head()


# In[ ]:


AgriculturalMachines_df = pd.read_csv('../input/AgriculturalMachinery_PerUnitofArableLand.csv', skiprows = range(0,4))
AgriculturalMachines_df = AgriculturalMachines_df.drop(['Unnamed: 62'],axis = 1)
AgriculturalMachines_df = pd.melt(AgriculturalMachines_df,id_vars = ['Country Name','Country Code','Indicator Name','Indicator Code'])
AgriculturalMachines_df.head()


# In[ ]:


LiteracyRateAdult_df = pd.read_csv('../input/AdultPopulation_Literate.csv', skiprows = range(0,4))
#AgriculturalMachines_df = AgriculturalMachines_df.drop(['Unnamed: 62'],axis = 1)
LiteracyRateAdult_df = pd.melt(LiteracyRateAdult_df,id_vars = ['Country Name','Country Code','Indicator Name','Indicator Code'])
LiteracyRateAdult_df.head()


# In[ ]:


AccountsRatioFinancialInst_df = pd.read_csv('../input/AccountAtaFinancialInstitutionMale15Adults.csv', skiprows = range(0,4))
AccountsRatioFinancialInst_df = AccountsRatioFinancialInst_df.drop(['Unnamed: 62'],axis = 1)
AccountsRatioFinancialInst_df = pd.melt(AccountsRatioFinancialInst_df,id_vars = ['Country Name','Country Code','Indicator Name','Indicator Code'])
AccountsRatioFinancialInst_df.head()


# In[ ]:





# In[ ]:


GDP_Stacked_df['WomenMakingInformedChoices_df'] = WomenMakingInformedChoices_df.value
GDP_Stacked_df.head()


# In[ ]:


GDP_Stacked_df['RuralPopulationPerCent_df'] = RuralPopulationPerCent_df.value
GDP_Stacked_df.head()


# In[ ]:


GDP_Stacked_df['LegalRightsStrength_df'] = LegalRightsStrength_df.value
GDP_Stacked_df.head()


# In[ ]:


GDP_Stacked_df['CreditToPrivateSector_df'] = CreditToPrivateSector_df.value
GDP_Stacked_df.head()


# In[ ]:


GDP_Stacked_df['BirthsAttendedbySkilledStaff_df'] = BirthsAttendedbySkilledStaff_df.value
GDP_Stacked_df.head()


# In[ ]:


GDP_Stacked_df['ATMMachinesRatio_df'] = ATMMachinesRatio_df.value
GDP_Stacked_df.head()


# In[ ]:


GDP_Stacked_df['AgriculturalMachines_df'] = AgriculturalMachines_df.value
GDP_Stacked_df.head()


# In[ ]:


GDP_Stacked_df['LiteracyRateAdult_df'] = LiteracyRateAdult_df.value
GDP_Stacked_df.head()


# In[ ]:


GDP_Stacked_df['AccountsRatioFinancialInst_df'] = AccountsRatioFinancialInst_df.value
GDP_Stacked_df


# In[ ]:


sns.regplot(x="value", y="WomenMakingInformedChoices_df", data=GDP_Stacked_df)


# In[ ]:


sns.regplot(x="value", y="WomenMakingInformedChoices_df", data=GDP_Stacked_df);


# In[ ]:


sns.regplot(x="value", y="RuralPopulationPerCent_df", data=GDP_Stacked_df);


# In[ ]:


sns.regplot(x="value", y="LegalRightsStrength_df", data=GDP_Stacked_df);


# In[ ]:


sns.regplot(x="value", y="CreditToPrivateSector_df", data=GDP_Stacked_df);


# In[ ]:


sns.regplot(x="value", y="BirthsAttendedbySkilledStaff_df", data=GDP_Stacked_df);


# In[ ]:


sns.regplot(x="value", y="ATMMachinesRatio_df", data=GDP_Stacked_df);


# In[ ]:


sns.regplot(x="value", y="AgriculturalMachines_df", data=GDP_Stacked_df);


# In[ ]:


sns.regplot(x="value", y="LiteracyRateAdult_df", data=GDP_Stacked_df);


# In[ ]:


sns.regplot(x="value", y="AccountsRatioFinancialInst_df", data=GDP_Stacked_df);


# In[ ]:


print(GDP_Stacked_df.isnull().any())
# where value = GDP_per_Capita


# In[ ]:


# Counting missing values in a column

GDP_Stacked_df.dropna(subset=['value'],inplace = True)
print(GDP_Stacked_df['value'].isnull().sum())
print(GDP_Stacked_df['value'].notnull().sum())


# In[ ]:


# Counting missing values in a column
GDP_Stacked_df['WomenMakingInformedChoices_df'].fillna(value=GDP_Stacked_df['WomenMakingInformedChoices_df'].mean(),inplace=True)
print(GDP_Stacked_df['WomenMakingInformedChoices_df'].isnull().sum())
print(GDP_Stacked_df['WomenMakingInformedChoices_df'].notnull().sum())


# In[ ]:


# Counting missing values in a column
GDP_Stacked_df['RuralPopulationPerCent_df'].fillna(value=GDP_Stacked_df['RuralPopulationPerCent_df'].mean(),inplace=True)
print(GDP_Stacked_df['RuralPopulationPerCent_df'].isnull().sum())
print(GDP_Stacked_df['RuralPopulationPerCent_df'].notnull().sum())


# In[ ]:


# Counting missing values in a column
GDP_Stacked_df['LegalRightsStrength_df'].fillna(value=GDP_Stacked_df['LegalRightsStrength_df'].mean(),inplace=True)
print(GDP_Stacked_df['LegalRightsStrength_df'].isnull().sum())
print(GDP_Stacked_df['LegalRightsStrength_df'].notnull().sum())


# In[ ]:


# Counting missing values in a column
GDP_Stacked_df['CreditToPrivateSector_df'].fillna(value=GDP_Stacked_df['CreditToPrivateSector_df'].mean(),inplace=True)
print(GDP_Stacked_df['CreditToPrivateSector_df'].isnull().sum())
print(GDP_Stacked_df['CreditToPrivateSector_df'].notnull().sum())


# In[ ]:


# Counting missing values in a column
GDP_Stacked_df['BirthsAttendedbySkilledStaff_df'].fillna(value=GDP_Stacked_df['BirthsAttendedbySkilledStaff_df'].mean(),inplace=True)
print(GDP_Stacked_df['BirthsAttendedbySkilledStaff_df'].isnull().sum())
print(GDP_Stacked_df['BirthsAttendedbySkilledStaff_df'].notnull().sum())


# In[ ]:


# Counting missing values in a column
GDP_Stacked_df['ATMMachinesRatio_df'].fillna(value=GDP_Stacked_df['ATMMachinesRatio_df'].mean(),inplace=True)
print(GDP_Stacked_df['ATMMachinesRatio_df'].isnull().sum())
print(GDP_Stacked_df['ATMMachinesRatio_df'].notnull().sum())


# In[ ]:


# Counting missing values in a column
GDP_Stacked_df['AgriculturalMachines_df'].fillna(value=GDP_Stacked_df['AgriculturalMachines_df'].mean(),inplace=True)
print(GDP_Stacked_df['AgriculturalMachines_df'].isnull().sum())
print(GDP_Stacked_df['AgriculturalMachines_df'].notnull().sum())


# In[ ]:


# Counting missing values in a column
GDP_Stacked_df['LiteracyRateAdult_df'].fillna(value=GDP_Stacked_df['LiteracyRateAdult_df'].mean(),inplace=True)
print(GDP_Stacked_df['LiteracyRateAdult_df'].isnull().sum())
print(GDP_Stacked_df['LiteracyRateAdult_df'].notnull().sum())


# In[ ]:


# Counting missing values in a column
GDP_Stacked_df['AccountsRatioFinancialInst_df'].fillna(value=GDP_Stacked_df['AccountsRatioFinancialInst_df'].mean(),inplace=True)
print(GDP_Stacked_df['AccountsRatioFinancialInst_df'].isnull().sum())
print(GDP_Stacked_df['AccountsRatioFinancialInst_df'].notnull().sum())


# In[ ]:


X = GDP_Stacked_df[['CreditToPrivateSector_df','WomenMakingInformedChoices_df','RuralPopulationPerCent_df','BirthsAttendedbySkilledStaff_df','ATMMachinesRatio_df','AgriculturalMachines_df','LiteracyRateAdult_df','AccountsRatioFinancialInst_df','LegalRightsStrength_df']]
X = np.array(X)
X


# In[ ]:


y = GDP_Stacked_df['value']
y = np.array(y)
y


# In[ ]:


X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, y, test_size=0.15, random_state=0)
X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.15/0.85, random_state=0)


# In[ ]:


model = KNeighborsRegressor(n_neighbors=3)
model.fit(X_train, Y_train)


# In[ ]:


y_pred = model.predict(X_val)
y_actual = Y_val
mean_squared_error(y_actual, y_pred)


# In[ ]:


# Test R^2
print(model.score(X_val, y_actual))
plt.scatter(y_pred, y_actual, marker='.')
plt.xlabel('Predicted y')
plt.ylabel('Actual y')
plt.show()


# In[ ]:


y_pred_test = model.predict(X_test)
y_actual_test = Y_test
mean_squared_error(y_actual_test, y_pred_test)


# In[ ]:


# Test R^2
print(model.score(X_test, y_actual_test))
plt.scatter(y_pred_test, y_actual_test, marker='.')
plt.xlabel('Predicted y')
plt.ylabel('Actual y')
plt.show()


# In[ ]:


from pandas import Series, DataFrame
from sklearn.linear_model import LinearRegression


# In[ ]:


lreg = LinearRegression()


# In[ ]:


X = GDP_Stacked_df[['CreditToPrivateSector_df','WomenMakingInformedChoices_df','RuralPopulationPerCent_df','BirthsAttendedbySkilledStaff_df','ATMMachinesRatio_df','AgriculturalMachines_df','LiteracyRateAdult_df','AccountsRatioFinancialInst_df','LegalRightsStrength_df']]
# splitting into training and val sets for cross validation
X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, y, test_size=0.15, random_state=0)
X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.15/0.85, random_state=0)


# In[ ]:


#training the model
lreg.fit(X_train,Y_train)


# In[ ]:


pred = lreg.predict(X_val)


# In[ ]:


# calculating MSE
mse = np.mean((pred - Y_val)**2)


# In[ ]:


from pandas import Series, DataFrame
lreg.score(X_val,Y_val)


# In[ ]:


x_plot = plt.scatter(pred, (pred - Y_val), c='b')
plt.hlines(y=0, xmin= -1000, xmax=5000)

plt.title('Residual plot')


# In[ ]:


predictors = X_train.columns


# In[ ]:


coef = Series(lreg.coef_,predictors).sort_values()


# In[ ]:


coef.plot(kind='bar', title='Model Coefficients')


# In[ ]:


#Ridge Regression 

from sklearn.linear_model import Ridge
ridgeReg = Ridge(alpha=0.0001, normalize=True)
ridgeReg.fit(X_train,Y_train)


# In[ ]:


pred = ridgeReg.predict(X_val)


# In[ ]:


mse = np.mean((pred - Y_val)**2)


# In[ ]:


ridgeReg.score(X_train,Y_train)


# In[ ]:


ridgeReg.score(X_test,Y_test)


# In[ ]:


from sklearn import tree
from sklearn.tree import DecisionTreeRegressor


# In[ ]:


X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, y, test_size=0.15, random_state=0)
X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.15/0.85, random_state=0)


# In[ ]:


# Fit the regression model
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor()
regr_1.fit(X_train, Y_train)
regr_2.fit(X_train, Y_train)


# In[ ]:


regr_1.score(X_val,Y_val)


# In[ ]:


regr_2.score(X_val,Y_val)


# In[ ]:


regr_2.score(X_test,Y_test)


# In[ ]:


regr_2.feature_importances_


# In[ ]:


list(zip(regr_2.feature_importances_,X.columns))


# In[ ]:


from sklearn import ensemble
from sklearn.ensemble import RandomForestRegressor


# In[ ]:


X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, y, test_size=0.15, random_state=0)
X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.15/0.85, random_state=0)


# In[ ]:


regr_1 = RandomForestRegressor(n_estimators=500, min_samples_leaf=1)
regr_2 = RandomForestRegressor()
regr_1.fit(X_train, Y_train)
regr_2.fit(X_train, Y_train)


# In[ ]:


regr_1.score(X_val,Y_val)


# In[ ]:


regr_2.score(X_val,Y_val)


# In[ ]:


regr_2.feature_importances_


# In[ ]:


list(zip(regr_2.feature_importances_,X.columns))


# In[ ]:


regr_1.score(X_test,Y_test)


# In[ ]:


regr_2.feature_importances_


# In[ ]:


list(zip(regr_1.feature_importances_,X.columns))


# In[ ]:




