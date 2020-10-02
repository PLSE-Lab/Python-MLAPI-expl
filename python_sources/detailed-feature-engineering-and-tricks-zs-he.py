#!/usr/bin/env python
# coding: utf-8
This was ZS data science challenge, on hackerearth. Where I bagged 37 rank out of 18000 people. Here I had to predict sales for a country, based on lots of features and some extra datasets. Details of which are explained below.It might be a little lengthy, but have patience, it'd be worth it :P
# # Importing all libraries and modules

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import svm
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import Imputer
import seaborn as sns


get_ipython().run_line_magic('matplotlib', 'inline')


# ### We've following datasets to play

# #### Holiday

# In[ ]:


show_holiday=pd.read_excel('../input/holidays.xlsx')


# In[ ]:


show_holiday.head()


# As we can see, we've Date(not in the best format for processing) and then we've country and the Holiday we've for that country on that day. Which might affect the sales price we're to predict for that country

# #### Promotional expenses

# In[ ]:


show_promotional=pd.read_csv('../input/promotional_expense.csv')


# In[ ]:


show_promotional.head()

We've expense price, based on country and the product type. Expense price is primarily all the price involving advertisement and labor work. 
# ## Our dataset includes, holiday and promotional expenses(more on these datasets later)

# In[ ]:


promo=pd.read_csv('../input/promotional_expense.csv')


# #### This is our main training dataset, which has following fields as shown in data.head()

# In[ ]:


data=pd.read_csv('../input/yds_train2018.csv')


# ### Starting feature Engineering and creating training and test data sets
# 

# In[ ]:


data=data.drop(['S_No'],axis=1)


# In[ ]:


data.head()


# In[ ]:


data.iloc[1]


# ### Data has been grouped, as many of the countries had several week for a month , also we don't have Week and merchant ID feature in test data. So taking em out and grouping training data seemed like the best course of action

# In[ ]:


mod_data=data.groupby(['Year','Month','Product_ID','Country'])['Sales'].sum()


# In[ ]:


mod_data=mod_data.reset_index()


# In[ ]:


mod_data.head()


# In[ ]:


promo.head()


# ### Now we wanna merge data, promotional data with Training data. As we simply wanna use all the features we can get. So we merge the data using left join based on common feature of training and promotional data

# In[ ]:


new=mod_data.merge(promo,left_on=['Year','Month','Product_ID','Country'],
                   right_on=['Year','Month','Product_Type','Country'],
                  how='left')


# In[ ]:


new['Expense_Price'].shape


# ##### To be noticed, all the countries don't have expense price in the dataset. So for them we are simply filling them with mean. As they gotta have some expense price after all

# In[ ]:


# imp=Imputer(missing_values='NaN',strategy='mean',axis=0)
# new['Expense_Price']=imp.fit_transform([new['Expense_Price']])
new['Expense_Price'].fillna(new['Expense_Price'].mean(),inplace=True)


# #### Now we wanna import holidays, in order to merge it with training data set. Again, it'll start with processing date of Holidays in appropriate way and performing join based on common feature of holidays and training dataset

# In[ ]:


holidays=pd.read_excel('../input/holidays.xlsx',parse_dates=['Date'])


# In[ ]:


holidays.head()


# In[ ]:


holidays['Date'][0].year


# In[ ]:


holidays['Year']=holidays['Date'].map(lambda x: x.year)


# In[ ]:


holidays['Month']=holidays['Date'].map(lambda x: x.month)


# In[ ]:


holidays=holidays.drop(['Date'],axis=1)


# In[ ]:


holidays.head()


# In[ ]:


holiday_new=holidays.groupby(['Year','Month','Country'])['Holiday'].count()


# In[ ]:


holiday_new=holiday_new.reset_index()


# In[ ]:


new=new.drop(['Product_Type'],axis=1)


# In[ ]:


new.head()


# #### We've done enough cleaning and aligning holidays dataset, and now we wanna perform merge with the dataset, previously merged with promotional expenses

# In[ ]:


calc=new.merge(holiday_new,on=['Year','Month','Country'],
              how='left')


# In[ ]:


calc.head()


# In[ ]:


calc['Holiday'].fillna(0,inplace=True)


# In[ ]:


calc.shape


# In[ ]:


mod_data=calc


# In[ ]:


mod_data.head()


# In[ ]:


mod_data.columns


# In[ ]:


plt.plot(mod_data['Sales'],mod_data['Expense_Price'])


# In[ ]:


plt.plot(mod_data['Sales'],mod_data['Holiday'])


# In[ ]:


features=['Year', 'Month', 'Product_ID', 'Country', 'Sales', 'Expense_Price',
       'Holiday']
sns.heatmap(mod_data[features].corr(), linewidths=0.25, vmax=1.0, square=True, cmap="BuGn_r", linecolor='k', annot=True)


# ## Now this is one of the trick that really helped me increase my score.
# 
# #### I divided my dataset, according to the country, and trained my algorithm seperately on each country, it helped my algorithm generalize to a country, better.

# ## Diving DataSet
# 

# In[ ]:


mod_data['Country'].unique()


# In[ ]:


argentina=mod_data.loc[mod_data['Country']=='Argentina']
belgium=mod_data.loc[mod_data['Country']=='Belgium']
columbia=mod_data.loc[mod_data['Country']=='Columbia']
denmark=mod_data.loc[mod_data['Country']=='Denmark']
england=mod_data.loc[mod_data['Country']=='England']
finland=mod_data.loc[mod_data['Country']=='Finland']


# In[ ]:


argentina.head()


# ##### Ofc, we don't need a country feature explicitly , if our dataframe just contains a feature regarding a country. 

# In[ ]:


argentina.drop(['Country'],inplace=True,axis=1)
belgium.drop(['Country'],inplace=True,axis=1)
columbia.drop(['Country'],inplace=True,axis=1)
denmark.drop(['Country'],inplace=True,axis=1)
england.drop(['Country'],inplace=True,axis=1)
finland.drop(['Country'],inplace=True,axis=1)


# ### Another one of the tricks, that helped me increase my score. Divide a country wrt product id, then train algorithms, based on each product id

# ### Designing algo of all Countries with respective Product ID

# #####  Linear Regression

# In[ ]:


qalg_ar1=linear_model.LinearRegression() #For Argentina 3 prod id that is 1,2,3
qalg_ar2=linear_model.LinearRegression()
qalg_ar3=linear_model.LinearRegression()

qalg_bl2=linear_model.LinearRegression() #For Belgium 1 prod id that is 2

qalg_co1=linear_model.LinearRegression() #for columbia 3 prod id that is 1,2,3
qalg_co2=linear_model.LinearRegression()
qalg_co3=linear_model.LinearRegression()

qalg_dn2=linear_model.LinearRegression() #For Denmark 1 prod id that is 2

qalg_fn4=linear_model.LinearRegression() #For Finland 1 prod id that is 4


qalg_en4=linear_model.LinearRegression() #For England 2 prod id that is 4,5
qalg_en5=linear_model.LinearRegression()

AdaBoostRegressor(n_estimators=100)


# #####  Elastic Net

# In[ ]:


walg_ar1=AdaBoostRegressor(n_estimators=100) #For Argentina 3 prod id that is 1,2,3
walg_ar2=AdaBoostRegressor(n_estimators=100)
walg_ar3=AdaBoostRegressor(n_estimators=100)

walg_bl2=AdaBoostRegressor(n_estimators=100) #For Belgium 1 prod id that is 2

walg_co1=AdaBoostRegressor(n_estimators=100) #for columbia 3 prod id that is 1,2,3
walg_co2=AdaBoostRegressor(n_estimators=100)
walg_co3=AdaBoostRegressor(n_estimators=100)

walg_dn2=AdaBoostRegressor(n_estimators=100) #For Denmark 1 prod id that is 2

walg_fn4=AdaBoostRegressor(n_estimators=100) #For Finland 1 prod id that is 4


walg_en4=AdaBoostRegressor(n_estimators=100) #For England 2 prod id that is 4,5
walg_en5=AdaBoostRegressor(n_estimators=100)




# #####  XGBoost
# 

# In[ ]:


ealg_ar1=XGBRegressor() #For Argentina 3 prod id that is 1,2,3
ealg_ar2=XGBRegressor()
ealg_ar3=XGBRegressor()

ealg_bl2=XGBRegressor() #For Belgium 1 prod id that is 2

ealg_co1=XGBRegressor() #for columbia 3 prod id that is 1,2,3
ealg_co2=XGBRegressor()
ealg_co3=XGBRegressor()

ealg_dn2=XGBRegressor() #For Denmark 1 prod id that is 2

ealg_fn4=XGBRegressor() #For Finland 1 prod id that is 4


ealg_en4=XGBRegressor() #For England 2 prod id that is 4,5
ealg_en5=XGBRegressor()
# XGBRegressor()



# #####  Lasso
# 

# In[ ]:


ralg_ar1=linear_model.Lasso(alpha=0.1) #For Argentina 3 prod id that is 1,2,3
ralg_ar2=linear_model.Lasso(alpha=0.1)
ralg_ar3=linear_model.Lasso(alpha=0.1)

ralg_bl2=linear_model.Lasso(alpha=0.1) #For Belgium 1 prod id that is 2

ralg_co1=linear_model.Lasso(alpha=0.1) #for columbia 3 prod id that is 1,2,3
ralg_co2=linear_model.Lasso(alpha=0.1)
ralg_co3=linear_model.Lasso(alpha=0.1)

ralg_dn2=linear_model.Lasso(alpha=0.1) #For Denmark 1 prod id that is 2

ralg_fn4=linear_model.Lasso(alpha=0.1) #For Finland 1 prod id that is 4


ralg_en4=linear_model.Lasso(alpha=0.1) #For England 2 prod id that is 4,5
ralg_en5=linear_model.Lasso(alpha=0.1)



# ### Starting with argentina

# In[ ]:


argentina['Product_ID'].unique()


# In[ ]:


ar_pd1=argentina[argentina['Product_ID']==1]
ar_pd2=argentina[argentina['Product_ID']==2]
ar_pd3=argentina[argentina['Product_ID']==3]


ar_pd1.drop(['Product_ID'],axis=1,inplace=True)
ar_pd2.drop(['Product_ID'],axis=1,inplace=True)
ar_pd3.drop(['Product_ID','Expense_Price'],axis=1,inplace=True)


# In[ ]:


ar_pd1_feat=ar_pd1.drop(['Sales'],axis=1)
ar_pd2_feat=ar_pd2.drop(['Sales'],axis=1)
ar_pd3_feat=ar_pd3.drop(['Sales'],axis=1)


ar_pd1_label=ar_pd1['Sales']
ar_pd2_label=ar_pd2['Sales']
ar_pd3_label=ar_pd3['Sales']


# In[ ]:


ar_pd1_feat.head()


# In[ ]:





# In[ ]:


qalg_ar1.fit(ar_pd1_feat,ar_pd1_label)
qalg_ar2.fit(ar_pd2_feat,ar_pd2_label)
qalg_ar3.fit(ar_pd3_feat,ar_pd3_label)

walg_ar1.fit(ar_pd1_feat,ar_pd1_label)
walg_ar2.fit(ar_pd2_feat,ar_pd2_label)
walg_ar3.fit(ar_pd3_feat,ar_pd3_label)

ealg_ar1.fit(ar_pd1_feat,ar_pd1_label)
ealg_ar2.fit(ar_pd2_feat,ar_pd2_label)
ealg_ar3.fit(ar_pd3_feat,ar_pd3_label)

ralg_ar1.fit(ar_pd1_feat,ar_pd1_label)
ralg_ar2.fit(ar_pd2_feat,ar_pd2_label)
ralg_ar3.fit(ar_pd3_feat,ar_pd3_label)


# ###  Now Belgium

# In[ ]:


belgium['Product_ID'].unique()


# In[ ]:


belgium=belgium.drop(['Product_ID'],axis=1)


# In[ ]:


belgium.columns


# In[ ]:


belgium.head()


# In[ ]:


bel_feat=belgium.drop(['Sales'],axis=1)

bel_label=belgium['Sales']


# In[ ]:


qalg_bl2.fit(bel_feat,bel_label)
walg_bl2.fit(bel_feat,bel_label)
ealg_bl2.fit(bel_feat,bel_label)
ralg_bl2.fit(bel_feat,bel_label)


# ### Now Columbia

# In[ ]:


columbia['Product_ID'].unique()

co_pd3
# In[ ]:


co_pd1=columbia[columbia['Product_ID']==1]
co_pd2=columbia[columbia['Product_ID']==2]
co_pd3=columbia[columbia['Product_ID']==3]


co_pd1.drop(['Product_ID'],axis=1,inplace=True)
co_pd2.drop(['Product_ID'],axis=1,inplace=True)
co_pd3.drop(['Product_ID','Expense_Price'],axis=1,inplace=True)


# In[ ]:


co_pd1_feat=co_pd1.drop(['Sales'],axis=1)
co_pd2_feat=co_pd2.drop(['Sales'],axis=1)
co_pd3_feat=co_pd3.drop(['Sales'],axis=1)


co_pd1_label=co_pd1['Sales']
co_pd2_label=co_pd2['Sales']
co_pd3_label=co_pd3['Sales']


# In[ ]:


qalg_co1.fit(co_pd1_feat,co_pd1_label)
qalg_co2.fit(co_pd2_feat,co_pd2_label)
qalg_co3.fit(co_pd3_feat,co_pd3_label)

walg_co1.fit(co_pd1_feat,co_pd1_label)
walg_co2.fit(co_pd2_feat,co_pd2_label)
walg_co3.fit(co_pd3_feat,co_pd3_label)

ealg_co1.fit(co_pd1_feat,co_pd1_label)
ealg_co2.fit(co_pd2_feat,co_pd2_label)
ealg_co3.fit(co_pd3_feat,co_pd3_label)

ralg_co1.fit(co_pd1_feat,co_pd1_label)
ralg_co2.fit(co_pd2_feat,co_pd2_label)
ralg_co3.fit(co_pd3_feat,co_pd3_label)


# ### Now Denmark

# In[ ]:


denmark['Product_ID'].unique()


# In[ ]:


denmark.drop(['Product_ID'],axis=1,inplace=True)


# In[ ]:


denmark.head()


# In[ ]:


den_pd2_feat=denmark.drop(['Sales'],axis=1)

den_pd2_label=denmark['Sales']


# In[ ]:


qalg_dn2.fit(den_pd2_feat,den_pd2_label)

walg_dn2.fit(den_pd2_feat,den_pd2_label)

ealg_dn2.fit(den_pd2_feat,den_pd2_label)

ralg_dn2.fit(den_pd2_feat,den_pd2_label)


# ### Now Finland

# In[ ]:


finland['Product_ID'].unique()


# In[ ]:


finland.head()


# In[ ]:


finland=finland.drop(['Product_ID'],axis=1)


# In[ ]:


fn_pd4_feat=finland.drop(['Sales'],axis=1)

fn_pd4_label=finland['Sales']


# In[ ]:


qalg_fn4.fit(fn_pd4_feat,fn_pd4_label)

walg_fn4.fit(fn_pd4_feat,fn_pd4_label)

ealg_fn4.fit(fn_pd4_feat,fn_pd4_label)

ralg_fn4.fit(fn_pd4_feat,fn_pd4_label)


# ### Now England

# In[ ]:


england['Product_ID'].unique()


# In[ ]:


en_pd4=england[england['Product_ID']==4]
en_pd4=en_pd4.drop(['Product_ID'],axis=1)

en_pd5=england[england['Product_ID']==5]
en_pd5=en_pd5.drop(['Product_ID'],axis=1)


# In[ ]:


en_pd4.head()


# In[ ]:


en_pd4_feat=en_pd4.drop(['Sales'],axis=1)
en_pd5_feat=en_pd5.drop(['Sales'],axis=1)

en_pd4_label=en_pd4['Sales']
en_pd5_label=en_pd5['Sales']


# In[ ]:


qalg_en4.fit(en_pd4_feat,en_pd4_label)
qalg_en5.fit(en_pd5_feat,en_pd5_label)

walg_en4.fit(en_pd4_feat,en_pd4_label)
walg_en5.fit(en_pd5_feat,en_pd5_label)

ealg_en4.fit(en_pd4_feat,en_pd4_label)
ealg_en5.fit(en_pd5_feat,en_pd5_label)

ralg_en4.fit(en_pd4_feat,en_pd4_label)
ralg_en5.fit(en_pd5_feat,en_pd5_label)


# ### We've to apply same process on the test data set too, divide it by country then product id, then predict using algo trained with same country and product id of training data set. 

# ## This is final test data set

# In[ ]:


final_jest=pd.read_csv('../input/yds_test2018.csv')
combine=pd.read_csv('../input/yds_test2018.csv')


# In[ ]:


final_jest.head()


# In[ ]:


final_jest_new=final_jest.merge(promo,left_on=['Year','Month','Product_ID','Country'],
                   right_on=['Year','Month','Product_Type','Country'],
                  how='left')


# ### Final test data is merged, with expense price, NA values filled with mean, same with Holidays, NA values filled with 0

# In[ ]:


final_jest_new.shape
final_jest_new['Expense_Price'].fillna(final_jest_new['Expense_Price'].mean,inplace=True)


# In[ ]:


final_jest_update=final_jest_new.merge(holiday_new,on=['Year','Month','Country'],
              how='left')


# In[ ]:


final_jest_update['Holiday'].fillna(0,inplace=True)


# In[ ]:


final_jest_update.head()


# In[ ]:


# mod_data=data.groupby(['Year','Month','Product_ID','Country'])['Sales'].stes
test_data=final_jest_update.drop(['S_No'],axis=1)


# In[ ]:


# test_data=test_data.reset_index()
test_data.head()


# In[ ]:


test_data['Country'].unique()


# In[ ]:


targentina=test_data.loc[test_data['Country']=='Argentina']
tbelgium=test_data.loc[test_data['Country']=='Belgium']
tcolumbia=test_data.loc[test_data['Country']=='Columbia']
tdenmark=test_data.loc[test_data['Country']=='Denmark']
tengland=test_data.loc[test_data['Country']=='England']
tfinland=test_data.loc[test_data['Country']=='Finland']


# In[ ]:


targentina.head()


# In[ ]:


targentina.head()


# In[ ]:


targentina.drop(['Country'],inplace=True,axis=1)
tbelgium.drop(['Country'],inplace=True,axis=1)
tcolumbia.drop(['Country'],inplace=True,axis=1)
tdenmark.drop(['Country'],inplace=True,axis=1)
tengland.drop(['Country'],inplace=True,axis=1)
tfinland.drop(['Country'],inplace=True,axis=1)


# ### TEST-Designing algo of all Countries with respective Product ID

# ### Test-Starting with argentina

# In[ ]:


targentina['Product_ID'].unique()


# In[ ]:


tar_pd1=targentina[targentina['Product_ID']==1]
tar_pd2=targentina[targentina['Product_ID']==2]
tar_pd3=targentina[targentina['Product_ID']==3]


tar_pd1.drop(['Product_ID','Product_Type'],axis=1,inplace=True)
tar_pd2.drop(['Product_ID','Product_Type'],axis=1,inplace=True)
tar_pd3.drop(['Product_ID','Expense_Price','Product_Type'],axis=1,inplace=True)


# In[ ]:


ar_pd1_feat.shape


# ## The last trick I used, was combine various prediction, I used 0.55 of prediction with linear regression and 0.45 with Elastic net, Feel free to try with other combination.

# In[ ]:


tar_pd1_feat=tar_pd1.drop(['Sales'],axis=1)
tar_pd2_feat=tar_pd2.drop(['Sales'],axis=1)
tar_pd3_feat=tar_pd3.drop(['Sales'],axis=1)


tar_pd1_label=0.55*qalg_ar1.predict(tar_pd1_feat)+0.45*walg_ar1.predict(tar_pd1_feat)
tar_pd2_label=0.55*qalg_ar2.predict(tar_pd2_feat)+0.45*walg_ar2.predict(tar_pd2_feat)
tar_pd3_label=0.55*qalg_ar3.predict(tar_pd3_feat)+0.45*walg_ar3.predict(tar_pd3_feat)


# In[ ]:


tar_pd1_feat.head()


# In[ ]:


tar_pd2_feat.head()


# ### TEST  Now Belgium

# In[ ]:


tbelgium['Product_ID'].unique()


# In[ ]:


tbelgium=tbelgium.drop(['Product_ID','Product_Type'],axis=1)


# In[ ]:


tbelgium.columns


# In[ ]:


tbelgium.head()


# In[ ]:


tbel_feat=tbelgium.drop(['Sales'],axis=1)

tbel_label=0.55*qalg_bl2.predict(tbel_feat)+0.45*walg_bl2.predict(tbel_feat)


# In[ ]:


tbel_label


# ### TEST Now Columbia

# In[ ]:


tcolumbia['Product_ID'].unique()

co_pd3
# In[ ]:


tco_pd1=tcolumbia[tcolumbia['Product_ID']==1]
tco_pd2=tcolumbia[tcolumbia['Product_ID']==2]
tco_pd3=tcolumbia[tcolumbia['Product_ID']==3]


tco_pd1.drop(['Product_ID','Product_Type'],axis=1,inplace=True)
tco_pd2.drop(['Product_ID','Product_Type'],axis=1,inplace=True)
tco_pd3.drop(['Product_ID','Expense_Price','Product_Type'],axis=1,inplace=True)


# In[ ]:


tco_pd1_feat=tco_pd1.drop(['Sales'],axis=1)
tco_pd2_feat=tco_pd2.drop(['Sales'],axis=1)
tco_pd3_feat=tco_pd3.drop(['Sales'],axis=1)


tco_pd1_label=0.55*qalg_co1.predict(tco_pd1_feat)+0.45*walg_co1.predict(tco_pd1_feat)

tco_pd2_label=0.55*qalg_co2.predict(tco_pd2_feat)+0.45*walg_co2.predict(tco_pd2_feat)

tco_pd3_label=0.55*qalg_co3.predict(tco_pd3_feat)+0.45*walg_co3.predict(tco_pd3_feat)


# ### TEST Now Denmark

# In[ ]:


tdenmark['Product_ID'].unique()


# In[ ]:


tdenmark.drop(['Product_ID','Product_Type'],axis=1,inplace=True)


# In[ ]:


tdenmark.head()


# In[ ]:


tden_pd2_feat=tdenmark.drop(['Sales'],axis=1)

tden_pd2_label=0.55*qalg_dn2.predict(tden_pd2_feat)+0.45*walg_dn2.predict(tden_pd2_feat)


# ### TEST Now Finland

# In[ ]:


tfinland['Product_ID'].unique()


# In[ ]:


tfinland.head()


# In[ ]:


tfinland=tfinland.drop(['Product_ID','Product_Type'],axis=1)


# In[ ]:


tfn_pd4_feat=tfinland.drop(['Sales'],axis=1)

tfn_pd4_label=0.55*qalg_fn4.predict(tfn_pd4_feat)+0.45*walg_fn4.predict(tfn_pd4_feat)


# ### TEST Now England

# In[ ]:


tengland['Product_ID'].unique()


# In[ ]:


ten_pd4=tengland[tengland['Product_ID']==4]
ten_pd4=ten_pd4.drop(['Product_ID','Product_Type'],axis=1)

ten_pd5=tengland[tengland['Product_ID']==5]
ten_pd5=ten_pd5.drop(['Product_ID','Product_Type'],axis=1)


# In[ ]:


ten_pd4.head()


# In[ ]:


ten_pd4_feat=ten_pd4.drop(['Sales'],axis=1)
ten_pd5_feat=ten_pd5.drop(['Sales'],axis=1)

ten_pd4_label=0.55*qalg_en4.predict(ten_pd4_feat)+0.45*walg_en4.predict(ten_pd4_feat)
ten_pd5_label=0.55*qalg_en5.predict(ten_pd5_feat)+0.45*walg_en5.predict(ten_pd5_feat)


# ## Placing Value in TEST DATA

# In[ ]:


to_place=pd.read_csv('../input/yds_test2018.csv',index_col=0)


# In[ ]:


to_place.drop(['Sales'],axis=1,inplace=True)


# ## Finally, combine all the distributed dataset in the original test data and export

# In[ ]:


to_place.loc[(to_place['Country']=='Argentina') & (to_place['Product_ID']==1),'Sales']=tar_pd1_label
to_place.loc[(to_place['Country']=='Argentina') & (to_place['Product_ID']==2),'Sales']=tar_pd2_label
to_place.loc[(to_place['Country']=='Argentina') & (to_place['Product_ID']==3),'Sales']=tar_pd3_label


to_place.loc[(to_place['Country']=='Columbia') & (to_place['Product_ID']==1),'Sales']=tco_pd1_label
to_place.loc[(to_place['Country']=='Columbia') & (to_place['Product_ID']==2),'Sales']=tco_pd2_label
to_place.loc[(to_place['Country']=='Columbia') & (to_place['Product_ID']==3),'Sales']=tco_pd3_label

to_place.loc[(to_place['Country']=='Denmark') & (to_place['Product_ID']==2),'Sales']=tden_pd2_label

to_place.loc[(to_place['Country']=='England') & (to_place['Product_ID']==4),'Sales']=ten_pd4_label
to_place.loc[(to_place['Country']=='England') & (to_place['Product_ID']==5),'Sales']=ten_pd5_label

to_place.loc[(to_place['Country']=='Belgium') & (to_place['Product_ID']==2),'Sales']=tbel_label

to_place.loc[(to_place['Country']=='Finland') & (to_place['Product_ID']==4),'Sales']=tfn_pd4_label




# In[ ]:





# In[ ]:


#to_place.to_csv('../input/output.csv')

