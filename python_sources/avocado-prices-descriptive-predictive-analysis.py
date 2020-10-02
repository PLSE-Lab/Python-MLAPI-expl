#!/usr/bin/env python
# coding: utf-8

# **The goal in this project is to understand the price trends and the elements that have an impact on avocado sales in United States. If we assume all avocados here are being sold by one supplier (or a group of suppliers), a specific objective can be to see what composition of packaging can increase revenue for avocado producers. This project also includes a number of linear regressions to predict the average price of avocado.  **

# In[ ]:


import pandas as pd
import numpy as np 
import seaborn as sns
from pandas import DataFrame, Series
import matplotlib.pyplot as plt


# In[ ]:


avocado=pd.read_csv('../input/avocado.csv')


# In[ ]:


avocado.head(5)


# **Cleaning the data**

# In[ ]:


del avocado['Unnamed: 0']
avocado['Date']=pd.to_datetime(avocado['Date'])


# In[ ]:


avocado.isnull().sum()


# Descriptive analysis:
# 
# 1. How has the average price changed over time since 2015?
# 2. How has the price changed with the season?
# 3. How does avocado type affect the pricing?

# In[ ]:


avocado_date=avocado.groupby('Date').mean()


# In[ ]:


avocado_date.head(5)


# In[ ]:


avocado_date.AveragePrice.plot(figsize=(15,4))
plt.ylabel('Average Price')


# There is a seasonal trend and a general upward trajectory visible here.

# In[ ]:


ax=sns.factorplot(x='Date', y='AveragePrice', data=avocado, hue='type',aspect=3)


# In[ ]:


avocado_date_t=avocado.pivot_table(index='Date', columns='type', aggfunc='mean')['AveragePrice']


# In[ ]:


avocado_date_t.plot(figsize=(15,4))
plt.text(x='2017-1-15', y=2.04, s='March 2017', color='green', fontsize=12)
plt.vlines(x='2017-3-1', ymin=0.8, ymax=2, color='green', linestyles=':', linewidth=3, label='March 2017')
plt.ylabel('Average Price')


# In March 2017 the average price for conventional and organic avocado was almost the same (not sure why, an interesting observation). 

# In[ ]:


sns.boxplot(data=avocado_date_t, palette='bone')


# **The price of organic avocado is on average 45% higher than conventional avocado.**

# In[ ]:


x=[]
for i in range(len(avocado)):
    m=avocado['Date'].loc[i].strftime("%B")
    x.append(m)


# In[ ]:


avocado['month']=x


# In[ ]:


avocado_year=avocado.pivot_table(index='month', columns='year', aggfunc='mean')['AveragePrice']


# In[ ]:


month_order={
    'January':1,
    'February':2,
    'March':3,
    'April':4,
    'May':5,
    'June':6,
    'July':7,
    'August':8,
    'September':9,
    'October':10,
    'November':11,
    'December':12
}


# In[ ]:


fig=avocado_year.loc[month_order].plot(figsize=(15,4), xticks=range(0,13), cmap='jet')
fig.set_facecolor('#7F7F7F')
plt.text(x=0.7, y=1.9, s='Winter', color='#98F5FF', fontsize=15)
plt.vlines(x=0, ymin=0.8, ymax=2, color='lightblue', linestyles=':', linewidth=3)
plt.text(x=3.8, y=1.9, s='Spring', color='#7FFF00', fontsize=15)
plt.vlines(x=2.5, ymin=0.8, ymax=2, color='#7FFF00', linestyles=':', linewidth=3)
plt.text(x=6.8, y=1.9, s='Summer', color='#FFB6C1', fontsize=15)
plt.vlines(x=5.5, ymin=0.8, ymax=2, color='pink', linestyles=':', linewidth=3)
plt.text(x=9.5, y=1.9, s='Fall', color='orange', fontsize=15)
plt.vlines(x=8.5, ymin=0.8, ymax=2, color='orange', linestyles=':', linewidth=3)
plt.ylabel('Average Price')


# **Price fluctuations in each year**

# In[ ]:


year_15=avocado.loc[avocado['year']==2015]
year_16=avocado.loc[avocado['year']==2016]
year_17=avocado.loc[avocado['year']==2017]
year_18=avocado.loc[avocado['year']==2018]


# In[ ]:


fig1=year_15.groupby('Date').mean().plot(y='AveragePrice', figsize=(10,3), kind='line', sharex=True, color='#98F5FF')
plt.text(x='2015-1-1', y=1.45, s='2015', color='#98F5FF', fontsize=15)
fig1.set_facecolor('#7F7F7F')

fig2=year_16.groupby('Date').mean().plot(y='AveragePrice', figsize=(10,3), kind='line', sharex=True, color='#FFE4C4')
plt.text(x='2016-1-1', y=1.55, s='2016', color='#FFE4C4', fontsize=15)
fig2.set_facecolor('#7F7F7F')

fig3=year_17.groupby('Date').mean().plot(y='AveragePrice', figsize=(10,3), kind='line', sharex=True, color='#CAFF70')
plt.text(x='2017-1-1', y=1.8, s='2017', color='#CAFF70', fontsize=15)
fig3.set_facecolor('#7F7F7F')


# In[ ]:


y=[]
a=avocado_date.index
for i in range(len(avocado_date)):
    m=a[i].strftime("%B")
    y.append(m)
avocado_date['month']=y


# In[ ]:


avocado_volume=avocado.pivot_table(index='Date', aggfunc='sum')
b=avocado_volume.index
z=[]
for i in range(len(avocado_volume)):
    m=b[i].strftime("%B")
    z.append(m)
ab=[]
for i in range(len(avocado_volume)):
    m=b[i].strftime("%Y")
    ab.append(m)
avocado_volume['month']=z
avocado_volume['year']=ab


# In[ ]:


plt.figure(figsize=(15,4))
sns.pointplot(x='month', y='AveragePrice', data=avocado_date, hue='year', palette='Set1')
plt.title('Average Price')
plt.ylabel('USD')


# In[ ]:


plt.figure(figsize=(15,4))
sns.pointplot(x='month', y='Total Volume', data=avocado_volume, hue='year', palette='Set1')
plt.title('Total Volume')


# **The market share based on type:**

# In[ ]:


avocado_type_volume=avocado.pivot_table(index='Date', columns='type', aggfunc='sum')


# In[ ]:


avocado_type_volume.plot(y='Total Volume', figsize=(15,4), title='Total Volume')


# In[ ]:


e=avocado_type_volume['Total Volume'].sum(axis=1)
f=avocado_type_volume['Total Volume']['conventional']/e*100
g=100-avocado_type_volume['Total Volume']['conventional']/e*100


# In[ ]:


plt.figure(figsize=(10,4))
plt.text(x='2017-10-15', y=10, s='Organic', color='#00CD00', fontsize=18)
plt.bar(x=a, height=f, width=6, color='#FFE4B5')
plt.bar(x=a, height=g, width=7, color='#7CFC00')
plt.ylabel('Market share%')


# **So the market share for organic avocados is a very small percent, BUT, there is an upward trajectory for it. **

# **4- Is the price of avocado different in different regions of US? Can we explain the price distribution?**

# In[ ]:


avocado.region.unique()


# In[ ]:


west=avocado.loc[avocado['region']=='West']
southeast=avocado.loc[avocado['region']=='Southeast']
southcentral=avocado.loc[avocado['region']=='SouthCentral']
plains=avocado.loc[avocado['region']=='Plains']
northeast=avocado.loc[avocado['region']=='Northeast']
midsouth=avocado.loc[avocado['region']=='Midsouth']
totalus=avocado.loc[avocado['region']=='TotalUS']


# In[ ]:


df=pd.merge(west, southeast, on='Date', suffixes=('_west', '_southeast'))
df=pd.concat([west, southeast])
df1=pd.concat([df,southcentral])
df2=pd.concat([df1,plains])
df3=pd.concat([df2,northeast])
df4=pd.concat([df3,midsouth])
df5=pd.concat([df4,totalus])
df5_price=df5.pivot_table(index='Date', columns='region', aggfunc='mean')['AveragePrice']


# **Moving Average of the price**

# In[ ]:


ma_days=[10,20,50]
for ma in ma_days:
    column_name='MA for {} days'.format(ma)
    df5_price[column_name]=df5_price['TotalUS'].rolling(ma).mean()


# In[ ]:


test=df5_price.groupby('Date')[['TotalUS', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].mean()


# In[ ]:


test.drop('TotalUS', axis=1, inplace=True)


# In[ ]:


test.columns.name='Moving Average'


# In[ ]:


fig2=test.plot( figsize=(15,4), kind='line', cmap='Blues')
fig2.set_facecolor('#7A8B8B')


# In[ ]:


df5_price.plot(kind='line', figsize=(12,4))
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.ylabel('Average Price')


# In[ ]:


r=df5_price.index
c=[]
for i in range(len(df5_price)):
    m=r[i].strftime("%B")
    c.append(m)
df5_price['month']=c
#------------------------------------
f=[]
for i in range(len(df5_price)):
    m=r[i].strftime("%Y")
    f.append(m)
df5_price['year']=f    


# In[ ]:


med=df5_price.median()


# In[ ]:


med.sort_values()


# In[ ]:


median=['SouthCentral', 'West', 'Midsouth', 'Southeast', 'Plains', 'Northeast']


# In[ ]:


plt.figure(figsize=(15,4))
sns.boxplot( data=df5_price[['West','Midsouth','Northeast', 'Plains', 'SouthCentral', 'Southeast']], palette='summer', saturation=1, order=median)
plt.ylabel('Average Price')


# In[ ]:


plt.figure(figsize=(15,4))
sns.boxplot(x='month',y='TotalUS', data=df5_price, palette='cool', saturation=1)
plt.ylabel('Average Price')
plt.title('Average price fluctuations in US (years= 2015, 2016, 2017, 2018)')


# Looks like the average price changes drastically with region and time of the year. As for regions, looks like the further we are from the Southern border the higher the average price will be. After a little bit of research, we can find that the majority of Hass avocados are imported to US from Peru, Chile, and Mexico. Looks like the additional cost of transportation is well reflected on the price people pay in grocery stores. 
# The average price in total US seems to be higher in colder months too. Considering the fact that most of the avocado is coming from south America and that the seasons are switched comapred to north America, this could be related to the amount of avocados that's available during colder months in south.  

# **Now, let's see what the consumption looks like in different seasons and regions: **

# In[ ]:


df5_volume=df5.pivot_table(index='Date', columns='region', aggfunc='sum')['Total Volume']


# In[ ]:


df5_volume[['SouthCentral', 'West', 'Midsouth', 'Southeast', 'Plains', 'Northeast']].plot(kind='line', figsize=(12,4), cmap='Set1')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.ylabel('Total Volume')


# In[ ]:


medi=df5_volume.median()


# In[ ]:


medi.sort_values()


# In[ ]:


order=['Plains','Midsouth', 'Southeast', 'Northeast', 'SouthCentral', 'West']


# In[ ]:


plt.figure(figsize=(15,4))
sns.violinplot( data=df5_volume[['West','Midsouth','Northeast', 'Plains', 'SouthCentral', 'Southeast']], palette='winter', saturation=1, order=order, orient='v')
plt.ylabel('Total Volume')


# In[ ]:


v=[]
for i in range(len(df5_volume)):
    m=r[i].strftime("%B")
    v.append(m)
df5_volume['month']=v    
    


# In[ ]:


plt.figure(figsize=(15,4))
sns.boxplot(x='month',y='TotalUS', data=df5_volume, palette='ocean', saturation=1)
plt.ylabel('Total Volume')
plt.title('Total volume fluctuations in US (years= 2015, 2016, 2017, 2018)')


# Now, let's look at the revenue:

# In[ ]:


avocado['Total indiv']=avocado['Total Volume']-avocado['Total Bags']
avocado['Revenue indiv']=avocado['Total indiv']*avocado['AveragePrice']
avocado['Revenue Bagged']=avocado['Total Bags']*avocado['AveragePrice']
avocado['Revenue Total']=avocado['Revenue Bagged']+avocado['Revenue indiv']


# In[ ]:


avocado_r_t=avocado.pivot_table(index='Date', columns='type', aggfunc='sum')[['Revenue indiv', 'Revenue Bagged', 'Revenue Total']]


# In[ ]:


avocado_r_t.plot(cmap='Set1', figsize=(10,4))
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# Although the average price of organic avocados is higher, the revenue from organic avocados doesn't account for much of the total revenue.

# In[ ]:


avocado_rev=avocado.pivot_table(index='Date', aggfunc='sum')[['Revenue indiv', 'Revenue Bagged', 'Revenue Total']]


# In[ ]:


avocado_rev.plot(cmap='Set1', figsize=(10,4))


# Now this graph is very interesting. The red line is the revenue from individual avocados, and it looks like the seasonal trend is clearly visible here. This means that the revenue from sales of individual avocados has a seasonal pattern which is dictating the seasonal pattern in our total revenue. Now the orange line, is the revenue from bagged avocados. There is a clear upward trajectory in the past three years here. The seasonal trend is also weaker. 
# 
# Now let's see if this observation is confirmed from our data: 

# In[ ]:


import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split


# In[ ]:


revenue=avocado[['Total indiv','Revenue indiv','Total Bags', 'Revenue Bagged', 'Total Volume', 'Revenue Total']]


# In[ ]:


rename=['Total_indiv', 'Revenue_indiv', 'Total_Bags', 'Revenue_Bagged','Total_Volume', 'Revenue_Total' ]


# In[ ]:


revenue.columns=rename


# In[ ]:


sns.pairplot(data=revenue)


# Out of all these graphs, I am interested in the relatiosnhip between Total revenue, total individual, and total bagged. This relationship will determine which type of packaging has a larger effect on the total revenue. 

# In[ ]:


lm3=smf.ols(formula='Revenue_Total ~ Total_indiv + Total_Bags', data=revenue).fit()


# In[ ]:


lm3.summary()


# In[ ]:


revenue['lm2']=0.8766*revenue['Total_indiv']+1.5155*revenue['Total_Bags']+29190  


# In[ ]:


revenue.plot(x='Revenue_Total', y='lm2', kind='scatter')
x=[0,7*10**7]
y=[0,7*10**7]
plt.plot(x,y, '-.', color='red')
plt.text(x=6.5*10**7, y=6*10**7, s='y=x', color='red', fontsize=13)


# In this simple regression model, we can see that the coefficient for bagged avocados (~1.5) is almost twice as large as the coefficient for individual ones (~0.87). 

# **Forecasting the Average Price**

# In[ ]:


test=df5_price.loc[df5_price['year']=='2018']


# In[ ]:


train=df5_price.loc[(df5_price['year']=='2017') ^ (df5_price['year']=='2016') ^ (df5_price['year']=='2015') ]


# We already know that region has a large effect on the average price. 

# In[ ]:


price_reg=DataFrame(avocado.groupby('region')['AveragePrice'].mean())


# In[ ]:


price_reg.sort_values(by='AveragePrice').plot(kind='bar', figsize=(15,4))


# To turn the regions into dummy variables, I will group them into 5 classes based on mean price. 

# In[ ]:


price_reg.reset_index(inplace=True)


# In[ ]:


Bin=(price_reg['AveragePrice'].max()-price_reg['AveragePrice'].min())/5


# In[ ]:


class1=[]
class2=[]
class3=[]
class4=[]
class5=[]

for i in price_reg.index: 
    if (price_reg['AveragePrice'].loc[i]<price_reg['AveragePrice'].min()+Bin):
        class1.append(price_reg['region'].loc[i])


# In[ ]:


for i in price_reg.index: 
    if (price_reg['AveragePrice'].loc[i]>=price_reg['AveragePrice'].min()+Bin) & (price_reg['AveragePrice'].loc[i]<price_reg['AveragePrice'].min()+2*Bin):
        
        class2.append(price_reg['region'].loc[i])
#---------------------------------------------------------
for i in price_reg.index: 
    if (price_reg['AveragePrice'].loc[i]>=price_reg['AveragePrice'].min()+2*Bin) & (price_reg['AveragePrice'].loc[i]<price_reg['AveragePrice'].min()+3*Bin):
        
        class3.append(price_reg['region'].loc[i])
#---------------------------------------------------------
for i in price_reg.index: 
    if (price_reg['AveragePrice'].loc[i]>=price_reg['AveragePrice'].min()+3*Bin) & (price_reg['AveragePrice'].loc[i]<price_reg['AveragePrice'].min()+4*Bin):
        
        class4.append(price_reg['region'].loc[i])
#---------------------------------------------------------
for i in price_reg.index: 
    if (price_reg['AveragePrice'].loc[i]>=price_reg['AveragePrice'].min()+4*Bin):
        class5.append(price_reg['region'].loc[i])

        


# In[ ]:


temp=[]
for i in avocado['region']:
    if i in class1: 
        temp.append('class1')
    if i in class2:
        temp.append('class2')
    if i in class3:
        temp.append('class3')
    if i in class4:
        temp.append('class4')
    if i in class5:
        temp.append('class5')


# In[ ]:


avocado['class']=temp


# In[ ]:


avocado.head(3)


# Now instead of and individual region, each record belongs to a class. Good. 
# The next thing I want to tackle is the average price itself. I want to capture the seasonality in the prediction I make so I'm going to focus not on the average price, but on the difference between price on a given day and a baseline price. For the baseline price I will pick the weekly average price in year 2015. 

# In[ ]:


week=[]
for i in avocado['Date']:
    week.append(i.isocalendar()[1])


# In[ ]:


avocado['week']=week


# In[ ]:


year2015=avocado.loc[avocado['year']==2015]


# In[ ]:


year2015=year2015.pivot_table(index='week', columns='type', aggfunc='mean')['AveragePrice']


# In[ ]:


for i in avocado.index:
    if (avocado['week'].loc[i]==53) & (avocado['month'].loc[i]=='January'):
        avocado['week'].loc[i]=1


# In[ ]:


for i in avocado.index:
    if (avocado['week'].loc[i]==52) & (avocado['month'].loc[i]=='January'):
        avocado['week'].loc[i]=1


# In[ ]:


d=[]
for i in avocado.index:
    a=avocado['week'].loc[i]
    if avocado['type'].loc[i]=='conventional':
        d.append(year2015['conventional'].loc[a])
    else:
        d.append(year2015['organic'].loc[a])


# In[ ]:


avocado['base_price']=d


# In[ ]:


avocado['delta']=avocado['AveragePrice']-avocado['base_price']


# In[ ]:


df_modeling=avocado.drop(['Date', 'AveragePrice', 'year','region', 'month', 'base_price', 'Revenue indiv', 'Revenue Bagged', 'Revenue Total','Total indiv', 'Total Bags'], axis=1)


# In[ ]:


type_dum=pd.get_dummies(df_modeling['type'])
class_dum=pd.get_dummies(df_modeling['class'])


# In[ ]:


df_modeling=pd.concat([df_modeling, type_dum], axis=1)
df_modeling=pd.concat([df_modeling, class_dum], axis=1)


# In[ ]:


df_modeling.drop(['organic', 'class1', 'type', 'class'], axis=1, inplace=True)


# In[ ]:


df_modeling.head()


# Now my DataFrame is ready for some regression modeling! 

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_multi=df_modeling.drop('delta', axis=1)


# In[ ]:


Y=df_modeling['delta']


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X_multi, Y)


# In[ ]:


lreg=LinearRegression()
lreg.fit(X_train, Y_train)


# In[ ]:


coef=DataFrame(lreg.coef_)


# In[ ]:


features=DataFrame(X_multi.columns)


# In[ ]:


line1=pd.concat([features, coef], axis=1)
line1.columns=['feature', 'coefficient']
line1.index=line1['feature']


# In[ ]:


line1.sort_values(by='coefficient').plot(kind='bar')


# In[ ]:


pred_train=lreg.predict(X_train)
pred_test=lreg.predict(X_test)


# In[ ]:


plt.scatter(x=Y_test, y=pred_test, marker='.')
plt.plot(Y_test, Y_test, color='red')
plt.title('test data')


# In[ ]:


resid=Y_train-pred_train
resid2=Y_test-pred_test


# In[ ]:


train=plt.scatter(x=Y_train, y=resid, alpha=0.5)
test=plt.scatter(x=Y_test, y=resid2, marker='x', alpha=0.5)
line=plt.hlines(y=0, xmin=-1.5, xmax=1.5, color='red')
plt.legend((train,test), ('Training', 'Test'), loc='lower right')
plt.ylabel('residual')
plt.xlabel('delta')


# In[ ]:





# In[ ]:




