#!/usr/bin/env python
# coding: utf-8

# # Kaggle-WHO suicide statistics dataset
# 
# ## 2019-02-21

# ## Summary
# 
# - The suicides population of male is increasing
# 
# 
# - The suicides percentage of male is increasing
# 
# 
# - 35-54 year suicides population is highest and increasing
# 
# 
# - Russian.USA and Japan are top3 country that has more suicides population
# 
# 
# - Suicides rate after 2013 is increasing
# 
# 
# - Suicides rate between 1979-2013 is increasing
# 
# 
# - Suicides population and population may decrease in the future
# 
# 
# - Suicides rate may increase in the future

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


original=pd.read_csv('../input/who_suicide_statistics.csv')


# In[ ]:


print(original.head())
print(original.shape)
print(original.dtypes)
print(original.isnull().sum())
print(original.describe())


# In[ ]:


x=list(original['country'].unique())
for i in x:
    if original.loc[original['country']==str(i),'suicides_no'].isnull().sum()>0:
        original.loc[original['country']==str(i),'suicides_no'] =         original.loc[original['country']==str(i),'suicides_no'].fillna(         round(np.mean(original.loc[original['country']==str(i),'suicides_no']),0))
    else:
        pass
else:
    pass


# In[ ]:


original.isnull().sum()


# In[ ]:


original.loc[:,'population']=original.loc[:,'population'].fillna(np.mean(original.loc[:,'population']))


# In[ ]:


original.isnull().sum()


# In[ ]:


print(original.loc[original['year']==2016].size)
print(original.loc[original['year']==2015].size)
print(original.loc[original['year']==2012].size)


# ## Visualization-by year

# In[ ]:


original.head()


# In[ ]:


groupbyyear=original.loc[:,['year','suicides_no']].groupby(by='year')
yearcount=groupbyyear.sum()
yearcount.tail()


# In[ ]:


yearcount.drop(index=2016,inplace=True)
yearcount.drop(index=2015,inplace=True)


# In[ ]:


sns.set()
p0=yearcount.plot.barh(title=r'1979-2014 suicide number by year bar plot',figsize=(16,15),legend=False)
p0.get_figure()
plt.xlabel('year')
plt.ylabel('number')
plt.yticks(range(yearcount.index.shape[0]),yearcount.index)
x=np.arange(yearcount.index.shape[0])
y=np.array(yearcount['suicides_no'])
for i,j in zip(x,y):
    plt.text(j,i-0.2,'%d'%j,ha='center',va='bottom')
else:
    pass
plt.savefig('total_by_year_bar.png')
plt.show()


# In[ ]:


sns.set()
yearcount.index=yearcount.index.astype('str')
z1=yearcount.plot(figsize=(15,14),linestyle='--',marker='o',legend=False)
z1.get_figure()
plt.title(r'1979-2014 suicides_no by year plot')
plt.xticks(range(yearcount.index.shape[0]),yearcount.index,rotation=45)
plt.savefig('./total_by_year_plot.png')
plt.show()


# ## by year
# 
# With bar chart,
# 
# 1998-2003 have higher suicides population
# 
# And with plot chart,
# 
# suicides population has small fluctuation
# 
# but seems to be decrease

# ## Visualization-by gender

# In[ ]:


gender=pd.pivot_table(original,index='year',columns='sex',values='suicides_no',aggfunc=np.sum)


# In[ ]:


gender.drop(index=2015,inplace=True)
gender.drop(index=2016,inplace=True)


# In[ ]:


gender.tail()


# In[ ]:


sns.set()
p1=gender.plot.barh(title=r'1979-2014 suicide number by gender bar plot',                    figsize=(15,14),stacked=True)
p1.get_figure()
plt.ylabel('year')
plt.xlabel('number')
x=np.arange(gender.index.shape[0])
y1=np.array(gender['female'])
y2=np.array(gender['male'])
for i,j in zip(x,y1):
    plt.text(j,i-0.2,'%d'%j,ha='right',va='bottom')
else:
    for k,l in zip(x,y2):
        plt.text(l,k-0.2,'%d'%l,ha='left',va='bottom')

plt.savefig('./total_by_gender_bar_plot.png')
plt.show()


# In[ ]:


sns.set()
p2=gender.boxplot(figsize=(15,14))
p2.get_figure()
plt.title(r'1979-2014 suicide number by gender boxplot')
plt.savefig('./total_by_gender_boxplot.png')
plt.show()


# In[ ]:


sns.set()
n1=plt.figure(figsize=(30,29))
explode=[0,0.1]
label=gender.columns
for i in range(36):
    n1.add_subplot(9,4,i+1)
    plt.title(r'%d suicides_no gender ratio pie plot'%gender.index[i])
    plt.pie(gender.iloc[i,:],autopct='%1.1f%%',explode=explode,labels=label)
else:
    pass
n1.savefig('./total_by_gender_pie.png')
plt.show()


# ## by gender
# 
# With bar chart,male is more than female,
# 
# male's population is increase,
# 
# and female's population is decrease.
# 
# With box chart,male is gently increase,
# 
# and female has no obvious change.
# 
# Because female has no obvious change,
# 
# smaller and bigger values are seem as abnormal values.
# 
# With pie chart,the percentage of male is increasing,
# 
# in 1987-1988,the percentage of female has abnormal change.

# ## Visualization-by age

# In[ ]:


agePivot=pd.pivot_table(original,index='year',columns='age',values='suicides_no',aggfunc=np.sum)
agePivot.tail()


# In[ ]:


agePivot.drop(index=2016,inplace=True)


# In[ ]:


agePivot.tail()


# In[ ]:


sns.set()
p3=agePivot.plot.barh(figsize=(15,14),stacked=True)
p3.get_figure()
plt.title(r'1979-2015 suicides number by age barplot')
plt.xlabel('year')
plt.ylabel('number')
plt.savefig('./total_by_age_barh.png')
plt.show()


# In[ ]:


sns.set()
p4=agePivot.boxplot(figsize=(15,14))
p4.get_figure()
plt.title(r'1979-2015 suicide number by age boxplot')
plt.savefig('./total_by_age_boxplot.png')
plt.show()


# ## by age
# 
# With bar chart,age 35-54 is more than other age,
# 
# in 1983-1984,age 5-14 is increasing,but decrease by next year.
# 
# other age seems to be gently trend.
# 
# With box chart,age 35-54 and age 55-74 has gently increasing,
# 
# And other age have no obvious change,especially 5-14 years.
# 
# Because other age have no obvious change,some values are seem as abnormal values.

# ## Visualization-by country

# In[ ]:


original.head()


# In[ ]:


countryPivot=pd.pivot_table(original,index='year',columns='country',values='suicides_no',aggfunc=np.sum)
countryPivot.head()


# In[ ]:


countryPivot=countryPivot.fillna(0)
countryPivot.head()


# In[ ]:


countryPivot.index.shape


# In[ ]:


countrysum=pd.DataFrame(countryPivot.apply(np.sum))
countrysum=countrysum.rename(columns={0:'total'})
countrysum=countrysum.sort_values(by='total',ascending=False)


# In[ ]:


countrysum.head()


# In[ ]:


top5=countrysum.iloc[0:5,:]


# In[ ]:


top5


# In[ ]:


sns.set()
p5=top5.plot.bar(figsize=(15,14),legend=False)
p5.get_figure()
plt.title(r'1979-2015 top5 suicides_no country bar plot')
plt.xlabel('country')
plt.ylabel('number')
plt.xticks(range(top5.index.shape[0]),top5.index,rotation=45)
x=np.arange(top5.index.shape[0])
y=np.array(top5.loc[:,'total'])
for i,j in zip(x,y):
    plt.text(i,j,'%d'%j,ha='center')
else:
    pass
plt.savefig('./top5_country_bar_plot.png')
plt.show()


# ## top5 country
# 
# With bar chart,Russian.USA and Japan has higher values.
# 
# Then,France and Ukraine are much smaller than top3.
# 
# 

# ## Visualization--year rate

# In[ ]:


original.head()


# In[ ]:


yearRateGroup=original.groupby(by='year')
yearsum=yearRateGroup.sum()
yearsum.head()


# In[ ]:


yearsum['suicides_rate']=yearsum['suicides_no']/yearsum['population']
yearsum.tail()


# In[ ]:


yearsum.drop(index=[2015,2016],inplace=True)


# In[ ]:


rate=pd.DataFrame(yearsum.loc[:,'suicides_rate'])
rate.tail()


# In[ ]:


sns.set()
p6=rate.plot.barh(figsize=(15,14),legend=False)
p6.get_figure()
plt.title(r'1979-2014 suicides rate bar plot')
plt.xlabel('rate')
x=np.array(rate['suicides_rate'])
y=np.arange(rate.index.shape[0])
for i,j in zip(x,y):
    plt.text(i,j-0.2,'%.4f%%'%(i*100),ha='center')
else:
    pass
plt.savefig('./suicides_rate_by_year_bar.png')
plt.show()


# In[ ]:


sns.set()
rate.index=rate.index.astype('str')
z2=rate.plot(figsize=(15,14),linestyle='--',marker='o',legend=False)
z2.get_figure()
plt.title(r'1979-2014 suicides rate plot')
plt.xticks(range(rate.index.shape[0]),rate.index,rotation=45)
plt.savefig('./suicides_rate_by_year_plot.png')
plt.show()


# ## year rate
# 
# With plot chart,the trend of suicides rate is decreasing.
# 
# But after 2013,suicides rate has increasing trend.
# 
# And 1980 is highest,
# 
# 1992-1998 is another higher section.

# ## Predict--with population predict formula
# 
# With data that is total by year,using population predict formula,
# 
# and using previous population for basic period.
# 
# ### population predict formula:
# 
# $${M_n}={M_0}*{(1+V)}^n$$
# 
# ${M_n}$=predicted period.
# 
# ${M_0}$=basic period.
# 
# V=Growth rate
# 
# n=year value
# 
# 
# ### growth rate formula:
# 
# $$\frac{{a_i}-{a_{i-1}}}{a_{i-1}}$$

# In[ ]:


yearsum.index[-1]


# In[ ]:


def populationpredict(data):
    basic_suicide=float(data.iloc[-2,0])
    basic_population=float(data.iloc[-2,1])
    suicidesrate=(data.iloc[-1,0]-basic_suicide)/(basic_suicide)
    suicidespredict=data.iloc[-1,0]*(1+suicidesrate)
    populationrate=(data.iloc[-1,1]-basic_population)/(basic_population)
    populationpredict=data.iloc[-1,1]*(1+populationrate)
    rate=suicidespredict/populationpredict
    x=[suicidespredict,populationpredict,rate]
    x=pd.DataFrame(x)
    x=x.T
    x=x.rename(columns={0:'suicides_no',
                      1:'population',
                      2:'suicides_rate'})
    x=x.rename(index={0:data.index[-1]+1})
    return x


# In[ ]:


suicides2015=populationpredict(yearsum)
yearsum=pd.concat([yearsum,suicides2015])
suicides2016=populationpredict(yearsum)
yearsum=pd.concat([yearsum,suicides2016])
yearsum.tail()


# In[ ]:


suicides2017=populationpredict(yearsum)
yearsum=pd.concat([yearsum,suicides2017])
suicides2018=populationpredict(yearsum)
yearsum=pd.concat([yearsum,suicides2018])
suicides2019=populationpredict(yearsum)
yearsum=pd.concat([yearsum,suicides2019])


# In[ ]:


suicides2020=populationpredict(yearsum)
yearsum=pd.concat([yearsum,suicides2020])
suicides2021=populationpredict(yearsum)
yearsum=pd.concat([yearsum,suicides2021])
suicides2022=populationpredict(yearsum)
yearsum=pd.concat([yearsum,suicides2022])
suicides2023=populationpredict(yearsum)
yearsum=pd.concat([yearsum,suicides2023])


# In[ ]:


suicides2024=populationpredict(yearsum)
yearsum=pd.concat([yearsum,suicides2024])
suicides2025=populationpredict(yearsum)
yearsum=pd.concat([yearsum,suicides2025])


# In[ ]:


#yearsum.to_csv('./predict_no_population.csv')
yearsum.tail()


# In[ ]:


year0025=yearsum.iloc[21:,:]
year0025.head()
year0025.index=year0025.index.astype('str')


# In[ ]:


sns.set()
p7=plt.figure(figsize=(15,14))
plt.plot(year0025.iloc[:,0],linestyle='--',marker='o')
plt.title(r'suicides_no predict by formula plot')
plt.xlabel('year')
plt.ylabel(r'predict number')
plt.xticks(range(year0025.index.shape[0]),year0025.index,rotation=45)
p7.savefig('./predict_suicides_no_plot.png')
plt.show()


# In[ ]:


year0025.head()


# In[ ]:


sns.set()
p8=plt.figure(figsize=(15,14))
plt.plot(year0025.iloc[:,1],linestyle='--',marker='o')
plt.xlabel('year')
plt.ylabel(r'predict population')
plt.xticks(range(year0025.index.shape[0]),year0025.index,rotation=45)
plt.title(r'population predict by formula plot')
p8.savefig('./predict_population_plot.png')
plt.show()


# In[ ]:


sns.set()
p9=plt.figure(figsize=(15,14))
plt.plot(year0025.iloc[:,2],linestyle='--',marker='o')
plt.xlabel('year')
plt.ylabel('rate')
plt.xticks(range(year0025.index.shape[0]),year0025.index,rotation=45)
plt.title(r'predict suicides rate by formula plot')
p9.savefig('./predict_suicides_rate_plot.png')
plt.show()


# ## Predict
# 
# With using population predict formula,
# 
# and using previous population for basic period,
# 
# suicides population and population is decreasing,
# 
# but suicides rate is increasing,
# 
# so we should be more seriously to face suicides.

# In[ ]:




