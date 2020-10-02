
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

data=pd.read_csv('../input/mow_led.csv', delimiter=',',skipinitialspace=True)
data.head()
data.columns
data['fact_pjs'].dtype
data['plan_pjs'].dtype
data['fact_rev'].dtype
data['plan_rev'].dtype
data['fact_rev']=data['fact_rev'].str.replace(",",".")
data['fact_rev']=data['fact_rev'].astype(float
data['plan_rev']=data['plan_rev'].str.replace(",",".")
data['plan_rev']=data['plan_rev'].str.replace(" ","")
data['plan_rev']=data['plan_rev'].astype(float)

data['plan_pjs']=data['plan_pjs'].str.replace(" ","")
data['plan_pjs']=data['plan_pjs'].astype(int)


data['week']=pd.to_datetime(data['week'], format="%d.%m.%Y")

#1residual error = expected - predicted

data['residuals']=data.apply(lambda data: data['plan_pjs']-data['fact_pjs'], axis=1)
data.head()

#group analysis 
res_byroute1=data.groupby(['route', 'terr'])['residuals'].median()
res_byroute1.to_csv('res_byroute1.csv',sep=';', encoding='utf-8-sig', index = True)
#res_byroute1=res_byroute1.to_frame()
res_byroute=data.groupby(['route', 'terr'])['residuals'].describe()
res_byroute.to_csv('res_byroute.csv',sep=';', encoding='utf-8-sig', index = True)
#res_byroute=res_byroute.to_frame()
#table=pd.merge(res_byroute,res_byroute1, how='left', on=['route'])

#residials statistics
data['residuals'][data['route']=='MOW-REK'].plot()
data['residuals'][data['route']=='MOW-REK'].describe()
data['residuals'][data['route']=='MOW-REK'].median()

#residials plot
sns.set(style="whitegrid")
sns.residplot(data['plan_pjs'][data['route']=='MOW-REK'], data['fact_pjs'][data['route']=='MOW-REK'], lowess=True, color="g")

plt.scatter(data['residuals'][data['route']=='MOW-REK'],data['fact_pjs'][data['route']=='MOW-REK'])

res_byroute2=data.groupby(['route', 'terr'])['plan_pjs', 'fact_pjs', 'plan_rev', 'fact_rev'].sum()
res_byroute2.to_csv('res_byroute2.csv',sep=';', encoding='utf-8-sig', index = True)

