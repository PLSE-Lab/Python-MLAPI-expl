#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from datetime import datetime
pd.options.display.max_rows = 999
pd.options.display.max_columns = 999
train_file = "../input/covid19-global-forecasting-week-3/train.csv"
test_file = "../input/covid19-global-forecasting-week-3/test.csv"
sub_file = "../input/covid19-global-forecasting-week-3/submission.csv"
Optimistic=False


# In[ ]:


df = pd.read_csv(train_file)


# In[ ]:


df


# In[ ]:


loc_group = ["Province_State", "Country_Region"]
def preprocess(df,datecol):
    df["Date"] = df[datecol].astype("datetime64[ms]")
    for col in loc_group:
        df[col].fillna("none", inplace=True)
    return df

df= preprocess(df,"Date")


# In[ ]:



TRAIN_X = datetime.strptime('20-03-11', '%y-%m-%d')


# In[ ]:


#df["Date"].max()-timedelta(days=5)


# In[ ]:


from datetime import timedelta
TEST_END = df["Date"].max()
TRAIN_LAST = datetime.strptime('20-03-23', '%y-%m-%d')
TRAIN_FIRST = df["Date"].max() - timedelta(days=5) #Use in cases where it's not clear when min date should be


# In[ ]:





# In[ ]:


df=df.sort_values(['Province_State','Country_Region','Date'])


# In[ ]:


TARGETS = ["ConfirmedCases", "Fatalities"]
for col in TARGETS:
    df["prev_{}".format(col)] = df[col].shift()
    df[f'delta_{col}'] =  df[col]-df[f'prev_{col}']
    df[f'rolling_{col}'] = df[f'delta_{col}'].rolling(window=3).mean()
    df[f'div_rolling_{col}'] =  (df[f'rolling_{col}']-df[f'rolling_{col}'].shift())/df[f'rolling_{col}'].shift()
df['div_rolling_ConfirmedCases']=df['div_rolling_ConfirmedCases'].replace([np.inf, -np.inf], np.nan).fillna(0)
df['div_rolling_Fatalities']=df['div_rolling_Fatalities'].replace([np.inf, -np.inf], np.nan).fillna(0)  


# In[ ]:


df[df['Province_State']=="New York"]


# In[ ]:


df = df[df.Date>df.Date.min() + timedelta(days=3)]


# In[ ]:


dfSmall = df[df.rolling_ConfirmedCases>100].reset_index()


# In[ ]:


MaxDate = dfSmall.loc[dfSmall.groupby(loc_group)['rolling_ConfirmedCases'].idxmax()][['Province_State','Country_Region','Date']].reset_index()


# In[ ]:


df = df.merge(MaxDate, how = 'left',on=['Province_State','Country_Region'],suffixes=['','_Max'])


# In[ ]:


MinDate = dfSmall.loc[dfSmall.groupby(loc_group)['rolling_ConfirmedCases'].idxmin()][['Province_State','Country_Region','Date']].reset_index()


# In[ ]:


df = df.merge(MinDate, how = 'left',on=['Province_State','Country_Region'],suffixes=['','_Min'])


# In[ ]:


#df['Date_Min'] = np.where((df.Date_Min>TRAIN_FIRST) ,pd.NaT,df['Date_Min'])


# In[ ]:


df['Date_Max'] = df['Date_Max'].fillna(TEST_END)
#df['Date_Min'] = df['Date_Min'].fillna(TRAIN_FIRST)


# In[ ]:


#df['Check'] = df["Date_Max"] - timedelta(days=5)
df['Date_Min'] =df["Date_Max"] - timedelta(days=7)


# In[ ]:





# In[ ]:


df[['Date_Max','Date_Min']].drop_duplicates()


# In[ ]:


#df["Date_Min"] = df["Date_Min"].astype("datetime64[ms]")
#df.dtypes


# In[ ]:


df['Days'] = (df['Date']-df['Date_Min'].min()).dt.days.astype('int16')


# In[ ]:


#df[df['Date_Max'].isna()]


# In[ ]:


df['div_rolling_ConfirmedCases'] = np.clip(df['div_rolling_ConfirmedCases'],0,1)
df3=df[(df['Date']>df['Date_Min']) & (df['Date']<=df['Date_Max'])].groupby(loc_group)['div_rolling_ConfirmedCases'].mean().reset_index()
df3.columns = ['Province_State','Country_Region','Coeff']
#group_predictions(df, '01-10-2016')


# In[ ]:


df['div_rolling_Fatalities'] = np.clip(df['div_rolling_Fatalities'],0,1)
df4=df[(df['Date']>df['Date_Min']) & (df['Date']<=(df['Date_Max']))].groupby(loc_group)['div_rolling_Fatalities'].mean().reset_index()
df4.columns = ['Province_State','Country_Region','Death_Coeff']


# In[ ]:


#x=df[(df['Date']<=df['Date_Max']].groupby(loc_group)['div_rolling_ConfirmedCases'].mean().reset_index()
df4[df4['Country_Region']=='Turkey']


# In[ ]:


#LastGood = df.groupby(loc_group)['Date'].max().reset_index()
#df = df.merge(LastGood,how='left',on=loc_group,suffixes=['','_LastGood'])


# In[ ]:


if Optimistic:
    df['Check'] = df["Date"].max() + timedelta(days=7)
else:
    df['Check'] = df["Date"].max() + timedelta(days=14)
df['Date_Max'] = np.where(df['Date_Max']>df["Date"].max()- timedelta(days=1),df["Check"],df['Date_Max'])


# In[ ]:


test_df = preprocess(pd.read_csv(test_file),"Date")
df = df.merge(test_df,how='outer', on =['Province_State','Country_Region','Date'])
df=df.sort_values(['Province_State','Country_Region','Date'])
df['Date_Max']=df.groupby(loc_group)['Date_Max'].fillna(method='ffill')
if Optimistic:
    df['Death_Date_Max'] = df['Date_Max']+ timedelta(days=3)
else:
    df['Death_Date_Max'] = df['Date_Max']+ timedelta(days=8)


# In[ ]:


df = df.merge(df3,how='left',on=['Province_State','Country_Region'])
df = df.merge(df4,how='left',on=['Province_State','Country_Region'])


# In[ ]:


print(df.Coeff.mean())
print(df.Death_Coeff.mean())


# In[ ]:


df['Coeff']=df.groupby(loc_group)['Coeff'].fillna(method='ffill')
df['Coeff'] = np.where(df['Coeff'].isna(),0,df['Coeff'])
df['Coeff'] = np.clip(df['Coeff'],0.05,0.75)


df['Death_Coeff']=df.groupby(loc_group)['Death_Coeff'].fillna(method='ffill')
df['Death_Coeff'] = np.where(df['Death_Coeff'].isna(),0,df['Death_Coeff'])
df['Death_Coeff'] = np.clip(df['Death_Coeff'],0.05,0.75)
df['Death_Coeff'] = np.where(df['Death_Coeff']>df['Coeff'],df['Coeff'],df['Death_Coeff'])

df['Coeff'] = np.where(df.Date <= df['Date_Max'],df['Coeff'],-0.7*df['Coeff'])
df['Death_Coeff'] = np.where(df.Date <= df['Death_Date_Max'],df['Death_Coeff'],-0.7*df['Death_Coeff'])


# In[ ]:


#df2


# In[ ]:


#df2['pred_deltaCC'] =df2['delta_ConfirmedCases']
#df2['pred_deltaCC'] = df2['pred_deltaCC'].shift(1)+df2['Coeff']


# In[ ]:


#df2.iloc[0]['pred_deltaCC']


# In[ ]:





# In[ ]:


#df2.head()


# In[ ]:


list1 =[]
list2=[]
pred = 0
change = 0
pred_death = 0
change_death = 0
for index, row in df.iterrows():
    if (row['Date']<=TRAIN_LAST) | (row['Date']==TRAIN_FIRST):
        change = (1+row['Coeff'])*row['rolling_ConfirmedCases']
        pred = row['ConfirmedCases']-change
        change_death = (1+row['Death_Coeff'])*row['rolling_Fatalities']
        pred_death = row['Fatalities']-change_death
    else:
        change = change*(1+row['Coeff'])
        change_death = change_death*(1+row['Death_Coeff'])
        
    if row['Date']==TEST_END:
        change = (1+row['Coeff'])*row['rolling_ConfirmedCases']
        pred = row['ConfirmedCases']-change
        change_death = (1+row['Death_Coeff'])*row['rolling_Fatalities']
        pred_death = row['Fatalities']-change_death
        
        
    if change<0:
        change =0
    pred = pred+change
    list1.append(pred)
    
    if change_death<0:
        change_death =0
    pred_death = pred_death+change_death
    if pred_death>pred*0.1:
        pred_death = pred*0.1
    list2.append(pred_death)
df['pred_CC']= list1
df['pred_Fatalities']= list2


# In[ ]:


df.to_csv("df2.csv")


# In[ ]:


df.groupby('Date')[['ConfirmedCases','Fatalities','pred_CC','pred_Fatalities']].sum().to_csv('sum.csv')


# In[ ]:


TRAIN_LAST


# In[ ]:


df[df['Province_State']=='New York'][['Date','ConfirmedCases','Fatalities','pred_CC','pred_Fatalities','Coeff','Death_Coeff','rolling_ConfirmedCases','Date_Max','Death_Date_Max']]


# In[ ]:


#df


# In[ ]:





# In[ ]:


df['Error_CC']= np.where(df['ConfirmedCases']==0,0,(abs((df['pred_CC']-df['ConfirmedCases'])/df['ConfirmedCases'])))
df['Error_Death']= np.where(df['Fatalities']==0,0,abs((df['pred_Fatalities']-df['Fatalities'])/df['Fatalities']))
df.groupby('Date')[['Error_CC','Error_Death']].sum().to_csv('error.csv')


# In[ ]:


df[df['Country_Region']=='Spain'][['Date','ConfirmedCases','Fatalities','pred_CC','pred_Fatalities','Coeff','Death_Coeff','rolling_ConfirmedCases','Date_Max','Death_Date_Max']]


# In[ ]:


#test_df = preprocess(pd.read_csv("../input/covid19-global-forecasting-week-1/test.csv"))
sub_df = pd.read_csv(sub_file)
#df2 = df.merge(sub_df,how='outer', on =['Province/State','Country/Region','Date'])


# In[ ]:


sub_df = sub_df.merge(df[['ForecastId','pred_CC','pred_Fatalities']],how='left',on='ForecastId')


# In[ ]:


sub_df['ConfirmedCases']= sub_df['pred_CC'].astype('int')
sub_df['Fatalities']= sub_df['pred_Fatalities'].astype('int')
sub_df = sub_df.drop(['pred_CC','pred_Fatalities'],axis=1)


# In[ ]:


sub_df.to_csv("submission.csv", index=False)


# In[ ]:


#sub_df = sub_df.merge(test, how = 'left', on =(['Province/State','Date']))


# In[ ]:


#sub_df['PredCC2'] = sub_df.groupby(['Province/State'])['pred_ConfirmedCases'].shift()*1.33


# In[ ]:


#sub_df.head(50)


# In[ ]:




