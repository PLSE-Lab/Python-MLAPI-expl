#!/usr/bin/env python
# coding: utf-8

# # Aviation Accident Analysis

# I am still new to Kaggle and Data Science. However this dataset is one of my school projects. It has been fun learning from this dataset.

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
sb.set()
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression , LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
lm = LinearRegression()
logmodel = LogisticRegression()


# In[ ]:


# df = pd.read_csv('AviationData.csv', encoding = 'latin1')
df = pd.read_csv('../input/aviation-accident-database-synopses/AviationData.csv', sep=',', header=0, encoding = 'iso-8859-1')


# In[ ]:


df.info()


# In[ ]:


df['Country'].value_counts().head()


# In[ ]:


df['Investigation.Type'].value_counts()


# - It appears that the data is more dominant towards USA. (since its from NTSB) We will narrow down the data set to only USA.
# - Since our analysis is on accidents, we will narrow it to purely Accidents in the dataset.

# # 1. Cleaning of Data Set

# In[ ]:


df = df[df['Country']=='United States']


# In[ ]:


df['Country'].value_counts()


# In[ ]:


df = df[df['Investigation.Type']=='Accident']


# In[ ]:


df['Investigation.Type'].value_counts()


# In[ ]:


df.info()


# - Initial Dropping Data that is not very useful in our analysis

# In[ ]:


df.drop(['Event.Id','Accident.Number','Airport.Code','Airport.Name','Location','Injury.Severity','Registration.Number','FAR.Description','Air.Carrier','Report.Status','Publication.Date','Number.of.Engines'],axis=1,inplace=True)


# In[ ]:


sb.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='Blues')


# - According to the heatmap, it seems that Latitude, Longitude, Aircraft Category and Schedule has way too many missing data to be used. 

# In[ ]:


df.drop(['Latitude','Longitude','Aircraft.Category','Schedule'],axis=1,inplace=True)


# In[ ]:


# Cleaning of Data
df['Total.Fatal.Injuries'].fillna(0, inplace = True)
df['Total.Serious.Injuries'].fillna(0, inplace = True)
df['Total.Minor.Injuries'].fillna(0, inplace = True)
df['Total.Uninjured'].fillna(0, inplace = True)
df['Broad.Phase.of.Flight'].fillna('UNKNOWN',inplace = True)
df['Weather.Condition'].fillna('UNKNOWN',inplace = True)
df['Weather.Condition'].replace({'UNK':'UNKNOWN'},inplace=True)
df['Aircraft.Damage'].fillna('UNKNOWN',inplace=True)
df['Engine.Type'].fillna('UNKNOWN',inplace=True)
df['Purpose.of.Flight'].fillna('Other Work Use',inplace=True)
df['Amateur.Built'].fillna('No',inplace=True)


# In[ ]:


sb.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='Blues')


# # 2. Adding New Columns to Data Set for Analysis

# In[ ]:


df['Total Injuries'] = df['Total.Fatal.Injuries'] + df['Total.Serious.Injuries'] + df['Total.Minor.Injuries']


# In[ ]:


df['Event.Date'] = pd.to_datetime(df['Event.Date'])
df['Year'] = df['Event.Date'].apply(lambda time : time.year)
df['Month']=df['Event.Date'].apply(lambda time:time.month)
# Only want data after 1982
df = df[df['Year']>=1982]


# In[ ]:


df.head()


# # 3. Accident Over The Years (Time Based Analysis)

# **3.1 Exploratory Analysis**

# ##### Accident Trend Over the Years

# In[ ]:


plt.figure(figsize=(20,8))
sb.countplot(df['Year'],palette = 'coolwarm')


# - Accident Counts are on a downward trend

# ##### Accident Trend by Month

# In[ ]:


plt.figure(figsize=(20,8))
sb.countplot(df['Month'],palette='coolwarm')


# - Most Accidents occur between June and August

# **3.2 Time Based Linear Regression** 

# In[ ]:


accYear=pd.DataFrame(df.groupby("Year").count())
accYear=accYear.drop(columns=['Event.Date'])
accYear=accYear.rename(columns={'Month':'Count'})
accYear.head()

X=[ [y] for y in accYear.index.values ]
y=[[e] for e in accYear['Count']]

lm.fit(X,y)


# In[ ]:


accPredict_X=[[y] for y in range (1982, 2025)]
accPredict=lm.predict(accPredict_X)

f, axes = plt.subplots(1, 1, figsize=(15, 8))
plt.plot(X,y)
plt.plot(accPredict_X,accPredict, alpha = 0.5)

print("Accident prediction for the next 5 years:\n" )
for i in range (0,5):
    year=2021+i
    n=-5+i
    print('Year %d: %d' % (year,accPredict[n]))


# # 4. Total Injury Count Over The Years (Time Based Analysis)

# **4.1 Exploratory Analysis**

# ##### Total Injury Count Over the Years

# In[ ]:


by_year = df.groupby('Year').sum()


# In[ ]:


plt.figure(figsize=(12,6))
by_year['Total Injuries'].plot(color='blue',fontsize=15,lw=3,markersize=10,marker='o',markerfacecolor='r')
plt.xlabel('Year',fontsize=13)
plt.ylabel('Total Injury Count',fontsize=13)


# ##### Different Kind of Injuries Count Over the Years

# In[ ]:


by_year[['Total.Fatal.Injuries','Total.Serious.Injuries','Total.Minor.Injuries']].plot(lw = 2, figsize=(12,6))
# to move the legend outside of graph
plt.legend(bbox_to_anchor = (1.05,1), loc=2, borderaxespad=0)


# - Based on the two graphs above, we can see that Injury Count is on a downward trend for all kinds of injuries.

# ##### Uninjured Count Over the Years

# In[ ]:


plt.figure(figsize=(12,6))
by_year['Total.Uninjured'].plot(color='blue', fontsize = 15,lw = 3,markersize=10,marker='o',markerfacecolor='r')
plt.xlabel('Year',fontsize = 13)


# - Surprisingly, the trend is also going downward for Uninjured Personnel from 2017 onwards

# **4.2 Time Based Linear Regression**

# #### Linear Regression for Total Injury Count

# In[ ]:


by_year_2 = by_year.reset_index()


# In[ ]:


X = by_year_2[['Year']]
y = np.asarray(by_year_2['Total.Fatal.Injuries'])
lm.fit(X,y)


# In[ ]:


injury_predict_X = [[y] for y in range (1982,2025)]
injury_predict = lm.predict(injury_predict_X)

f,axes = plt.subplots(1,1,figsize = (16,8))
plt.plot(X,y)
plt.plot(injury_predict_X,injury_predict, alpha = 1.0)

print("Total Injuries Predictions for the next 5 years:\n" )

for i in range (0,5):
    year=2021+i
    n=-5+i
    print('Year %d: %d' % (year,injury_predict[n]))


# **Linear Regression for Uninjured Count**

# In[ ]:


X2 = by_year_2[['Year']]
y2 = np.asarray(by_year_2['Total.Uninjured'])
lm.fit(X2, y2)


# In[ ]:


injury_predict_X2 = [[y] for y in range (1982,2025)]
injury_predict2 = lm.predict(injury_predict_X2)

f,axes = plt.subplots(1,1,figsize = (16,8))
plt.plot(X2,y2)
plt.plot(injury_predict_X2,injury_predict2, alpha = 1.0)

print("Total Uninjured Predictions for the next 5 years:\n" )

for i in range (0,5):
    year=2021+i
    n=-5+i
    print('Year %d: %d' % (year,injury_predict2[n]))


# # 5. Phase of Flight Analysis

# **5.1 Exploratory Analysis**

# In[ ]:


by_phase = df.groupby('Broad.Phase.of.Flight').sum().reset_index()
by_phase = by_phase.drop(['Year','Month'], axis=1)
by_phase


# In[ ]:


plt.figure(figsize = (14,8))
sb.barplot(x = 'Broad.Phase.of.Flight',y='Total Injuries' , data = by_phase.reset_index() , palette = 'coolwarm', ec = 'black')
plt.title('Phase Of Flight ' , size = 20)
plt.xlabel('')
plt.ylabel('Total Injury Count', size = 20)
plt.tight_layout()


# - Most Accident and Injuries Sustained occured during TAKEOFF, CRUISE, MANEUVERING

# In[ ]:


yearPhase = df.groupby(by = ['Year','Broad.Phase.of.Flight']).sum()['Total Injuries'].unstack()
yearPhase.head()


# In[ ]:


plt.figure(figsize = (20,10))
sb.heatmap(yearPhase, cmap = 'Blues')
plt.xlabel('')


# - As seen from the heatmap, we can observe that during 'Takeoff', it is still very prone to accidents with substantial amount of injury count.

# - Since the common few are Landing, Takeoff, Cruise, Maneuvering and Approach. We will classify the rest as others.

# In[ ]:


def other_phases(phase):
    if phase in (['UNKNOWN','TAXI','DESCENT','CLIMB','GO-AROUND','STANDING']):
        return 'OTHER'
    else:
        return phase


# In[ ]:


df['Phases'] = df['Broad.Phase.of.Flight'].apply(other_phases)


# In[ ]:


plt.figure(figsize=(8,4))
sb.countplot(df['Phases'], palette='coolwarm')


# In[ ]:


df.groupby('Aircraft.Damage')['Phases'].value_counts()


# In[ ]:


plt.figure(figsize=(12,6))
sb.countplot(df['Aircraft.Damage'],hue=df['Phases'],palette='coolwarm')
plt.legend(bbox_to_anchor = (1.05,1), loc=2, borderaxespad=0)


# In[ ]:


plt.figure(figsize=(12,6))
sb.countplot(df['Weather.Condition'],hue=df['Phases'],palette='coolwarm')
plt.legend(bbox_to_anchor = (1.05,1), loc=2, borderaxespad=0)


# **5.2 RandomForest to predict Phase of Flight**

# - Predictors: Weather Condition, Aircraft Damage, Injuries, Non Injuries

# In[ ]:


rfc = RandomForestClassifier(n_estimators=300)


# In[ ]:


df_phase = pd.get_dummies(df,columns=['Aircraft.Damage','Weather.Condition'],drop_first=True)


# In[ ]:


df_phase.columns


# In[ ]:


X = df_phase[['Aircraft.Damage_Minor', 'Aircraft.Damage_Substantial',
       'Aircraft.Damage_UNKNOWN','Total Injuries','Total.Uninjured',
       'Weather.Condition_UNKNOWN','Weather.Condition_VMC']]

y = df_phase['Phases']


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30)


# In[ ]:


rfc.fit(X_train,y_train)


# In[ ]:


predictions = rfc.predict(X_test)


# In[ ]:


print(classification_report(y_test,predictions))


# In[ ]:


print('Classification Accuracy: {:.3f}'.format(rfc.score(X_test,y_test)))


# # 6. Purpose of Flight Analysis

# **6.1 Exploratory Analysis**

# In[ ]:


df['Purpose.of.Flight'].value_counts()


# - Since Personal Flights hold more than 50% of the counts, we shall categorise it into Personal and Non Personal Flights to compare.

# In[ ]:


def personal(purpose):
    if purpose == 'Personal':
        return 1
    
    else:
        return 0


# In[ ]:


df['Personal Flight'] = df['Purpose.of.Flight'].apply(personal)


# In[ ]:


sb.countplot(df['Personal Flight'])


# **Comparing Personal Flights & Weather Conditions**

# In[ ]:


sb.countplot(df['Personal Flight'], hue=df['Weather.Condition'])
plt.legend(bbox_to_anchor = (1.05,1), loc=2, borderaxespad=0)


# - It seems both Personal and Non Personal Flights, most accidents occured at VMC Weather Condition

# **Comparing Personal Flights & Aircraft Damage**

# In[ ]:


sb.countplot(df['Personal Flight'],hue=df['Aircraft.Damage'])
plt.legend(bbox_to_anchor = (1.05,1), loc=2, borderaxespad=0)


# - For both Personal and Non Personal Flights, most aircraft suffered Substantial Damage. It is worth noting that Personal Flight Aircrafts are either 'Destroyed' or 'Substantial' damaged.

# **Comparing Personal Flights & Phase**

# In[ ]:


plt.figure(figsize=(12,8))
sb.countplot(df['Personal Flight'],hue=df['Broad.Phase.of.Flight'],palette='coolwarm')
plt.legend(bbox_to_anchor = (1.05,1), loc=2, borderaxespad=0)


# - Judging from the Phase of Flight where accident occured, it seems like personal flight has more 'Cruising' Accidents and lesser 'Maneuvering' Accidents.

# **Comparing Personal Flights with Engine Type**

# In[ ]:


plt.figure(figsize=(18,10))
sb.countplot(df['Personal Flight'],hue=df['Engine.Type'],palette = 'coolwarm')
plt.legend(bbox_to_anchor = (1.05,1), loc=2, borderaxespad=0)


# - It seems like Non Personal Flight Purpose has more variety in engines.

# **Comparing Injuries with Flight Purpose**

# In[ ]:


df_new = df[df['Total Injuries']>=1]
df_new = df_new[df_new['Total.Uninjured']>=1]


# In[ ]:


f,axes = plt.subplots(2,1,figsize=(18,8))
sb.boxplot(x='Total Injuries',y='Personal Flight',data=df_new,orient ='h',ax=axes[0])
sb.boxplot(x='Total.Uninjured',y='Personal Flight',data=df_new,orient='h',ax=axes[1])


# - From the above boxplot, we can see that for personal flights, there are much lesser outliars.

# **6.2 Logistic Regression to Predict if flight is Personal**

# - Predictors: Total Injuries, Total Uninjured, Engine Type

# In[ ]:


df_new = pd.get_dummies(df_new,columns=['Engine.Type'],drop_first=True)


# In[ ]:


df_new.columns


# In[ ]:


df_new = df_new[['Personal Flight','Engine.Type_Reciprocating',
       'Engine.Type_Turbo Fan', 'Engine.Type_Turbo Jet',
       'Engine.Type_Turbo Prop', 'Engine.Type_Turbo Shaft',
       'Engine.Type_UNKNOWN', 'Engine.Type_Unknown','Total Injuries','Total.Uninjured']]


# In[ ]:


df_new.head(2)


# In[ ]:


X = df_new.drop('Personal Flight',axis=1)
y = df_new['Personal Flight']


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30)


# In[ ]:


logmodel.fit(X_train,y_train)


# In[ ]:


predictions = logmodel.predict(X_test)


# In[ ]:


print(classification_report(y_test,predictions))


# In[ ]:


print(confusion_matrix(y_test,predictions))


# In[ ]:


print('Classification Accuracy: {:.3f}'.format(logmodel.score(X_test,y_test)))


# # 7. Aircraft Manufacturer Analysis

# In[ ]:


df.columns


# In[ ]:


df['Make'].value_counts().head(30)


# - We will focus the analysis on the top 3 companies with highest accident count.

# In[ ]:


def get_company(company):
    
    if company in ['Cessna','Piper','Beech']:
        
        return company.upper()
    
    else:
        
        if company in ['CESSNA','PIPER','BEECH']:
            
            return company
        
        else:
            
            return 'OTHER'


# In[ ]:


df['Make'].apply(get_company).value_counts()


# In[ ]:


df_selected = df[df['Make'].isin(['CESSNA','PIPER','BEECH'])]


# **Company and Phase of Accident**

# In[ ]:


plt.figure(figsize=(12,6))
sb.countplot(df_selected['Make'],hue=df['Phases'],palette='coolwarm')
plt.legend(bbox_to_anchor = (1.05,1), loc=2, borderaxespad=0)


# **Company and State of Aircraft after accident**

# In[ ]:


plt.figure(figsize=(12,6))
sb.countplot(df_selected['Make'],hue=df['Aircraft.Damage'],palette='coolwarm')
plt.legend(bbox_to_anchor = (1.05,1), loc=2, borderaxespad=0)


# **Company and Number of Injuries/Non Injuries**

# In[ ]:


f,axes = plt.subplots(3,1,figsize=(18,12))
sb.boxplot(x='Total Injuries',y='Make',data=df_selected,orient ='h',ax=axes[0])
sb.boxplot(x='Total.Uninjured',y='Make',data=df_selected,orient='h',ax=axes[1])
sb.countplot(df_selected['Make'])


# In[ ]:




