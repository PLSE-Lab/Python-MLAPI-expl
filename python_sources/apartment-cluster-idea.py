#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


train = pd.read_csv("../input/train.csv", parse_dates=['timestamp'])
test = pd.read_csv("../input/test.csv", parse_dates=['timestamp'])
macro = pd.read_csv("../input/macro.csv", parse_dates=['timestamp'])


# All of following `features` are dependant on sub_area. Now if we have sub_area as feature, we don't need other features which are totally dependant on sub_area. This will make data much more manageable.

# In[ ]:


train.drop('area_m',axis=1,inplace=True)
train.drop('raion_popul',axis=1,inplace=True)
train.drop('green_zone_part',axis=1,inplace=True)
train.drop('indust_part',axis=1,inplace=True)
train.drop('children_preschool',axis=1,inplace=True)
train.drop('preschool_quota',axis=1,inplace=True)
train.drop('preschool_education_centers_raion',axis=1,inplace=True)
train.drop('children_school',axis=1,inplace=True)
train.drop('school_quota',axis=1,inplace=True)
train.drop('school_education_centers_raion',axis=1,inplace=True)
train.drop('school_education_centers_top_20_raion',axis=1,inplace=True)
train.drop('hospital_beds_raion',axis=1,inplace=True)
train.drop('healthcare_centers_raion',axis=1,inplace=True)
train.drop('university_top_20_raion',axis=1,inplace=True)
train.drop('sport_objects_raion',axis=1,inplace=True)
train.drop('additional_education_raion',axis=1,inplace=True)
train.drop('culture_objects_top_25',axis=1,inplace=True)
train.drop('culture_objects_top_25_raion',axis=1,inplace=True)
train.drop('shopping_centers_raion',axis=1,inplace=True)
train.drop('office_raion',axis=1,inplace=True)
train.drop('thermal_power_plant_raion',axis=1,inplace=True)
train.drop('incineration_raion',axis=1,inplace=True)
train.drop('oil_chemistry_raion',axis=1,inplace=True)
train.drop('radiation_raion',axis=1,inplace=True)
train.drop('railroad_terminal_raion',axis=1,inplace=True)
train.drop('big_market_raion',axis=1,inplace=True)
train.drop('nuclear_reactor_raion',axis=1,inplace=True)
train.drop('detention_facility_raion',axis=1,inplace=True)
train.drop('full_all',axis=1,inplace=True)
train.drop('male_f',axis=1,inplace=True)
train.drop('female_f',axis=1,inplace=True)
train.drop('young_all',axis=1,inplace=True)
train.drop('young_male',axis=1,inplace=True)
train.drop('young_female',axis=1,inplace=True)
train.drop('work_all',axis=1,inplace=True)
train.drop('work_male',axis=1,inplace=True)
train.drop('work_female',axis=1,inplace=True)
train.drop('ekder_all',axis=1,inplace=True)
train.drop('ekder_male',axis=1,inplace=True)
train.drop('ekder_female',axis=1,inplace=True)
train.drop('0_6_all',axis=1,inplace=True)
train.drop('0_6_male',axis=1,inplace=True)
train.drop('0_6_female',axis=1,inplace=True)
train.drop('7_14_all',axis=1,inplace=True)
train.drop('7_14_male',axis=1,inplace=True)
train.drop('7_14_female',axis=1,inplace=True)
train.drop('0_17_all',axis=1,inplace=True)
train.drop('0_17_male',axis=1,inplace=True)
train.drop('0_17_female',axis=1,inplace=True)
train.drop('16_29_all',axis=1,inplace=True)
train.drop('16_29_male',axis=1,inplace=True)
train.drop('16_29_female',axis=1,inplace=True)
train.drop('0_13_all',axis=1,inplace=True)
train.drop('0_13_male',axis=1,inplace=True)
train.drop('0_13_female',axis=1,inplace=True)
train.drop('raion_build_count_with_material_info',axis=1,inplace=True)
train.drop('build_count_block',axis=1,inplace=True)
train.drop('build_count_wood',axis=1,inplace=True)
train.drop('build_count_frame',axis=1,inplace=True)
train.drop('build_count_brick',axis=1,inplace=True)
train.drop('build_count_monolith',axis=1,inplace=True)
train.drop('build_count_panel',axis=1,inplace=True)
train.drop('build_count_foam',axis=1,inplace=True)
train.drop('build_count_slag',axis=1,inplace=True)
train.drop('build_count_mix',axis=1,inplace=True)
train.drop('raion_build_count_with_builddate_info',axis=1,inplace=True)
train.drop('build_count_before_1920',axis=1,inplace=True)
train.drop('build_count_1921-1945',axis=1,inplace=True)
train.drop('build_count_1946-1970',axis=1,inplace=True)
train.drop('build_count_1971-1995',axis=1,inplace=True)
train.drop('build_count_after_1995',axis=1,inplace=True)


# In[ ]:


train.head(5)


# Let's find % of missing records in each column.

# In[ ]:


#Finding % of missing records
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])


# In[ ]:


plt.rcParams['figure.figsize'] = (10, 8)
missing_data[missing_data.Percent>0].plot(kind = "barh")


# In[ ]:


missing_data[missing_data.Percent>0]


# ****Work In Progress.**** I want to create apartment cluster feature based on distance based features. Assumption is that if more than one building shares same distance features than they belong to same apartment.

# In[ ]:


plt.figure(figsize=(8,6))
plt.scatter(train.ID_railroad_station_walk,train.ID_railroad_station_avto)
plt.xlabel('ID_railroad_station_walk', fontsize=12)
plt.ylabel('ID_railroad_station_avto', fontsize=12)
plt.show()


# In[ ]:


train.describe(include=['O'])


# In[ ]:


railroad_diff = train.ID_railroad_station_walk-train.ID_railroad_station_avto


# In[ ]:


railroad_diff.hist()


# Strong co-relation between ID_railroad_station_walk and ID_railroad_station_avto. Both value for most of the cases is same. So let's replace null values of ID_railroad_station_walk  with ID_railroad_station_avto.

# In[ ]:


train['ID_railroad_station_walk']=train['ID_railroad_station_walk'].fillna(train.ID_railroad_station_avto)


# Now let's explore other features: railroad_station_walk_min & railroad_station_walk_min

# In[ ]:


#correlation matrix
corrmat = train.corr()


# In[ ]:


fig, ax = plt.subplots(figsize=(12,12))
corrmat['railroad_station_walk_min'].sort_values().tail(10).plot(kind = "barh")
ax.set_title("Co-relation of other features with railroad_station_walk_min")
plt.show()


# In[ ]:


plt.figure(figsize=(8,6))
plt.scatter(train.ID_railroad_station_walk,train.ID_railroad_station_avto)
plt.xlabel('railroad_station_walk_min', fontsize=12)
plt.ylabel('railroad_station_avto_min', fontsize=12)
plt.show()


# In[ ]:


train['railroad_station_walk_min']=train['railroad_station_walk_min'].fillna(train.railroad_station_avto_min)


# In[ ]:


fig, ax = plt.subplots(figsize=(12,12))
corrmat['railroad_station_walk_km'].sort_values().tail(10).plot(kind = "barh")
ax.set_title("Co-relation of other features with railroad_station_walk_km")
plt.show()


# In[ ]:


train['railroad_station_walk_km']=train['railroad_station_walk_km'].fillna(train.railroad_station_avto_km)


# In[ ]:


fig, ax = plt.subplots(figsize=(10,10))
corrmat['metro_min_walk'].sort_values().tail(10).plot(kind = "barh")
ax.set_title("Co-relation of other features with metro_min_walk")
plt.show()


# In[ ]:


train['metro_km_walk']=train['metro_km_walk'].fillna(train.metro_km_avto)
train['metro_min_walk']=train['metro_min_walk'].fillna(train.metro_min_avto)


# Before moving to new features. Lets convert non-numeric features to numeric one.

# In[ ]:


from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler


# In[ ]:


#convert objects / non-numeric data types into numeric
for f in train.columns:
    if train[f].dtype=='object':
        lbl = LabelEncoder()
        lbl.fit(list(train[f].values)) 
        train[f] = lbl.transform(list(train[f].values))


# Now an interesting feature after long time. **floor**. Let's see co-relation of this feature.

# In[ ]:


fig, ax = plt.subplots(figsize=(10,12))
corrmat['floor'].sort_values().tail(10).plot(kind = "barh")
#corrmat['floor'].sort_values().plot(kind = "barh")
ax.set_title("Co-relation of other features with floor")
plt.show()


# In[ ]:


train["floor"].describe()


# In[ ]:


train["max_floor"].describe()


# Floor is depend on max_floor. so Lets explore max_floor as well.

# In[ ]:


fig, ax = plt.subplots(figsize=(10,10))
imp_coef = pd.concat([corrmat['max_floor'].sort_values().head(10),
                      corrmat['max_floor'].sort_values().tail(10)]).plot(kind = "barh")
ax.set_title("Co-relation of other features with max_floor")
plt.show()


# so max_floor is largely depend on floor. and Floor valued doesn't have as many null as max_floor. so let's try to populate floor first.

# In[ ]:


fig, ax = plt.subplots(figsize=(10,10))
imp_coef = pd.concat([corrmat['floor'].sort_values().head(10),
                      corrmat['floor'].sort_values().tail(10)]).plot(kind = "barh")
ax.set_title("Co-relation of other features with floor")
plt.show()


# In[ ]:


floor_group=train.groupby('floor').size()
floor_group.describe()


# In[ ]:


floor_group.plot(kind="bar")


# In[ ]:


train[['floor', 'max_floor']].groupby(['floor']).agg(['mean', 'count'])


# In[ ]:


train["prize_per_sqrmtr"]=train["price_doc"]/train["full_sq"]


# In[ ]:


fig, ax = plt.subplots(figsize=(10,12))
sns.boxplot(x="floor",y="prize_per_sqrmtr",data= train[(train.floor <= 25) & (train.prize_per_sqrmtr<=600000)])
ax.set_title("Box chart of floor vs prize per Square meter")
plt.xticks(rotation='vertical')


# In[ ]:


fig, ax = plt.subplots(figsize=(10,12))
sns.boxplot(x="floor",y="prize_per_sqrmtr",data= train[(train.floor <= 25) & (train.prize_per_sqrmtr<=200000) & (train.sub_area==102)])
ax.set_title("Box chart of floor vs prize per Square meter")
plt.xticks(rotation='vertical')


# In[ ]:


train.groupby("full_sq").size().sort_values()

