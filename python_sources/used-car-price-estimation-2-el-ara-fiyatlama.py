#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter        


# In[ ]:


df = pd.read_csv("../input/car-data/cardata.csv")


# In[ ]:


df.info()


# As we can see there are some null values on "Beygir" which is kW of engine, we will take care of it after

# In[ ]:


df.columns=[i.lower() for i in df.columns]
df.head(2)


# yil = Made year of the car
# marka = Brand
# model = Model
# yakit = Fuel Type
# sanziman = Gearbox
# km = Km.
# boya = Number of painted body parts
# degisen = Number of changed body parts
# kaput_tavan = Number of changed or painted crucial body parts (which alters the price a lot)
# fiyat = Price of the car in Turkish Liras (The main feature we would like to estimate)
# beygir = Power of the car in kW (not in hp)
# tramer = Record of total accident value of the car in Turkish Liras
# hafta = Cumulative week of which the car was sold

# In[ ]:


def countplot(baslik): 
    sns.countplot(x=baslik, data=df)
    plt.xticks(rotation=50)
    plt.show()
    
    print(df[baslik].value_counts())


# In[ ]:


countplot("yakit")


# In[ ]:


countplot("sanziman")


# In[ ]:


countplot("boya")


# In[ ]:


countplot("degisen")


# In[ ]:


g = sns.FacetGrid(df, height= 5)
g.map(sns.distplot, "km", bins=50)
plt.show()

print("Average car km:",df.km.mean())


# In[ ]:


g = sns.FacetGrid(df, height= 5)
g.map(sns.distplot, "yil", bins=15)
g.add_legend()
plt.show()

print("Average car year:",df.yil.mean())


# In[ ]:


g = sns.catplot(x = "yakit", y = "fiyat", data = df, kind="bar",height= 6 )
g.set_ylabels("Fiyat")
plt.show()


# In[ ]:


g = sns.catplot(x = "sanziman", y = "fiyat", data = df, kind="bar",height= 6 )
g.set_ylabels("Fiyat")
plt.show()


# In[ ]:


g = sns.catplot(x = "boya", y = "fiyat", data = df, kind="bar",height= 6 )
g.set_ylabels("Fiyat")
plt.show()


# In[ ]:


g = sns.catplot(x = "degisen", y = "fiyat", data = df, kind="bar",height= 6 )
g.set_ylabels("Fiyat")
plt.show()


# In[ ]:


df.isnull().sum()


# In[ ]:


beygirsiz = list(df["beygir"][df["beygir"].isnull()].index) 


# In[ ]:


for i in beygirsiz:
    tahmin_beygir1= df["beygir"][((df["marka"] == df.iloc[i]["marka"]) & (df["model"] == df.iloc[i]["model"]) & (df["sanziman"] == df.iloc[i]["sanziman"]) & (df["yakit"] == df.iloc[i]["yakit"]))].median()
    tahmin_beygir2= df["beygir"][((df["marka"] == df.iloc[i]["marka"]) & (df["model"] == df.iloc[i]["model"]) & (df["sanziman"] == df.iloc[i]["sanziman"]))].median()
    tahmin_beygir3= df["beygir"][((df["marka"] == df.iloc[i]["marka"]) & (df["model"] == df.iloc[i]["model"]))].median()
    tahmin_beygir4= df["beygir"][((df["marka"] == df.iloc[i]["marka"]))].median()
    tahmin_beygir5= df["beygir"].mean()
    
    if  np.isnan(tahmin_beygir1) == False:
        df["beygir"].iloc[i] = tahmin_beygir1
    if  np.isnan(tahmin_beygir1) == True & np.isnan(tahmin_beygir2) == False:
        df["beygir"].iloc[i] = tahmin_beygir2
    if  np.isnan(tahmin_beygir1) == True & np.isnan(tahmin_beygir2) == True & np.isnan(tahmin_beygir3) == False:
        df["beygir"].iloc[i] = tahmin_beygir3
    if  np.isnan(tahmin_beygir1) == True & np.isnan(tahmin_beygir2) == True & np.isnan(tahmin_beygir3) == True & np.isnan(tahmin_beygir4) == False:
        df["beygir"].iloc[i] = tahmin_beygir4
    else:
        df["beygir"].iloc[i] = tahmin_beygir5


# In[ ]:


plt.subplots(figsize=(10, 10))
sns.heatmap(df.corr(), annot=True,cmap="YlGnBu")
plt.show()


# In[ ]:


df["yas"] = [2017-i+1 for i in df.yil]
del df["yil"]


# In[ ]:


g = sns.FacetGrid(df, height= 5)
g.map(sns.distplot, "yas", bins=15)
g.add_legend()
plt.show()

print("Average car age:",df.yas.mean())


# In[ ]:


df["yakit"] = [2 if i == "DIESEL" else 1 for i in df.yakit]
df.yakit=df["yakit"].astype("category")


# In[ ]:


df["sanziman"] = [2 if i == "AUTOMATIC" else 1 for i in df.sanziman]
df.sanziman=df["sanziman"].astype("category")


# In[ ]:


from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
df["brand_code"] = lb_make.fit_transform(df["marka"])

lb_make = LabelEncoder()
df["model_code"] = lb_make.fit_transform(df["model"])

df.brand_code=df["brand_code"].astype("category")
df.model_code=df["model_code"].astype("category")

brands= df[["marka", "brand_code"]]
models= df[["model", "model_code"]]

del df["marka"]
del df["model"]


# In[ ]:


df.head()


# In[ ]:


x = df.drop(labels="fiyat", axis=1)
y = df.fiyat


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state = 42)

print("X_train:", len(X_train))
print("X_test:", len(X_test))
print("y_train:", len(y_train))
print("y_test:", len(y_test))


# In[ ]:


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=70)
rf.fit(X_train,y_train)
rf_score= rf.score(X_test,y_test)

rf_score


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

interval=np.arange(100, 120)
grid = {"n_estimators":interval}

rf = RandomForestRegressor()
rf_cv = GridSearchCV(rf, grid, cv = 10, n_jobs = -1,verbose = 1)
rf_cv.fit(X_train,y_train)

rf_cv.best_score_, rf_cv.best_params_


# In[ ]:


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=119)
rf.fit(X_train,y_train)
rf_score= rf.score(X_test,y_test)

rf_score

