#!/usr/bin/env python
# coding: utf-8

# In[169]:


import numpy as np
import pandas as pd


# class: edible=e, poisonous=p
# 
# cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
# 
# cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s
# 
# cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y
# 
# bruises: bruises=t,no=f
# 
# odor: almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s
# 
# gill-attachment: attached=a,descending=d,free=f,notched=n
# 
# gill-spacing: close=c,crowded=w,distant=d
# 
# gill-size: broad=b,narrow=n
# 
# gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y
# 
# stalk-shape: enlarging=e,tapering=t
# 
# stalk-root: bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?
# 
# stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
# 
# stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
# 
# stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
# 
# stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
# 
# veil-type: partial=p,universal=u
# 
# veil-color: brown=n,orange=o,white=w,yellow=y
# 
# ring-number: none=n,one=o,two=t
# 
# ring-type: cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z
# 
# spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y
# 
# population: abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y
# 
# habitat: grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d

# In[170]:


df = pd.read_csv("../input/mushrooms.csv")
print(df.shape)
for col in df.columns:
    df.rename(columns={col:col.capitalize()}, inplace=True)   #I just like it capitalized
df.describe()


# Class, Bruises and Gill-size only have 2 unique values
# 
# Class will be called Edible
# 
# **Veil-Type always p so it will be dropped**

# In[171]:


def SporeNamer(x):
    if x == 'k':
        return 'Black'
    if x == 'n':
        return 'Brown'
    if x == 'b':
        return 'Buff'
    if x == 'h':
        return 'Chocolate'
    if x == 'r':
        return 'Green'
    if x == 'o':
        return 'Orange'
    if x == 'u':
        return 'Purple'
    if x == 'w':
        return 'White'
    return 'Yellow'
def OdorNamer(x):
    if x == 'a':
        return 'Almond'
    if x == 'l':
        return 'Anise'
    if x == 'c':
        return 'Creosote'
    if x == 'y':
        return 'Fishy'
    if x == 'f':
        return 'Foul'
    if x == 'm':
        return 'Musty'
    if x == 'n':
        return 'None'
    if x == 'p':
        return 'Pungent'
    return 'Spicy'
def GillNamer(x):
    if x == 'k':
        return 'Black'
    if x == 'n':
        return 'Brown'
    if x == 'b':
        return 'Buff'
    if x == 'h':
        return 'Chocolate'
    if x == 'r':
        return 'Green'
    if x == 'o':
        return 'Orange'
    if x == 'p':
        return 'Pink'
    if x == 'e':
        return 'Red'
    if x == 'u':
        return 'Purple'
    if x == 'w':
        return 'White'
    return 'Yellow'
def PopNamer(x):
    if x == 'a':
        return 'Abundant'
    if x == 'c':
        return 'Clustered'
    if x == 'n':
        return 'Numerous'
    if x == 's':
        return 'Scattered'
    if x == 'v':
        return 'Several'
    return 'Solitary'


# In[172]:


df['Class'] = df['Class'].apply(lambda x: 'Edible' if x == 'e' else 'Poisonous')
df['Bruises'] = (df['Bruises'] == 't')

df['Odor'] = df['Odor'].apply(OdorNamer)
df['Spore-print-color'] = df['Spore-print-color'].apply(SporeNamer)
df['Gill-color'] = df['Population'].apply(GillNamer)
df['Population'] = df['Population'].apply(PopNamer)
df.rename(columns={'Class':'Edible'}, inplace=True)
df.drop('Veil-type',axis=1,  inplace=True)
df.head()


# In[173]:


from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
dfEncoded = df.apply(lambda col: LE.fit_transform(col))
dfEncoded.head()


# In[174]:


dfEncoded.describe()


# In[175]:


# Classification
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Modelling Helpers :
from sklearn.model_selection import train_test_split


# In[176]:


RFC = RandomForestClassifier(n_estimators=666, random_state=82)
KNN = KNeighborsClassifier(n_neighbors = 1)
BAG = BaggingClassifier(random_state = 222, n_estimators=92)
GradBost = GradientBoostingClassifier(random_state = 15)
ADA = AdaBoostClassifier(random_state = 37)
DT = DecisionTreeClassifier(random_state=12)


# In[177]:


x = dfEncoded.copy()
y = x.pop('Edible')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 37)


# In[187]:


RFC.fit(x_train,y_train)
RFC_pred = RFC.predict(x_test)
print("accuracy: {} %".format((RFC.score(x_test,y_test)*100)))
for Counter, i in enumerate(RFC.feature_importances_):
    if i > 0.10:
        print("{} makes up {} % of the decision making process".format(x.columns[Counter], ("%.2f" % (i*100))))


# In[190]:


DT.fit(x_train,y_train)
DT_pred = DT.predict(x_test)
print("accuracy: "+ str(DT.score(x_test,y_test)*100) + "%")
for Counter, i in enumerate(DT.feature_importances_):
    if i > 0.10:
        print("{} makes up {} % of the decision making process".format(x.columns[Counter], ("%.2f" % (i*100))))


# In[189]:


ADA.fit(x_train,y_train)
ADA_pred = ADA.predict(x_test)
print("accuracy: "+ str(ADA.score(x_test,y_test)*100) + "%")
for Counter, i in enumerate(ADA.feature_importances_):
    if i > 0.10:
        print("{} makes up {} % of the decision making process".format(x.columns[Counter], ("%.2f" % (i*100))))


# In[195]:


GradBost.fit(x_train,y_train)
GradBost_pred = GradBost.predict(x_test)
print("accuracy: "+ str(("%.2f" %(GradBost.score(x_test,y_test)*100))) + "%")
for Counter, i in enumerate(GradBost.feature_importances_):
    if i > 0.10:
        print("{} makes up {} % of the decision making process".format(x.columns[Counter], ("%.2f" % (i*100))))


# In[192]:


BAG.fit(x_train,y_train)
BAG_pred = BAG.predict(x_test)
print("accuracy: "+ str(BAG.score(x_test,y_test)*100) + "%")


# In[193]:


KNN.fit(x_train,y_train)
KNN_pred = KNN.predict(x_test)
print("accuracy: "+ str(KNN.score(x_test,y_test)*100) + "%")


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

fig, ax = plt.subplots(figsize=(18,6))
g = sns.countplot(df["Spore-print-color"], ax=ax, data = df[["Spore-print-color", 'Edible']],
                  hue='Edible', palette='hls')


# In[ ]:


fig, ax = plt.subplots(figsize=(18,6))
g = sns.countplot(df["Odor"], ax=ax, data = df[["Odor", 'Edible']], hue='Edible', palette='hls')


# In[ ]:


fig, ax = plt.subplots(figsize=(18,6))
g = sns.countplot(df["Gill-color"], ax=ax, data = df[["Gill-color", 'Edible']], hue='Edible', palette='hls')


# In[ ]:


fig, ax = plt.subplots(figsize=(18,6))
g = sns.countplot(df["Population"], ax=ax, data = df[["Population", 'Edible']], hue='Edible', palette='hls')


# In conclusion, look for:
# 
# Odor
# 
# Spore-print-color
# 
# Gill-color
# 
# Population
