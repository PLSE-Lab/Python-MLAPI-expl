#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# Hai kagglers, here is another dataest to explore. Mushroom Classification Dataset.
# In this kernel, i'm going to do some exploratory data analysis, build the models and run prediction as well.
# Ler's get started...

# ## Import Modules

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ## Quick Look
# 
# We got 8124 row and 22 columns on this dataset, with no missing value at all, we are good to go.

# In[ ]:


df = pd.read_csv('../input/mushroom-classification/mushrooms.csv')
df.head()


# In[ ]:


df.info()


# In[ ]:


df.columns


# ## Opitional
# 
# In this dataset, everything was display on their initial words. For me, it is much easier to understand when we can get their full words since we are going to do some EDA, if you want to keep it as it is, thats fine. This is just my personal prefrence.

# In[ ]:


df['class'].replace(to_replace=['e','p'], value=['edible','poisonous'],inplace=True)
df['cap-shape'].replace(to_replace=['b', 'c','f','x','k','s'], value=['bell','conical','convex','flat','knobbed','sunken'],inplace=True)
df['cap-surface'].replace(to_replace=['f','g','y','s'], value=['fibrous','grooves','scaly','smooth'],inplace=True)
df['cap-color'].replace(to_replace=['n','b','c','g','r','p','u','e','w','y'], value=['brown','buff','cinnamon','gray','green','pink','purple','red','white','yellow'],inplace=True)
df['bruises'].replace(to_replace=['t', 'f'], value=['bruises','no'],inplace=True)
df['odor'].replace(to_replace=['a','l','c','y','f','m','n','p','s'], value=['almond','anise','creosote','fishy','foul','musty','none','pungent','spicy'],inplace=True)
df['gill-attachment'].replace(to_replace=['a','d','f','n'], value=['attached','descending','free','notched'],inplace=True)
df['gill-spacing'].replace(to_replace=['c','w','d'], value=['close','crowded','distant'],inplace=True)
df['gill-size'].replace(to_replace=['b', 'n'], value=['broad','narrow'],inplace=True)
df['gill-color'].replace(to_replace=['k','n','b','h','g','r','o','p','u','e','w','y'], value=['black','brown','buff','chocolate','gray','green','orange','pink','purple','red','white','yellow'],inplace=True)
df['stalk-shape'].replace(to_replace=['e', 't'], value=['enlarging','tapering'],inplace=True)
df['stalk-root'].replace(to_replace=['b','c','u','e','z','r','?'], value=['bulbous','club','cup','equal','rhizomorphs','rooted','missing'],inplace=True)
df['stalk-surface-above-ring'].replace(to_replace=['f','y','k','s'], value=['fibrous','scaly','silky','smooth'],inplace=True)
df['stalk-surface-below-ring'].replace(to_replace=['f','y','k','s'], value=['fibrous','scaly','silky','smooth'],inplace=True)
df['stalk-color-above-ring'].replace(to_replace=['n','b','c','g','o','p','e','w','y'], value=['brown','buff','cinnamon','gray','orange','pink','red','white','yellow'],inplace=True)
df['stalk-color-below-ring'].replace(to_replace=['n','b','c','g','o','p','e','w','y'], value=['brown','buff','cinnamon','gray','orange','pink','red','white','yellow'],inplace=True)
df['veil-type'].replace(to_replace=['p', 'u'], value=['partial','universal'],inplace=True)
df['veil-color'].replace(to_replace=['n','o','w','y'], value=['brown','orange','white','yellow'],inplace=True)
df['ring-number'].replace(to_replace=['n','o','t'], value=['none','one','two'],inplace=True)
df['ring-type'].replace(to_replace=['c','e','f','l','n','p','s','z'], value=['cobwebby','evanescent','flaring','large','none','pendant','sheathing','zone'],inplace=True)
df['spore-print-color'].replace(to_replace=['k','n','b','h','r','o','u','w','y'], value=['black','brown','buff','chocolate','green','orange','purple','white','yellow'],inplace=True)
df['population'].replace(to_replace=['a','c','n','s','v','y'], value=['abundant','clustered','numerous','scattered','several','solitary'],inplace=True)
df['habitat'].replace(to_replace=['g','l','m','p','u','w','d'], value=['grasses','leaves','meadows','paths','urban','waste','woods'],inplace=True)


# Let's see how the data looks like.

# In[ ]:


df.head()


# ## EDA

# Perfect, let's move on to correlation. I love to use heatmap, but you can use anything you want

# In[ ]:


corr = df.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1)
plt.figure(figsize=(16,16))
sns.heatmap(corr, cmap = "RdBu_r", vmax=0.9, square=True)


# Some column that has strong correlation with class is gill-size and gill-spacing. 
# 
# BUT WAIT...
# 
# Something is wrong with veil-type, let's check what's in veil type

# In[ ]:


df['veil-type'].value_counts()


# Veil-type only has 1 kind of value in its columns, that is why we cant see variation of veil type in heat map.
# Well, that is fine i guess. Let's just dig a little but deeper on those column with high correlation on class columns.

# In[ ]:


IF = corr['class'].sort_values(ascending=False).head(10).to_frame()
IF.head(8)


# Those are some columns that has a high correlation with class columns, even most of them are quite low on their correlation value, it is worth to check.

# In[ ]:


print(df.groupby('gill-size')['class'].value_counts())
df.groupby('gill-size')['class'].value_counts().unstack().plot.barh()


# ### Gill-size
# 
# in gill-size, the highest correlation with class column say almost all mushroom with narrow gill zise are posionous and around 60% of broad gill-size are edible, i guess it is safe to say that broad gill size mushroom are highly edible than narrow gill size mushroom.

# In[ ]:


print(df.groupby('gill-spacing')['class'].value_counts())
df.groupby('gill-spacing')['class'].value_counts().unstack().plot.barh()


# ### Gill-Spacing
# 
# How about gill spacing,
# in gill-spacing, the second correlation with class column say almost all crowded gill spacing mushroom are edible and around 40% of close gill-spacing are edible, it's better to play it safe and only eat crowded gill spacing mushroom.

# In[ ]:


print(df.groupby('cap-surface')['class'].value_counts())
df.groupby('cap-surface')['class'].value_counts().unstack().plot.barh()


# ### Cap-surface
# 
# in Cap surface, data show that almost all mushroom with smooth and scaly cap surface are highly posionous, while mushroom with grooves cap surface is also posionous (since the data only have a few record of it, i guess this is the rare mushroom), and fibrous cap surface are highly edible, i think only fibrous cap surfaced mushroom are safe to consume.

# In[ ]:


print(df.groupby('ring-number')['class'].value_counts())
df.groupby('ring-number')['class'].value_counts().unstack().plot.barh()


# ### Ring Number
# 
# Ring number in mushroom, the easiest thing to notice say all the mushroom with two ring number is highly edible than the other, while one is also edible but the chance is quite low since it's only 50% of them safe to consume, and mushroom with no ring number are not the option, it's 100% posionous.

# In[ ]:


print(df.groupby('gill-attachment')['class'].value_counts())
df.groupby('gill-attachment')['class'].value_counts().unstack().plot.barh()


# ### Gill-attachment
# 
# in Gill-attachment, almost all the mushroom with attached gill is 100% edible than the other, while free is also edible but the chance is quite low since it's only 50% of them safe to consume,

# In[ ]:


print(df.groupby('veil-color')['class'].value_counts())
df.groupby('veil-color')['class'].value_counts().unstack().plot.barh()


# ### Veil Color
# 
# Unlike veil type, veil color has a lot of variation, another easy thing to notice i guess. The data say all the mushroom with orange or brown are highly edible than the others, while white veil colored is also edible but the chance is quite low since it's only 50% of them safe to consume, and yellow veil colored mushroom (another rare mushroom) is not the option, it's 100% posionous.

# In[ ]:


print(df.groupby('stalk-shape')['class'].value_counts())
df.groupby('stalk-shape')['class'].value_counts().unstack().plot.barh()


# ### Stalk Shape
# 
# Stalk shape, the last one, say all mushroom with tapering stalk shape is highly edible while on the other side enlarging stalk shape is highky posionous.

# ## Modeling

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[ ]:


## Feature engineering

df['class'].replace(to_replace=['edible','poisonous'], value=['0','1'],inplace=True)
df.head()


# In[ ]:


# Split the data

X = df.drop('class', axis=1)
y = df['class']


# In[ ]:


# handling categorical features

X = pd.get_dummies(X)
X.head()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=44)


# ### Random Forest Classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

# Fitting Random Forest Classification
classifier = RandomForestClassifier(n_estimators = 200)
classifier.fit(X_train, y_train)

# predict
RF_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, RF_pred)
accuracy


# ### SGDClassifier

# In[ ]:


from sklearn.linear_model import SGDClassifier

# fit to SGDClassifier
sgd= SGDClassifier()
sgd.fit(X_train, y_train)

# predict
SGD_pred = sgd.predict(X_test)
acc = accuracy_score(y_test, SGD_pred)
print(acc)


# ### Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression

#fit to LogReg
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Predict
LR_pred = lr.predict(X_test)
acc = accuracy_score(y_test, LR_pred)
print(acc)


# In[ ]:


## Distribution Comparison

f = plt.figure(figsize=(15,12))

# Basic Distribution
ax = f.add_subplot(221)
ax = sns.distplot(y_test)
ax.set_title('Basic Distribution')

# Random Forest Predicted result
ax = f.add_subplot(222)
xx = pd.DataFrame(RF_pred)
ax = sns.distplot(RF_pred, label="Predicted Values")
ax.set_title('Random Forest Predicted result')

# SGDClassifier Predicted result
ax = f.add_subplot(223)
ax = sns.distplot(SGD_pred, label="Predicted Values")
ax.set_title('SGDClassifier Predicted result')

# Logistic Regression Predicted result
ax = f.add_subplot(224)
ax = sns.distplot(LR_pred, label="Predicted Values")
ax.set_title('Logistic Regression Predicted result')


# ## End
# 
# Those models we made is perfect models, mostly 100% accuracy, and i put the distribution comparison as well, but there is hardly any difference. Well in the end of the day we knew what kind of mushroom is edible and posionous, i'm not a mushroom expert and all the data is form the dataset itself, i'm sorry if there is anything wrong with this kernel.
# 
# Hope you like it, have a good day kagglers
