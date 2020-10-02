#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../input/kc_house_data.csv")


# In[ ]:


df.head()


# In[ ]:


df.columns


# In[ ]:


df.info()


# In[ ]:


len(df)


# # It appears there are no non-null values, and there are no categorical variables either, so no need for imputing or dummy variables.

# In[ ]:


cor = df.corr()
sns.set(font_scale=1.25)
plt.figure(figsize=(12,12))
hm = sns.heatmap(cor, cbar=True, annot=True, square=True,
                 fmt='.2f', annot_kws={'size': 10}, 
                 yticklabels=list(cor.columns), xticklabels=list(cor.columns))
plt.show()


# price appears to be highly correlated with sqft_above, grade, sqft_living, bathrooms, sqft_living15, and, to a lesser extent, lat, view, sqft_basement, waterfront, floors, bedrooms.
# 
# sqft_living15 and sft_lot15 don't seem to add much of value compared to their non-15 counterparts.
# 
# sqft_above, grade and sqft_living probably collinear, could be wise to discard one or two.

# # Some data exploration.

# In[ ]:


plt.scatter(df.sqft_living,df.price)
plt.show()


# Likely dealing with non-normal distributions.

# In[ ]:


plt.hist(df.sqft_living,bins=30)
plt.show()


# In[ ]:


plt.hist(df.price,bins=30)
plt.show()


# Indeed, both price and sqft_living are left-skewed. Will np.log() them both.

# In[ ]:


plt.scatter(np.log(df.sqft_living),np.log(df.price))
plt.show()


# Better, but still far from satisfactory. While log(price) and log(sqft_living) are clearly related, it is obvious that something is causing the relation to "shift". It may be a good idea to look for variables that correlate with price, but not with sqft_living.

# In[ ]:


(cor["price"] - cor["sqft_living"]).sort_values()


# It seems that lat is our best bet. Waterfront, view, and condition may also be worth looking into.

# In[ ]:


plt.hist(df.lat,bins=30)
plt.show()


# Latitude distribution appears to be distinctly multimodal.

# In[ ]:


cm = plt.cm.get_cmap('RdYlBu')
#cm = plt.cm.get_cmap('gnuplot')
plt.figure(figsize=(12,9))
plt.scatter(np.log(df.sqft_living),np.log(df.price),
           c = df.lat, cmap = cm, s=5)
plt.colorbar()
plt.show()


# Better. That's a nice gradient. As a zeroth order approximation, one could say that lower latitudes imply lower prices. Upon closer inspection, one can see that there is a thin belt of high latitude values in the middle. There also seem to be a few exceptions to the rule on both sides. Let's see if our second suspect, waterfront, can account for some of those.

# In[ ]:


cm = plt.cm.get_cmap('RdYlBu')
#cm = plt.cm.get_cmap('gnuplot')
plt.figure(figsize=(12,9))
plt.scatter(np.log(df.sqft_living),np.log(df.price),
           c = df.lat, cmap = cm, s=5)
plt.colorbar()
plt.show()


# In[ ]:


df.waterfront.unique()


# In[ ]:


cm = plt.cm.get_cmap('RdYlBu')
#cm = plt.cm.get_cmap('gnuplot')
plt.figure(figsize=(12,9))
plt.scatter(np.log(df.sqft_living),np.log(df.price),
           c = df.lat, cmap = cm, s=(df.waterfront*40 + 5))
plt.colorbar()
plt.show()


# Indeed, waterfront does seem to push data points up. Let's see what happens with view.

# In[ ]:


print(df.view.unique())


# In[ ]:


#cm = plt.cm.get_cmap('Set1')
cm = plt.cm.get_cmap('RdYlBu')
#cm = plt.cm.get_cmap('gnuplot')
plt.figure(figsize=(12,9))
plt.scatter(np.log(df.sqft_living),np.log(df.price),
           c = df.lat, cmap = cm, s=(df.view+1)*20)# +1 because we don't want 0s to be invisible.
plt.colorbar()
plt.show()


# Attempting linear regression with raw data.

# In[ ]:


f = ["bedrooms","bathrooms","sqft_living","sqft_lot","floors","waterfront","view","condition","yr_built","long"]


# In[ ]:


trn,tst = train_test_split(df)
lm = LinearRegression()
lm.fit(trn[f],trn["price"])
lm.score(tst[f],tst["price"])


# Going to scale data.

# In[ ]:


df1 = df.copy(deep=True)
df1 = df1.drop(["date","id"],axis=1)
mapper = DataFrameMapper([(df1.columns, StandardScaler())])
scaled_features = mapper.fit_transform(df1.copy(), 4)
scaled_features_df = pd.DataFrame(scaled_features, index=df1.index, columns=df1.columns)
trn,tst=train_test_split(scaled_features_df)
lm.fit(trn[f],trn["price"])
lm.score(tst[f],tst["price"])


# Scaling did not improve results. Going to take a closer look at raw data.

# In[ ]:


plt.scatter(np.log(df.price),df.condition)
plt.show()


# In[ ]:





# Some humble feature engineering.

# In[ ]:


df["family"] = (df.view+df.condition)


# In[ ]:





# In[ ]:


df.groupby(["condition","view"]).mean()["price"]/100000


# It's not perfect, but it does seem that price tends to go up with increasing view and 

# In[ ]:


plt.scatter((np.log(df.sqft_living)*(df.view+1+df.condition)),np.log(df.price),
            c=(df.view+1+df.condition),cmap = cm, s=5)
plt.show()


# In[ ]:


plt.scatter((np.log(df.sqft_living)*(df.view+df.condition)),np.log(df.price),
            c=(df.lat),cmap = cm, s=5)
plt.colorbar()
plt.show()


# In[ ]:


df.columns


# In[ ]:


f = ["bedrooms","bathrooms","sqft_living","sqft_lot","floors","waterfront","view","condition","yr_built","long"]


# In[ ]:


trn,tst=train_test_split(df)


# In[ ]:


lm = LinearRegression()
lm.fit(trn[f],trn["price"])
lm.score(tst[f],tst["price"])


# Separating by latitude. Arbitrarily based on the color map above for the time being, will do something more refined and quantitative later.

# In[ ]:


l0 = df1.loc[df1.lat<47.45]
l1 = df1.loc[(df1.lat >= 47.45) & (df1.lat < 47.55)]
l2 = df1.loc[(df1.lat >= 47.55) & (df1.lat < 47.72)]
l3 = df1.loc[(df1.lat >= 47.72)]


# In[ ]:


print(len(l0),len(l1),len(l2),len(l3))


# In[ ]:


trn0,tst0 = train_test_split(l0)
trn1,tst1 = train_test_split(l1)
trn2,tst2 = train_test_split(l2)
trn3,tst3 = train_test_split(l3)


# In[ ]:


#x = trn0.loc[(trn0.family==7) & (trn0.lat < 45.45)]
#lm0 = RidgeCV(alphas=(0.0001,0.001,0.01,0.1,1,10))
lm0 = LinearRegression()
lm0.fit(trn0[f],trn0["price"])


# In[ ]:


lm0.score(tst0[f],tst0["price"])


# In[ ]:


lm1 = LinearRegression()
lm1.fit(trn1[f],trn1["price"])
lm1.score(tst1[f],tst1["price"])


# In[ ]:


lm2 = LinearRegression()
lm2.fit(trn2[f],trn2["price"])
lm2.score(tst2[f],tst2["price"])


# Dividing by latitude range improved the score improved from 0.58 to 0.69, 0.66, 0.68, 0.69. Better, but there is still room for improvement. 

# In[ ]:




