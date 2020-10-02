#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


df = pd.read_csv('../input/kc_house_data.csv')

df


# In[ ]:


df.shape


# In[ ]:


feature_columns = ['bedrooms','bathrooms','sqft_living','sqft_lot']
x = df.iloc[:801, 3:7]
x


# In[ ]:


y = df.price.iloc[:801]
from sklearn.linear_model import LinearRegression 
clf = LinearRegression()
clf.fit(x,y)


# In[ ]:


x_new = df.iloc[802:1602, 3:7]
x_new


# In[ ]:


new_predict = clf.predict(x_new)
new_predict


# In[ ]:


plt.scatter(df['price'], df['sqft_living'])


# In[ ]:


plt.scatter(df['price'],df['sqft_lot'])


# In[ ]:


plt.scatter(df['price'],df['bathrooms'])


# In[ ]:


plt.scatter(df['price'],df['bedrooms'])


# In[ ]:


#Distribution
from scipy import stats
from scipy.stats import skew,norm
from scipy.stats.stats import pearsonr
#density plot
sb.distplot(df.condition,fit=norm)
plt.ylabel('frequency')
plt.title('price distribution')
(mu,sigma) = norm.fit(df['condition'])

fig = plt.figure()
res = stats.probplot(df['condition'],plot=plt)
plt.show()

print('skewness %f' % df['condition']).skew()
print('kurtosis %f' % df['condition']).kurt()



# In[ ]:


from scipy import stats
from scipy.stats import skew,norm
from scipy.stats.stats import pearsonr
#density plot
sb.distplot(df.condition,fit=norm)
plt.ylabel('frequency')
plt.title('price distribution')
(mu,sigma) = norm.fit(df['sqft_above'])

fig = plt.figure()
res = stats.probplot(df['sqft_above'],plot=plt)
plt.show()

print('skewness %f' % df['sqft_above']).skew()
print('kurtosis %f' % df['sqft_above']).kurt()



# In[ ]:


from scipy import stats
from scipy.stats import skew,norm
from scipy.stats.stats import pearsonr
#density plot
sb.distplot(df.condition,fit=norm)
plt.ylabel('frequency')
plt.title('price distribution')
(mu,sigma) = norm.fit(df['sqft_lot15'])

fig = plt.figure()
res = stats.probplot(df['sqft_lot15'],plot=plt)
plt.show()

print('skewness %f' % df['sqft_lot15']).skew()
print('kurtosis %f' % df['sqft_lot15']).kurt()


# In[ ]:


#Normalization
df['sqft_lot15'] = np.log1p(df['sqft_lot15'])
#kernel density plot

sb.distplot(df.sqft_lot15,fit=norm)
plt.ylabel=('frequency')
plt.title=('price distribution')
#get the fitted parameter suited to the function
(mu,sigma)=norm.fit(df['sqft_lot15'], plot=plt)
fig = plt.figure()
stats = probplot(df['sqft_lot15'], plot=plt)
print("skewness: %f" % data['sqft_lot15'].skew())
print("kurtosis: %f" % data['sqft_lot15'].kurt())

#notebook error


# In[ ]:


cor = df.corr()
cols = cor.nlargest(21, 'price')['price'].index #specify number of columns to display i.e 21
f, ax = plt.subplots(figsize=(18, 10)) #size of matrix
cm = np.corrcoef(df[cols].values.T)
sb.set(font_scale=1.25)
hm = sb.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size':12}, yticklabels=cols.values,
                 xticklabels=cols.values)
plt.yticks(rotation=0, size=15)
plt.xticks(rotation=90, size=15)
#plt.title("Correlation Matrix",style='oblique', size= 20)
plt.savefig('dist.png')
#files.download('dist.png')

plt.show()


# In[ ]:


cor=df.corr()

# Generate a mask for the upper triangle

mask = np.zeros_like(cor, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Generate a custom diverging colormap
cmap = sb.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
f, ax = plt.subplots(figsize=(11, 9))

sb.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[ ]:


plt.bar(df['grade'],df['price'])
plt.xlabel('Grade')


# In[ ]:


df["bedrooms"].value_counts().plot(kind='bar')
plt.title('Count vs bedrooms Bar Graph')
plt.ylabel("Count")
plt.xlabel('Number of bedrooms')


# In[ ]:


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(20,20))
ax = Axes3D(fig)
ax.scatter(df['long'],df['lat'],df['price'])


# In[ ]:




