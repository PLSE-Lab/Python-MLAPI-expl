#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #for plotting
import seaborn as sns #for plotting
get_ipython().run_line_magic('matplotlib', 'inline')


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df = pd.read_csv('../input/forest/train.csv')


# overview of the DATA

# In[ ]:


df.head(3)


# In[ ]:


df.tail(3)


# In[ ]:


df.shape


# In[ ]:


df.describe() # mean median sd all the basic statistic is obseved.


# In[ ]:


df.info()


# In[ ]:


df.columns


# In[ ]:


df.dtypes #datatype of each column is observed and it is seen that all are int type


# here we see that all the column is related to cover type column so we have to do our all analysis with respect to cover type

# In[ ]:


df['Cover_Type'].value_counts() # value counts frelated to cover_type


# In[ ]:


df["Cover_Type"].value_counts().plot(kind='bar',color='purple') #plotting the value counts
plt.ylabel("occurences")
plt.xlabel("Cover Type")


# In[ ]:


plt.bar(df['Cover_Type'], df['Elevation']) #plottinfg with cover type and elevation in bar plot and scatter plot.


# In[ ]:


plt.scatter(df['Cover_Type'], df['Elevation'])


# In[ ]:


#box plot for cover type and elevation
x=df.groupby('Cover_Type')
x.boxplot(column='Elevation')


# In[ ]:


x=df.groupby('Cover_Type')
x.boxplot(column='Aspect') #with Aspect


# In[ ]:


x=df.groupby('Cover_Type')
x.boxplot(column='Slope') #with slope


# In[ ]:


x=df.groupby('Cover_Type')
x.boxplot(column='Horizontal_Distance_To_Hydrology')


# In[ ]:


x=df.groupby('Cover_Type')
x.boxplot(column='Horizontal_Distance_To_Roadways')


# in the above box plot for all 7 types of cover type we can observed the mean, median, quartile w.r.t 1st 5 columns

# In[ ]:


# Extract column from the dataset to do specific plotting
cl = df.columns.tolist()


# Now with continous variable we are going to do the plotting. means we are going plot normal univariate distribution for the 1st 10 columns means excluiding Soil, wild, id and cover type columns in cl:

# In[ ]:


for name in cl:
    if name[0:4] != 'Soil' and name[0:4] != 'Wild' and name != 'Id' and name != 'Cover_Type':
        plt.figure()
        sns.distplot(df[name])


# we have seen plotting are skewed, So normalization is done w.r.t cover type and thn plot it

# In[ ]:


for name in cl:
    if name[0:4] != 'Soil' and name[0:4] != 'Wild' and name != 'Id' and name != 'Cover_Type':
        title = name + ' vs Cover Type'
        plt.figure()
        sns.stripplot(df["Cover_Type"],df[name],jitter=True)
        plt.title(title);


# cover type 7 is basically found in higher elevation and covr type 3 is found from lowest elevation to a medium elevation.
# aspet is same accross all cover type
# slope is also same accross all cover type the average is around 35.
# horizental distance average for cover type is around 600.
# vertical distance average is 250 to 300
# 

# for find the correlation, heatmap is a good plotting So excluing Soil_type , plotting the heatmap

# In[ ]:


y = [x for x in df.columns.tolist() if "Soil_Type" not in x]
y = [x for x in y if "Wilderness" not in x]
dfnew = df.reindex(columns=y)


# In[ ]:


dfnew.head(4)


# In[ ]:


cor1=dfnew.corr()
f, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(cor1, vmax=.7, square=True);


# from the diagonal part we can observed that ttehy are highly correlated. darker the color manes highly correaled and lighter the color less correlated

# pairs of variables give the idea between the cover types. So pair plotting will be done on on the dfnew dataset

# In[ ]:


y1 = dfnew.columns.tolist()
remove_cl = ['Id', 'Slope', 'Aspect', 'Cover_Type']
y1 = [x for x in y1 if y1 not in remove_cl]
y1


# In[ ]:


plt1=sns.pairplot(df, vars=y1, hue="Cover_Type") #A pairplot plot a pairwise relationships in a dataset.


# work with Wilderness variable

# In[ ]:


cwild = [x for x in df.columns.tolist() if "Wilderness" in x]
t = df[cwild].groupby(df['Cover_Type']).sum()
m = t.T.plot(kind='bar', figsize=(10, 10), legend=True, fontsize=15) #here we cant put scatter plot as we now it need x and y variable
m.set_xlabel("Wilderness_Type", fontsize=15)
m.set_ylabel("Count", fontsize=15)
plt.show()


# in  the above plot we can obsere that the cover type 2 is is in every area.

# count of each soil type

# In[ ]:


s = np.array(cl)
st = [item for item in s if "Soil" in item]
for soil_type in st:
    print (soil_type, df[soil_type].sum())


# In[ ]:


z = df[st].groupby(df['Cover_Type']).sum() #plotting the soli type w.r.t cover type
z.T.plot(kind='barh', stacked=True, figsize=(15,10))

here cover type 5 and 7 can be found in every soil type
# In[ ]:




