#!/usr/bin/env python
# coding: utf-8

# # Problem Statement
# 
# EDA on car_ daat

# In[ ]:


#Import all the necessary modules
#Import all the necessary modules
import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
import warnings
warnings.filterwarnings('ignore')

import gc
gc.enable()


# # Load the Cars Data file into Python DataFrame
# 
# Get the data from here - https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/

# In[ ]:


df=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data",header=None, delimiter=r"\s+")


# In[ ]:


df.head()


# In[ ]:


cdf=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.names", header=None,sep=":",skiprows=32,nrows=9)
cdf.rename(columns={0:'Row',1:"datatype"},inplace=True)


# In[ ]:


cdf = pd.DataFrame(cdf["Row"].str.split(' ',10).tolist())
cdf.rename(columns={5:'Row'},inplace=True)


# In[ ]:


def rename(df1,df2):
    t=df1.columns.values
    s=df2['Row'].tolist()
    for x in range(df.shape[1]):
        if x!=9:
            df1.rename(columns={t[x]:s[x]},inplace=True)  
            #print("_____________")
            #print(t[x])
            #print(s[x])
    return df1

Car_df=rename(df,cdf)


# In[ ]:


Car_df.head()


# ##### From the overall look on the dataset , we can see some ? mark present in the data set. We will look at lt later.

# In[ ]:


Car_df.tail()


# In[ ]:


Car_df.sample(5)


# # IDA of the Data frame

# In[ ]:


Car_df.shape


# In[ ]:


def datatypes_insight(data):
    display(data.dtypes.to_frame())
    data.dtypes.value_counts().plot(kind="barh")


# #### "horsepower"- Looks like it set as Object. We may need to change the datatype.

# In[ ]:


#Car_df.apply(lambda x: len(x.unique()))
datatypes_insight(Car_df)


# In[ ]:


def datatypes_insight(data):
    display(data.apply(lambda x: len(x.unique())).to_frame())
    data.apply(lambda x: len(x.unique())).plot(kind="barh")


# In[ ]:


datatypes_insight(Car_df)


# In[ ]:


Car_df.describe().T


# Standard divisoin is very high for weight and displacement.

# # Missing value check, incorrect data and data imputation with mean, median, mode as necessary.

# In[ ]:


Car_df = Car_df.replace('?', np.nan)


# In[ ]:


Car_df.isnull().sum()


# In[ ]:


Car_df = Car_df.drop('car', axis=1)


# In[ ]:


Car_df["horsepower"] = Car_df.astype('float64')


# In[ ]:


pd.DataFrame({'count' : Car_df.groupby(["horsepower","origin"] ).size()}).head().reset_index()


# In[ ]:


Car_df["horsepower"] = Car_df.groupby(["origin"])["horsepower"]    .transform(lambda x: x.fillna(x.median()))


# In[ ]:


Car_df.isnull().sum()


# In[ ]:


Car_df.head()


# # EDA

# #### Uni-variate Analysis

# In[ ]:


def distploting(df):
    col_value=df.columns.values.tolist()
    sns.set(context='notebook',style='whitegrid', palette='dark',font='sans-serif',font_scale=1.2,color_codes=True)
    
    fig, axes = plt.subplots(nrows=2, ncols=4,constrained_layout=True)
    count=0
    for i in range (2):
        for j in range (4):
            s=col_value[count+j]
            #axes[i][j].hist(df[s].values,color='c')
            sns.distplot(df[s].values,ax=axes[i][j],bins=30,color="b")
            axes[i][j].set_title(s,fontsize=17)
            fig=plt.gcf()
            fig.set_size_inches(15,10)
            plt.tight_layout()
        count=count+j+1
        
             
distploting(Car_df)


# ### Observation:
# 
# 1. All the data distribution showed mixed gaussian distribution. That indicate, its mixed data from different types of cars.
# 2. Acceleration - Looks like fairly normal distribution.Slight mixed distribution present.
# 3. Car are having more observation from region 1. That indicates, the sample data may be skweed.
# 4. Car having 4 cylinder has more observation. That also indicate that the data is biased.

# In[ ]:


def boxplot(df):
    col_value=['mpg',
 'cylinders',
 'displacement',
 'horsepower',
 'weight',
 'acceleration',
 'model',"origin"]
    sns.set(context='notebook', palette='pastel',font='sans-serif',font_scale=1.5,color_codes=True,style='whitegrid')
    fig, axes = plt.subplots(nrows=2, ncols=4,constrained_layout=True)
    count=0
    for i in range (2):
        for j in range (4):
            s=col_value[count+j]
            #axes[i][j].boxplot(df[s])
            sns.boxplot(df[s],ax=axes[i][j],orient="v")
            fig=plt.gcf()
            fig.set_size_inches(15,20)
            plt.tight_layout()
        count=count+j+1
        
             
boxplot(Car_df)


# ### Observation:
# 1. Acceleration- feature hase more outliers compared to any othe feature.
# 2. Mpg feature has few outlier.
# 
# we will check on the outlier and try to figure out the the explanation.

# In[ ]:


fig, ax = plt.subplots(nrows=1, ncols=2,squeeze=True)
fig.set_size_inches(20,8)
Age_frequency_colums= pd.crosstab(index=Car_df["origin"],columns="count")
Age_frequency_colums.plot(kind='bar',ax=ax[0],color="c",legend=False)
Age_frequency_colums.plot(kind='pie',ax=ax[1],subplots=True,legend=False,autopct='%.2f')
ax[0].set_title('Frequency Distribution of Dependent variable: origin')
ax[1].set_title('Pie chart representation of Dependent variable: origin')

#adding the text labels
rects = ax[0].patches
labels = Age_frequency_colums["count"].values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax[0].text(rect.get_x() + rect.get_width()/2, height +1,label, ha='center', va='bottom',fontsize=15)
plt.show()


# ### Observation:
# As we can see the dataset has more observation from Origin-1 -62.56 % of the observation are from the Origin 1.This data was taken/used at 1983.Assumssion may be:
# 1. At 1983, Region-1 might be leading car manufacturing company. 
# 2. Survay may be regional/not global. Hence produced bias dataset.
# 3. Exported/Imported car may be less during the timeframe.

# #### Bi-variate analysis

# In[ ]:


def diffplot(df):
    sns.set(context='notebook',style='whitegrid', palette='pastel',font='sans-serif',font_scale=1.5)
    column_names=df.describe().columns.values.tolist()
    number_of_column=len(column_names)

    fig,ax=plt.subplots(nrows=2,ncols=4,figsize=(15,15))

    counter=0
    for i in range(2):
        for j in range(4):
            sns.boxplot(x='origin', y=column_names[counter],data=df,ax=ax[i][j])
            plt.tight_layout()
            counter+=1
            if counter==(number_of_column-1,):
                break
                 
diffplot(Car_df)


# ### Observation:
# 1. Look at the MPG, its high for car produced/originated at number 3 region compared to others.
# 2. Cylinder has more varity at #1 region/originated cars. Whereas other 2 region has some varity but in this data its showing as outliers. It could be wrong entry or highend car which was produced at origine 1 and 2.
# 3. Engine displacement is high for origine 1 cars. There are no much difference in between other 2 region.
# 4. Origine 1 cars are heavy compare to othe 2 region. That expain, Low millage and low horsepower.
# 

# In[ ]:


def diffplot(df):
    sns.set(context='notebook',style='whitegrid', palette='pastel',font='sans-serif',font_scale=1.5)
    column_names=df.describe().columns.values.tolist()
    number_of_column=len(column_names)

    fig,ax=plt.subplots(nrows=2,ncols=4,figsize=(15,15))

    counter=0
    for i in range(2):
        for j in range(4):
            sns.violinplot(x='origin', y=column_names[counter],data=df,ax=ax[i][j])
            plt.tight_layout()
            counter+=1
            if counter==(number_of_column-1,):
                break
                 
diffplot(Car_df)


# ### Jointplot for Bivariate Analysis

# In[ ]:


sns.set(context='notebook',style='whitegrid', palette='pastel',font='sans-serif',font_scale=1.5)
sns.jointplot(x=Car_df.mpg,y=Car_df["horsepower"], kind="scatter", color="c")


# #### Horse power and mpg are highly co-related. As show pearson corelation value as 1. But we need to check further more that if by only horsepower feature can we predict the mgp more accurately. 

# In[ ]:


sns.set(context='notebook',style='whitegrid', palette='pastel',font='sans-serif',font_scale=1.5)
sns.jointplot(x=Car_df.mpg,y=Car_df["cylinders"], kind="scatter", color="g")


# In[ ]:


sns.set(context='notebook',style='whitegrid', palette='pastel',font='sans-serif',font_scale=1.5)
sns.jointplot(x=Car_df.mpg,y=Car_df["displacement"], kind="scatter", color="b")


# ### MPG and Displacement features are inversely exponential.

# In[ ]:


sns.set(context='notebook',style='whitegrid', palette='pastel',font='sans-serif',font_scale=1.5)
sns.jointplot(x=Car_df.mpg,y=Car_df["weight"], kind="scatter", color="r")


# #### MPG and Displacement features are inversely exponential

# In[ ]:


sns.set(context='notebook',style='whitegrid', palette='pastel',font='sans-serif',font_scale=1.5)
sns.jointplot(x=Car_df.mpg,y=Car_df.acceleration, kind="scatter", color="b")


# In[ ]:


sns.set(context='notebook',style='whitegrid', palette='pastel',font='sans-serif',font_scale=1.5)
sns.jointplot(x=Car_df.mpg,y=Car_df.model, kind="scatter", color="g")


# In[ ]:





# In[ ]:


sns.pairplot(Car_df,hue="origin")


# #### Observation:
# 1. As shown in jointplot, mpg and horsepower showed high corelation. But as we saw in distribution , data has mixed distribution.All the features ares overlapping to each other for all differant origin.

# In[ ]:


# Draw a heatmap with the numeric values in each cell
cor_mat= Car_df.corr()
mask = np.array(cor_mat)
mask[np.tril_indices_from(mask)] = False
fig=plt.gcf()
fig.set_size_inches(28,28)
sns.set(context='notebook',style='whitegrid', palette='pastel',font='sans-serif',font_scale=1.5)
sns.heatmap(data=cor_mat,mask=mask,square=True,annot=True,cbar=True,cmap="coolwarm",linewidths=1)


# #### Observation: 
# 1. As we can see , with MPG, most of the features has high corelation.
# 2. Weight and cylinders has high corelation,Explained as number of cylinders engine will increase the weight of the car.
# 3. Displacement and cylinders has high corelation,Explained as number of cylinders engine will increase the weight of the car.
