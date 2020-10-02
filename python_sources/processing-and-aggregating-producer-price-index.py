#!/usr/bin/env python
# coding: utf-8

# # Bureau of Labor Statistics: Processing Producer Price Index
# _By Nick Brooks, Date: December 2017_
# 
# ## Content
# 1. [Reading Multiple Files](#p1)
# 1. Regular Expressions
# 1. [New Variables: Age (how long since present), Minimum (Starting Year)](#p3)
# 1. [Datetime Conversion](#p4)
# 1. [Merging datasets by ID to append important categorizing information](#p5)
# 1. [Normalize: Calculate value difference by Series_ID](#p6)
# 1. [Data Visualization](#p7)

# In[ ]:


# General
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import os 

# Viz
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Warnings OFF
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# File Names
ppi_df = pd.DataFrame(os.listdir('../input/ppi/'), columns=["name"])

# Determine file purpose (prices or supportive files)
ppi_df["type"] = ppi_df.name.apply(lambda x: x.split('.')[1])

# Extract Industry Name
ppi_df["industry"] = "none"
ppi_df["industry"] = ppi_df.loc[ppi_df.type == "data","name"].apply(lambda x: x.split('.')[3])


# ## Reading Multiple Files and Concatenate
# <a id="ch1"></a>

# In[ ]:


# Read Name and Industry Name
indus_file = ppi_df.loc[ppi_df.type == "data",["name","industry"]]

out = pd.DataFrame(index=None)
for index, row in indus_file.iterrows():
    df = pd.read_csv("../input/ppi/{}".format(row[0]),
                               usecols=["series_id","year","period","value"])
    df["industry"] = row[1]
    df.period = df.period.str.replace("M","").astype(int)
    out = pd.concat([out, df], axis=0)
out.shape


# ## New Variables: Age (how long since present), Minimum (Starting Year)
# <a id="ch3"></a>

# In[ ]:


# Find minimum by Year
out["minimum"] = 'none'
indus_min = []
# Loop by Industry
for x in out.industry.unique():
    # Minimum
    mini = out.loc[out.industry=="Wood", "year"].min()
    # List of Minimums
    indus_min.append([x,mini])
    # Variable
    out.minimum[out.industry==x] = mini
indus_min= pd.DataFrame(indus_min, columns=["Industry","MinimumYear"])

# Time from present
out["frompresent"]= (2017 - out.year)
out['age'] = (out.year - out.minimum)


# ## Datetime Conversion
# <a id="ch4"></a>

# In[ ]:


# Annual Aggregates
aggregate = out.loc[out.period>12,:]

# Remove Aggregates, saved under month 13, to convert to date time
out = out[out.period != 13]

# Convert to pandas datetime format
out.loc[:,'date'] = pd.to_datetime(out.apply(
    lambda x:'%s-%s-01' % (x['year'],x['period']),axis=1))


# In[ ]:


# View Supportive Files
helper_file = ppi_df.loc[(ppi_df.type != "data") &
    (~ppi_df['name'].isin(["pc.contacts","pc.txt"])),["name"]]

print(ppi_df.loc[ppi_df.type!= "data",["name","type"]])

for index, row in helper_file.iterrows():
    print("\n",row[0])
    df = pd.read_csv("../input/ppi/{}".format(row[0]))
    print(df.head())


# In[ ]:


# Load Helper Files to ID product types
series = pd.read_csv("../input/ppi/pc.series.csv",index_col=0)
industry = pd.read_csv("../input/ppi/pc.industry.csv").iloc[:,0:2]
industry.columns = ["industry_code","industry_name"]
product = pd.read_csv("../input/ppi/pc.product.csv").iloc[:,0:3]
product.columns = ["industry_code","product_code","product_name"]


# ## Merging datasets by ID to append important categorizing information
# <a id="ch5"></a>

# In[ ]:


# oh baby a triple
data = pd.merge(
        pd.merge(series.iloc[:,0:3],
             pd.merge(product, industry,
                      on='industry_code', how='outer'),
                 on=["industry_code","product_code"], how="outer"),
        out, on="series_id", how="right", suffixes=["sup", " "])

# Series_id has empty spaces at the end.
data.series_id = [x.replace(" ", "") for x in data.series_id ]


# In[ ]:


data.sample(15)


# In[ ]:


[print(y, x.shape) for (x,y) in [(industry, 'industry_df'), (product, 'product_df'),
                                 (industry, 'industry_df'),(out, 'main_dataset')]]
print("\nUnique Categories:")
print("Unique Product Category Count:", len(product.product_name.unique()))
print("Unique Industry Category Count:", len(industry.industry_name.unique()))
print("Data Unique Industry Count:", len(out.industry.unique()))
print("\nOuter-Merged Dataset:")
[print("{}:".format(x), len(data[x].unique()))for x in ["product_name","industry_name","industry"]]
print("shape:", data.shape)
print("\nMissing? ", data.isnull().values.any())

del out, series, industry, product


# In[ ]:


current = data[data.industry=="Current"]
print("'Current' Industry:")
print("Observation Count:", current.shape[0])
print("Unique Sub-Industries:", len(current.industry_name.unique()))
print("Unique Products:", len(current.product_name.unique()))


# In[ ]:


# Show the sub-industries and sub-products
industry_unique = pd.DataFrame()
for x in data.industry.unique():
    industry_unique = industry_unique.append({"Industry": x,
                                              "Sub-Industries": len(data[data.industry==x].industry_name.unique()),
                                              "Total Products": len(data[data.industry==x].product_name.unique())},
                                            ignore_index=True)
    #print(x, len(data[data.industry==x].industry_name.unique()), len(data[data.industry==x].product_name.unique()))
    
industry_unique.sort_values(by=["Total Products"], ascending=False, inplace=True)
industry_unique.reset_index(drop=True, inplace=True)
pd.concat([industry_unique[:35], industry_unique[35:].reset_index(drop=True)], axis=1)


# ##  Normalize: Calculate value difference by Series_ID 
# <a id="p6"></a>

# In[ ]:


# Calculate Difference by period and by Series_Id
data.set_index("date", inplace=True)
data.sort_index(inplace=True)
data['value_diffs'] = data.groupby('series_id')['value'].transform(pd.Series.diff)


# ## Exploratory Data Analysis
# <a id="p7"></a>

# In[ ]:


cols = ["OilAndGas", "GasolineStations", "PetroleumCoalProducts"]


# In[ ]:


# Histogram
#data.loc[(data["value_diffs"] > 1000) | (data["value_diffs"] < -1000), :]
data.loc[data.series_id== "PCU33441333441312", "value_diffs"].hist()


# In[ ]:


for x in cols:
    sns.kdeplot(data.value_diffs[data.industry==x], label=x, shade=True)
    #plt.xlabel('Name_length');
plt.show()


# In[ ]:


f, axarr = plt.subplots(len(cols), sharex=True, figsize=(15,15))
for i, cat in enumerate(cols):
    for x in set(data.product_name[data.industry == cat]):
        data.value_diffs[(data.index > '1974-01-01') & (data.product_name == x)].rolling(window = 40).mean()        .plot(alpha= 0.7, label = x, title=cat, ax=axarr[i], legend=False)

    #axarr[i].legend(fontsize='large', loc='center left', bbox_to_anchor=(1, 0.5))
    #axarr[i].ylabel("Price Index Value Difference")
plt.show()


# In[ ]:


# Exclude Micro-Conductors
#sns.kdeplot(data.loc[~(data.series_id == "PCU33441333441312"), ["value_diffs"]], shade=True);


# ## Output Processed Data

# In[ ]:


data.to_csv("Aggregated_BLS_Producer_Price.csv")

