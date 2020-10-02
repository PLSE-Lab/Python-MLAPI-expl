#!/usr/bin/env python
# coding: utf-8

# ### Wrangling the D.C. Real Property Dataset
# This notebook describes the steps required to inspect, clean, and merge several sources of information on residential properties in Washington, D.C.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#flag missing data
na_sentinels = {'SALEDATE':['1900-01-01T00:00:00.000Z'],'PRICE':[0,''],'AYB':[0],'EYB':[0]}

#three data sources
residential = pd.read_csv("../input/raw_residential_data.csv", index_col='SSL', na_values=na_sentinels).drop(columns=["OBJECTID"])
condo = pd.read_csv("../input/raw_condominium_data.csv", index_col='SSL', na_values=na_sentinels).drop(columns=["OBJECTID"])
address = pd.read_csv("../input/raw_address_points.csv")


# First, we will merge the [residential ](http://http://opendata.dc.gov/datasets/computer-assisted-mass-appraisal-condominium?page=2)and [condominium ](http://http://opendata.dc.gov/datasets/computer-assisted-mass-appraisal-condominium) datasets provided by the [DC Geographic Informtion Service](http://https://octo.dc.gov/service/dc-gis-services).
# 
# The **SSL** - the square, suffix, and lot - will serve as an index for the properties:

# In[ ]:


residential["SOURCE"] = "Residential"
condo["SOURCE"] = "Condominium"

df = pd.concat([residential,condo], sort=False)
df.info()


# Next, eliminate redundant dummy codes for categorical values.

# In[ ]:


#Identify all categorical variables
categories = [['CNDTN_D','CNDTN'],['HEAT_D','HEAT'],['STYLE_D','STYLE'],['STRUCT_D','STRUCT'],['GRADE_D','GRADE'],['ROOF_D','ROOF'],['EXTWALL_D','EXTWALL'],['INTWALL_D','INTWALL']]
cat_drop = []
for c in categories:
    df[c[1]] = df[c[0]].astype('category')
    cat_drop.append(c[0])

df['SOURCE'] = df['SOURCE'].astype('category')    
#eliminate redundant dummy variables
df.drop(cat_drop, inplace=True, axis=1)


# Now let's see if there are any missing values in our dataset.

# In[ ]:


print(df.isnull().sum())


# It seems like there a few dozen entries that are missing key descriptors. Let's remove these rows.

# In[ ]:


df.dropna(subset=['ROOMS','BEDRM','BATHRM','HF_BATHRM','FIREPLACES','EYB','QUALIFIED'], inplace=True)

print(df.isnull().sum())


# ...and change data types to integer where appropriate.

# In[ ]:


int_col = ['BATHRM','HF_BATHRM','ROOMS','BEDRM','EYB','SALE_NUM','BLDG_NUM','FIREPLACES','LANDAREA']
#con_col = ['BATHRM','HF_BATHRM','NUM_UNITS','ROOMS','BEDRM','EYB','STORIES','SALE_NUM','KITCHENS','FIREPLACES','LANDAREA']

for i in int_col:
    df[i] = df[i].astype('int64')


# There are two timestamp fields. Any suspicious outlier values?

# In[ ]:


print(df["SALEDATE"].sort_values(ascending=True).head(5))
print(df["SALEDATE"].sort_values(ascending=False).head(5))


# Do not allow sale dates outside the 20th or 21st century:

# In[ ]:


import re

def valid_datestring(x):
    if re.match(r'(19|20)\d{2}\-',str(x)):
        return x
    else:
        return None

df["SALEDATE"] = df['SALEDATE'].map(valid_datestring)
df['GIS_LAST_MOD_DTTM'] =  df['GIS_LAST_MOD_DTTM'].map(valid_datestring)


# ...and change data type to *datetime* where appropriate:

# In[ ]:


df['SALEDATE'] = pd.to_datetime(df['SALEDATE'], dayfirst=False)
df['GIS_LAST_MOD_DTTM'] = pd.to_datetime(df['GIS_LAST_MOD_DTTM'], dayfirst=False)


# A correlogram o
# 

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('seaborn-colorblind')
# Basic correlogram
sns.pairplot(df[['ROOMS','BATHRM','HF_BATHRM','BEDRM']], kind="scatter", diag_kind = 'kde', plot_kws = {'alpha': 0.33, 's': 80, 'edgecolor': 'k'}, size = 4)
plt.show()


# This reveals several impossible properties:
# * The ROOMS X BTHRM plot includes at least one place with 101 rooms and only 3 bathrooms. 
# * The BEDRM X ROOMS plot includes properties with more bedrooms than rooms.
# * The BEDRM X BATHRM plot includes one point representing 24 bedrooms and 24 bathrooms. As it turns out, this represents a[ pile of shipping containers](http://https://dc.curbed.com/2014/9/30/10041494/dc-gets-first-apartments-made-of-shipping-containers) that would be more accureately described as four separate 6 Bed/6 Bath units.
# 
# We will eliminate all these suspicious rows:

# In[ ]:


df = df[( (df["ROOMS"]<100) & (df["ROOMS"]>=df["BEDRM"]) & (df["BATHRM"]<24) )]


# Next, we will explore the [Address Points database](http://http://opendata.dc.gov/datasets/address-residential-units ). This source will complement the property information with spatial information and links to census data.
# 

# In[ ]:


#df.head()
address.head(5)


# A number of columns reflect the jurisdictions of municipal agencies. For the main dataset, we will only include a subset of rows of interest.
# 
# Also note we will drop rows duplicate SSL rows. More than one address/row may be associated with a single  SSL (square,suffix,lot), such as a corner address with separate addresses facing each adjacent street ([source](http://https://octo.dc.gov/sites/default/files/dc/sites/octo/publication/attachments/DCGIS-MarFAQ_0.pdf)). The descriptive information we are interested in - other than mailing address - will typically be identical for both rows, so we only need to keep one.

# In[ ]:


address_subset = address.drop_duplicates(['SSL'], keep='last').set_index("SSL")[["FULLADDRESS","CITY","STATE","ZIPCODE","NATIONALGRID","LATITUDE","LONGITUDE","ASSESSMENT_NBHD","ASSESSMENT_SUBNBHD","CENSUS_TRACT","CENSUS_BLOCK","WARD"]]


# Merge the Address Points columns with the combined property database, using the SSL index:

# In[ ]:


premaster = pd.merge(df,address_subset,how="left",on="SSL")


# Many entries, including most condiminiums, do not have a specific SSL match in the address points database. In these cases, we will impute with data from nearnby properties (in the same by *square*).
# 
# First, we will build a lookup DataFrame that includes summarized data for properties in each square:

# In[ ]:


address["SQUARE"] = address["SQUARE"].apply(lambda x: str(x)[0:4])

address_impute = address[((address["SQUARE"]!="0000") & (address["SQUARE"].str.match(r'\d+')) )]     .groupby("SQUARE")     .agg({'X':'median','Y':'median','QUADRANT':'first','ASSESSMENT_NBHD':'first','ASSESSMENT_SUBNBHD':'first','CENSUS_TRACT': 'median','WARD':'first','ZIPCODE':'median','LATITUDE':'median','LONGITUDE':'median'})  

print(address_impute.head())


# Next, we will impute the fields from the address dataset for any properties lacking this information

# In[ ]:


#create a SQUARE key on premaster
premaster["SQUARE"] = df.apply(axis=1, func=lambda x: str(x.name)[0:4]) 
master = pd.merge(premaster,address_impute,how="left",on="SQUARE", suffixes=('', '_impute')) 
cols_to_impute = ["CENSUS_TRACT","LATITUDE","LONGITUDE","ZIPCODE","WARD","ASSESSMENT_NBHD","ASSESSMENT_SUBNBHD"]
for c in cols_to_impute:
    master[c] = master[c].fillna(master[(c + "_impute")])

master.drop(["CENSUS_TRACT_impute","LATITUDE_impute","LONGITUDE_impute","ZIPCODE_impute","WARD_impute","ASSESSMENT_NBHD_impute","ASSESSMENT_SUBNBHD_impute"],axis=1,inplace=True)


# Inspect the new master dataset

# In[ ]:


master.describe()


# Finally, save the master dataframe to *DC_Properties.csv*

# In[ ]:


master.to_csv("DC_Properties.csv", header=True)

