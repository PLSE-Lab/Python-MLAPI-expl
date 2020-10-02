#!/usr/bin/env python
# coding: utf-8

# In this analysis, we will try to understand the impact of oil prices on the Algerian economy and compare it to other Maghreb countries.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import util_eco


# ### Economic indicators
# Data about the Algerian economy is unfortunately scarce and unstructured. The only usable data at the moment is from the World Bank website. We picked the following indicators to study the impact of oil prices on the economy: 

# In[ ]:


indicators = ["GDP growth (annual %)", 
            "Broad money (% of GDP)",
            "Inflation, consumer prices (annual %)",
            "Unemployment, total (% of total labor force) (national estimate)",
            "CO2 emissions (metric tons per capita)", 
            "Trade (% of GDP)",
            "Military expenditure (% of GDP)"
            ]


# ### Utilitary functions to plot and clean the data

# In[ ]:


def plot_sns(df, title):
    corr = df.corr()

    mask = np.triu(np.ones_like(corr, dtype=np.bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    ax = sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    ax.set_title(title)
    
    
def prepare_econ_df(df, indicators, country):
    df = df.drop(columns=['Country Code','Indicator Code']).iloc[:,:-1]
    df = df.loc[df['Indicator Name'].isin(indicators), :].T

    df.columns = df.iloc[1,:]

    df = df.rename(columns={"GDP growth (annual %)": "GDP growth",
                      "Broad money (% of GDP)": "Money supply", 
                      "Inflation, consumer prices (annual %)" : "Inflation", 
                      "Unemployment, total (% of total labor force) (national estimate)" : "Unemployment",
                      "CO2 emissions (metric tons per capita)" : "CO2 emissions", 
                      "Trade (% of GDP)" : "Trade", 
                      "Military expenditure (% of GDP)" : "Military spending"
                      })

    df = df.iloc[2:,:]    
    df = df.dropna(thresh=20, axis=1)
    df["Date"] = pd.to_datetime(df.index)
    df["Country Name"] = country
    
    df = df.infer_objects()
    return df


# ### Read & format the data 

# In[ ]:


df = pd.read_csv("/kaggle/input/econdata/API_DZA_DS2_en_csv_v2_888940.csv", sep = ',')
c_df = pd.read_csv("/kaggle/input/econdata/API_MAR_DS2_en_csv_v2_936889.csv",  sep = ',' )
t_df = pd.read_csv("/kaggle/input/econdata/API_TUN_DS2_en_csv_v2_942194.csv",  sep = ',' )

df = prepare_econ_df(df, indicators, 'Algeria')
c_df = prepare_econ_df(c_df, indicators, 'Morocco')
t_df = prepare_econ_df(t_df, indicators, 'Tunisia')
df = pd.concat([df, c_df, t_df])

df.head()


# ### Read oil prices data

# In[ ]:


odf = pd.read_csv("/kaggle/input/econdata/brent-annual_csv.csv")
odf['Date'] = pd.to_datetime(odf['Date'], format = "%Y")
odf = odf.rename(columns={"Price": "Annual oil price"})


# ### Merge the datasets

# In[ ]:


df = pd.merge(df, odf, how='left', on='Date')

min_oil_date = df.loc[pd.notnull(df['Annual oil price']), 'Date'].min()

df = df.loc[df['Date'] >= min_oil_date, :]
df = df.dropna(axis=1, thresh=20)
df.head()


# In[ ]:


plot_sns(df, "Correlation matrix of economic indicators - Maghreb")


# In[ ]:


plot_sns(df.loc[df['Country Name'] == 'Algeria', :], "Correlation matrix of economic indicators - Algeria")


# We notice a strong negative correlation between unemployment and average annual oil prices in Algeria. Let's compare it to the other two countries.

# In[ ]:


sns.set()
ax = sns.lmplot(x="Annual oil price", y="Unemployment", hue="Country Name", data=df, height=7)


# #### Relationship between variation in oil price and unemployment in the Maghreb
# (Complete)

# In[ ]:




