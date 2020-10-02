#!/usr/bin/env python
# coding: utf-8

# ****Objective****
# 
# The main objective of this study is to examine the trends in India's exports and imports in terms of value and to examine the structural changes in composition of India's exports and imports.

# ****1. Importing Libraries and Loading Data****
# 
# We will import libraries for data processing and preparing charts

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O 

# charts
import seaborn as sns 
sns.set_style("whitegrid")
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

get_ipython().run_line_magic('matplotlib', 'inline')

#ignore warning 
import warnings
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ****Data Loading****
# 
# We need to load two files one for import and other for export . The files will contain import and export data from 2010 to 2018.

# In[ ]:


df_import = pd.read_csv("/kaggle/input/india-trade-data/2018-2010_import.csv")
df_export = pd.read_csv("/kaggle/input/india-trade-data/2018-2010_export.csv")


# ****2. Exploring the Dataset****

# In[ ]:


df_export.head()


# In[ ]:


df_import.head()


# ****Findings:****
# 
# In both the input files we have 5 columns each.
# 
# i) HSCode - HS stands for Harmonized System. It is an identification code which follows international product nomenclature and describes the type of good that is shipped. All the commodities are grouped under 99 chapters or commodity groups or HSCode as they are commonly called.
# 
# ii) Commodity - The column contains commodity category. In each commodity category there are various commodities. A commodity is an economic goods or service.
# 
# iii) Value - Amount for export and import of commodities in million USD.
# 
# iv) Country - Country from where the goods are imported from or exported to.
# 
# v) Year - Year in which comodities where imported or exported (lies between 2010 to 2018).

# In[ ]:


df_export.describe()


# **From the describe function we find that for exports:**
# 
# i) HSCode has data falling between 1 to 99.
# 
# ii) Value has data falling between 0 and 19805. We assume that some items exported are very expensive.It shows a huge outlier as 75% data has value below 3.77 and maximum is 19805. Also we assume minimum is zero as some values might be too small to be rounded off in decimals.
# 
# iii) Year has data falling between 2010 and 2018, which we already know.

# In[ ]:


df_import.describe()


# **From the describe function we find that for imports:**
# 
# i) HSCode has data falling between 1 to 99.
# 
# ii) Value has data falling between 0 and 32781. We assume that some items imported are very expensive.It shows a huge outlier as 75% data has value below 4.78 and maximum is 32781. Also we assume minimum is zero as some values might be too small to be rounded off in decimals.
# 
# iii) Year has data falling between 2010 and 2018, which we already know.

# In[ ]:


df_export.info()


# **We find that:**
# 
# i) Export file has 137023 rows of data.
# 
# ii)Value column contains null values.

# In[ ]:


df_import.info()


# **We find that:**
# 
# i) Import file has 76124 rows of data.
# 
# ii) Value column contains null values.

# In[ ]:


df_export.isnull().sum()


# In[ ]:


df_export[df_export.value==0].count()


# In[ ]:


country_list=list(df_export.country.unique())
country_list


# In[ ]:


print("Duplicate exports : "+str(df_export.duplicated().sum()))


# **Findings in Export File:**
# 
# i) Value column has Null values.
# 
# ii) Value column has zero value.
# 
# iii) Country column has 'unspecified' as value.
# 
# There can be various way to handle it but for now we will delete these rows.**

# In[ ]:


df_import.isnull().sum()


# In[ ]:


df_import[df_import.value==0].count()


# In[ ]:


country_list1=list(df_import.country.unique())
country_list1


# In[ ]:


print("Duplicate imports : "+str(df_import.duplicated().sum()))


# **Findings in the Import File:**
# 
# i) Value column has Null values.
# 
# ii) Value column has zero value.
# 
# iii) Country column has 'unspecified' as value.
# 
# iv) Duplicate rows for imports.
# 
# There can be various way to handle it but for now we are deleting the rows.

# **3. Cleaning the Dataset**
# 
# We will cleanup the data so that we can visualize the data better.

# In[ ]:


def cleanup(df_data):
    df_data['country']= df_data['country'].apply(lambda x : np.NaN if x == "UNSPECIFIED" else x)
    df_data.dropna(inplace=True)
    df_data = df_data[df_data.value!=0] 
    df_data.drop_duplicates(keep="first",inplace=True)
    df_data = df_data.reset_index(drop=True)
    return df_data


# In[ ]:


df_export = cleanup(df_export)
df_import = cleanup(df_import)


# In[ ]:


df_import.isnull().sum()


# In[ ]:


df_import[df_import.value==0].count()


# **4. Commodity based Data Analysis**
# 
# 
# We can now analyse the data since it has been cleaned up.

# **Number of Commodities Exported or Imported**

# In[ ]:


print("Count of Commodities Exported: "+ str(len(df_export['Commodity'].unique())))
print("Count of Commodities Imported: "+ str(len(df_import['Commodity'].unique())))


# While exploring data, we found there are 99 groups. But here we see that commodities exported or imported have a count of 98 only. One group with HSCode 77 is missing because it is reserved for future use.

# **Expensive Commodities Exported**
# 
# While exploring the dataset, we found that some expensive commodities (coming as outliers) are exported. Lets find them and the total exports amount involved.

# In[ ]:


df_import_temp = df_import.copy(deep=True)
df_export_temp = df_export.copy(deep=True)
df_import_temp['commodity_sum'] = df_import_temp['value'].groupby(df_import_temp['Commodity']).transform('sum')
df_export_temp['commodity_sum'] = df_export_temp['value'].groupby(df_export_temp['Commodity']).transform('sum')
df_import_temp.drop(['value','country','year','HSCode'],axis=1,inplace=True)
df_export_temp.drop(['value','country','year','HSCode'],axis=1,inplace=True)

df_import_temp.sort_values(by='commodity_sum',inplace=True,ascending=False)
df_export_temp.sort_values(by='commodity_sum',inplace=True,ascending=False)

df_import_temp.drop_duplicates(inplace=True)
df_export_temp.drop_duplicates(inplace=True)


# In[ ]:


# Top 7 Goods exported as per their aggregate values
df_export_temp['Commodity'] = df_export_temp['Commodity'].apply(lambda x:x.split()[0])
px.bar(data_frame=df_export_temp.head(7),y='Commodity', x='commodity_sum', orientation='h',
       color='commodity_sum', title='Expensive Goods Exported from India Between 2010-2018 According to their Aggregate Value',
       labels={'commodity_sum':'Commoditiy Value in Million US $'})


# **Goods Exported Net Value**

# In[ ]:


pd.DataFrame(df_export.groupby(df_export['Commodity'])['value'].sum().sort_values(ascending=False).head(7))


# **Expensive Exports based on HSCode**

# In[ ]:


exp_exports = pd.DataFrame(data=df_export[df_export.value>700])
px.box(x="HSCode", y="value", data_frame=exp_exports, title='Expensive Exports HSCodewise', 
            color='HSCode', hover_name='value', height=700, width = 1400)


# **Expensive Commodities Imported**
# 
# While exploring the dataset, we found that some expensive commodities (coming as outliers) are imported. Lets find them and the total imports amount involved.

# In[ ]:


# Top 7 Goods imported as per their aggergate values
df_import_temp['Commodity'] = df_import_temp['Commodity'].apply(lambda x:x.split()[0])

px.bar(data_frame=df_import_temp.head(7),y='Commodity', x='commodity_sum', orientation='h',
       color='commodity_sum', title='Expensive Goods Imported to India Between 2010-2018 According to their Aggregate Value',
       labels={'commodity_sum':'Commoditiy Value in Million USD'})


# **Goods Imported Net Value** 

# In[ ]:


pd.DataFrame(df_import.groupby(df_import['Commodity'])['value'].sum().sort_values(ascending=False).head(7))


# **Expensive Imports based on HSCode**

# In[ ]:


exp_imports = pd.DataFrame(data=df_import[df_import.value>1000])
px.box(x="HSCode", y="value", data_frame=exp_imports, title='Expensive Imports HSCodewise', 
            color='HSCode',  height=700, width = 1400)


# **Findings:**
# 
# From above graphs, it is clear that the most expensive commodities exported or imported are: Mineral Fuels & Oils and Natural or Cultured Pearls, Stones etc. Their falling under outliers is justified from the graphs.

# **Commodities Exported Most Often or Popular Exported Commodities**

# In[ ]:


df = pd.DataFrame(df_export['Commodity'].value_counts())
df.head(10)


# **Trend followed by the Popular Commodities Exported(In Value) From 2010 to 2018**

# In[ ]:


exp_temp = df_export.copy()
exp_temp.drop(['HSCode', 'country'], axis=1, inplace=True)
exp_temp['Commodity'] = exp_temp['Commodity'].apply(lambda x:x.split(';')[0])
exp_temp.set_index('Commodity', inplace=True)
exp_temp


# In[ ]:


g= pd.DataFrame(exp_temp.loc[["ELECTRICAL MACHINERY AND EQUIPMENT AND PARTS THEREOF"]].groupby(['year', 'Commodity'])['value'].sum()).reset_index()
g1= pd.DataFrame(exp_temp.loc[["NUCLEAR REACTORS, BOILERS, MACHINERY AND MECHANICAL APPLIANCES"]].groupby(['year', 'Commodity'])['value'].sum()).reset_index()
g2= pd.DataFrame(exp_temp.loc[["OPTICAL, PHOTOGRAPHIC CINEMATOGRAPHIC MEASURING, CHECKING PRECISION, MEDICAL OR SURGICAL INST. AND APPARATUS PARTS AND ACCESSORIES THEREOF"]].groupby(['year', 'Commodity'])['value'].sum()).reset_index()
g3= pd.DataFrame(exp_temp.loc[["PHARMACEUTICAL PRODUCTS"]].groupby(['year', 'Commodity'])['value'].sum()).reset_index()
g4= pd.DataFrame(exp_temp.loc[["ARTICLES OF APPAREL AND CLOTHING ACCESSORIES, NOT KNITTED OR CROCHETED."]].groupby(['year', 'Commodity'])['value'].sum()).reset_index()

# Initialize figure with subplots
fig = make_subplots(
    rows=5, cols=1, subplot_titles=("Trend for Electrical Machinery & Equipments and Parts",
                                    "Trend for Nuclear Reactors",
                                    "Trend for Medical or Surgical Apparatus & Equipments",
                                    "Trend for Pharmaceutical Products",
                                    "Trend for Apparel and Clothing Accessories"
                                   )
)

# Add traces
fig.add_trace(go.Scatter(x=g.year, y=g.value), row=1, col=1)
fig.add_trace(go.Scatter(x=g1.year, y=g1.value), row=2, col=1)
fig.add_trace(go.Scatter(x=g2.year, y=g2.value), row=3, col=1)
fig.add_trace(go.Scatter(x=g3.year, y=g3.value), row=4, col=1)
fig.add_trace(go.Scatter(x=g4.year, y=g4.value), row=5, col=1)


# Update xaxis properties
fig.update_xaxes(title_text="Year", row=1, col=1)
fig.update_xaxes(title_text="Year", row=2, col=1)
fig.update_xaxes(title_text="Year", row=3, col=1)
fig.update_xaxes(title_text="Year", row=4, col=1)
fig.update_xaxes(title_text="Year", row=5, col=1)

# Update yaxis properties
fig.update_yaxes(title_text="Million US $", row=1, col=1)
fig.update_yaxes(title_text="Million US $", row=2, col=1)
fig.update_yaxes(title_text="Million US $", row=3, col=1)
fig.update_yaxes(title_text="Million US $", row=4, col=1)
fig.update_yaxes(title_text="Million US $", row=5, col=1)

# Update title and height
fig.update_layout(title_text="Trade Trends for Some Popular Commodities Exported", showlegend=False, height = 1500 )
fig.show()


# **Findings**
# 
# i) The top 5 most often exported goods have seen an increase in the value of exports between 2010 and 2018; except for the Apparel and Clothing Accessories which see a dip after 2016.
# 
# ii) Electrical Machinery & Equipments and Parts saw a dip in the exports between 2011 and 2016; but shows increasing trade offlate.
# 
# iii) Exports for Nuclear Reactors and Part thereof, Medical or Surgical Apparatus and Equipments and Pharmaceutical Products show increasing trade consistently.

# **Commodities Imported Most Often or Popular Imported Commodities**

# In[ ]:


df1 = pd.DataFrame(df_import['Commodity'].value_counts())
df1.head(10)


# **Trend followed by the Popular Commodities Imported(In Values) From 2010 to 2018**

# In[ ]:


imp_temp = df_import.copy()
imp_temp.drop(['HSCode', 'country'], axis=1, inplace=True)
imp_temp['Commodity'] = imp_temp['Commodity'].apply(lambda x:x.split(';')[0])
imp_temp.set_index('Commodity', inplace=True)
imp_temp


# In[ ]:


g= pd.DataFrame(imp_temp.loc[["ELECTRICAL MACHINERY AND EQUIPMENT AND PARTS THEREOF"]].groupby(['year', 'Commodity'])['value'].sum()).reset_index()
g1= pd.DataFrame(imp_temp.loc[["NUCLEAR REACTORS, BOILERS, MACHINERY AND MECHANICAL APPLIANCES"]].groupby(['year', 'Commodity'])['value'].sum()).reset_index()
g2= pd.DataFrame(imp_temp.loc[["MISCELLANEOUS GOODS."]].groupby(['year', 'Commodity'])['value'].sum()).reset_index()
g3= pd.DataFrame(imp_temp.loc[["PLASTIC AND ARTICLES THEREOF."]].groupby(['year', 'Commodity'])['value'].sum()).reset_index()
g4= pd.DataFrame(imp_temp.loc[["IRON AND STEEL"]].groupby(['year', 'Commodity'])['value'].sum()).reset_index()

# Initialize figure with subplots
fig = make_subplots(
    rows=5, cols=1, subplot_titles=("Trend for Electrical Machinery & Equipments and Parts",
                                    "Trend for Nuclear Reactors",
                                    "Trend for Miscellaneous Goods",
                                    "Trend for Plastic and Articles",
                                    "Trend for Iron and Steel"
                                   )
)

# Add traces
fig.add_trace(go.Scatter(x=g.year, y=g.value), row=1, col=1)
fig.add_trace(go.Scatter(x=g1.year, y=g1.value), row=2, col=1)
fig.add_trace(go.Scatter(x=g2.year, y=g2.value), row=3, col=1)
fig.add_trace(go.Scatter(x=g3.year, y=g3.value), row=4, col=1)
fig.add_trace(go.Scatter(x=g4.year, y=g4.value), row=5, col=1)


# Update xaxis properties
fig.update_xaxes(title_text="Year", row=1, col=1)
fig.update_xaxes(title_text="Year", row=2, col=1)
fig.update_xaxes(title_text="Year", row=3, col=1)
fig.update_xaxes(title_text="Year", row=4, col=1)
fig.update_xaxes(title_text="Year", row=5, col=1)

# Update yaxis properties
fig.update_yaxes(title_text="Million US $", row=1, col=1)
fig.update_yaxes(title_text="Million US $", row=2, col=1)
fig.update_yaxes(title_text="Million US $", row=3, col=1)
fig.update_yaxes(title_text="Million US $", row=4, col=1)
fig.update_yaxes(title_text="Million US $", row=5, col=1)

# Update title and height
fig.update_layout(title_text="Trade Trends for Some Popular Commodities Imported", showlegend=False, height = 1500 )
fig.show()


# **Findings**
# 
# i) The top 5 most often imported goods have seen an increase in the value of imports between 2010 and 2018; except for the Miscellaneous Goods which see a dip after 2013.
# 
# ii) Imports for Nuclear Reactors and Parts saw a dip in the exports between 2011 and 2015; but shows increasing trade offlate.
# 
# iii) Imports for Iron and Steel products shows inconsistent trend between 2012 and 2016; but shows increasing imports after that.
# 
# iv) Imports for Electrical Machinery & Equipments and Plastic & Articles show increasing trade consistently. 

# **5. Country based Data Analysis:**

# In[ ]:


print("Number of Countries to whom we export comodities: " + str(df_export['country'].nunique()))
print("Number of Countries from whom we import comodities: " + str(df_import['country'].nunique()))


# **Top Countries to Export from India**

# In[ ]:


import warnings
warnings.filterwarnings("ignore")

exp_country = df_export.groupby('country').agg({'value':'sum'})
exp_country = exp_country.rename(columns={'value': 'Export'})
exp_country = exp_country.sort_values(by = 'Export', ascending = False)
exp_country = exp_country[:20]
exp_country_tmp = exp_country[:10]


# In[ ]:


px.bar(data_frame = exp_country_tmp, x=exp_country_tmp.index, y ='Export',
labels={'country':"Countries", 'Export': "Total Exports in Million US$" } , color='Export', width=1200)


# **Top Countries From Whom India Imports** 

# In[ ]:


imp_country = df_import.groupby('country').agg({'value':'sum'})
imp_country = imp_country.rename(columns={'value': 'Import'})
imp_country = imp_country.sort_values(by = 'Import', ascending = False)
imp_country = imp_country[:20]
imp_country_tmp = imp_country[:10]


# In[ ]:


px.bar(data_frame = imp_country_tmp, x=imp_country_tmp.index, y ='Import',
labels={'country':"Countries", 'Import': "Total Exports in Million US$" } , color='Import', width=1200 )


# **Findings:**
# 
# i) USA is biggest importer from India followed by UAE and China Republic.
# 
# ii) China has the biggest market of Goods in India followed by UAE, Saudi Arabia and USA.

# **6. Calculating Trade Deficit:**
# 
# Let us calculate the trade deficit between exports and imports.

# In[ ]:


total_trade = pd.concat([exp_country, imp_country], axis = 1)
total_trade['Trade Deficit'] = exp_country.Export - imp_country.Import
total_trade = total_trade.sort_values(by = 'Trade Deficit', ascending = False)
total_trade = total_trade[:11]

print('Countrywise Trade Export/Import and Trade Balance of India')
display(total_trade)


# In[ ]:


px.bar(data_frame = total_trade, x=total_trade.index, y=['Import', 'Export', 'Trade Deficit'], barmode='group', labels={'index':'Countries', 'value':'Million US $'})


# **Findings:**
# 
# i) India has a trade surplus with USA, U Arab Emts, Hong Kong and Singapore.
# 
# ii) India has a huge trade deficit with China, Suadi Arab and Indonesia etc.

# **7. Year wise Data Analysis**

# In[ ]:


Import =df_import.groupby(['year']).agg({'value':'sum'}).reset_index()
Export =df_export.groupby(['year']).agg({'value':'sum'}).reset_index()
Import['Deficit'] = Export.value - Import.value


# In[ ]:


fig = go.Figure()

# Create and style traces
fig.add_trace(go.Scatter(x=Import.year, y=Import.value, name='Import',mode='lines+markers',
                         line=dict(color='blue', width=4)))
fig.add_trace(go.Scatter(x=Export.year, y=Export.value, name = 'Export',mode='lines+markers',
                         line=dict(color='green', width=4)))
fig.add_trace(go.Scatter(x=Import.year, y=Import.Deficit, name='Deficit',mode='lines+markers',
                         line=dict(color='red', width=4)))

fig.update_layout(
    title=go.layout.Title(
        text="Indian Trade Over The Years 2010-2018",
        xref="paper",
        x=0
    ),
    xaxis=go.layout.XAxis(
        title=go.layout.xaxis.Title(
            text="Year",
            font=dict(
                family="Times New",
                size=18,
                color="#7f7f7f"
            )
        )
    ),
    yaxis=go.layout.YAxis(
        title=go.layout.yaxis.Title(
            text="Million US $",
            font=dict(
                family="Times New",
                size=18,
                color="#7f7f7f"
            )
        )
    )
)

fig.show()


# **Yearly Export or Import Trends for Some Chosen Countries:** 

# In[ ]:


exp_country = df_export.copy()
exp_country.drop(['HSCode', 'Commodity'], axis=1, inplace=True)
exp_country = pd.DataFrame(exp_country.groupby(['country', 'year'])['value'].sum())
exp_country.reset_index('year', inplace=True)


imp_country = df_import.copy()
imp_country.drop(['HSCode', 'Commodity'], axis=1, inplace=True)
imp_country = pd.DataFrame(imp_country.groupby(['country', 'year'])['value'].sum())
imp_country.reset_index('year', inplace=True)
imp_country


# In[ ]:


# Initialize figure with subplots
fig = make_subplots(
    rows=4, cols=1, subplot_titles=("Chinese Trade with India Over The Years 2010-2018",
                                    "Saudi Arab Trade with India Over The Years 2010-2018",
                                    "USA Trade with India Over The Years 2010-2018",
                                    "United Arab Emts Trade with India Over The Years 2010-2018"
                                   )
)


# Create traces
g1 = pd.DataFrame(imp_country.loc[["CHINA P RP"]]).groupby(['year'])['value'].sum().reset_index()
g2 = pd.DataFrame(exp_country.loc[["CHINA P RP"]]).groupby(['year'])['value'].sum().reset_index()
g3 = pd.DataFrame(imp_country.loc[["SAUDI ARAB"]]).groupby(['year'])['value'].sum().reset_index()
g4 = pd.DataFrame(exp_country.loc[["SAUDI ARAB"]]).groupby(['year'])['value'].sum().reset_index()
g5 = pd.DataFrame(imp_country.loc[["U S A"]]).groupby(['year'])['value'].sum().reset_index()
g6 = pd.DataFrame(exp_country.loc[["U S A"]]).groupby(['year'])['value'].sum().reset_index()
g7 = pd.DataFrame(imp_country.loc[["U ARAB EMTS"]]).groupby(['year'])['value'].sum().reset_index()
g8 = pd.DataFrame(exp_country.loc[["U ARAB EMTS"]]).groupby(['year'])['value'].sum().reset_index()


# Add traces
fig.add_trace(go.Scatter(x=g1.year, y=g1.value, name='Import to India',mode='lines+markers',
                         line=dict(color='red', width=4)), row=1, col=1)
fig.add_trace(go.Scatter(x=g2.year, y=g2.value, name = 'Export to China',mode='lines+markers',
                         line=dict(color='blue', width=4)), row=1, col=1)

fig.add_trace(go.Scatter(x=g3.year, y=g3.value, name='Import to India',mode='lines+markers',
                         line=dict(color='orange', width=4)), row=2, col=1)
fig.add_trace(go.Scatter(x=g4.year, y=g4.value, name = 'Export to Saudi Arab',mode='lines+markers',
                         line=dict(color='green', width=4)), row=2, col=1)

fig.add_trace(go.Scatter(x=g5.year, y=g5.value, name='Import to India',mode='lines+markers',
                         line=dict(color='gold', width=4)), row=3, col=1)
fig.add_trace(go.Scatter(x=g6.year, y=g6.value, name = 'Export to USA',mode='lines+markers',
                         line=dict(color='purple', width=4)), row=3, col=1)

fig.add_trace(go.Scatter(x=g7.year, y=g7.value, name='Import to India',mode='lines+markers',
                         line=dict(color='olive', width=4)), row=4, col=1)
fig.add_trace(go.Scatter(x=g8.year, y=g8.value, name = 'Export to U Arab Emts',mode='lines+markers',
                         line=dict(color='yellow', width=4)), row=4, col=1)



# Update xaxis properties
fig.update_xaxes(title_text="Year", row=1, col=1)
fig.update_xaxes(title_text="Year", row=2, col=1)
fig.update_xaxes(title_text="Year", row=3, col=1)
fig.update_xaxes(title_text="Year", row=4, col=1)


# Update yaxis properties
fig.update_yaxes(title_text="Million US $", row=1, col=1)
fig.update_yaxes(title_text="Million US $", row=2, col=1)
fig.update_yaxes(title_text="Million US $", row=3, col=1)
fig.update_yaxes(title_text="Million US $", row=4, col=1)


# Update title and height
fig.update_layout(title_text="Trade Trends for Some Popular Commodities Imported", height = 1500 )

fig.show()


# **Findings:**
# 
# i) The graph for China and Saudi Arab show a huge trade deficit over the years. India is at a loss over there.
# 
# ii) The graph for USA and U Arab Emts show a trade surplus over the years. USA is the biggest importer from India.

# **Conclusion:**
# 
# i) The change in trade policies and tendency to buy more imported Goods has caused a huge increase in the import bill, which is pushing India towards a deficit Country.
# 
# ii) New Initiatives taken by Goverment as "Skill India" , "Make In India", "Startup India" can help to reduce the import and increase the export.
# 
# iii) Need of the hour is to **Use Products That are Made In India**.

# **Thank You**

# In[ ]:




