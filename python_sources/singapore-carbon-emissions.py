#!/usr/bin/env python
# coding: utf-8

# This kernal explores the data from the Singapore Biennial Update Report released in 2017.  The data only contains carbon dioxide (CO2) emissions from energy consumption.  This dataset does not include other greenhouse gas inventories such as methane, fugitive emissions nor does it include carbon capture.  The data was manually transcribed from the PDF report.  
# 
# There is some minor rounding error difference between the final GtCO2eq figures in this database and the report believed to be due to the actual value in the molecular weight conversion factor applied (44/12).
# 
# Data source
# Table 1A1-4 CO2 emissions
# Singapore National Climate Secretariat - [3rd Biennial Update Report to the UN - 2017]
# [https://www.nccs.gov.sg/docs/default-source/default-document-library/singapore's-fourth-national-communication-and-third-biennial-update-repo.pdf](http://)

# In[ ]:


# Using Python 3 environment 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sqlite3
import seaborn as sb


# In[ ]:


SQL_DB = '../input/carbon.db'
TABLES = {'energy':'energy_consumption','em_factors':'emission_factors'}
con = sqlite3.connect(SQL_DB)


# In[ ]:


em_factors = pd.read_sql('select * from ' + TABLES['em_factors'],con=con)
energy = pd.read_sql('select * from ' + TABLES['energy'],con=con)


# In[ ]:


em_factors


# In[ ]:


energy.head()


# In[ ]:


ghg = pd.merge(energy[[x for x in energy.columns if not x=='Fuel class']]
               ,em_factors,on='Fuel sub-class')


# Add a new calculated field of Gigatons of carbon dioxide using the conversion factors

# In[ ]:


ghg['GtCO2eq'] = ghg.apply(lambda x: 
        x['Qty consumed']*x['TJ/unit']*x['tc / TJ']*x['CO2eq'],axis=1)


# In[ ]:


inv = pd.pivot_table(ghg,index=['Souce group','Source sub group'],columns='Fuel sub-class',values='GtCO2eq',aggfunc='sum')


# The complete table showing source of energy consumption by sector and type of fuel

# In[ ]:


inv


# In[ ]:


subtotals = inv.transpose().sum().reset_index()
subtotals.rename(columns={0:'GtCO2eq'},inplace=True)
subtotals


# In[ ]:


g = sb.barplot(x='Souce group',y='GtCO2eq',data=subtotals)
g.set_xticklabels(g.get_xticklabels(), rotation=90)

