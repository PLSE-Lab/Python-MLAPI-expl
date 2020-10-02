#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from pandas_highcharts.core import serialize
from pandas_highcharts.display import display_charts
from IPython.display import display
pd.options.display.max_columns = None
import numpy as np
fileEL = pd.ExcelFile('../input/elementary_data1.xlsx')
dataEL1 = pd.read_excel(fileEL, '2015-16_1')
dataEL2 = pd.read_excel(fileEL, '2015-16_2')
dataEL3 = pd.read_excel(fileEL, '2014-15_PY')
dataDIS = pd.read_csv(fileDIS , encoding='utf-8')


# # Elementary school data 2015-16

# In[21]:


dls = dataEL1.iloc[16:]
dls = dls.replace(np.NaN, '', regex=True).reset_index()
dls.columns = dls.iloc[0].astype(str) + ' '+'('+dls.iloc[1].astype(str) + ')'
dls = dls.drop([0,1]).drop(columns=['16 (17)',' (STATCD)',' (DISTCD)',' (DISTRICTS)',' (BLOCKS)',' (VILLAGES)',' (CLUSTERS)'])
dls = dls.rename(columns={' (YEAR)':'YEAR',' (STATNAME)':'State Name',' (DISTNAME)':'District name'})
dls


# # Elementary school data 2014-15

# In[22]:


dls3 = dataEL3.iloc[16:]
dls3 = dls3.replace(np.NaN, '', regex=True).reset_index()
dls3.columns = dls3.iloc[0].astype(str) + ' '+'('+dls3.iloc[2].astype(str) + ')'
dls3 = dls3.drop([0,1,2]).drop(columns=['16 (18)',' (DISTCD)'])
dls3 = dls3.rename(columns={' ()':'YEAR',' (State Name)':'State Name',' (DISTNAME)':'District name'})
dls3


#  # State wise Professionally Qualified Teachers: Government

# In[23]:


dfComELFinal = dls.replace('NA',0,regex=True)
SWPQdf = dfComELFinal.groupby(['State Name'])[['Regular Teachers with Professional Qualification : Male  (PGRMTCH)','Regular Teachers with Professional Qualification : Female  (PGRFTCH)','Total Regular Teachers: Male  (GRMTCH)','Total Regular Teachers: Female  (GRFTCH)','Contractual Teachers with Professional Qualification : Male  (PGCMTCH)','Contractual  Teachers with Professional Qualification : Female  (PGCFTCH)','Total Contractual  Teachers: Male  (PCMTCH)','Total Contractual  Teachers: Female  (PCFTCH)']].sum()
display_charts(SWPQdf , kind="bar", title="State wise Professionally Qualified Teachers: Government",figsize = (1000, 700))


# # State wise  male OBC Teachers by School Category 

# In[25]:


SWOTSCdf = dls.groupby(['State Name'])[['Primary Only (TCHOBCM1)','Primary with Upper Primary (TCHOBCM2)','Primary with upper Primary Sec/H.Sec (TCHOBCM3)','Upper Primary Only (TCHOBCM4)','Upper Primary with Sec./H.Sec (TCHOBCM5)','Primary with upper Primary Sec (TCHOBCM6)','Upper Primary with  Sec. (TCHOBCM7)']].sum()
display_charts(SWOTSCdf, kind="barh", title="State wise  male OBC Teachers by School Category ",figsize = (1000, 700))


# # State wise Professionally Qualified Teachers: Private

# In[26]:


SWPQTMdf = dls.groupby(['State Name'])[['Teachers with Professional Qualification : Female  (PPFTCH)','Teachers with Professional Qualification : Male  (PPMTCH)','Total  Teachers: Male  (PMTCH)','Total  Teachers: Female  (PFTCH)']].sum()
display_charts(SWPQTMdf, title="State wise Professionally Qualified Teachers: Private", kind="line",figsize = (1000, 700))


# # State wise Number of Classrooms by School Category

# In[27]:


SWNCSTdf = dls.groupby(['State Name'])[['Primary Only (CLS1)','Primary with Upper Primary (CLS2)','Primary with upper Primary Sec/H.Sec (CLS3)','Upper Primary Only (CLS4)','Upper Primary with Sec./H.Sec (CLS5)','Primary with upper Primary Sec (CLS6)','Upper Primary with  Sec. (CLS7)','Total (CLSTOT)']].sum()
display_charts(SWNCSTdf, title="State wise Number of Classrooms by School Category", kind="bar",figsize = (1000, 700))


# # State wise Elementary School Enrolment by School Category

# In[28]:


SWEESCdf = dls.groupby(['State Name'])[['Primary Only (ENR1)','Primary with Upper Primary (ENR2)','Primary with upper Primary Sec/H.Sec (ENR3)','Upper Primary Only (ENR4)','Upper Primary with Sec./H.Sec (ENR5)','Primary with upper Primary Sec (ENR6)','Upper Primary with  Sec. (ENR7)','No Response (ENR9)','Total (ENRTOT)']].sum()
display_charts(SWEESCdf, title="State wise Elementary School Enrolment by School Category", kind="bar",figsize = (1000, 700))


# # State wise no of schools in India

# In[29]:


SWNOSdf = dls.groupby(['State Name'])[['Total (SCHTOT)','Total (SCHTOTG)','Total (SCHTOTP)','Total (SCHTOTM)']].sum().rename(columns={'Total (SCHTOT)':'Total','Total (SCHTOTG)':'Government','Total (SCHTOTP)':'Private','Total (SCHTOTM)':'Madarsa & Unrecognised'})
display_charts(SWNOSdf, title="State wise no of schools in India", kind="area",figsize = (1000, 700))


# # Percentage of different types of schools in India

# In[30]:


PODTOSdf = dls[['Total (SCHTOTG)','Total (SCHTOTP)','Total (SCHTOTM)']].rename(columns={'Total (SCHTOTG)':'Government','Total (SCHTOTP)':'Private','Total (SCHTOTM)':'Madarsa & Unrecognised'}).sum()
PODTOSdf = pd.DataFrame(PODTOSdf)
display_charts(PODTOSdf, kind='pie', title='Schools In India', tooltip={'pointFormat': '{series.name}: <b>{point.percentage:.1f}%</b>'})


# # State wise government school by category in rural areas

# In[31]:


SWGSIRAdf = dls.groupby(['State Name'])[['Primary Only (SCH1GR)','Primary with Upper Primary (SCH2GR)','Primary with upper Primary Sec/H.Sec (SCH3GR)','Upper Primary Only (SCH4GR)','Upper Primary with Sec./H.Sec (SCH5GR)','Primary with upper Primary Sec (SCH6GR)','Upper Primary with  Sec. (SCH7GR)','No Response (SCH9GR)','Total (SCHTOTGR)']].sum()
display_charts(SWGSIRAdf, title="State wise government school by category in rural areas", kind="bar",figsize = (1000, 700))


# # State wise no of teachers by school category in India

# In[32]:


SWTBSCdf = dls3.groupby(['State Name'])[['Primary Only (TCH1)','Primary with Upper Primary (TCH2)','Primary with upper Primary Sec/H.Sec (TCH3)','Upper Primary Only (TCH4)','Upper Primary with Sec./H.Sec (TCH5)','Primary with upper Primary Sec (TCH6)','Upper Primary with  Sec. (TCH7)','No Response (TCH9)','Total (TCHTOT)']].sum()
display_charts(SWTBSCdf, title="State wise no of teachers by school category in India", kind="line",figsize = (1000, 700))


# # State wise no of single-classroom schools by school category

# In[33]:


SWSCSdf = dls3.groupby(['State Name'])[['Primary Only (SCLS1)','Primary with Upper Primary (SCLS2)','Primary with upper Primary Sec/H.Sec (SCLS3)','Upper Primary Only (SCLS4)','Upper Primary with Sec./H.Sec (SCLS5)','Primary with upper Primary Sec (SCLS6)','Upper Primary with  Sec. (SCLS7)','Total (SCLSTOT)']].sum()
display_charts(SWSCSdf, title="State wise no of single-classroom schools by school category", kind="bar",figsize = (1000, 700))


# # State wise no of single teacher school by school category

# In[34]:


SWSTSdf = dls3.groupby(['State Name'])[['Primary Only (STCH1)','Primary with Upper Primary (STCH2)','Primary with upper Primary Sec/H.Sec (STCH3)','Upper Primary Only (STCH4)','Upper Primary with Sec./H.Sec (STCH5)','Primary with upper Primary Sec (STCH6)','Upper Primary with  Sec. (STCH7)','Total (STCHTOT)']].sum()
display_charts(SWSTSdf, title="State wise no of single teacher school by school category", kind="area",figsize = (1000, 700))


# # State wise no of schools approachable by all weather road by school category

# In[35]:


SWSAWRdf = dls3.groupby(['State Name'])[['Primary Only (ROAD1)','Primary with Upper Primary (ROAD2)','Primary with upper Primary Sec/H.Sec (ROAD3)','Upper Primary Only (ROAD4)','Upper Primary with Sec./H.Sec (ROAD5)','Primary with upper Primary Sec (ROAD6)','Upper Primary with  Sec. (ROAD7)','Total (ROADTOT)']].sum()
display_charts(SWSAWRdf, title="State wise no of schools approachable by all weather road by school category", kind="bar",figsize = (1000, 700))


# # State wise no of schools with playground facility by school category

# In[36]:


SWSWPFdf = dls3.groupby(['State Name'])[['Primary Only (SPLAY1)','Primary with Upper Primary (SPLAY2)','Primary with upper Primary Sec/H.Sec (SPLAY3)','Upper Primary Only (SPLAY4)','Upper Primary with Sec./H.Sec (SPLAY5)','Primary with upper Primary Sec (SPLAY6)','Upper Primary with  Sec. (SPLAY7)','Total (SPLAYTOT)']].sum()
display_charts(SWSWPFdf, title="State wise no of schools with playground facility by school category", kind="barh",figsize = (1000, 700))


# # State wise no of schools with Boundarywall by school category

# In[37]:


SWSWBWdf = dls3.groupby(['State Name'])[['Primary Only (SBNDR1)','Primary with Upper Primary (SBNDR2)','Primary with upper Primary Sec/H.Sec (SBNDR3)','Upper Primary Only (SBNDR4)','Upper Primary with Sec./H.Sec (SBNDR5)','Primary with upper Primary Sec (SBNDR6)','Upper Primary with  Sec. (SBNDR7)','Total (SBNDRTOT)']].sum()
display_charts(SWSWBWdf, title="State wise no of schools with Boundarywall by school category", kind="area",figsize = (1000, 700))


# # Schools with basic facility in India

# In[38]:


SWBFdf = dls3[['Total (SGTOILTOT)','Total (SBTOILTOT)','Total (SWATTOT)','Total (SELETOT)','Total (SCOMPTOT)']].rename(columns={'Total (SGTOILTOT)':'School with girls toilet','Total (SBTOILTOT)':'School with boys toilet','Total (SWATTOT)':'School with drinking water facility','Total (SELETOT)':'School with electricity','Total (SCOMPTOT)':'School with computer'}).sum()
SWBFdf = pd.DataFrame(SWBFdf)
display_charts(SWBFdf, kind='pie', title='Schools with basic facility in India', tooltip={'pointFormat': '{series.name}: <b>{point.percentage:.1f}%</b>'})

