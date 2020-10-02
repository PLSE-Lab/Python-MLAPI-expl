#!/usr/bin/env python
# coding: utf-8

# # The African American Gender Gap and Massachusetts Criminal Justice Budgets****
# 
# Summary:
# 
# 1) According to 5 Year 2018 U.S. Census ACS 5 Year data, the  ratio of males to females in communities with significant African Americans appears to decline with each age group. Are African American males not being counted? Or does this represent something more troubling? Is it the product of violence or overly harsh policing in these communities? The eventual goal is to determine how counties/zips with low ratios have been fairing in other socioeconomic features.  Do low ratio communities face difficulties with education, health, arrests, or violence? Note: Before drawing conclusions, keep in mind this data is from a single point in time. 
# 
# 2) This Notebook also looks at Massachusetts Criminal Justice Spending according massbudget.org and uses a Plotly visualization to illustrate the small amount specifically earmarked for substance abuse treatment. If nothing more, it shows the need for more detailed data. 
# 
# 

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go


# In[ ]:


black_gender_zip = pd.read_csv("../input/washington-post-police-killings-data-2010-census/2018_ACS_black_by_sex_zipcodeonly.csv")
black_gender_zip.shape


# In[ ]:


###COUNT NA's####
black_gender_zip.isna().sum()


# In[ ]:


black_gender_county = pd.read_csv("../input/washington-post-police-killings-data-2010-census/2018_ACS_black_by_sex.csv")
black_gender_county.shape


# In[ ]:


black_gender_county.isna().sum()


# In[ ]:


black_gender_zip = black_gender_zip[black_gender_zip['ET_African_American_Male'].notna()]
black_gender_zip = black_gender_zip[black_gender_zip['ET_African_American_Female'].notna()]
black_gender_zip['Zip_Male_Female_Ratio']=black_gender_zip['ET_African_American_Male']/black_gender_zip['ET_African_American_Female']
black_gender_zip = black_gender_zip.drop("State", axis=1)
black_gender_zip = black_gender_zip.rename(columns = {'County':'Zipcode'})
black_gender_zip.shape


# In[ ]:


#### We are removing ZCTA5 23891 because there are 0 females and the black male population may be a product of Sussex II State Prison

black_gender_zip=black_gender_zip[black_gender_zip['ET_African_American_Female']!=0]
black_gender_zip.shape


# In[ ]:


black_gender_zip['Zip_Male_Female_Ratio_15_17']=black_gender_zip['ET_African_American_Male_15_17']/black_gender_zip['ET_African_American_Female_15_17']
black_gender_zip['Zip_Male_Female_Ratio_18_19']=black_gender_zip['ET_African_American_Male_18_19']/black_gender_zip['ET_African_American_Female_18_19']
black_gender_zip['Zip_Male_Female_Ratio_20_24']=black_gender_zip['ET_African_American_Male_20_24']/black_gender_zip['ET_African_American_Female_20_24']
black_gender_zip['Zip_Male_Female_Ratio_25_29']=black_gender_zip['ET_African_American_Male_25_29']/black_gender_zip['ET_African_American_Female_25_29']
black_gender_zip['Zip_Male_Female_Ratio_30_34']=black_gender_zip['ET_African_American_Male_30_34']/black_gender_zip['ET_African_American_Female_30_34']
black_gender_zip['Zip_Male_Female_Ratio_35_44']=black_gender_zip['ET_African_American_Male_35_44']/black_gender_zip['ET_African_American_Female_35_44']
black_gender_zip['Zip_Male_Female_Ratio_45_54']=black_gender_zip['ET_African_American_Male_45_54']/black_gender_zip['ET_African_American_Female_45_54']
black_gender_zip.head()


# In[ ]:


zip_stats=pd.DataFrame(black_gender_zip.describe())
print(zip_stats)


# In[ ]:


black_gender_zip=black_gender_zip[black_gender_zip['ET_African_American_Male_15_17']>=50]
black_gender_zip=black_gender_zip[black_gender_zip['ET_African_American_Male_18_19']>=50]
black_gender_zip=black_gender_zip[black_gender_zip['ET_African_American_Male_20_24']>=50]
black_gender_zip=black_gender_zip[black_gender_zip['ET_African_American_Male_25_29']>=50]
black_gender_zip=black_gender_zip[black_gender_zip['ET_African_American_Male_30_34']>=50]
black_gender_zip=black_gender_zip[black_gender_zip['ET_African_American_Male_35_44']>=50]
black_gender_zip=black_gender_zip[black_gender_zip['ET_African_American_Male_45_54']>=50]
black_gender_zip.shape


# In[ ]:


black_gender_zip=black_gender_zip[black_gender_zip['ET_African_American_Female_15_17']>=50]
black_gender_zip=black_gender_zip[black_gender_zip['ET_African_American_Female_18_19']>=50]
black_gender_zip=black_gender_zip[black_gender_zip['ET_African_American_Female_20_24']>=50]
black_gender_zip=black_gender_zip[black_gender_zip['ET_African_American_Female_25_29']>=50]
black_gender_zip=black_gender_zip[black_gender_zip['ET_African_American_Female_30_34']>=50]
black_gender_zip=black_gender_zip[black_gender_zip['ET_African_American_Female_35_44']>=50]
black_gender_zip=black_gender_zip[black_gender_zip['ET_African_American_Female_45_54']>=50]
black_gender_zip.shape


# In[ ]:


zip_stats=pd.DataFrame(black_gender_zip.describe())
print(zip_stats)


# In[ ]:


black_gender_county=black_gender_county[black_gender_county['ET_African_American_Population']>=2000]
black_gender_county.shape


# In[ ]:



black_gender_county=black_gender_county[black_gender_county['ET_African_American_Male_15_17']>=50]
black_gender_county=black_gender_county[black_gender_county['ET_African_American_Male_18_19']>=50]
black_gender_county=black_gender_county[black_gender_county['ET_African_American_Male_20_24']>=50]
black_gender_county=black_gender_county[black_gender_county['ET_African_American_Male_25_29']>=50]
black_gender_county=black_gender_county[black_gender_county['ET_African_American_Male_30_34']>=50]
black_gender_county=black_gender_county[black_gender_county['ET_African_American_Male_35_44']>=50]
black_gender_county=black_gender_county[black_gender_county['ET_African_American_Male_45_54']>=50]
black_gender_county.shape


# In[ ]:


black_gender_county=black_gender_county[black_gender_county['ET_African_American_Female_15_17']>=50]
black_gender_county=black_gender_county[black_gender_county['ET_African_American_Female_18_19']>=50]
black_gender_county=black_gender_county[black_gender_county['ET_African_American_Female_20_24']>=50]
black_gender_county=black_gender_county[black_gender_county['ET_African_American_Female_25_29']>=50]
black_gender_county=black_gender_county[black_gender_county['ET_African_American_Female_30_34']>=50]
black_gender_county=black_gender_county[black_gender_county['ET_African_American_Female_35_44']>=50]
black_gender_county=black_gender_county[black_gender_county['ET_African_American_Female_45_54']>=50]
black_gender_county.shape


# In[ ]:


black_gender_county['County_Male_Female_Ratio']=black_gender_county['ET_African_American_Male']/black_gender_county['ET_African_American_Female']
black_gender_county['County_Male_Female_Ratio_15_17']=black_gender_county['ET_African_American_Male_15_17']/black_gender_county['ET_African_American_Female_15_17']
black_gender_county['County_Male_Female_Ratio_18_19']=black_gender_county['ET_African_American_Male_18_19']/black_gender_county['ET_African_American_Female_18_19']
black_gender_county['County_Male_Female_Ratio_20_24']=black_gender_county['ET_African_American_Male_20_24']/black_gender_county['ET_African_American_Female_20_24']
black_gender_county['County_Male_Female_Ratio_25_29']=black_gender_county['ET_African_American_Male_25_29']/black_gender_county['ET_African_American_Female_25_29']
black_gender_county['County_Male_Female_Ratio_30_34']=black_gender_county['ET_African_American_Male_30_34']/black_gender_county['ET_African_American_Female_30_34']
black_gender_county['County_Male_Female_Ratio_35_44']=black_gender_county['ET_African_American_Male_35_44']/black_gender_county['ET_African_American_Female_35_44']
black_gender_county['County_Male_Female_Ratio_45_54']=black_gender_county['ET_African_American_Male_45_54']/black_gender_county['ET_African_American_Female_45_54']
black_gender_county.head()


# **Observation: Reviewing the summary below, we see median and average gender ratios may be declining for each subsequent generation.**

# In[ ]:


county_stats=pd.DataFrame(black_gender_county.describe())
columns3 = ['Key', 'ET_African_American_Population','ET_African_American_Male', 'ET_African_American_Female_35_44','ET_African_American_Male_15_17', 'ET_African_American_Male_18_19', 'ET_African_American_Male_20_24','ET_African_American_Male_25_29', 'ET_African_American_Male_30_34', 'ET_African_American_Male_35_44', 'ET_African_American_Male_45_54', 'ET_African_American_Male_55_64', 'ET_African_American_Male_65_74', 'ET_African_American_Female', 'ET_African_American_Female_15_17', 'ET_African_American_Female_18_19', 'ET_African_American_Female_20_24', 'ET_African_American_Female_25_29', 'ET_African_American_Female_30_34', 'ET_African_American_Female_45_54','ET_African_American_Female_55_64']
county_stats.drop(columns3, inplace=True, axis=1)

county_stats


# In[ ]:


# Make boxplot for one group only
sns.boxplot(y = 'County_Male_Female_Ratio_45_54', data = black_gender_county)


# In[ ]:


sns.boxplot( y=black_gender_county["County_Male_Female_Ratio"])


# In[ ]:


black_gender_county_over_10=black_gender_county[black_gender_county["County_Male_Female_Ratio_45_54"]>=10]


# In[ ]:


import plotly.express as px
fig = px.violin(black_gender_county, y="County_Male_Female_Ratio")
fig.update_layout(
    title_text="Black Male to Black Female Ratio for Counties in 2018 ACS 5 Year Estimates")
fig.show(rendering = "kaggle")


# In[ ]:


import plotly.express as px
fig = px.violin(black_gender_county, y="County_Male_Female_Ratio",  box=True, points='all')
fig.update_layout(
    title_text="Black Male to Black Female Ratio for Counties in 2018 ACS 5 Year Estimates")
fig.show(rendering = "kaggle")


# In[ ]:


fig = px.box(black_gender_county, y="County_Male_Female_Ratio", points="all")
fig.update_layout(
    title_text="Black Male to Black Female Ratio for Counties in 2018 ACS 5 Year Estimates")
fig.show(rendering = "kaggle")


# In[ ]:


import plotly.graph_objects as go


# In[ ]:


black_gender_county.dtypes


# In[ ]:




black_gender_subset_A=black_gender_county[['County_Male_Female_Ratio']]
black_gender_subset_A['Group']='County_Male_Female_Ratio'
black_gender_subset_B=black_gender_county[['County_Male_Female_Ratio_15_17']]
black_gender_subset_B['Group']='County_Male_Female_Ratio_15_17'
black_gender_subset_B = black_gender_subset_B.rename(columns = {'County_Male_Female_Ratio_15_17':'County_Male_Female_Ratio'})

black_gender_subset_C=black_gender_county[['County_Male_Female_Ratio_20_24']]
black_gender_subset_C['Group']='County_Male_Female_Ratio_20_24'
black_gender_subset_C = black_gender_subset_C.rename(columns = {'County_Male_Female_Ratio_20_24':'County_Male_Female_Ratio'})
black_gender_subset_D=black_gender_county[['County_Male_Female_Ratio_25_29']]
black_gender_subset_D['Group']='County_Male_Female_Ratio_25_29'
black_gender_subset_D = black_gender_subset_D.rename(columns = {'County_Male_Female_Ratio_25_29':'County_Male_Female_Ratio'})
black_gender_subset_E=black_gender_county[['County_Male_Female_Ratio_30_34']]
black_gender_subset_E['Group']='County_Male_Female_Ratio_30_34'
black_gender_subset_E = black_gender_subset_E.rename(columns = {'County_Male_Female_Ratio_30_34':'County_Male_Female_Ratio'})
black_gender_subset_F=black_gender_county[['County_Male_Female_Ratio_35_44']]
black_gender_subset_F['Group']='County_Male_Female_Ratio_35_44'
black_gender_subset_F = black_gender_subset_F.rename(columns = {'County_Male_Female_Ratio_35_44':'County_Male_Female_Ratio'})
black_gender_subset_G=black_gender_county[['County_Male_Female_Ratio_45_54']]
black_gender_subset_G['Group']='County_Male_Female_Ratio_45_54'
black_gender_subset_G = black_gender_subset_G.rename(columns = {'County_Male_Female_Ratio_45_54':'County_Male_Female_Ratio'})

frames = [black_gender_subset_B, black_gender_subset_C, black_gender_subset_D, black_gender_subset_E,black_gender_subset_F,black_gender_subset_G]

black_gender_subset = pd.concat(frames)
black_gender_subset.head()
#black_gender_subset=black_gender_subset.T

#black_gender__subset_melted=pd.melt(black_gender_subset)


# In[ ]:


black_gender_subset=black_gender_subset[black_gender_subset['County_Male_Female_Ratio']<=4]
fig = px.box(black_gender_subset, x="Group", y="County_Male_Female_Ratio", points="all")
fig.update_layout(
    title_text="Declining Black Male to Black Female Ratio by Age Group In Counties from <br> 2018 ACS 5 Year Estimates")
fig.show(rendering = "kaggle")


# In[ ]:



black_gender_subset=black_gender_subset[black_gender_subset['County_Male_Female_Ratio']<=1.5]
fig = px.box(black_gender_subset,  x="Group", y="County_Male_Female_Ratio", color="Group")
fig.update_layout(
    title_text="Declining Black Male to Black Female Ratio by Age Group In Counties from <br> 2018 ACS 5 Year Estimates")
fig.show(rendering = "kaggle")


# In[ ]:


fig = px.violin(black_gender_subset,  x="Group", y="County_Male_Female_Ratio", color="Group", box=True, points="all",
          hover_data=black_gender_subset.columns)
fig.update_layout(
    title_text="Declining Black Male to Black Female Ratio by Age Group In Counties from <br> 2018 ACS 5 Year Estimates")
fig.show(rendering = "kaggle")


# In[ ]:


ma_crim_justice_spending_inf_adj = pd.read_csv("../input/washington-post-police-killings-data-2010-census/MA_Overall_Public_Safety_Spending_Inf_Adj.csv")
ma_crim_justice_spending_not_inf_adj = pd.read_csv("../input/washington-post-police-killings-data-2010-census/MA_Overall_Public_Safety_Spending_Not_Inf_Adj.csv")
ma_prisons_inf_adj = pd.read_csv("../input/washington-post-police-killings-data-2010-census/MA_Prisons_Probation_Parole_Inflation_Adjusted.csv")
ma_prisons_not_inf_adj = pd.read_csv("../input/washington-post-police-killings-data-2010-census/MA_Prisons_Probation_Parole_Not_Inflation_Adjusted.csv")


# In[ ]:


ma_crim_justice_spending_inf_adj=ma_crim_justice_spending_inf_adj.T
new_header = ma_crim_justice_spending_inf_adj.iloc[0] #grab the first row for the header
ma_crim_justice_spending_inf_adj = ma_crim_justice_spending_inf_adj[1:] #take the data less the header row
ma_crim_justice_spending_inf_adj.columns = new_header #set the header row as the df header


# In[ ]:


ma_crim_justice_spending_inf_adj.reset_index(drop=False, inplace=True)
ma_crim_justice_spending_inf_adj.rename(columns = {'index':'Program'}, inplace = True)
ma_crim_justice_spending_inf_adj.rename(columns = {'FY21':'FY21 Gov'}, inplace = True)
ma_crim_justice_spending_inf_adj['Treatment_Dummy']=0
ma_crim_justice_spending_inf_adj['Line_Item']="XXXXX"


# In[ ]:


ma_crim_justice_spending_inf_adj.head()


# In[ ]:


columns_for_append=ma_crim_justice_spending_inf_adj[['Line_Item','Program','Treatment_Dummy','FY01','FY02','FY03','FY04','FY05','FY06','FY07','FY08','FY09','FY10','FY11','FY12','FY13','FY14','FY15','FY16','FY17','FY18','FY19','FY20','FY21 Gov']]


# In[ ]:


ma_prisons_inf_adj.head()


# In[ ]:


ma_prisons_inf_adj.shape


# In[ ]:


columns_for_append = columns_for_append[columns_for_append['Program'] != "Prisons_Probation_Parole"]
columns_for_append = columns_for_append[columns_for_append['Program'] != "Totals"]
complete_crim_spending = pd.concat([ma_prisons_inf_adj,columns_for_append])


# In[ ]:


complete_crim_spending.shape


# In[ ]:


complete_crim_spending.head()


# In[ ]:


####Prison, Probation, and Parole Budget Only####

sub_treat=ma_prisons_inf_adj['FY20']* ma_prisons_inf_adj['Treatment_Dummy']
sub_treat=sub_treat.sum()
non_sub_treat=ma_prisons_inf_adj['FY20'].sum()-sub_treat
labels = ['Substance_Abuse_Treatment','Prison, Probation, and Parole Budget']
values = [sub_treat, non_sub_treat]
# pull is given as a fraction of the pie radius
fig = go.Figure(data=[go.Pie(labels=labels, values=values, pull=[0, 0.2])])
fig.update_layout(
    title_text="FY2020 MA Prison, Probation, and Parole Budget <br> Percent Spent on Substance Abuse Treatment")
fig.show(rendering = "kaggle")


# In[ ]:


####Total Criminal Justice Budget####
courts=complete_crim_spending.loc[complete_crim_spending['Program']=='Courts_Legal_Assistance',['FY20']]
courts=courts['FY20'][0]
law_enf=complete_crim_spending.loc[complete_crim_spending['Program']=='Law_Enforcement',['FY20']]
law_enf=law_enf['FY20'][1]
OPS=complete_crim_spending.loc[complete_crim_spending['Program']=='Other_Law_Public_Safety',['FY20']]
prosc=complete_crim_spending.loc[complete_crim_spending['Program']=='Prosecutors',['FY20']]
prosc=prosc['FY20'][4]

labels2 = ['Substance_Abuse_Treatment','Prison, Probation, and Parole Budget','Courts_Legal_Assistance', 'Law_Enforcement', 'Other_Law_Public_Safety', 'Prosecutors']
values2 = [sub_treat, non_sub_treat, courts,law_enf, OPS, prosc]




# pull is given as a fraction of the pie radius
fig = go.Figure(data=[go.Pie(labels=labels2, values=values2, pull=[0.2, 0,0,0,0,0])])
fig.update_layout(
    title_text="FY2020 MA Total Criminal Justice Budget <br> Percent Spent on Substance Abuse Treatment")
fig.show(rendering = "kaggle")


# In[ ]:


complete_crim_spending_2020 = complete_crim_spending[complete_crim_spending['FY20'] != 0]

MA_Criminal_Justice_2020_Spending_On_Treatment = complete_crim_spending_2020[complete_crim_spending_2020['Treatment_Dummy'] == 1]
MA_Criminal_Justice_2020_Spending_On_Treatment = MA_Criminal_Justice_2020_Spending_On_Treatment[['Program','Treatment_Dummy','FY20']]


MA_Criminal_Justice_2020_Spending_On_Non_Treatment = complete_crim_spending_2020[complete_crim_spending_2020['Treatment_Dummy'] == 0]
MA_Criminal_Justice_2020_Spending_On_Non_Treatment = MA_Criminal_Justice_2020_Spending_On_Non_Treatment[['Program','Treatment_Dummy','FY20']]


# In[ ]:


MA_Criminal_Justice_2020_Spending_On_Treatment.head(10)


# In[ ]:


MA_Criminal_Justice_2020_Spending_On_Non_Treatment.head(50)


# In[ ]:




