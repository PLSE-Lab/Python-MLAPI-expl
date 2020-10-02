#!/usr/bin/env python
# coding: utf-8

# ## What insights can be derived by the survey results published by Kaggle? In this kernal, I have tried to come up with 15 questions, the answers for which I thought could be interesting. This is my first public kernal and I have not worked much on visualization before. I am trying to figure what visualizations would portray the answers to each of these questions the best. Currently working on that and will update the kernal with Visualizations soon.

# In[ ]:


import pandas as pd


# In[ ]:


data_freeform=pd.read_csv('..//input//freeFormResponses.csv')
data_mc_responses=pd.read_csv('..//input//multipleChoiceResponses.csv')
data_survey_schema=pd.read_csv('..//input//SurveySchema.csv')


# ## Before looking at my set of 15 questions, let us look at the questions posed by Kaggle as part of the survey. 

# In[ ]:


for col in data_survey_schema:
    q_num=str(col)
    print(q_num,' - ',data_survey_schema.loc[0,col], ' Number of responses ', data_survey_schema.loc[1,col])


# ## Now, the questions for which we will try to get insights into:
# 
# 1. What Prog language do people use and does it have any coorelation to the amount of time they spend doing coding
# 2. What ML framework do people use and do males and females have any specific preference
# 3. Does years of expereince have any dependency on the amount of time doing coding
# 4. Industry wise amount of time spent coding.
# 5. What are the avg salaries across different countries
# 6. What is the percentage of female respondents by country
# 7. What are the avg salaries across different countries and is there a gender bias with Salaries
# 8. Which IDEs are popular in which locations
# 9. Do people of specific age group prefer any specific IDEs
# 10. Is the choice of IDE dependent on the years of expereince of an individual
# 11. Popularity of different cloud services across locations
# 12. Does type of data that people work on have any say on the salaries? Do people who work on images/videos/speech earn more than people who work on tabular data?
# 13. How are salaries dependent on years of experience across locations
# 14. Is there any correlation between the age of individual and type of data they work on 
# 15. Is there any correlation between the years of exp of individual and type of data they work on 

# ## Qs - 1 What Prog language do people use and does it have any coorelation to the amount of time they spend coding

# In[ ]:


data_mc_responses['Q18'].value_counts(normalize=True).round(3)*100


# In[ ]:


group_23_18_df=data_mc_responses[['Q23','Q18']]
group_23_18_df['Count_row']=1
group_23_18_df_1=group_23_18_df.groupby(['Q23','Q18']).agg({'Count_row': 'sum'})
group_23_df=group_23_18_df[['Q23','Count_row']].groupby(['Q23']).agg({'Count_row': 'sum'})
group_23_18_df=group_23_18_df_1.div(group_23_df,level='Q23')* 100
group_23_18_df=group_23_18_df.reset_index()
group_23_18_df[group_23_18_df['Q23']=='0% of my time'].sort_values(by='Count_row',ascending=False)


# In[ ]:


group_23_18_df[group_23_18_df['Q23']=='1% to 25% of my time'].sort_values(by='Count_row',ascending=False)


# In[ ]:


group_23_18_df[group_23_18_df['Q23']=='25% to 49% of my time'].sort_values(by='Count_row',ascending=False)


# In[ ]:


group_23_18_df[group_23_18_df['Q23']=='50% to 74% of my time'].sort_values(by='Count_row',ascending=False)


# In[ ]:


group_23_18_df[group_23_18_df['Q23']=='75% to 99% of my time'].sort_values(by='Count_row',ascending=False)


# In[ ]:


group_23_18_df[group_23_18_df['Q23']=='100% of my time'].sort_values(by='Count_row',ascending=False)


# ### Inference - Python is marked as one of the prog lang which people use by ~75% of the respondents. I expected those who answered the qs-23 (How much of time you spend coding) as 0% would answer the programming language they use as 'None'. Surprisingly its not, many of those (65%) of those have responded either Python or R, with only 7% of them saying None. 7% is significantly higher than the other cases where None features among less than 2% of respondents   

# ## Qs - 2 What ML framework do people use and do males and females have any specific preference

# In[ ]:


list_q=['Q19']
rel_cols_mc_q19=[]
for i in list_q:
    for col in data_mc_responses.columns:
        if i in col:
            rel_cols_mc_q19.append(col)
rel_cols_mc_q19=set(rel_cols_mc_q19)
relevant_uc_q19_data=data_mc_responses.loc[:,rel_cols_mc_q19]


# In[ ]:


ml_framework=pd.DataFrame()
for i,col in enumerate(relevant_uc_q19_data.columns):
    if col != 'Q19_OTHER_TEXT':
        vc=relevant_uc_q19_data[col].value_counts()
        ml_framework.loc[i,'Framework']=vc.index[0]
        #print(vc.index[0])
        ml_framework.loc[i,'Count']=int(vc[0])
ml_framework


# In[ ]:


relevant_uc_q19_data.shape


# In[ ]:


age_framework_df=pd.concat([relevant_uc_q19_data,data_mc_responses['Q1']],axis=1)
female_Framework_df=age_framework_df[age_framework_df['Q1']=='Female']
male_Framework_df=age_framework_df[age_framework_df['Q1']=='Male']


# In[ ]:


male_Framework_df.head()


# In[ ]:


ml_framework_male=pd.DataFrame()
ml_framework_female=pd.DataFrame()
for i,col in enumerate(male_Framework_df.columns):
    if (col != 'Q19_OTHER_TEXT'):
        if (col != 'Q1'):
            vc=male_Framework_df[col].value_counts()
            vc_f=female_Framework_df[col].value_counts()
            ml_framework_male.loc[i,'Framework']=vc.index[0]
            ml_framework_female.loc[i,'Framework']=vc_f.index[0]
            #print(vc.index[0])
            ml_framework_male.loc[i,'Count']=int(vc[0])
            ml_framework_female.loc[i,'Count']=int(vc_f[0])
ml_framework_male


# In[ ]:


sum_female=ml_framework_female['Count'].sum()
ml_framework_female['Count']=ml_framework_female['Count']/sum_female


# In[ ]:


ml_framework_female['Count']=ml_framework_female['Count']*100


# In[ ]:


sum_male=ml_framework_male['Count'].sum()
ml_framework_male['Count']=ml_framework_male['Count']*100/sum_male


# In[ ]:


ml_framework_female.sort_values(by='Count',ascending=False).head(10)


# In[ ]:


ml_framework_male.sort_values(by='Count',ascending=False).head(10)


# ### Inference - Not too much of a difference here, except that Random forest is slightly more popular among women then men! Men seem to prefer, XG and LGBM slightly

# ## Qs - 3 Does years of expereince have any dependency on the amount of time spent coding

# In[ ]:


exp_time_coding_df=data_mc_responses[['Q8','Q23']]
exp_time_coding_df['Aggr']=1
exp_time_coding_df_1=exp_time_coding_df.groupby(['Q23','Q8']).agg({'Aggr': 'sum'}).reset_index()
#exp_time_coding_df_2=exp_time_coding_df[['Q8','Aggr']].groupby(['Q8']).agg({'Aggr': 'sum'})
#exp_time_coding_df_3=exp_time_coding_df_1.div(exp_time_coding_df_2,level='Q8')* 100
exp_time_coding_df_1


# In[ ]:


sum_aggr=exp_time_coding_df_1['Aggr'].sum()
exp_time_coding_df_1['Aggr']=exp_time_coding_df_1['Aggr']*100/sum_female
exp_time_coding_df_1


# In[ ]:


exp_time_coding_df_1[exp_time_coding_df_1['Q23']=='100% of my time']


# In[ ]:


exp_time_coding_df_1[exp_time_coding_df_1['Q23']=='75% to 99% of my time']


# In[ ]:


exp_time_coding_df_1[exp_time_coding_df_1['Q23']=='50% to 74% of my time']


# In[ ]:


exp_time_coding_df_1[exp_time_coding_df_1['Q23']=='25% to 49% of my time']


# In[ ]:


exp_time_coding_df_1[exp_time_coding_df_1['Q23']=='1% to 25% of my time']


# In[ ]:


exp_time_coding_df_1[exp_time_coding_df_1['Q23']=='0% of my time']


# > ### Inference - Need to relook

# ## Qs - 4 Industry wise amount of time spent coding.

# In[ ]:


exp_time_coding_df=data_mc_responses[['Q7','Q23']]
exp_time_coding_df['Aggr']=1
exp_time_coding_df_1=exp_time_coding_df.groupby(['Q23','Q7']).agg({'Aggr': 'sum'}).reset_index()


# In[ ]:


exp_time_coding_df_1[exp_time_coding_df_1['Q7']=='Academics/Education']


# In[ ]:


df_temp=pd.DataFrame()
df_all=[]
for industry in exp_time_coding_df_1['Q7'].unique():
    df_temp=exp_time_coding_df_1[exp_time_coding_df_1['Q7']==industry]
    sum_aggr=df_temp['Aggr'].sum()
    #print(sum_aggr)
    df_temp['Aggr']=df_temp['Aggr']*100/sum_aggr
    df_all.append(df_temp)


# In[ ]:


df_all


# ### Inference - People who spend 0% of their work time coding are more in 'Non Profit service' and 'Manufacturing and fabrication' industry and expectedly least in 'Computers and Technology' industry. Also people who spend 100% of their work time coding are expectedly more in 'Computers & Tech' and in 'Retail & Sales' industry

# ## Qs - 5 What are the avg salaries across different countries

# In[ ]:


import numpy as np
temp_df=data_mc_responses[['Q3','Q9']]
temp_df=temp_df[temp_df['Q9']!=np.nan]
temp_df=temp_df[temp_df['Q9'].notnull()]
temp_df['Aggr']=1


# In[ ]:


temp_df=temp_df.groupby(['Q3','Q9']).agg({'Aggr':'sum'}).reset_index()


# In[ ]:


country_df=pd.DataFrame()
list_country_df=[]
for country in data_mc_responses['Q3'].unique():
    if country != 'In which country do you currently reside?':
        country_df=temp_df[temp_df['Q3']==country]
        list_country_df.append(country_df)
list_country_df[0]


# > ### Inference -Need to relook at this again, as the salaries also depend on years of experience and not just on the Country. 

# ## Qs - 6 Percentage of female respondents by country

# In[ ]:


temp_df=data_mc_responses[['Q3','Q9','Q1']]
temp_df['Aggr']=1


# In[ ]:


temp_df.head()


# In[ ]:


temp_group=temp_df.groupby(['Q3','Q1']).agg({'Aggr':'count'}).reset_index()


# In[ ]:


gender_df=pd.DataFrame()
list_gender_df=pd.DataFrame()
for country in data_mc_responses['Q3'].unique():
    if country != 'In which country do you currently reside?':
        gender_df=temp_group[temp_group['Q3']==country]
        temp_sum=gender_df['Aggr'].sum()
        gender_df['Aggr']=gender_df['Aggr']*100/temp_sum
        if temp_sum>250:
            list_gender_df=list_gender_df.append(gender_df[gender_df['Q1']=='Female'])


# In[ ]:


list_gender_df.sort_values(by='Aggr',ascending=False)


# ### Inference - US has the highest % of female respondents, followed by Canada, UK and Poland. This is among those countries from where there are atleast 250 responses. 

# ## Qs - 7 What are the avg salaries across different countries and is there a gender bias with Salaries

# In[ ]:


temp_df=data_mc_responses[['Q3','Q9','Q1','Q8']]
temp_df['Aggr']=1


# In[ ]:


temp_group=temp_df.groupby(['Q3','Q1','Q8','Q9']).agg({'Aggr':'count'}).reset_index()


# In[ ]:


temp_group=temp_group[temp_group['Q9']!='I do not wish to disclose my approximate yearly compensation']


# In[ ]:


temp_group.head(20)


# ****## Inference - Incomplete. 

# ## Qs - 8 Popularity of IDEs in different countries

# In[ ]:


list_q=['Q13']
rel_cols_mc_q13=[]
for i in list_q:
    for col in data_mc_responses.columns:
        if i in col:
            rel_cols_mc_q13.append(col)
rel_cols_mc_13=set(rel_cols_mc_q13)
relevant_uc_q13_data=data_mc_responses.loc[:,rel_cols_mc_q13]


# In[ ]:


relevant_uc_q13_data=pd.concat([relevant_uc_q13_data,data_mc_responses['Q3']],axis=1)


# In[ ]:


relevant_uc_q13_data=relevant_uc_q13_data.loc[1:,:]


# In[ ]:


rel_cols_mc_13= [i for i in list(rel_cols_mc_13) if i != 'Q13_OTHER_TEXT']
ide_mapping={}
for i in rel_cols_mc_13:
    ide=relevant_uc_q13_data[i].loc[relevant_uc_q13_data[i].first_valid_index()]
    ide_mapping[i]=ide
print (ide_mapping)


# In[ ]:


cols=['Country','PopularIDE','SecMostPopIDE']
ide_df=pd.DataFrame(columns=cols)
for i,country in enumerate(data_mc_responses['Q3'].unique()):
    rel_cols_mc_13_temp=rel_cols_mc_13
    if country != 'In which country do you currently reside?':
        #print(country)
        temp_df=relevant_uc_q13_data[relevant_uc_q13_data['Q3']==country]
        max_count=max(temp_df[list(rel_cols_mc_13_temp)].count())
        count_series=temp_df[list(rel_cols_mc_13)].count()
        max_col=count_series[count_series==max_count].index[0]
        pop_ide=ide_mapping[max_col]
        rel_cols_mc_13_temp=[j for j in rel_cols_mc_13_temp if j != max_col]
        max_count=max(temp_df[list(rel_cols_mc_13_temp)].count())
        count_series=temp_df[list(rel_cols_mc_13)].count()
        max_col=count_series[count_series==max_count].index[0]
        sec_pop_ide=ide_mapping[max_col]
        ide_df.loc[i,:]=[country,pop_ide,sec_pop_ide]
ide_df


# ### Inference - Jupyter notebook, is unanimously popular across the countries except in Newzeland seem to prefer R Studio more! Undisputed No 1. 
# 
# ### What is surpising though is close to 75% have responded saying that work on Python and only ~15% say they work on R. I would have expected Pycharm to be a close second among the IDEs after Jupyter. It looks majority of the R practisioners use RStudio and a huge majority of Python practisioners use Jupyter

# ## Qs 9 - Do people of specific age group prefer any specific IDEs?

# In[ ]:


list_q=['Q13']
rel_cols_mc_q13=[]
for i in list_q:
    for col in data_mc_responses.columns:
        if i in col:
            rel_cols_mc_q13.append(col)
rel_cols_mc_13=set(rel_cols_mc_q13)
relevant_uc_q13_data=data_mc_responses.loc[:,rel_cols_mc_q13]


# In[ ]:


relevant_uc_q13_data=pd.concat([relevant_uc_q13_data,data_mc_responses['Q2']],axis=1)
relevant_uc_q13_data=relevant_uc_q13_data.loc[1:,:]


# In[ ]:


rel_cols_mc_13= [i for i in list(rel_cols_mc_13) if i != 'Q13_OTHER_TEXT']
ide_mapping={}
for i in rel_cols_mc_13:
    ide=relevant_uc_q13_data[i].loc[relevant_uc_q13_data[i].first_valid_index()]
    ide_mapping[i]=ide
print (ide_mapping)


# In[ ]:


cols=['Age-Group','PopularIDE','SecMostPopIDE']
age_ide_df=pd.DataFrame(columns=cols)
for i,group in enumerate(data_mc_responses['Q2'].unique()):
    rel_cols_mc_13_temp=rel_cols_mc_13
    if group != 'What is your age (# years)?':
        #print(country)
        temp_df=relevant_uc_q13_data[relevant_uc_q13_data['Q2']==group]
        max_count=max(temp_df[list(rel_cols_mc_13_temp)].count())
        count_series=temp_df[list(rel_cols_mc_13)].count()
        max_col=count_series[count_series==max_count].index[0]
        pop_ide=ide_mapping[max_col]
        rel_cols_mc_13_temp=[j for j in rel_cols_mc_13_temp if j != max_col]
        max_count=max(temp_df[list(rel_cols_mc_13_temp)].count())
        count_series=temp_df[list(rel_cols_mc_13)].count()
        max_col=count_series[count_series==max_count].index[0]
        sec_pop_ide=ide_mapping[max_col]
        age_ide_df.loc[i,:]=[group,pop_ide,sec_pop_ide]
age_ide_df


# ### Inference - Again Jupyter notebook is undisputed No 1 across the age groups. R Studio is second most popular among all age groups except among the age groups of 18-24 who like to use Pycharm. Shows the rising popularity of python and diminishing popularity of R? 

# ## Qs - 10 Is the choice of IDE dependent on the years of expereince of an individual?

# In[ ]:


list_q=['Q13']
rel_cols_mc_q13=[]
for i in list_q:
    for col in data_mc_responses.columns:
        if i in col:
            rel_cols_mc_q13.append(col)
rel_cols_mc_13=set(rel_cols_mc_q13)
relevant_uc_q13_data=data_mc_responses.loc[:,rel_cols_mc_q13]


# In[ ]:


relevant_uc_q13_data=pd.concat([relevant_uc_q13_data,data_mc_responses['Q8']],axis=1)
relevant_uc_q13_data=relevant_uc_q13_data.loc[1:,:]


# In[ ]:


rel_cols_mc_13= [i for i in list(rel_cols_mc_13) if i != 'Q13_OTHER_TEXT']
ide_mapping={}
for i in rel_cols_mc_13:
    ide=relevant_uc_q13_data[i].loc[relevant_uc_q13_data[i].first_valid_index()]
    ide_mapping[i]=ide
print (ide_mapping)


# In[ ]:


data_mc_responses['Q8'].unique()


# In[ ]:


cols=['Exp','PopularIDE','SecMostPopIDE']
exp_ide_df=pd.DataFrame(columns=cols)
for i,exp in enumerate(data_mc_responses['Q8'].unique()):
    rel_cols_mc_13_temp=rel_cols_mc_13
    if exp != 'How many years of experience do you have in your current role?':
        #print(country)
        temp_df=relevant_uc_q13_data[relevant_uc_q13_data['Q8']==exp]
        max_count=max(temp_df[list(rel_cols_mc_13_temp)].count())
        count_series=temp_df[list(rel_cols_mc_13)].count()
        max_col=count_series[count_series==max_count].index[0]
        pop_ide=ide_mapping[max_col]
        rel_cols_mc_13_temp=[j for j in rel_cols_mc_13_temp if j != max_col]
        max_count=max(temp_df[list(rel_cols_mc_13_temp)].count())
        count_series=temp_df[list(rel_cols_mc_13)].count()
        max_col=count_series[count_series==max_count].index[0]
        sec_pop_ide=ide_mapping[max_col]
        exp_ide_df.loc[i,:]=[exp,pop_ide,sec_pop_ide]
exp_ide_df


# ### Inference - Well, there is no real correlation between Years of experience and the IDE choice

# ## Qs - 11 Popularity of different cloud services across locations

# In[ ]:


data_mc_responses['Q3'].value_counts()


# In[ ]:


list_q=['Q15']
rel_cols_mc_q15=[]
for i in list_q:
    for col in data_mc_responses.columns:
        if i in col:
            rel_cols_mc_q15.append(col)
rel_cols_mc_15=set(rel_cols_mc_q15)
relevant_uc_q15_data=data_mc_responses.loc[:,rel_cols_mc_q15]


# In[ ]:


relevant_uc_q15_data=pd.concat([relevant_uc_q15_data,data_mc_responses['Q3']],axis=1)
relevant_uc_q15_data=relevant_uc_q15_data.loc[1:,:]


# In[ ]:


cols=[i for i in relevant_uc_q15_data.columns if i != 'Q15_OTHER_TEXT']
relevant_uc_q15_data=relevant_uc_q15_data[cols]
relevant_uc_q15_data.head()


# In[ ]:


rel_cols_mc_15= [i for i in relevant_uc_q15_data.columns if i != 'Q3']
cloud_mapping={}
for i in rel_cols_mc_15:
    cloud=relevant_uc_q15_data[i].loc[relevant_uc_q15_data[i].first_valid_index()]
    cloud_mapping[i]=cloud
print (cloud_mapping)


# In[ ]:


cols=['Country','PopularIDE','SecMostPopIDE']
country_cloud_df=pd.DataFrame(columns=cols)
for i,country in enumerate(data_mc_responses['Q3'].unique()):
    rel_cols_mc_15_temp=rel_cols_mc_15
    if country != 'In which country do you currently reside?':
        #print(country)
        temp_df=relevant_uc_q15_data[relevant_uc_q15_data['Q3']==country]
        max_count=max(temp_df[list(rel_cols_mc_15_temp)].count())
        count_series=temp_df[list(rel_cols_mc_15)].count()
        max_col=count_series[count_series==max_count].index[0]
        pop_cloud=cloud_mapping[max_col]
        rel_cols_mc_15_temp=[j for j in rel_cols_mc_15_temp if j != max_col]
        max_count=max(temp_df[list(rel_cols_mc_15_temp)].count())
        count_series=temp_df[list(rel_cols_mc_15)].count()
        max_col=count_series[count_series==max_count].index[0]
        sec_pop_cloud=cloud_mapping[max_col]
        country_cloud_df.loc[i,:]=[country,pop_cloud,sec_pop_cloud]
country_cloud_df


# ### Inference - Majority of the respondents among 4 of the top 6 countries (Ind,China,Russia,Germany ) do not use Cloud. The other two countries in top 6 (US, Brazil) prefer Amazon over Azure. Amazon has overwhelming popularity compared to Azure or GCP. 
# 
# ### People who use Cloud do it when their personal laptop/desktop does not support high end computation typically needed for deep learning. These results indicate majority of the respondents in the 4 of the top 6 country work on traditional machine learning whereas the respondents from US and Brazil work more on Deep learning. 

# ## Qs - 12 Is there any correlation between the age of individual and type of data they work on 

# In[ ]:


list_q=['Q31']
rel_cols_mc_q31=[]
for i in list_q:
    for col in data_mc_responses.columns:
        if i in col:
            rel_cols_mc_q31.append(col)
rel_cols_mc_31=set(rel_cols_mc_q31)
relevant_uc_q31_data=data_mc_responses.loc[:,rel_cols_mc_q31]


# In[ ]:


relevant_uc_q31_data=pd.concat([relevant_uc_q31_data,data_mc_responses['Q2']],axis=1)
relevant_uc_q31_data=relevant_uc_q31_data.loc[1:,:]


# In[ ]:


cols=[i for i in relevant_uc_q31_data.columns if i != 'Q31_OTHER_TEXT']
relevant_uc_q31_data=relevant_uc_q31_data[cols]
relevant_uc_q31_data.head()


# In[ ]:


rel_cols_mc_31= [i for i in relevant_uc_q31_data.columns if i != 'Q2']
dtype_mapping={}
for i in rel_cols_mc_31:
    dtype=relevant_uc_q31_data[i].loc[relevant_uc_q31_data[i].first_valid_index()]
    dtype_mapping[i]=dtype
print (dtype_mapping)


# In[ ]:


cols=['AgeGroup','Dtype','SecPopDtype']
age_dtype_df=pd.DataFrame(columns=cols)
for i,age in enumerate(data_mc_responses['Q2'].unique()):
    rel_cols_mc_31_temp=rel_cols_mc_31
    if age != 'What is your age (# years)?':
        #print(country)
        temp_df=relevant_uc_q31_data[relevant_uc_q31_data['Q2']==age]
        max_count=max(temp_df[list(rel_cols_mc_31_temp)].count())
        count_series=temp_df[list(rel_cols_mc_31)].count()
        max_col=count_series[count_series==max_count].index[0]
        dtype=dtype_mapping[max_col]
        rel_cols_mc_31_temp=[j for j in rel_cols_mc_31_temp if j != max_col]
        max_count=max(temp_df[list(rel_cols_mc_31_temp)].count())
        count_series=temp_df[list(rel_cols_mc_31)].count()
        max_col=count_series[count_series==max_count].index[0]
        sec_pop_cloud=dtype_mapping[max_col]
        age_dtype_df.loc[i,:]=[age,dtype,sec_pop_cloud]
age_dtype_df


# ### Inference - Well, everyone starts with Numerical data. Nothing surprising here. 

# ## Qs - 13 Is there any correlation between the years of exp of individual and type of data they work on 

# In[ ]:


list_q=['Q31']
rel_cols_mc_q31=[]
for i in list_q:
    for col in data_mc_responses.columns:
        if i in col:
            rel_cols_mc_q31.append(col)
rel_cols_mc_31=set(rel_cols_mc_q31)
relevant_uc_q31_data=data_mc_responses.loc[:,rel_cols_mc_q31]


# In[ ]:


relevant_uc_q31_data=pd.concat([relevant_uc_q31_data,data_mc_responses['Q8']],axis=1)
relevant_uc_q31_data=relevant_uc_q31_data.loc[1:,:]


# In[ ]:


cols=[i for i in relevant_uc_q31_data.columns if i != 'Q31_OTHER_TEXT']
relevant_uc_q31_data=relevant_uc_q31_data[cols]
relevant_uc_q31_data.head()


# In[ ]:


rel_cols_mc_31= [i for i in relevant_uc_q31_data.columns if i != 'Q8']
dtype_mapping={}
for i in rel_cols_mc_31:
    dtype=relevant_uc_q31_data[i].loc[relevant_uc_q31_data[i].first_valid_index()]
    dtype_mapping[i]=dtype
print (dtype_mapping)


# In[ ]:


cols=['Exp','Dtype','SecPopDtype']
exp_dtype_df=pd.DataFrame(columns=cols)
for i,exp in enumerate(data_mc_responses['Q8'].unique()):
    rel_cols_mc_31_temp=rel_cols_mc_31
    if exp != 'How many years of experience do you have in your current role?':
        #print(country)
        temp_df=relevant_uc_q31_data[relevant_uc_q31_data['Q8']==exp]
        max_count=max(temp_df[list(rel_cols_mc_31_temp)].count())
        count_series=temp_df[list(rel_cols_mc_31)].count()
        max_col=count_series[count_series==max_count].index[0]
        dtype=dtype_mapping[max_col]
        rel_cols_mc_31_temp=[j for j in rel_cols_mc_31_temp if j != max_col]
        max_count=max(temp_df[list(rel_cols_mc_31_temp)].count())
        count_series=temp_df[list(rel_cols_mc_31)].count()
        max_col=count_series[count_series==max_count].index[0]
        sec_pop_cloud=dtype_mapping[max_col]
        exp_dtype_df.loc[i,:]=[exp,dtype,sec_pop_cloud]
exp_dtype_df


# ### Inference - No visible correlation. Probably, the choice of third most popular and forth most popular datatypes will be more interesting to see here. 
