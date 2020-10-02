#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


df = pd.read_csv('../input/survey_results_public.csv', index_col='Respondent')


# In[ ]:


df


# In[ ]:


df.shape


# In[ ]:


df.info


# In[ ]:


df.info()


# In[ ]:


pd.set_option('display.max_columns', 85)


# In[ ]:


schema_df = pd.read_csv('../input/survey_results_schema.csv', index_col='Column')


# In[ ]:


schema_df


# In[ ]:


pd.set_option('display.max_rows', 85)


# In[ ]:


df.head(10)


# In[ ]:


type(df)


# In[ ]:


type(df.Hobbyist)


# In[ ]:


df.columns


# In[ ]:


df.iloc[2, 2]


# In[ ]:


df.iloc[2]


# In[ ]:


df.loc[1:5, 'Hobbyist':'Employment']


# In[ ]:


df.loc[1:2, 'Hobbyist':'Employment']


# In[ ]:


df


# In[ ]:


df.loc[1]


# In[ ]:


df.iloc[0]


# In[ ]:


schema_df.loc['Hobbyist']


# In[ ]:


schema_df.loc['MgrIdiot']


# In[ ]:


schema_df.loc['MgrIdiot', 'QuestionText']


# In[ ]:


schema_df.sort_index()


# In[ ]:


schema_df.sort_index(ascending=False)


# In[ ]:


schema_df.sort_index(inplace=True)


# In[ ]:


schema_df


# In[ ]:


high_salary = (df['ConvertedComp'] > 70000)


# In[ ]:


df.loc[high_salary]


# In[ ]:


df.loc[high_salary, ['LanguageWorkedWith', 'ConvertedComp', 'Country']]


# In[ ]:


df.loc[high_salary, 'LanguageWorkedWith']


# In[ ]:


countries = ['United States', 'India']
filt1 = df['Country'].isin(countries)


# In[ ]:


df.loc[filt1, ['LanguageWorkedWith', 'ConvertedComp', 'Country']]


# In[ ]:


df['LanguageWorkedWith']


# In[ ]:


filt2 = df['LanguageWorkedWith'].str.contains('Python', na=False)


# In[ ]:


df.loc[filt2, ['LanguageWorkedWith', 'ConvertedComp']]


# In[ ]:


df.loc[~filt2, ['LanguageWorkedWith', 'ConvertedComp']]


# In[ ]:


df['Country']


# In[ ]:


df['Country'].str.lower()


# In[ ]:


df['Country'].str.upper()


# In[ ]:


df['Country']


# In[ ]:


df.rename(columns={'ConvertedComp' : 'SalaryUSD'}, inplace=True)


# In[ ]:


df['SalaryUSD']


# In[ ]:


df['Hobbyist'].map({'Yes' : 'True', 'No' : 'False'})


# In[ ]:


df['Hobbyist'] = df['Hobbyist'].map({'Yes' : 'True', 'No' : 'False'})


# In[ ]:


df['Hobbyist']


# In[ ]:


df['Hobbyist'].apply(len)


# In[ ]:


df.apply(len)


# In[ ]:


df.rename(columns = {'SalaryUSD' : 'ConvertedComp'}, inplace=True)


# In[ ]:


df


# In[ ]:


df.sort_values(['Country'], inplace=True)


# In[ ]:


df['Country']


# In[ ]:


df.sort_values(['Country', 'ConvertedComp'], ascending=[True, False], inplace=True)


# In[ ]:


df[['Country', 'ConvertedComp']].head(50)


# In[ ]:


df['ConvertedComp'].nlargest(10)


# In[ ]:


df.nlargest(5, 'ConvertedComp')


# In[ ]:


df[['LanguageWorkedWith', 'ConvertedComp', 'DevEnviron']].nlargest(10, 'ConvertedComp') 


# In[ ]:


df


# In[ ]:


df.sort_index(inplace=True)


# In[ ]:


df.describe()


# In[ ]:


df['Hobbyist'].value_counts()


# In[ ]:


df['SocialMedia'].value_counts()


# In[ ]:


df['SocialMedia'].value_counts(normalize=True)


# In[ ]:


df['Country'].value_counts()


# In[ ]:


country_grp = df.groupby(['Country'])


# In[ ]:


country_grp.get_group('India')


# In[ ]:


country_grp_india = country_grp.get_group('India')


# In[ ]:


country_grp_india 


# In[ ]:


country_grp_india['SocialMedia'].value_counts()


# In[ ]:


df.head(2)


# In[ ]:


country_grp_india['Age'].value_counts()


# In[ ]:


country_grp['ConvertedComp'].median().loc['India']


# In[ ]:


filt = df['Country'] == 'India'


# In[ ]:


df.loc[filt]['LanguageWorkedWith'].str.contains('Python').sum()


# In[ ]:


country_grp['LanguageWorkedWith'].apply(lambda x: x.str.contains('Python').sum()).loc['India']


# In[ ]:


country_res = df['Country'].value_counts()
country_res


# In[ ]:


uses_python = country_grp['LanguageWorkedWith'].apply(lambda x: x.str.contains('Python').sum())
uses_python


# In[ ]:


python_df = pd.concat([country_res, uses_python], axis='columns', sort=False)
python_df


# In[ ]:


python_df.rename(columns={'Country': 'Number of Respond', 'LanguageWorkedWith':'Knows Python'}, inplace=True)


# In[ ]:


python_df


# In[ ]:


python_df['pct'] = (python_df['Knows Python']/python_df['Number of Respond'] * 100)
python_df


# In[ ]:


na_val = ['Na', 'Missing']
df_test = pd.read_csv('../input/survey_results_public.csv', index_col='Respondent', na_values = na_val)


# In[ ]:


df['YearsCode']


# In[ ]:


df_test['YearsCode'] =  df['YearsCode'].astype(float)


# In[ ]:


df_test['YearsCode'].unique()


# In[ ]:


df_test['YearsCode'].replace('Less than 1 year', 0, inplace=True)


# In[ ]:


df_test['YearsCode'].replace('More than 50 years', 51, inplace=True)


# In[ ]:


df_test['YearsCode'].unique()


# In[ ]:


df_test['YearsCode'] = df_test['YearsCode'].astype(float)


# In[ ]:


df_test['YearsCode'].describe()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




