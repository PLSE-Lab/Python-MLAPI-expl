#!/usr/bin/env python
# coding: utf-8

# ### Summary
# 
# This notebook creates a panel that puts mean forecasts for widely followed epidemeological models into a single dataframe/CSV.
# 
# For now, it covers mean fatality forecasts for LANL, IHME, Columbia University and the leading Kaggle model from week 1 to week 4 challenges. 
# 
# ### Next steps
# 
# I'm hoping to add more models, forecast uncertainties and possibly more target variables over time. 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re

import os
from datetime import datetime, timedelta

pd.set_option('max_rows',100)


# In[ ]:


def reformat_benchmark_df(df_deaths_matrix,df_benchmark_panel):
    
    df_deaths_locations = df_deaths_matrix[['Country_Region','Province_State']]

    
    for column in df_deaths_matrix.columns:
        date_search = re.compile('[0-9]+/[0-9]+/20')
        if date_search.match(column):
            df_tmp = df_deaths_locations
            df_tmp.loc[:,'Date'] = [column] * len(df_tmp) 
            df_tmp.loc[:,'Fatalities'] = df_deaths_matrix[column]
            df_benchmark_panel= df_benchmark_panel.append(df_tmp,sort=False)
    
    return df_benchmark_panel


# In[ ]:


def add_population(df_benchmark_panel):
    df_population = pd.read_csv('../input/covid19-global-forecasting-week-5/test.csv')
    df_population = df_population[df_population['County'].isnull()][['Province_State','Country_Region','Population']].drop_duplicates()
    df_benchmark_panel = df_benchmark_panel.merge(df_population,left_on=['Province_State','Country_Region'],right_on=['Province_State','Country_Region'],how="left")
    return df_benchmark_panel


# In[ ]:


def add_location_lowest_level(df_benchmark_panel):
    df_benchmark_panel['Location_Lowest_Level'] = df_benchmark_panel['Province_State']
    df_benchmark_panel.loc[df_benchmark_panel['Location_Lowest_Level'].isna(),'Location_Lowest_Level'] = df_benchmark_panel[df_benchmark_panel['Location_Lowest_Level'].isna()]['Country_Region']    
    return df_benchmark_panel


# In[ ]:


def setup_benchmark_panel():

    df_benchmark_panel = pd.DataFrame(columns=['Country_Region','Province_State','Date','Fatalities'])

    
    df_global_deaths_matrix = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')
    df_global_deaths_matrix.rename(columns={"Province/State": "Province_State", "Country/Region": "Country_Region"},inplace=True)
    df_benchmark_panel = reformat_benchmark_df(df_global_deaths_matrix,df_benchmark_panel)
           
    df_us_deaths_matrix = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths_US.csv')
    df_us_deaths_matrix = df_us_deaths_matrix.groupby('Province_State').sum()
    df_us_deaths_matrix['Province_State'] = df_us_deaths_matrix.index
    df_us_deaths_matrix['Country_Region'] = 'US'
    df_benchmark_panel = reformat_benchmark_df(df_us_deaths_matrix,df_benchmark_panel)
    
    df_benchmark_panel['Fatalities'] = pd.to_numeric(df_benchmark_panel['Fatalities'])
    df_benchmark_panel['Date'] = pd.to_datetime(df_benchmark_panel['Date']).astype(str)
 
    df_benchmark_panel = add_location_lowest_level(df_benchmark_panel)

    
    return df_benchmark_panel


# In[ ]:


def merge_ihme_to_benchmark_panel(ihme_file,forecast_date,df_benchmark_panel):
    
    print("ihme {}".format(forecast_date))
    
    col_name = 'ihme_{}'.format(forecast_date)
    df_ihme = pd.read_csv(ihme_file)
    
    if 'date' in df_ihme.columns:
        date_col = 'date'    
    else: 
        date_col = 'date_reported'


    df_ihme = df_ihme.rename(columns={"totdea_mean": col_name ,'location_name' : 'Location_Lowest_Level',date_col : 'Date' })
    
    df_benchmark_panel = df_benchmark_panel.merge(df_ihme[['Location_Lowest_Level','Date',col_name]],left_on=['Location_Lowest_Level','Date'],right_on=['Location_Lowest_Level','Date'],how='outer')
            
    df_benchmark_panel.loc[df_benchmark_panel['Date'] < forecast_date,col_name] = df_benchmark_panel[df_benchmark_panel['Date'] < forecast_date]['Fatalities'] #set history to actual number

    return df_benchmark_panel


# In[ ]:


def parse_date(file_str,sep):
    date_re = re.compile('2020{}0[0-9]{}[0-9]+'.format(sep,sep))
    forecast_date = date_re.search(file_str).group().replace(sep,'-')
    return forecast_date


# In[ ]:


def add_ihme_to_benchmark_panel(df_benchmark_panel):

    date_list = []

    for dirname, _, filenames in os.walk('/kaggle/input/covid19-epidemiological-benchmarking-dataset/Benchmarking Data/IHME/'):
        for filename in filenames:
            if filename[1:] == 'ospitalization_all_locs.csv': 
                ihme_file = os.path.join(dirname, filename)

                forecast_date = parse_date(ihme_file,'_')

                if forecast_date not in date_list: #sometimes IHME has two files for the same date. Just picking one
                    date_list.append(forecast_date)

                    df_benchmark_panel = merge_ihme_to_benchmark_panel(ihme_file,forecast_date,df_benchmark_panel)


    return df_benchmark_panel


# In[ ]:


def add_lanl_to_benchmark_panel(df_benchmark_panel):
    for dirname, _, filenames in os.walk('/kaggle/input/covid19-epidemiological-benchmarking-dataset/Benchmarking Data/LANL/'):
        for filename in filenames:
            if filename[-4:] == '.csv': 
                lanl_file = os.path.join(dirname, filename)

                if lanl_file[93:112] == 'deaths_quantiles_us':
                    forecast_date = parse_date(lanl_file,'-')
                    
                    print("lanl {}".format(forecast_date))

                    forecast_col = "lanl_{}".format(forecast_date)

                    df_lanl = pd.read_csv(lanl_file)

                    df_lanl = df_lanl.rename(columns={"q.50": forecast_col,'state': 'Province_State','dates':'Date'})

                    df_benchmark_panel = df_benchmark_panel.merge(df_lanl[['Province_State','Date',forecast_col]],left_on=['Province_State','Date'],right_on=['Province_State','Date'],how='outer')
                    
                    df_benchmark_panel = add_location_lowest_level(df_benchmark_panel)

                    df_benchmark_panel.loc[df_benchmark_panel['Date'] < forecast_date,forecast_col] = df_benchmark_panel[df_benchmark_panel['Date'] < forecast_date]['Fatalities'] #set history to actual number
                    df_benchmark_panel.loc[(df_benchmark_panel[forecast_col].notnull() ),'Country_Region'] = 'US'
                   
    
    return df_benchmark_panel


# In[ ]:


def add_kaggle_leader_to_benchmark_panel(df_benchmark_panel):
    base_dir = '/kaggle/input/covid19-epidemiological-benchmarking-dataset/Benchmarking Data/'
    kaggle_leaders = ['{}Kaggle/2020_04_02/Submissions/week1winner.csv'.format(base_dir),'{}Kaggle/2020_04_09/Submissions/week2winner.csv'.format(base_dir),'{}Kaggle/2020_04_16/Submissions/week3winner.csv'.format(base_dir)]

    for kaggle_file in kaggle_leaders:
        forecast_date = parse_date(kaggle_file,'_')

        df_kaggle = pd.read_csv(kaggle_file,index_col=0)
        df_test = pd.read_csv('/kaggle/input/covid19-epidemiological-benchmarking-dataset/Benchmarking Data/Kaggle/{}/test.csv'.format(forecast_date.replace('-','_'),index_col=0))
        df_kaggle = df_kaggle.merge(df_test,left_index=True,right_index=True)

        print("kaggle_previous_winner_{}".format(forecast_date))
        if forecast_date == '2020-03-26':
            df_kaggle = df_kaggle.rename(columns={"Province/State": "Province_State","Country/Region":"Country_Region" })

        col_name = "kaggle_previous_winner_{}".format(forecast_date)
        df_kaggle = df_kaggle.rename(columns={"Fatalities": col_name})
        df_benchmark_panel = df_benchmark_panel.merge(df_kaggle[['Country_Region','Province_State','Date',col_name]],left_on=['Country_Region','Province_State','Date'],right_on=['Country_Region','Province_State','Date'],how='outer')

        df_benchmark_panel.loc[df_benchmark_panel['Date'] < forecast_date,col_name] = df_benchmark_panel[df_benchmark_panel['Date'] < forecast_date]['Fatalities'] #set history to actual number

        
    return df_benchmark_panel


# In[ ]:



def merge_cu_to_benchmark_panel(cu_file,df_benchmark_panel):
    
    df_cu = pd.read_csv(cu_file,encoding = "ISO-8859-1")

    date_re = re.compile('Projection_[A-Za-z0-9]+\/')
    date_str = date_re.search(cu_file).group(0)[11:-1]
    date_str = datetime.strftime(datetime.strptime('{}_2020'.format(date_str),'%B%d_%Y'),'%Y-%m-%d')

    if (re.search('bed_nointer[a-z]+.csv',cu_file)):
        col_name = "cu_nointer_{}".format(date_str)
    elif (re.search('bed_[0-9]+contact.csv',cu_file)):
        col_name = "cu{}_{}".format(cu_file[-13:-11],date_str)
        
    print(col_name)
    
    df_cu['StateCode'] = df_cu['county'].str[-2:]
    df_state_codes = pd.read_csv('/kaggle/input/two-letter-us-state-codes/StateCode.csv',index_col=1)

    df_cu = df_cu.merge(df_state_codes,left_on='StateCode',right_index=True)
    df_cu['Date'] = pd.to_datetime(df_cu['Date']).astype(str)

    df_cu = df_cu.groupby(['State','Date'])[['death_50']].sum()
    df_cu.reset_index(inplace=True) #make indexes columns

    df_cu = df_cu.rename(columns={"death_50": col_name,'State':'Province_State'})
    
    df_benchmark_panel = df_benchmark_panel.merge(df_cu[['Province_State','Date',col_name]],left_on=['Province_State','Date'],right_on=['Province_State','Date'],how='outer')

    df_benchmark_panel.loc[df_benchmark_panel['Date'] < date_str,col_name] = df_benchmark_panel[df_benchmark_panel['Date'] < date_str]['Fatalities']

    for location_name in df_benchmark_panel['Province_State'].unique():
        history_mask = (df_benchmark_panel['Province_State'] == location_name) & ( pd.to_datetime(df_benchmark_panel['Date']) >= datetime.strptime(date_str,'%Y-%m-%d')-timedelta(days=1))
        df_benchmark_panel.loc[history_mask,col_name] = df_benchmark_panel[history_mask][col_name].cumsum()

    df_benchmark_panel.loc[(df_benchmark_panel[col_name].notnull() ),'Country_Region'] = 'US'
    
    
    return df_benchmark_panel

    


# In[ ]:


def add_cu_to_benchmark_panel(df_benchmark_panel):
    for dirname, _, filenames in os.walk('/kaggle/input/columbia-university-shaman-lab-covid19-forecasts/'):
        for filename in filenames:

            cu_file = os.path.join(dirname, filename)
            path_re = re.compile('/kaggle/input/columbia-university-shaman-lab-covid19-forecasts/Projection_[A-Za-z0-9]+/bed_([0-9]+contact|nointer[a-z]+).csv')
            if (path_re.match(cu_file)): 
                df_benchmark_panel = merge_cu_to_benchmark_panel(cu_file,df_benchmark_panel)

    return df_benchmark_panel


# In[ ]:


def add_kaggle_leader_w5_to_benchmark_panel(df_benchmark_panel):
    base_dir = '/kaggle/input/covid19-epidemiological-benchmarking-dataset/Benchmarking Data/'

    forecast_date = "2020-05-11"
    forecast_col = "kaggle_previous_winner_{}".format(forecast_date)
    print(forecast_col)

    
    df_kaggle_w5 = pd.read_csv('{}/Kaggle/2020_05_11/Submissions/week_5_leader.csv'.format(base_dir))
    df_kaggle_w5 = df_kaggle_w5[df_kaggle_w5['ForecastId_Quantile'].str[-3:] == "0.5"]
    df_kaggle_w5.index = df_kaggle_w5['ForecastId_Quantile'].str[:-4]
    df_kaggle_w5.index = pd.to_numeric(df_kaggle_w5.index)

    df_test_w5 = pd.read_csv('{}/Kaggle/2020_05_11/test.csv'.format(base_dir),index_col=0)
    df_test_w5 = df_test_w5[df_test_w5['Target'] == 'Fatalities']

    df_kaggle_w5 = df_test_w5.merge(df_kaggle_w5,left_index=True,right_index=True,how='inner')
    
    df_kaggle_w5 = df_kaggle_w5.groupby(['Country_Region','Province_State','Date'])[['TargetValue']].sum()
    df_kaggle_w5.reset_index(inplace=True)

    df_kaggle_w5 = df_kaggle_w5.rename(columns={"TargetValue": forecast_col })
    
    df_benchmark_panel = df_benchmark_panel.merge(df_kaggle_w5[['Country_Region','Province_State','Date',forecast_col]],left_on=['Country_Region','Province_State','Date'],right_on=['Country_Region','Province_State','Date'],how='outer')

    df_benchmark_panel.loc[df_benchmark_panel['Date'] < forecast_date,forecast_col] = df_benchmark_panel[df_benchmark_panel['Date'] < forecast_col]['Fatalities']
    
    for location_name in df_benchmark_panel['Province_State'].unique():
        history_mask = (df_benchmark_panel['Province_State'] == location_name) & ( pd.to_datetime(df_benchmark_panel['Date']) >= datetime.strptime(forecast_date,'%Y-%m-%d')-timedelta(days=1))
        df_benchmark_panel.loc[history_mask,forecast_col] = df_benchmark_panel[history_mask][forecast_col].cumsum()


    df_benchmark_panel.loc[(df_benchmark_panel[forecast_col].notnull() ),'Country_Region'] = 'US'

    
    return df_benchmark_panel


# In[ ]:


df_benchmark_panel = setup_benchmark_panel()
df_benchmark_panel = add_lanl_to_benchmark_panel(df_benchmark_panel)
df_benchmark_panel = add_cu_to_benchmark_panel(df_benchmark_panel)
df_benchmark_panel = add_kaggle_leader_to_benchmark_panel(df_benchmark_panel)
df_benchmark_panel = add_kaggle_leader_w5_to_benchmark_panel(df_benchmark_panel) #treated seperately because these are county based
df_benchmark_panel = add_ihme_to_benchmark_panel(df_benchmark_panel) #must do ihme last for location joining reasons
df_benchmark_panel = add_population(df_benchmark_panel)



# In[ ]:


df_benchmark_panel.to_csv('benchmark_panel.csv')


# In[ ]:


df_benchmark_panel

