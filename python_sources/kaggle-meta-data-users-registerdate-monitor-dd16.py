#!/usr/bin/env python
# coding: utf-8

# ### notes / updates
# * this analysis concentrates on the users->registerdate field
# * I only do quick checks, so please use results with caution

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
import sys
# Any results you write to the current directory are saved as output.
from datetime import datetime
from IPython.core.display import display, HTML
from matplotlib import pyplot    
import math
import csv
bln_create_df_all_csv_file = True
bln_create_words_csv_file = False
int_df_all_version = 6
bln_ready_to_commit = True
bln_create_estimate_files = False
bln_upload_input_estimates = False
bln_recode_variables = True
pd.set_option("display.max_rows", 150)
pd.set_option("display.max_columns", 100)
pd.set_option('max_colwidth', 60)

df_time_check = pd.DataFrame(columns=['Stage','Start','End', 'Seconds', 'Minutes'])
int_time_check = 0
dat_start = datetime.now()
dat_program_start = dat_start

if not bln_ready_to_commit:
    int_read_csv_rows = 100000
else:
    int_read_csv_rows= None
    
# generate crosstabs  {0 = nothing; 1 = screen}
int_important_crosstab = 1
int_past_crosstab = 0
int_current_crosstab = 1

bln_read_train = True
bln_read_test = True


# In[ ]:


def get_translations_analysis_description(df_input, str_language, str_group, int_code):
    # created by darryldias 25may2018
    df_temp = df_input[(df_input['language']==str_language) & (df_input['group']==str_group) & (df_input['code']==int_code)]                     ['description']
    return df_temp.iloc[0]

#translations_analysis = pd.read_csv('../input/ulabox-translations-analysis/translations_analysis.csv')
strg_count_column = 'count'   #get_translations_analysis_description(translations_analysis, str_language, 'special', 2)

def start_time_check():
    # created by darryldias 21may2018 - updated 8june2018
    global dat_start 
    dat_start = datetime.now()
    
def end_time_check(dat_start, str_stage):
    # created by darryldias 21may2018 - updated 8june2018
    global int_time_check
    global df_time_check
    int_time_check += 1
    dat_end = datetime.now()
    diff_seconds = (dat_end-dat_start).total_seconds()
    diff_minutes = diff_seconds / 60.0
    df_time_check.loc[int_time_check] = [str_stage, dat_start, dat_end, diff_seconds, diff_minutes]

def create_topline(df_input, str_item_column, str_count_column):
    # created by darryldias 21may2018; updated by darryldias 29may2018
    str_percent_column = 'percent'   #get_translations_analysis_description(translations_analysis, str_language, 'special', 3)
    df_temp = df_input.groupby(str_item_column).size().reset_index(name=str_count_column)
    df_output = pd.DataFrame(columns=[str_item_column, str_count_column, str_percent_column])
    int_rows = df_temp.shape[0]
    int_columns = df_temp.shape[1]
    int_total = df_temp[str_count_column].sum()
    flt_total = float(int_total)
    for i in range(int_rows):
        str_item = df_temp.iloc[i][0]
        int_count = df_temp.iloc[i][1]
        flt_percent = round(int_count / flt_total * 100, 1)
        df_output.loc[i] = [str_item, int_count, flt_percent]
    
    df_output.loc[int_rows] = ['total', int_total, 100.0]
    return df_output        

def get_size_raw(df_input):
    return sys.getsizeof(df_input)

def get_size_mb(df_input):
    int_size_raw = get_size_raw(df_input)
    flt_size = float(int_size_raw) / 1000000 
    return int(flt_size)

def get_dataframe_info(df_input, bln_output_csv = False, str_filename = None):
    # created by darryldias 24may2018 - updated 25jan2019
    int_rows = df_input.shape[0]
    int_cols = df_input.shape[1]
    flt_rows = float(int_rows)
    int_size_mb = get_size_mb(df_input)

    df_output = pd.DataFrame(columns=["Column", "Type", "Not Null", 'Null', '% Not Null', '% Null'])
    df_output.loc[0] = ['Table Row Count', '', int_rows, '', '', '']
    df_output.loc[1] = ['Table Column Count', '', int_cols, '', '', '']
    df_output.loc[2] = ['Table Size (MB)', '', int_size_mb, '', '', '']
    int_table_row = 2
    for i in range(int_cols):
        str_column_name = df_input.columns.values[i]
        str_column_type = df_input.dtypes.values[i]
        int_not_null = df_input[str_column_name].count()
        int_null = sum( pd.isnull(df_input[str_column_name]) )
        flt_percent_not_null = round(int_not_null / flt_rows * 100, 1)
        flt_percent_null = round(100 - flt_percent_not_null, 1)
        int_table_row += 1
        df_output.loc[int_table_row] = [str_column_name, str_column_type, int_not_null, int_null, flt_percent_not_null, flt_percent_null]

    if bln_output_csv:
        df_output.to_csv(str_filename)
        print ('Dataframe information output created in file: ' + str_filename)
        return None
    return df_output

def check_numeric_var(str_question, int_groups):
    # created by darryldias 3jul2018  
    #print(df_output.iloc[3][2])
    flt_min = application_all[str_question].min()
    flt_max = application_all[str_question].max()
    flt_range = flt_max - flt_min 
    flt_interval = flt_range / int_groups 
    df_output = pd.DataFrame(columns=['interval', 'value', 'count', 'percent', 'code1', 'code2'])

    int_total = application_all[ (application_all[str_question] <= flt_max) ][str_question].count()
    for i in range(0, int_groups + 1):
        flt_curr_interval = i * flt_interval
        flt_value = flt_min + flt_curr_interval
        int_count = application_all[ (application_all[str_question] <= flt_value) ][str_question].count()
        flt_percent = int_count /  int_total * 100.0
        str_code_value = "{0:.6f}".format(flt_value)
        str_code1 = "if row['" + str_question + "'] <= " + str_code_value + ":"
        str_code2 = "return '(x to " + str_code_value + "]'"
        df_output.loc[i] = [flt_curr_interval, flt_value, int_count, flt_percent, str_code1, str_code2]

    return df_output

def find_file(str_input_variable):
    df_file_info = pd.read_csv('../input/dd15-files/file_information2.csv')
    int_rows = df_file_info.shape[0]
    for i in range(int_rows):
        str_dataset = df_file_info.iloc[i][1]
        str_file = df_file_info.iloc[i][2]
        str_column = df_file_info.iloc[i][3]
        if str_column == str_input_variable:
            str_file = str_file.replace('train', '')
            return [str_dataset, str_file]

def load_variable_data(str_variable):
    lst_file_info = find_file(str_variable)
    df_temp = pd.read_csv( '../input/' + lst_file_info[0] + '/train' + lst_file_info[1] )
    df_temp2 = pd.read_csv('../input/dd15-files/train_target.csv')
    df_temp = pd.merge(df_temp, df_temp2, how='left', on=['KId'])
    df_temp['train_or_test'] = 1
    df_temp2 = pd.read_csv( '../input/' + lst_file_info[0] + '/test' + lst_file_info[1] )
    df_temp2['train_or_test'] = 2
    df_temp = pd.concat([df_temp, df_temp2], sort=False)
    df_temp['overall'] = 1
    return df_temp

def show_folder_items(str_folder):
    # darryldias 8jan2019
    df_return = pd.DataFrame(columns=['Folder', 'Item'])
    lst_items = sorted( os.listdir(str_folder) )
    int_row = 0
    for str_item in lst_items:
        int_row += 1
        df_return.loc[int_row] = [str_folder, str_item]
    
    return df_return


# In[ ]:


def check_for_new_data():
    df_return = pd.DataFrame(columns=['new data check'])
    df_users = pd.read_csv('../input/meta-kaggle/Users.csv', usecols=['Id'])
    int_db_rows = df_users.shape[0]
    df_return.loc[1] = ['database rows: '+ str(int_db_rows)]

    df_stats_01 = pd.read_csv('../input/dd16s-files/stats_01.csv')
    df_stats_01 = df_stats_01[ df_stats_01['Variable']=='RegisterDateOverall' ]
    int_01_rows = df_stats_01.iloc[0][2] 
    #print('01 rows:', int_01_rows)
    int_stats_rows = int_01_rows
    df_return.loc[2] = ['stats rows count: ' + str(int_stats_rows)]
    
    if int_db_rows == int_stats_rows:
        df_return.loc[3] = ['no updating required']
    else:
        int_diff = int_db_rows - int_stats_rows 
        df_return.loc[3] = ['updating required - there are ' + str(int_diff) + ' missing rows']
    return df_return

def get_month_description1 (int_input_month):
    if int_input_month == 1 :   
        return 'January'
    elif int_input_month == 2 :   
        return 'February'
    elif int_input_month == 3 :   
        return 'March'
    elif int_input_month == 4 :   
        return 'April'
    elif int_input_month == 5 :   
        return 'May'
    elif int_input_month == 6 :   
        return 'June'
    elif int_input_month == 7 :   
        return 'July'
    elif int_input_month == 8 :   
        return 'August'
    elif int_input_month == 9 :   
        return 'September'
    elif int_input_month == 10 :   
        return 'October'
    elif int_input_month == 11 :   
        return 'November'
    elif int_input_month == 12 :   
        return 'December'
    else:
        return 'Unknown'

def get_month_description2 (int_input_month):
    return get_month_description1(int_input_month)[:3]

def ym_d1 (row, str_input_column):
    int_ymd = row[str_input_column]
    str_ymd = str(int_ymd)
    int_m = int(str_ymd[4:6])
    int_d = int(str_ymd[6:])
    str_return = get_month_description2(int_m) + ' ' + str(int_d)
    return str_return

def ym_d2 (row, str_input_column):
    int_ym = row[str_input_column]
    str_ym = str(int_ym)
    int_m = int(str_ym[4:6])
    str_return = get_month_description2(int_m) 
    return str_return

def show_information(str_message):
    df_return = pd.DataFrame(columns=['information'])
    df_return.loc[1] = [ str_message ]
    display(df_return)   

def get_y_str_1 (int_ym):
    str_input = str(int_ym)
    return str_input[:4]

def get_stats_count(str_input_variable, int_input_value):
    df_temp2 = df_stats[ (df_stats['Variable'] == str_input_variable) & (df_stats['Value'] == int_input_value) ]
    #display( df_temp2.head() )
    return df_temp2.iloc[0][2]


# ## register date counts

# In[ ]:


df_stats = pd.read_csv('../input/dd16s-files/stats_01.csv')
df_info = pd.DataFrame(columns=['information'])

# current month
df_temp2 = df_stats[ (df_stats['Variable'] == 'RegisterDateYM_C1') & (df_stats['Value'] > 0) ]
int_ym_current = df_temp2.iloc[0][1]
int_ym_current_count = df_temp2.iloc[0][2]
str_ym_current = str(df_temp2.iloc[0][1])
str_m_current = str_ym_current[4:] 
str_m_current_desc = get_month_description1 ( int(str_m_current) )
df_info.loc[1] = [ 'the count for the current month ' + str_m_current_desc + ' is ' + str(int_ym_current_count) ]

# overall
df_temp2 = df_stats[ df_stats['Variable'] == 'RegisterDateOverall' ]
int_overall_count = df_temp2.iloc[0][2]
df_info.loc[2] = [ 'the overall count since 2010 is ' + str(int_overall_count) ]

# last 14 days daily average
df_temp3 = df_stats[ (df_stats['Variable'] == 'RegisterDateYMD_C1') & (df_stats['Value'] > 0) ]
int_value_max = df_temp3['Value'].max()
df_temp3 = df_stats[ (df_stats['Variable'] == 'RegisterDateYMD_C1') & (df_stats['Value'] > 0) & (df_stats['Value'] < int_value_max) ]
int_mean = round( df_temp3['Count'].mean() )
df_info.loc[3] = [ 'the daily average for the last 14 days is ' + str(int_mean) ]

# 12 months prior
str_variable = 'RegisterDateYM_C2'
df_tempf = df_stats[ (df_stats['Variable'] == str_variable) ]
int_ym_12mprior = df_tempf['Value'].min()
str_y_12mprior = get_y_str_1(int_ym_12mprior)
int_ym_12mprior_count = get_stats_count(str_variable, int_ym_12mprior)
int_ym_current_count_estimate = get_stats_count(str_variable, int_ym_current)

df_info.loc[4] = [ 'for ' + str(str_m_current_desc) + ' ' + str(str_y_12mprior) + ' the count is ' + str(int_ym_12mprior_count) ]
df_info.loc[5] = [ 'the estimate for the whole current month is ' + str(int_ym_current_count_estimate) ]

df_info


# ### rolling 12 month average
# #### uses estimate for the whole current month

# In[ ]:


df_temp2 = pd.read_csv('../input/dd16s-files/stats_01.csv')
df_temp2 = df_temp2[ (df_temp2['Variable'] == 'RegisterDateYM_C5') ]
df_temp2['Value_D1'] = df_temp2.apply(ym_d2, axis=1, str_input_column='Value')
df_temp2.head(20)


# In[ ]:


plot_temp = df_temp2.plot(x='Value_D1', y='Count', figsize=(10, 5), legend=None)


# ### last 15 days counts
# #### the most recent day's count is usually a partial count

# In[ ]:


df_temp3 = pd.read_csv('../input/dd16s-files/stats_01.csv')
df_temp3 = df_temp3[ (df_temp3['Variable'] == 'RegisterDateYMD_C1') & (df_temp3['Value'] > 0) ]
df_temp3['Value_D1'] = df_temp3.apply(ym_d1, axis=1, str_input_column='Value')
df_temp3.head(20)


# In[ ]:


plot_temp = df_temp3.plot(x='Value_D1', y='Count', figsize=(10, 5), legend=None)


# ### monthly counts
# #### the current month shows estimate for the whole month

# In[ ]:


df_temp2 = pd.read_csv('../input/dd16s-files/stats_01.csv')
df_temp2 = df_temp2[ (df_temp2['Variable'] == 'RegisterDateYM_C3') & (df_temp2['Value'] > 0) ]
df_temp2['Value_D1'] = df_temp2.apply(ym_d2, axis=1, str_input_column='Value')
df_temp2.head(20)


# In[ ]:


plot_temp = df_temp2.plot(x='Value_D1', y='Count', figsize=(10, 5), legend=None)


# ### yearly counts

# In[ ]:


df_temp2 = df_stats[ df_stats['Variable'] == 'RegisterDateYear' ]
df_temp2.head(20)


# In[ ]:


#df_temp2 = df_stats[ (df_stats['Variable'] == 'RegisterDateYear') & (df_stats['Value'] <= 2018) ]
df_temp2 = df_stats[ (df_stats['Variable'] == 'RegisterDateYear') ]
plot_temp = df_temp2.plot(x='Value', y='Count', figsize=(10, 5), legend=None)


# ### other information (can be ignored)

# In[ ]:


show_folder_items('../input')    


# In[ ]:


show_folder_items('../input/meta-kaggle')    


# In[ ]:


show_folder_items('../input/dd16-files')


# In[ ]:


show_folder_items('../input/dd16s-files')


# In[ ]:


check_for_new_data()


# ### source users data

# In[ ]:


df_users = pd.read_csv('../input/meta-kaggle/Users.csv')
get_dataframe_info(df_users)


# In[ ]:


df_users.sample(10)


# ### external stats data (created from the kaggle meta data)

# In[ ]:


df_stats.sample(10)


# In[ ]:


end_time_check(dat_program_start, 'overall')
df_time_check

