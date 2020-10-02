#!/usr/bin/env python
# coding: utf-8

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
pd.set_option("display.max_rows", 101)
pd.set_option("display.max_columns", 25)

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

str_text = "Please note that as I can't guarantee the accuracy of the information provided in this kernel, " +            "you use the information at your own risk. I'm only using some of the data provided in the Chicago Crime dataset. " +            "This kernel is subject to change at any time."
display(HTML('<p style="color:orange;">' + str_text + '</p>'))


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


def get_column_analysis(int_analysis, int_code):
    # created by darryldias 24jul2018 
    df_date_ym_col = pd.read_csv('../input/dd19-files/date_ym_col.csv', nrows=int_read_csv_rows)
    if int_code == 1:
        lst = df_date_ym_col['description']
        return [lst[0], lst[1], lst[2], lst[3], lst[4], lst[5], lst[6], lst[7], lst[8], lst[9], lst[10], lst[11], lst[12] ]
        #return ['overall', 'homicide yes', 'homicide no', 'may18', 'jun18', 'jul18', 'aug18', 'sep18', 'oct18', \
        #         'nov18', 'dec18', 'jan19', 'feb19', 'mar19', 'apr19']
    elif int_code == 2:
        lst = df_date_ym_col['question']
        return [lst[0], lst[1], lst[2], lst[3], lst[4], lst[5], lst[6], lst[7], lst[8], lst[9], lst[10], lst[11], lst[12] ]
        #return ['overall', 'is_homicide', 'is_homicide', 'date_ym', 'date_ym', 'date_ym', 'date_ym', 'date_ym', 'date_ym', \
        #         'date_ym', 'date_ym', 'date_ym', 'date_ym', 'date_ym', 'date_ym']
    elif int_code == 3:
        lst = df_date_ym_col['value']
        return [lst[0], lst[1], lst[2], lst[3], lst[4], lst[5], lst[6], lst[7], lst[8], lst[9], lst[10], lst[11], lst[12] ]
        #return [1, 1, 0, 201805, 201806, 201807, 201808, 201809, 201810, 201811, 201812, 201901, 201902, 201903, 201904]
    else:
        return None

def create_crosstab_type1(df_input, str_row_question, int_output_destination):
    # created by darryldias 10jun2018 - updated 27sep2018 
    # got some useful code from:
    # https://chrisalbon.com/python/data_wrangling/pandas_missing_data/
    # https://www.tutorialspoint.com/python/python_lists.htm
    # https://stackoverflow.com/questions/455612/limiting-floats-to-two-decimal-points

    if int_output_destination == 0:
        return None
    
    str_count_desc = 'count'  #get_translations_analysis_description(translations_analysis, str_language, 'special', 3)
    str_colpercent_desc = 'col percent'
    
    list_str_column_desc = get_column_analysis(1, 1)
    list_str_column_question = get_column_analysis(1, 2)
    list_str_column_category = get_column_analysis(1, 3)
    int_columns = len(list_str_column_desc)
    list_int_column_base = []
    list_flt_column_base_percent = []
    
    df_group = df_input.groupby(str_row_question).size().reset_index(name='count')
    int_rows = df_group.shape[0]

    for j in range(int_columns):
        int_count = df_input[ df_input[str_row_question].notnull() & (df_input[list_str_column_question[j]]==list_str_column_category[j]) ]                                 [list_str_column_question[j]].count()
        list_int_column_base.append(int_count)
        if int_count == 0:
            list_flt_column_base_percent.append('')
        else:
            list_flt_column_base_percent.append('100.0')
        
    list_output = []
    list_output.append('row_question')
    list_output.append('row_category')
    list_output.append('statistic')
    for k in range(1, int_columns+1):
        str_temp = 'c' + str(k)
        list_output.append(str_temp)
    df_output = pd.DataFrame(columns=list_output)

    int_row = 1
    list_output = []
    list_output.append(str_row_question)
    list_output.append('')
    list_output.append('')
    for k in range(int_columns):
        list_output.append(list_str_column_desc[k])
    df_output.loc[int_row] = list_output
    
    int_row = 2
    list_output = []
    list_output.append(str_row_question)
    list_output.append('total')
    list_output.append(str_count_desc)
    for k in range(int_columns):
        list_output.append(list_int_column_base[k])
    df_output.loc[int_row] = list_output
    
    int_row = 3
    list_output = []
    list_output.append(str_row_question)
    list_output.append('total')
    list_output.append(str_colpercent_desc)
    for k in range(int_columns):
        list_output.append(list_flt_column_base_percent[k])
    df_output.loc[int_row] = list_output

    for i in range(int_rows):
        int_row += 1
        int_count_row = int_row
        int_row += 1
        int_colpercent_row = int_row

        str_row_category = df_group.iloc[i][0]

        list_int_column_count = []
        list_flt_column_percent = []
        for j in range(int_columns):
            int_count = df_input[ (df_input[str_row_question]==str_row_category) &                                   (df_input[list_str_column_question[j]]==list_str_column_category[j]) ]                                 [list_str_column_question[j]].count()
            list_int_column_count.append(int_count)
            flt_base = float(list_int_column_base[j])
            if flt_base > 0:
                flt_percent = round(100 * int_count / flt_base,1)
                str_percent = "{0:.1f}".format(flt_percent)
            else:
                str_percent = ''
            list_flt_column_percent.append(str_percent)
        
        list_output = []
        list_output.append(str_row_question)
        list_output.append(str_row_category)
        list_output.append(str_count_desc)
        for k in range(int_columns):
            list_output.append(list_int_column_count[k])
        df_output.loc[int_count_row] = list_output
        
        list_output = []
        list_output.append(str_row_question)
        list_output.append(str_row_category)
        list_output.append(str_colpercent_desc)
        for k in range(int_columns):
            list_output.append(list_flt_column_percent[k])
        df_output.loc[int_colpercent_row] = list_output
        
    return df_output        

def get_ct_statistic2(df_input, str_row_question, str_col_question, str_col_category, str_statistic):
    # created by darryldias 17jul2018
    if str_statistic == 'total':
        int_temp = df_input[ (df_input[str_col_question] == str_col_category) ][str_row_question].isnull().count() 
    elif str_statistic == 'notnull':
        int_temp = df_input[ (df_input[str_col_question] == str_col_category) ][str_row_question].count() 
    elif str_statistic == 'null':
        int_temp = df_input[ (df_input[str_col_question] == str_col_category) ][str_row_question].isnull().sum() 
    elif str_statistic == 'mean':
        int_temp = df_input[ (df_input[str_col_question] == str_col_category) ][str_row_question].mean() 
    elif str_statistic == 'median':
        int_temp = df_input[ (df_input[str_col_question] == str_col_category) ][str_row_question].median() 
    elif str_statistic == 'minimum':
        int_temp = df_input[ (df_input[str_col_question] == str_col_category) ][str_row_question].min() 
    elif str_statistic == 'maximum':
        int_temp = df_input[ (df_input[str_col_question] == str_col_category) ][str_row_question].max() 
    else:
        int_temp = None
    return int_temp
 
def create_crosstab_type2(df_input, str_row_question, int_output_destination):
    # created by darryldias 24jul2018
    if int_output_destination == 0:
        return None

    list_str_column_desc = get_column_analysis(1, 1)
    list_str_column_question = get_column_analysis(1, 2)
    list_str_column_category = get_column_analysis(1, 3)
    int_analysis_columns = len(list_str_column_question)

    list_str_statistics = ['total', 'notnull', 'null', 'mean', 'median', 'minimum', 'maximum']
    list_str_counts = ['total', 'notnull', 'null']
    int_statistics = len(list_str_statistics)

    df_output = pd.DataFrame(columns=['row_question', 'row_category', 'statistic', 'c1', 'c2', 'c3', 'c4', 'c5'])
    int_row = 1

    list_values = []
    list_values.append(str_row_question)
    list_values.append('')
    list_values.append('')
    for j in range(int_analysis_columns):
        list_values.append(list_str_column_desc[j])
    df_output.loc[int_row] = list_values

    for i in range(int_statistics):
        str_statistic = list_str_statistics[i] 
        list_values = []
        list_values.append(str_row_question)
        if str_statistic in list_str_counts:
            list_values.append(str_statistic)
            list_values.append('count')
        else:
            list_values.append('numeric')
            list_values.append(str_statistic)
    
        for j in range(int_analysis_columns):
            str_col_question = list_str_column_question[j]
            str_col_category = list_str_column_category[j]
            num_statistic = get_ct_statistic2(df_input, str_row_question, str_col_question, str_col_category, str_statistic)
            list_values.append(num_statistic)
        int_row += 1
        df_output.loc[int_row] = list_values
    return df_output


# In[ ]:


def percent_summary_1 (row, str_input_column):
    # created by darryldias 27may2018   
    if row[str_input_column] == 0 :   
        return 'no'
    elif row[str_input_column] > 0 :
        return 'yes'
    return 'Unknown'

def get_note(id):
    df_temp = df_notes[ df_notes['id']==id ]
    return df_temp.iloc[0][1]

def get_html_table(int_table_id):
    int_column_html = 2
    df_html_tables = pd.read_csv('../input/dd19-files/html_tables.csv', nrows=int_read_csv_rows)
    df_html_tables = df_html_tables[ df_html_tables['table_id'] == int_table_id ]
    int_rows = df_html_tables.shape[0]
    str_return = ''
    for i in range(int_rows):
        str_return = str_return + df_html_tables.iloc[i][int_column_html] + '\n'
    return str_return


df_notes = pd.read_csv('../input/dd19-files/notes.csv', nrows=int_read_csv_rows)

#str_note1 = get_note(1)
#display(HTML('<h4 style="color:purple;">' + str_note1 + '</br>' + str_note2 + '</h4>'))
str_html = '<ul style="color:purple;">'
for i in [1,2,3,4,5]:
    str_note = get_note(i)
    str_html += '<li>' + str_note +'</li>'
str_html += '</ul>'
display(HTML(str_html))
display(HTML('the following table shows the number of homicides each day for the last 28 days'))
str_html_table = get_html_table(2)
display(HTML(str_html_table))


# ## homicides per day

# ### month by month

# In[ ]:


#str_note = get_note(3)
#display(HTML('<h4 style="color:purple;">' + str_note + '</h4>'))

str_html_table = get_html_table(1)
#print(str_html_table)
display(HTML(str_html_table))

df_table = pd.read_csv('../input/dd19-files/tables.csv', nrows=int_read_csv_rows)
df_table = df_table[ df_table['table_id'] == 1 ]

plot_temp = df_table.plot(x='column', y='value', figsize=(10, 5), legend=None)
pyplot.xlabel('month')
pyplot.ylabel('homicides per day')
pyplot.grid()


# ### week by week (rolling 4 week values)

# In[ ]:


str_html_table = get_html_table(3)
display(HTML(str_html_table))

df_table = pd.read_csv('../input/dd19-files/tables.csv', nrows=int_read_csv_rows)
df_table = df_table[ df_table['table_id'] == 3 ]

plot_temp = df_table.plot(x='column', y='value', figsize=(10, 5), legend=None)
pyplot.xlabel('week')
pyplot.ylabel('homicides per day')
pyplot.grid()


# ## homicides by month crosstabs

# In[ ]:


df_crime = pd.read_csv('../input/dd19-files/homicides.csv', nrows=int_read_csv_rows)
#df_crime = df_crime[ (df_crime['is_homicide'] == 1) ]
create_crosstab_type1(df_crime, 'day_of_week_s1d', int_current_crosstab)


# In[ ]:


create_crosstab_type1(df_crime, 'hour_s1d', int_current_crosstab)


# In[ ]:


create_crosstab_type1(df_crime, 'district_s1d', int_current_crosstab)


# In[ ]:


create_crosstab_type1(df_crime, 'arrest', int_current_crosstab)


# In[ ]:


create_crosstab_type1(df_crime, 'domestic', int_current_crosstab)


# In[ ]:


create_crosstab_type1(df_crime, 'location_description_s1d', int_current_crosstab)


# In[ ]:


create_crosstab_type1(df_crime, 'month', int_current_crosstab)


# In[ ]:


create_crosstab_type1(df_crime, 'day', int_current_crosstab)


# ## homicides by district crosstabs
# * these crosstabs can be found in the relevant htm file in the dd19_files dataset
# * also included in the dataset are the homicides by month tables

# ## overall crime by month crosstabs
# * these crosstabs can now be found in the relevant htm file in the dd19_files dataset
# * there are only a few tables included for checking purposes

# In[ ]:


df_crime = pd.read_csv('../input/dd19-files/crime.csv', nrows=int_read_csv_rows)
#create_crosstab_type1(df_crime, 'overall', int_important_crosstab)
#create_crosstab_type1(df_crime, 'year', int_current_crosstab)
#create_crosstab_type1(df_crime, 'month', int_current_crosstab)
#create_crosstab_type1(df_crime, 'day', int_current_crosstab)
#create_crosstab_type1(df_crime, 'primary_type_s1d', int_current_crosstab)
#create_crosstab_type1(df_crime, 'district_s1d', int_current_crosstab)


# ## datasets

# In[ ]:


# a bug in the system - looks like it is fixed 25jun2019
show_folder_items('../input/dd19-files')


# In[ ]:


df_crime.sample(10)


# In[ ]:


get_dataframe_info(df_crime)


# ## other notes
# * community area tables are included in the htm crosstab files
# * for the htm crosstab files in the dd19_files dataset, if you click on a file then click the download icon the htm file should open in a new tab (and then you can copy / paste into Excel / Sheets for example)
# * district names were obtained from [https://home.chicagopolice.org/community/districts/](https://home.chicagopolice.org/community/districts/)
# * community area names were obtained from [https://home.chicagopolice.org/community/community-map/](https://home.chicagopolice.org/community/community-map/)
# * I am running a bigquery extraction (and some other data processing) in a private kernel prior to running this public kernel.

# In[ ]:


#create_crosstab_type1(df_crime, 'location_description', int_important_crosstab)
end_time_check(dat_program_start, 'overall')
df_time_check

