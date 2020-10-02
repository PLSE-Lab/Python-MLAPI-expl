# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

"""
This script helps to compare any 2 dataframes. The number of columns and rows need not be same. The script will sort columns and remove rows and columns that are 
unqiue to each dataframe and compare the common data set. It will also list all differences in a row and column wise format. User needs to pass 2 dataframes and
a column name that has to be common in both dataframes so that this column will be used to list differences.

User will get 4 dataframes returned from the function, first one is a pretty useful summary of differences, second is unqiue columns, third is unique rows and fourth
is a list of differences. 

A fourth optional argument (default True) is available which will save an excel report combining the results into different worksheets for an excel. This has been 
significantly useful utility script for me over last few years. Be it while comparing to data sets, after cleaning to see the summary of differences etc.

Hope this helps...
"""
import pandas as pd
import numpy as np

def compare_df(df1, df2, by, save=True):
    # Variable declarations...
    temp_list = []  # this list is used to store temporary values
    source_1 = 'df_1'
    source_2 = 'df_2'
    unique = by #assigning the value of by into a better named variable - unique

    df1.sort_index(axis=1, inplace=True)  # Sorting the data frame on column
    df1.sort_values(unique, ascending=True, inplace=True)  # Sorting the data frame on unique column
    df1.fillna('', inplace=True)  # convert NULL values to blank - this is needed for comparison
    df1 = df1.applymap(str)  # convert all values to string - this is needed for comparison

    # Refer to comments for 1st dataframe above
    df2.sort_index(axis=1, inplace=True)
    df2.sort_values(unique, ascending = True, inplace=True)
    df2.fillna('', inplace=True)
    df2 = df2.applymap(str)

    list_1 = df1[unique].tolist()   # storing all unique values in base column
    list_2 = df2[unique].tolist()   # storing all unique values in base column

    df1_header = list(df1)     # storing header to lists
    df2_header = list(df2)     # storing header to lists

    # identifying unique headers
    df1_unique_header = set(df1_header) - set(df2_header)
    df2_unique_header = set(df2_header) - set(df1_header)

    # Removing unique columns from dataframe so that same column values can be compared...
    if len(df1_unique_header) > 0:
        df1.drop(list(df1_unique_header), axis=1, inplace=True)
        df1_header = list(df1)
    if len(df2_unique_header) > 0:
        df2.drop(list(df2_unique_header), axis=1, inplace=True)
        df2_header = list(df2)

    df_comp_summary = pd.DataFrame(columns=['Item', source_1, source_2])

    df_comp_summary['Item'] = ['No: of Columns', 'No: of Rows', 'No: of Rows with matching ' + unique,\
                              'Unique Cols (Major Issue)', ' No: of Unique Rows (Major Issue)', \
                              'Duplicate Rows', 'Different rows with Duplicate ' + unique + 's',\
                              'Columns after optimizing', 'Rows after optimizing',\
                              'Comparisons: ', '', 'Failed Columns']

    df_comp_summary.at[0, source_1] = len(df1.columns)
    df_comp_summary.at[0, source_2] = len(df2.columns)
    df_comp_summary.at[1, source_1] = len(list_1)
    df_comp_summary.at[1, source_2] = len(list_2)
    df_comp_summary.at[5, source_1] = 'Unverified'
    df_comp_summary.at[5, source_2] = 'Unverified'
    df_comp_summary.at[6, source_1] = 'Unverifed'
    df_comp_summary.at[6, source_2] = 'Unverified'
    df_comp_summary.at[9, source_1] = 'Pass: '
    df_comp_summary.at[9, source_2] = 'Fail: '
    df_comp_summary.at[10, source_1] = ''
    df_comp_summary.at[10, source_2] = ''
    df_comp_summary.at[11, source_1] = 'Total Rows'
    df_comp_summary.at[11, source_2] = 'Mismatch #'

    df1_diff = df1[~df1[unique].isin(list_2)]
    df2_diff = df2[~df2[unique].isin(list_1)]

    # Identifying unique items in source and destination
    df_unique_rows = pd.DataFrame(index=np.arange(max(len(df1_diff), len(df2_diff))), columns=[source_1, source_2])
    df_unique_rows[source_1] = pd.Series(df1_diff[unique].tolist())
    df_unique_rows[source_2] = pd.Series(df2_diff[unique].tolist())
    df_unique_rows.insert(loc=0, column='Sl No', value=df_unique_rows.index+1)

    # Identifying duplicate columns in source and destination
    df_unique_cols = pd.DataFrame(index=np.arange(max(len(df1_unique_header), len(df2_unique_header))), \
                                  columns=[source_1, source_2])
    df_unique_cols[source_1] = pd.Series(list(df1_unique_header))
    df_unique_cols[source_2] = pd.Series(list(df2_unique_header))
    df_unique_cols.insert(loc=0, column='Sl No', value=df_unique_cols.index + 1)

    df_comp_summary.at[3, source_1] = len(df1_unique_header)
    df_comp_summary.at[3, source_2] = len(df2_unique_header)

    df_comp_summary.at[4, source_1] = len(pd.Series(df1_diff[unique].tolist()))
    df_comp_summary.at[4, source_2] = len(pd.Series(df2_diff[unique].tolist()))

    # Removing unique rows in each data frame
    df1 = df1[df1[unique].isin(list_2)]
    df2 = df2[df2[unique].isin(list_1)]

    df_comp_summary.at[2, source_1] = len(df1)
    df_comp_summary.at[2, source_2] = len(df2)

    dup_counter = 0 #counter for flagging duplicates
    if len(df1) != len(df2):  # remove duplicate rows if number of rows in both data frames are not the same
        row_dup = df1.duplicated(keep='last')
        df_dup = df1[row_dup]
        df_dup.to_csv('Dup_All_Rows_'+source_1+'.csv', index=False)
        df_comp_summary.at[5, source_1] = len(df_dup)
        if len(df_dup) > 0:
            dup_counter += len(df_dup)

        row_dup = df2.duplicated(keep='last')
        df_dup = df2[row_dup]
        df_dup.to_csv('Dup_All_Rows_'+source_2+'.csv', index=False)
        df_comp_summary.at[5, source_2] = len(df_dup)
        if len(df_dup) > 0:
            dup_counter += len(df_dup)

        df1.drop_duplicates(keep='last', inplace=True)
        df2.drop_duplicates(keep='last', inplace=True)

    # remove rows having duplicate ID if number of rows in both data frames are still not the same
    if len(df1) != len(df2):
        row_dup = df1.duplicated(subset=unique, keep='last')
        df_dup = df1[row_dup]
        #df_dup.to_csv('Dup_'+unique+'_'+source_1+'.csv', index=False)
        df_comp_summary.at[6, source_1] = len(df_dup)
        if len(df_dup) > 0:
            dup_counter += len(df_dup)
        row_dup = df2.duplicated(subset=unique, keep='last')
        df_dup = df2[row_dup]
       # df_dup.to_csv('Dup_'+unique+'_'+source_2+'.csv', index=False)
        df_comp_summary.at[6, source_2] = len(df_dup)
        if len(df_dup) > 0:
            dup_counter += len(df_dup)

        df1.drop_duplicates(subset=unique, keep='last', inplace=True)
        df2.drop_duplicates(subset=unique, keep='last', inplace=True)

    df1.reset_index(drop=True, inplace=True)
    df2.reset_index(drop=True, inplace=True)

    df_comp_summary.at[7, source_1] = len(df1.columns)
    df_comp_summary.at[7, source_2] = len(df2.columns)
    df_comp_summary.at[8, source_1] = len(df1)
    df_comp_summary.at[8, source_2] = len(df2)

    diff_counter = 0
    difference_list = []
    columns = ['Col No', 'Col Name', 'Row No', source_1 + ' Val', source_2 + ' Val', unique]

    if df1.equals(df2):
        None    # this will avoid looping through dataframes if they are the same
    else:  # Looping through dataframes to find and list differences
        for i in range (0, len(df1.columns)):
            df_to_db3_column_data = df1[list(df1)[i]].tolist()
            df_latest_sql_column_data = df2[list(df2)[i]].tolist()
            for j in range (0,len(df_to_db3_column_data)):
                if df_to_db3_column_data[j] != df_latest_sql_column_data[j]:
                    difference_list.append([i + 1, df1_header[i], j + 1, df_to_db3_column_data[j],\
                                            df_latest_sql_column_data[j], df1.loc[j, unique]])
                    diff_counter += 1

    df_diff = pd.DataFrame(difference_list,columns=columns)

    if diff_counter != 0:
        df_diff.drop(df_diff[df_diff[source_1 + ' Val'] == df_diff[source_2 + ' Val']].index, inplace=True)
        df_diff.reset_index(drop=True, inplace=True)
        df_diff.insert(loc=0, column='Sl No', value=df_diff.index+1)

    df_comp_summary.at[9, 'Item'] = 'Comparisons: ' + str(len(df1.columns) * len(df1))
    df_comp_summary.at[9, source_1] = 'Pass: ' + str((len(df1.columns) * len(df1)) - diff_counter)
    df_comp_summary.at[9, source_2] = 'Fail: ' + str(diff_counter)

    failed_cols = pd.Series(df_diff['Col Name'].unique()).tolist()

    for i in failed_cols:
        temp_list.append(i)
        temp_list.append(len(df1))
        temp_list.append((df_diff['Col Name'] == i).sum())
        if temp_list[1] == temp_list[2]:
            temp_list[2] = 'All'
        df_comp_summary.loc[len(df_comp_summary)] = temp_list
        temp_list.clear()

    if save == True:
        with pd.ExcelWriter('Result_Analysis_ops_comparison.xlsx') as writer:
            try:
                df_comp_summary.to_excel(writer, 'Summary', index=False)
                df_unique_rows.to_excel(writer, 'Unique_' + unique + '_list', index = False)
                df_unique_cols.to_excel(writer, 'Unique_Columns', index=False)
                df_diff.to_excel(writer, 'Mismatch_in_sorted_dfs', index=False)
            except:
                print('Cannot Save Report: Please check access permissions')

    return df_comp_summary, df_unique_rows, df_unique_cols, df_diff

# Loading files to data frame
#df_1 = pd.read_csv('1.csv') #sample file has been uploaded
#df_2 = pd.read_csv('2.csv') #sample file has been uploaded
#df_compare, df_unique_r, df_unique_c, df_differences = compare_df(df_1, df_2, by='ID', save=True)


# Any results you write to the current directory are saved as output.