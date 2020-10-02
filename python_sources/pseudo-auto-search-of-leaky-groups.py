#!/usr/bin/env python
# coding: utf-8

# ## PSEUDO-AUTO SEARCH OF LEAKY GROUPS
# 
# 
# Hi All,
# 
# This kernel aims to present my personal *pseudo-automated* approach on how to find more leaky groups. I chose to write some code to perform this task because searching manually in excel the right set of ordered rows & columns caused me a couple of headaches. Also because, as engineer, i wanted to solve this problem. Said that this kernel is no ML, but basic problem solving, coding and as well as optimization, debugging and a bit of math.
# 
# At a high level my approach consists of: 
# 1. Select the value of the target of the first row of the leaky group, this because we need to start somewhere. Below i show how i choose it. You can try multiple target values and wait a bit for the results.
# 2. Starting from the target, search for all possible groups of **5 rows X 3 columns** of leaky values. This is the core of the Kernel. It consists of a brute-force search of all possible next columns/rows assuming that the target is 2 step ahead the first column. As you can guess  this is really time consuming since there are loads of possible alternatives and more than 4 nested for loops are needed. However i optimized some steps and reduced the computing time.
# 3. Expand the leaky values trying to add more columns as tails (not as heads) using the  (slightly modified) function bf_search in [giba-s-property-extended-extended-result](https://www.kaggle.com/johnfarrell/giba-s-property-extended-extended-result/notebook)
# 4. Use the leaky groups found as new features only if group found has a precision higher than a threshold (see [baseline-with-lag-select-fake-rows-dropped](https://www.kaggle.com/johnfarrell/baseline-with-lag-select-fake-rows-dropped)). I tried 0.97, 0.98 and 0.99.
# 
# Below i present 1. and 2., being my contributions.
# 
# 
# 

# ### Import libraries

# In[ ]:


import numpy as np # math
import pandas as pd # data
pd.options.display.max_columns = 999
pd.options.display.max_rows = 999
from itertools import compress # for list

import os
print(os.listdir("../input"))


# ### Import train set

# In[ ]:


# import train
train = pd.read_csv('../input/train.csv', index_col=0)
cols = list(train.columns)[1:]

# matrix of data values
matrix_train = train[cols].values


# ## 1. select first target value of  5x3 leaky group

# In my opinion, the most basic approach to select a possible first target value consists of compute in how many columns  each target value appears, then check the distribution and see if we can detect weird.

# In[ ]:


# new df
df_train = pd.DataFrame(index=train.index)
df_train['target'] = train.target.values
df_train['in_df'] = 0


# In[ ]:


get_ipython().run_cell_magic('time', '', "\nif 1==1:\n    # count the number of columns for each target value\n    for enum, idx in enumerate(df_train.index):\n        df_train.loc[idx, 'in_df'] = sum((matrix_train == df_train.target[enum]).sum(axis=0) > 0)\n\n    df_train.to_csv('df_train.csv')\n\ndf_train = pd.read_csv('df_train.csv', index_col=0)\ndf_train.head()")


# In[ ]:


df_train.sort_values(by = 'in_df', inplace = True)
df_train.hist(column='in_df',bins=50, figsize = (15,4))


# *in_df* represents the number of columns in which the target value compares. Let's see in how many columns the targets of the first set of leaky values discovered by [Giba](https://www.kaggle.com/titericz) are presented:

# In[ ]:


giba_rows = ['7862786dc','c95732596','16a02e67a','ad960f947','8adafbb52','fd0c7cfc2','a36b78ff7','e42aae1b8','0b132f2c6',
             '448efbb28','ca98b17ca','2e57ec99f','fef33cb02']

df_train.loc[giba_rows,:]


# mmh... 37 and 36 looks weird...

# Now i do something more: assuming there are hidden leaky groups, i sub select the original train considering the rows which target compares the same number of times in the columns. Then i check again in how many columns this target compares in the sub train and then i sum up all the occurrences of the target values, just to have an insight.

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nif 1==1:\n    df_analysis = pd.DataFrame(index = df_train.in_df.unique())\n    df_analysis['sum_in_sub_df'] = 0\n\n    for idx in df_analysis.index:\n\n        # print('analysing index:', idx, ' ...')\n        # choose rows\n        df_train_sub = df_train[df_train.in_df == idx]\n\n        # select rows & associated columns\n        train_sub = train.loc[list(df_train_sub.index),:]\n        train_sub['in_sub_df'] = 0\n\n        # matrix of sub df\n        matrix_sub_train = train_sub[cols].values\n\n        # count the number of columns for each target value in the train_sub\n        for enum, idx_2 in enumerate(train_sub.index):\n            train_sub.loc[idx_2, 'in_sub_df'] = sum((matrix_sub_train == train_sub.target[enum]).sum(axis=0) > 0)\n\n        df_analysis.loc[idx, 'sum_in_sub_df'] = train_sub.in_sub_df.values.sum()\n        # print('sum is: ', df_analysis.loc[idx, 'sum_in_sub_df'])\n\n    df_analysis.to_csv('df_analysis.csv')\n\ndf_analysis = pd.read_csv('df_analysis.csv', index_col=0).sort_index()")


# Let's show df_analysis:

# In[ ]:


df_analysis


# 
# We see that most are zeros, while there is a weird pattern for rows 35, 36 and 37, and really high numbers for the last rows. 36 and 37 are the same that appear in the above leakky group.

# ## 2. EXTENSIVE AUTO SEARCH FOR LEAKY GROUP VALUES

#  From the above basic analysis i select all possible first rows which has *in_df* in [35, 36, 37]. For illustration purpose i compute 37.
#  
#  I could  search starting from all possible first row, but it could require months of computation time...

# In[ ]:


# choose all possible first rows
df_train_sub = df_train[df_train.in_df == 37]
# df_train_sub = df_train[(df_train.in_df == 35) | (df_train.in_df == 36) | (df_train.in_df == 37)]

# select rows and columns
train_sub = train.loc[list(df_train_sub.index),:]
print(train_sub.shape) # 27 possible first rows


# In[ ]:


# initialize dataframe in which we are going to save the found leaky groups
df_matches = pd.DataFrame(columns=['in_df', 
                                   'first_row', 'first_row_pt2', 
                                   'second_row', 'second_row_pt2', 'third_row',
                                   'first_column', 'second_column', 'third_column'])


# Note on the above notation:
# - first_row', 'first_row_pt2',  'second_row', 'second_row_pt2', 'third_row' represents the first, second, third, fourth and fifth rows. While coding i called *second_row* the *third_row* because since the target is two step ahead, we must search for the possible third rows before the possible second rows.
# 

#  The **idea of the algorithm** is:
# 
# -> Starting from the selected row:
#  - searching for all possible third rows
#      - searching for all possible first columns
#          - searching for all possible fifth rows
#              - searching for all possible second columns
#                  - searching for all possible second & fourth rows
#                      - searching for all possible third columns
#                  

#  Below the **full search algorithm**: it takes a approximatly a day to run the search for *in_df == 37*. To show the correctness of the algo i run it on the same first row of the original leak group found by Giba.

# In[ ]:


get_ipython().run_cell_magic('time', '', "\n#for index_row_0, row_target_to_find in enumerate(list(train_sub.index)):\nfor index_row_0, row_target_to_find in enumerate(list(train_sub.index)[9:10]):\n\n    target_to_find = train.loc[row_target_to_find, 'target']\n    rows_found = train.index[np.sum((matrix_train == target_to_find), axis=1)>0]\n    \n    print('\\n')\n    print('*'*110)\n    print('first_row number {}: {}'.format(index_row_0, row_target_to_find))\n    print('num possible next rows:', len(rows_found))\n    \n    for idx_row, row in enumerate(rows_found):\n\n        print('\\n')\n        print('*'*30)\n        print('searcing for second possible row number {}: {}'.format(idx_row,row), '\\n')\n        # select row:\n        row_selected = row\n\n        # columns of first row found\n        cols_found_of_row_temp = list(compress(cols, train.loc[row_selected, cols].values == target_to_find))\n        target_to_search_second_row = train.loc[row_selected, 'target']\n        print('target_to_search_second_row:', target_to_search_second_row)\n        cols_found_of_row = []\n        for col in cols_found_of_row_temp:\n            if sum(train[col].values == target_to_search_second_row) >= 1:\n                cols_found_of_row.append((col, sum(train[col].values == target_to_search_second_row)))\n        print('total columns found:', len(cols_found_of_row))\n        print('that is/are:', cols_found_of_row)\n\n        if len(cols_found_of_row) > 0:\n            for first_column, _ in cols_found_of_row:\n                all_possible_third_rows = list(train.index[train[first_column].values == target_to_search_second_row])\n                print('all_possible_third_rows:', all_possible_third_rows)\n                \n                value_to_search = train.loc[row_target_to_find, first_column]\n                print('value_to_search:', value_to_search)\n                \n                # first set of admissible second_columns:\n                cols_both_values_temp = []\n                for idx, col in enumerate(cols):\n                    temp_1 = 1*(sum(matrix_train[:,idx] == value_to_search)>=1)\n                    if temp_1 > 0:\n                        temp_2 = 1*(sum(matrix_train[:,idx] == target_to_find)>=1)\n                        if temp_2 > 0:\n                            temp_3 = 1*(sum(matrix_train[:,idx] == target_to_search_second_row)>=1)\n                            if temp_3 > 0:\n                                cols_both_values_temp.append(col)\n                # print('possible second columns temp:', cols_both_values_temp)\n                \n                for index_third_row, third_row in enumerate(all_possible_third_rows):\n                    print('third row n {} is: {}'.format(index_third_row, third_row))\n                    target_to_search_third_row = train.loc[third_row, 'target']\n                    print('target_to_search_third_row:', target_to_search_third_row)\n\n                    # now i search for other columns containing value_to_search, target_to_find e the target of the new row!!\n                    cols_both_values = []\n                    for idx, col in enumerate(cols_both_values_temp):\n                        occurrences = 1*(sum(train[col] == target_to_search_third_row)>=1)\n                        if occurrences >= 1:\n                            cols_both_values.append(col)\n                    \n                    print('All possible second columns are:', cols_both_values)\n                    for second_column in cols_both_values:\n                        if second_column != first_column:\n\n                            # check if E4 is in previous column (D3)\n                            coeff_e4 = train.loc[row_selected, second_column]\n                            first_row_pt2_temp_0 = list(train.index[train[first_column].values == coeff_e4])\n                            first_row_pt2_temp_1 = []\n                            for temp_first_row_pt2 in first_row_pt2_temp_0:\n                                if train.loc[row_target_to_find, first_column] == train.loc[temp_first_row_pt2,second_column]:\n                                    first_row_pt2_temp_1.append(temp_first_row_pt2)\n                            \n                            #print('all possible first_row_pt2_temp_1:', first_row_pt2_temp_1)\n                            # for all possible first_row_pt2_temp_1 check wheter its target is in first and second column\n                            first_row_pt2 = []\n                            possible_second_row_pt2 = []\n                            for temp_first_row_pt2 in first_row_pt2_temp_1:\n                                target_temp = train.loc[temp_first_row_pt2, 'target']\n                                # target_temp in third row & second_column?\n                                if train.loc[third_row, second_column] == target_temp:\n                                    # target_temp in any second row pt2 & first column?\n                                    possible_second_row_pt2_temp = list(compress(list(train.index), train[first_column] == target_temp))\n                                    if len(possible_second_row_pt2_temp) >= 1:\n                                        target_temp_2 = train.loc[row_selected, first_column]\n                                        # print('target_temp_2:', target_temp_2)\n                                        possible_second_row_pt2_temp_2 = list(compress(list(train.index), train[second_column] == target_temp_2))\n                                        # intersection:\n                                        possible_second_row_pt2_per_first_row_pt2 = list(set(possible_second_row_pt2_temp).intersection(set(possible_second_row_pt2_temp_2)))\n                                        if len(possible_second_row_pt2_per_first_row_pt2) >= 1:\n                                            first_row_pt2.append(temp_first_row_pt2)\n                                            possible_second_row_pt2.append(possible_second_row_pt2_per_first_row_pt2)\n                            #print('all possible possible_second_row_pt2:', possible_second_row_pt2)\n\n                            # save all possible matches\n                            for idx_temp_first_row_pt2 ,temp_first_row_pt2 in enumerate(first_row_pt2):\n                                for temp_second_row_pt2 in list(possible_second_row_pt2[idx_temp_first_row_pt2]):\n                                    # now i search a third column!!\n                                    cols_1 = list(compress(train.columns, \n                                                           train.loc[temp_first_row_pt2,:] == train.loc[row_target_to_find, second_column]))\n                                    cols_2 = list(compress(train.columns, \n                                                           train.loc[row_selected,:] == train.loc[temp_first_row_pt2,second_column]))\n                                    cols_3 = list(compress(train.columns,\n                                                           train.loc[temp_second_row_pt2,:] == train.loc[row_selected,second_column]))\n                                    cols_4 = list(compress(train.columns,\n                                                           train.loc[third_row,:] == train.loc[temp_second_row_pt2,second_column]))\n                                    # print(cols_1,cols_2,cols_3,cols_4)\n                                    possible_third_columns = list(set(cols_1).intersection(set(cols_2)).intersection(set(cols_3)).intersection(set(cols_4)))\n                                    # print('CALCOLATO THIRD COLUMNS:',possible_third_columns)\n                                    for third_column in possible_third_columns:\n                                        # print(third_column)\n                                        df_matches.loc[df_matches.shape[0]] = [35, \n                                                                               row_target_to_find,\n                                                                               temp_first_row_pt2,\n                                                                               row_selected, \n                                                                               temp_second_row_pt2, \n                                                                               third_row, \n                                                                               first_column, \n                                                                               second_column,\n                                                                               third_column]\n                                        print('FOUND {} SOLUTION: {}'.format(df_matches.shape[0], df_matches.loc[df_matches.shape[0]-1,:].values))\n\n# save\ndf_matches.to_csv('df_matches.csv')")


# Let's show all leaky group found

# In[ ]:


for idx in range(df_matches.shape[0]):
    rows_found = list(df_matches.loc[idx,['first_row','first_row_pt2','second_row','second_row_pt2','third_row']])
    cols_found = list(['target'] + list(df_matches.loc[idx,['first_column', 'second_column', 'third_column']]))
    print('*'*50)
    print(train.loc[rows_found,cols_found])
    print('*'*50)


# As you can see the second group coincides with Giba leaky group.
# 
# Here i stress that these groups should then be extended searching for next possible columns as done in the notebook cited above and select only those with a high precision obtained comparing the leaky values and the target values.

# In[ ]:



