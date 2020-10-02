# Shir Mani
import numpy as np 
import pandas as pd
from clean import *

class Control:
    """See information on the useful array for decision making"""

    @staticmethod
    def examining_values_by_col (dfs, dfs_name, col):
        """
        Prints values of each DF per column
        """
        for i in range(len(dfs)):
            print("\n" + dfs_name[i])
            if col in dfs[i].columns:
                print(dfs[i][col].value_counts())


    
    @staticmethod
    def build_columns_name_ls(dfs):
        """
        build list of columns name of many DF

        dfs: list
            [df1,df2,df3]
        """
        columns_name = []
        for i in dfs:
            i = i.columns
            for x in i:
                columns_name.append(x)

        columns_name = list(dict.fromkeys(columns_name))
        return columns_name
    
    
    @staticmethod
    def df_exam_columns_dfs(dfs, dfs_names, columns_name):
        """
         made df datasets VS columns (which columns contains every DS)
         
         dfs: list
             [df1,df2,df3]
        
        dfs: list
            list of names of dfs ["df1", "df2", "df3"]
        
        columns_df: list
             list of name of all columns without duplicate (output of Exam.build_columns_name_ls())
        """
        columns_name_value = []

        for j in columns_name:
            x =[]
            for i in dfs:
                x.append(i.keys().contains(j))
            columns_name_value.append(x)

        dict = {i:x for i,x in zip(columns_name, columns_name_value)}
        columns_df = pd.DataFrame(dict)

        columns_df.rename(index= {i:x for i,x in zip(np.arange(0,len(dfs_names)),dfs_names)}, 
                          inplace = True)
        return columns_df
    
    
    @staticmethod
    def full_common(exam_df):
        """
        Returns columns that all DATASETS have
        """
        full_common = []
        for j in exam_df.columns:
            boolyan = exam_df[j].all()
            if boolyan == True:
                full_common.append(j)
        return full_common

    
    @staticmethod 
    def check_na_all_cols(df):
        """
        Prints the names of all the columns in DATASET 
        and records how many missing values they have
        Returns a list of all columns that have missing values"""
        list_j_with_nan = []
        for j in df.columns:
            nan_num = len(df.index[df[j].isnull()])
            print( j,str(nan_num))
            if nan_num > 0:
                list_j_with_nan.append(j)
        return list_j_with_nan
    
    
    @staticmethod            
    def check_balances(df , col):
        """Prints the percentage of each category in the column"""
        value_in_col = dict.fromkeys(df[col].to_numpy())
        percent = len(df[col].to_numpy())/100

        for k in value_in_col.keys():
            len_k = len(df.index[df[col] == k])
            print(k, str(len_k/percent))
            
            
    @staticmethod
    def str_of_keys_from_dicts(ls_dicts):
        """
        Gets a list of dictionaries and returns a string of names of 
        all keys. the keys separated by ","

        input:
        ls_dicts: list
            list of dicts [dict, dict]

        return:
        keys:str
            "key, key, key"
        """
        keys = []
        for dic in ls_dicts:
            keys.append(list(dic.keys()))
        keys = Clean.organize_ls_to_str(keys)
        return keys
    
    @staticmethod
    def v():
        print(16)


