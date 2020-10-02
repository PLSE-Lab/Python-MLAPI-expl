# Shir Mani

import numpy as np 
import pandas as pd

class UpdateTime:
    
    @staticmethod
    def del_str_equal_x_from_ls(ls, x):
        """
        Deletes cells with empty str in the list
        
        input:
        ls: list
            list of str
        
        return:
        ls: list
            list of str
        """
        empty = []
        for i in range(len(ls)):
            ls[i] = ls[i].strip()
            if ls[i] == x :
                empty.append(i)
        for i in empty:
            ls.remove(x)
        return ls
    
    @staticmethod
    def insertChar(string, index, date_character):
        """
        Insert a character into the string in a selected location
        """
        longi = len(string)
        string   =  string[:index] + date_character + string[index:] 
        return string
    
    @staticmethod
    def ls_i_fit_date(ls, date_character):
        """
        Fits a date format string
        DD.MM.YYYY
        
        input:
        ls: list
            list of str
        
        date_character: str
            The character you want to separate from month to year and day
            For example: "." : DD.MM.YYYY  
        
        return:
        ls: list
            list of str
        
        child: UpdateTime.insertChar
        """
        for indx in range(len(ls)):
            i = ls[indx]
            if not i[2] == date_character:
                i  = UpdateTime.insertChar(i, 2, date_character)

            if not i[5]== date_character:
                i  = UpdateTime.insertChar(i, 5, date_character)
            ls[indx] = i

        return ls
    
    @staticmethod
    def redundant_numbers_date(ls, date_character):
        for i in ls:
            ls_i  = i.split(date_character)
            print(i)
            if  not (len(ls_i[0]) == 2 and len(ls_i[2]) == 4 and len(ls_i[0]) == len(ls_i[1])): 
                return False
        
        
    @staticmethod
    def make_ls_of_str_datatime(ls):
        """
        Makes a list of strings a list of pd.datetime
        
        input:
        ls: list
            list of str
            
        return:
        ls: list
            list of pd.datetime
        """
        for i in range(len(ls)):
            ls[i] = pd.to_datetime(ls[i], dayfirst=True, errors='ignore').date()
        return ls
    
    @staticmethod
    def  time_range_extremity(ls, earliest=False):
        """
        return the latest / earliest date from a date list
        
        input:
        ls: list
         list of pd.datatime
         
        earliest: True/Fasle
            True :return the earliest date
            False :return the latest date
        
        return:
            pd.datetime
        
        """
        df = pd.DataFrame({"series":ls})
        if earliest == True:
            value = df.series.max()
        else:
            value = df.series.min()
        return value
   
    @staticmethod
    def format_datatime(dataset, input_col, output_col, indexs, date_character, character_separator, earliest=False):
        """
        Takes a string containing multiple dates from column A
        and write  the latest / earliest date in column B
        -the type date pd.datetime
        
        input:
        dataset: pd.df
        
        input_col: str
            name of column A
        
        output_col: str
            name of column B
            A ==B or A!=B
        
        indexs: pd.index
            index contain data col.notnull()
            you can drop some contain data indexs too 
        date_character:
        
        date_character: str
            The character you want to separate from month to year and day
            For example: "." : DD.MM.YYYY
            
        character_separator: str
            Character that separates the dates in a string
            for example: "-" : "DD.MM.YYYY - DD.MM.YYYY"
        
        earliest: True/Fasle
            True :return the earliest date
            False :return the latest date
            
        return: 
        indexs: pd.index
            The indexs we received without the index whose values were changed
            
        child: 
        UpdateTime.del_str_equal_x_from_ls
        UpdateTime.ls_i_fit_date
        UpdateTime.make_ls_of_str_datatime
        UpdateTime.time_range_extremity
        """
        drop = []
        indexs_error = []
        for indx in indexs:
            i = dataset.loc[indx, input_col]
            ls = i.split(character_separator)
            ls = UpdateTime.del_str_equal_x_from_ls(ls, "")
            boo = UpdateTime.redundant_numbers_date(ls, date_character)
            if boo == False:
                    indexs_error.append(indx)
            else:
                if len(ls) > 1:
                    ls = UpdateTime.make_ls_of_str_datatime(ls)
                    value = UpdateTime.time_range_extremity(ls, earliest)
                    dataset.loc[indx, output_col] = value
                    drop.append(indx)
        indexs = indexs.drop(drop)
        return indexs, indexs_error
    
    @staticmethod
    def format_datatime_normal(dataset, input_col, output_col, indexs, date_character):
        """
        
        takes a string containing one date from column A
        format to pd.datetime
        and writes in column B
        
        input:
        dataset: pd.df
        
        input_col: str
            name of column A
        
        output_col: str
            name of column B
            A ==B or A!=B
        
        indexs: pd.index
            index contain data col.notnull()
            you can drop some contain data indexs too
        
        date_character: str
            The character you want to separate from month to year and day
            For example: "." : DD.MM.YYYY
        
        return: 
        indexs: pd.index
            The indexs we received without the index whose values were changed
        
        child:
        UpdateTime.ls_i_fit_date
        UpdateTime.make_ls_of_str_datatime
        """
        indexs_error = []
        drop = []
        for indx in indexs:
            i = dataset.loc[indx,input_col] 
            i = UpdateTime.ls_i_fit_date([i], date_character)
            boo  = UpdateTime.redundant_numbers_date(i, date_character)
            if boo == False:
                indexs_error.append(indx)
            else:
                dataset.loc[indx,input_col] = UpdateTime.make_ls_of_str_datatime(i)
                drop.append(indx)
        indexs = indexs.drop(drop)
        return indexs, indexs_error
    
    @staticmethod
    def updte_time(dataset, input_col, output_col, indexs, date_character, character_separator):
        """
        Takes column A dates
        Format includes multiple dates. Not listed well
        And writes in column B
        input:
        dataset: pd.df
        
        input_col: str
            name of column A
        
        output_col: str
            name of column B
            A ==B or A!=B
        
        indexs: pd.index
            index contain data col.notnull()
            you can drop some contain data indexs too 
        date_character:
        
        date_character: str
            The character you want to separate from month to year and day
            For example: "." : DD.MM.YYYY
            
        character_separator: list 
            list of kind Characters that separates the dates in a string in col 
            for example: "-" : "DD.MM.YYYY - DD.MM.YYYY"
        
        earliest: True/Fasle
            True :return the earliest date
            False :return the latest date
            
        return: 
        indexs: pd.index
            The indexs we received without the index whose values were changed
            
        child:
        UpdateTime.format_datatime
        UpdateTime.format_datatime_normal
        """
        error = []
        for i in range(len(character_separator)):
            indexs, indexs_error = UpdateTime.format_datatime(dataset, input_col, output_col, indexs, date_character, character_separator[i])
            error.append(indexs_error)
            
        indexs, indexs_error = UpdateTime.format_datatime_normal(dataset, input_col, output_col, indexs, date_character)
        error.append(indexs_error)
        return indexs, error

    @staticmethod
    def v():
        print(11)

