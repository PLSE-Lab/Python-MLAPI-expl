import os
import sys
from glob import glob
from tqdm.notebook import tqdm
import pandas as pd
from pandas.io.json import json_normalize
import concurrent.futures
import requests

class Uncover():
    def __init__(self, verbose=False):
        #Initalize variables
        self.base_path = "/kaggle/input/uncover/UNCOVER/"
        self.all_csv_files = self.get_files(self.base_path, ".csv")
        self.files_read = []
        self.files_skipped = []
        self.df_dictionary = self.get_dfs_from_csv(self.all_csv_files)
        self.col_df = self.get_col_df(self.df_dictionary)
        self.col_binary_matrix = self.get_col_bin(self.col_df)
        self.us_covid_stats = self.get_us_covid_data()
        self.nunique_cols = 0
        self.num_shared_cols = 0
        if verbose: self.to_print()

    def get_files(self, path = "", extension = ""):
        """
        Traverse a given folder path and return all fiels of the given extension.
        """
        all_files = []
        for path, subdir, files in os.walk(path):
            for file in glob(os.path.join(path, "*{}".format(extension))):
                all_files.append(file)
        #
        return all_files

    def get_dfs_from_csv(self, file_list = []):
        """
        Iterate through a list of csv files and return a dictionary as follows:
        {Filename with .csv removed: DataFrame}
        """
        df_dict = {}
        for file_path in tqdm(file_list):
            df_name = file_path.split("/")[-1].replace('.csv', '')
            try:
                df_dict[df_name] = pd.read_csv(file_path, low_memory=False)
                #Added for tracking but can be removed
                self.files_read.append(file_path)
            except:
                self.files_skipped.append(file_path)
                pass
        
        return df_dict

    def get_col_df(self, df_dictionary = {}):
        """
        Iterate through dictionary of Dfs and make new DF as follows:
        Features->[List of DFs with that Feature]
        """
        column_dict = {}
        for name, df in df_dictionary.items():
            all_cols = list(df.columns)
            for col in all_cols:
                if col in column_dict.keys():
                    column_dict[col].append(name)
                else:
                    column_dict[col] = list([name])

        # Drop any columns not found in other DataFrames
        len_before_drop = len(column_dict)
        to_pop = []
        for col, df_list in column_dict.items():
            if len(df_list) < 2:
                to_pop.append(col)

        # Run in seperate loop as can not change size in iterator
        for col in to_pop:
            column_dict.pop(col)

        #For printing
        self.nunique_cols = len_before_drop - len(column_dict)
        self.num_shared_cols = len(column_dict)
        
        # Make DF with index of cols and a column of dfs with that feature
        col_df = pd.DataFrame(pd.Series(column_dict)).reset_index()
        col_df.columns = ["Feature", "DataFrames"]
        return col_df
    
    def get_col_bin(self, col_df=None):
        """
        Take a column dataframe and return a binary matrix as follows:
        Axis=0(X)->List of all Features
        Axis=1(Y)->List of all DFs
        
        A 1 in a cell indicates the feature is in that dataframe
        1 0 in a cell indicates the feature in not in that dataframe
        """
        
        #Explode-Make a new row for each of the values found in the DataFrames lists
        col_df_explode = col_df.explode("DataFrames")
        
        #Add present columns to keep track of which are where
        col_df_explode["present"] = 1
        
        #Pivot to binary matrix
        col_binary_matrix = col_df_explode.pivot_table(index='Feature',
                            columns='DataFrames',
                            values='present',
                            aggfunc='sum',
                            fill_value=0)
        
        return col_binary_matrix
    
    def get_dfs_by_cols(self, col_list = []):
        """
        Take a list of columns/feature values and return all dfs in the corpus...
        ...that have all of the features present.
        """
        bin_matrix_transpose = self.col_binary_matrix.T
        return list(bin_matrix_transpose[bin_matrix_transpose[col_list].isin(["1"]).all(axis=1)].index)
    
    def get_us_covid_data(self):
        """
        Returns a DF with the most recent COVID stats by us county
        """
        #Linked ot Kaggle dataset, but pull right from Github
        us_county_data_path = "/kaggle/input/us-counties-covid-19-dataset/us-counties.csv"
        us_county_url_path = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv"
        us_county_timeseries = pd.read_csv(us_county_url_path)

        #Drop counties which are null, indicates its a state summary
        us_county_timeseries = us_county_timeseries[us_county_timeseries.county != "Unknown"]

        #Fix New York City and Kansas City
        us_county_timeseries.loc[us_county_timeseries['county'] == 'New York City', 'fips'] = 36061
        us_county_timeseries.loc[us_county_timeseries['county'] == 'Kansas City', 'fips'] = 20209

        #Set FIPS to into to enable merge
        us_county_timeseries["fips"] = us_county_timeseries["fips"].astype("int")

        #Get current information
        current_date_filter = us_county_timeseries["date"] == us_county_timeseries["date"].max()
        us_county_current = us_county_timeseries[current_date_filter]

        #Add in date first seen in county
        first_occurance_filter = ~us_county_timeseries.county.duplicated("first")
        first_occurance_df = us_county_timeseries[first_occurance_filter]
        data_county_first = first_occurance_df[["county", "date"]]
        us_county_current = pd.merge(us_county_current, data_county_first, on="county")

        #Fix date names and get spreaed time
        us_county_current.rename(columns={"date_x":"current_date","date_y":"first_seen"}, inplace=True)
        us_county_current["current_date"] = pd.to_datetime(us_county_current["current_date"])
        us_county_current["first_seen"] = pd.to_datetime(us_county_current["first_seen"])
        us_county_current["days_spread"] = us_county_current["current_date"] - us_county_current["first_seen"]
        us_county_current["days_spread"] = us_county_current["days_spread"].dt.days

        #Get specific features for investigation
        us_county_current["lethality_ratio"] = us_county_current["deaths"] / us_county_current["cases"]
        us_county_current["lethality_rate"] = us_county_current["deaths"] / us_county_current["days_spread"]
        us_county_current["cases_rate"] = us_county_current["cases"] / us_county_current["days_spread"]
        return us_county_current
    
    def to_print(self):
        print("There are a total of {} csv files in this dataset.".format(len(self.all_csv_files)))
        print("Read a total of {} files into Pandas dataframes and skipped {}.".format(len(self.files_read),
                                                                                       len(self.files_skipped)))
        print("A total of {} columns are unique to only one dataframe.".format(self.nunique_cols))
        print("A total of {} columns are shared by more than one dataframe.".format(self.num_shared_cols))
        
        
    
if __name__ == '__main__':
    uncover = Uncover(verbose=True)
    print(uncover.us_covid_stats)