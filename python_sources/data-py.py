# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import numpy as np
import pandas as pd

class dataset():
    '''
    # use this class to load data and do processing on it
    '''
    def __init__(self):
        '''
        # load datasets
        '''
        dir_loc = '../input/'
        print("Read datasets...")
        self.train = pd.read_csv(dir_loc + "gender_age_train.csv", dtype = {"device_id": np.str, "age": np.int8})
        self.test = pd.read_csv(dir_loc +  "gender_age_test.csv", dtype = {"device_id": np.str, "age": np.int8})
        self.events = pd.read_csv(dir_loc + "events.csv", dtype = {"device_id": np.str})
        self.brands = pd.read_csv(dir_loc + "phone_brand_device_model.csv", dtype = {"device_id": np.str}, encoding = "UTF-8")
        self.app_events = pd.read_csv(dir_loc + "app_events.csv")
        self.app_labels = pd.read_csv(dir_loc+ "app_labels.csv")
        self.label_categories = pd.read_csv(dir_loc + "label_categories.csv")
        
    def data_wrangling(self):
        '''
        # do data_wrangling and return some transform version of train and test
        '''
        self.events['counts'] = self.events.groupby(['device_id'])['event_id'].transform('count')
        self.events_small = self.events[['device_id', 'counts']].drop_duplicates('device_id', keep='first')
        
        self.brands.drop_duplicates('device_id', keep='first', inplace=True)
        self.brands = self.map_column(self.brands, 'phone_brand')
        self.brands = self.map_column(self.brands, 'device_model')
        
        self.train = self.map_column(self.train, 'group')
        self.train = self.train.drop(['age'], axis=1)
        self.train = self.train.drop(['gender'], axis=1)
        self.train = pd.merge(self.train, self.brands, how='left', on='device_id', left_index=True)
        self.train = pd.merge(self.train, self.events_small, how='left', on='device_id', left_index=True)
        self.train.fillna(-1, inplace=True)
        
        self.test = pd.merge(self.test, self.brands, how='left', on='device_id', left_index=True)
        self.test = pd.merge(self.test, self.events_small, how='left', on='device_id', left_index=True)
        self.test.fillna(-1, inplace=True)
        

    def map_column(self, table, f):
        labels = sorted(table[f].unique())
        mappings = dict()
        for i in range(len(labels)):
            mappings[labels[i]] = i
        table = table.replace({f: mappings})
        return table
        
    def data_wrangling_v2(self, TorT):
        '''
        # Take data after loading and merge in appropriate ways...
        # Merging datasets and dropping unnecessary items
        '''
        print(TorT)
        print('.....')
        ### events
        self.TorT = self.TorT.merge(self.events, how = "left", on = "device_id")
        # del self.events
        
        ### brands
        self.TorT = self.TorT.merge(self.brands, how = "left", on = "device_id")
        # del self.brands
        self.TorT.drop("device_id", axis = 1, inplace = True)
        # self.test.drop("device_id", axis = 1, inplace = True)
        
        ### app_events
        self.TorT = self.TorT.merge(self.app_events, how = "left", on = "event_id")
        # del self.app_events
        self.TorT.drop("event_id", axis = 1, inplace = True)
        
        ### app_labels and label_categories
        self.TorT = self.TorT.merge(self.app_labels, how = "left", on = "app_id")
        self.TorT = self.TorT.merge(self.label_categories, how = "left", on = "label_id")
        # del self.app_labels, self.label_categories
        self.TorT.drop(["app_id", "label_id"], axis = 1, inplace = True)
        
        print(self.TorT.head(5))
        print(self.TorT.info())
        
def define_pipeline():
    '''
    # Create the feature pipeline along with transformer_weights
    '''
    return pipeline([])

    
d = dataset()
d.data_wrangling()
# d.data_wrangling_v2('test')
# d.data_wrangling_v2('train')

