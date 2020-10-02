#!/usr/bin/env python
# coding: utf-8

# Predicting Incidences
# =================
# 
# Let's turn this into a classification/regression task. Let's predict insidents based on demographic attributes. Once we manage to achieve a reasonably useful model, we can ask more questions like, what does the model actually do? What are the important factor for the decision? And hence we can investigate better fairness.
# 
# In order to get to classification/regression, we'll need to collect data about locations, circumstances, demographic attributes to the neighbourhood, etc. Then we'll use Machince Learning and Data Mining, and see where this takes us.

# In[ ]:


import pandas as pd
import os
from IPython import display
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
#import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


def read_metadata(file):
    return pd.read_csv(file, index_col=0, header=None)
    
def read_data(file):
    return pd.read_csv(file, index_col=[0, 1, 2], header=[0,1], na_values=['(X)', '-', '**'])

def read_prepped(file):
    return pd.read_csv(file, header=[0,1], na_values=['-'])

def ingnore_DS_Store(directory):
    return filter(lambda f: f != '_DS_Store', os.listdir(directory))

def collect_info_for_dep (dept_dir):
    """
    This function collects the '.csv' files into pandas dataframes.
    The return value is a hash where the keys refer to the original file names.
    """
    base_dir = "../input/cpe-data/{}".format(dept_dir)
    data_directories = list(filter(lambda f: f.endswith("_data"), os.listdir(base_dir)))
    info = {'dept' : dept_dir}
    assert len(data_directories) == 1, "found {} data directories".format(len(data_directories))
    for dd in data_directories:
        directory = "{}/{}".format(base_dir, dd)
        dd_directories = ingnore_DS_Store(directory)
        #print(dd_directories)
        for ddd in dd_directories:
            ddd_directory = "{}/{}".format(directory, ddd)
            files = list(ingnore_DS_Store(ddd_directory))
            #print(files)
            assert len(files) == 2, "found {} files in {}".format(len(files), directory)
            full_file_names = ["{}/{}".format(ddd_directory, file) for file in files]
            dataframes = [read_metadata(file) if file.endswith('_metadata.csv') else read_data(file) for file in full_file_names]
            info[ddd] = dict(zip(files, dataframes))
    prepped_files = list(filter(lambda f: f.endswith("_prepped.csv"), os.listdir(base_dir)))
    for pf in prepped_files:
        info[pf] = read_prepped("{}/{}".format(base_dir, pf))
    return info


# In[ ]:


department_names = [
#    'Dept_11-00091',
#    'Dept_23-00089',
    'Dept_35-00103',
    'Dept_37-00027',
    'Dept_37-00049',
#    'Dept_49-00009',
]

departments = {dep: collect_info_for_dep(dep) for dep in department_names}


# In[ ]:


def investigate_dept(dept):
    print(dept['dept'])
    print('=' * 20)
    print(dept.keys())


# In[ ]:


for dep in departments.keys():
    investigate_dept(departments[dep])
    print()


# In[ ]:


prepped_dfs = [departments['Dept_35-00103']['35-00103_UOF-OIS-P_prepped.csv'], departments['Dept_37-00027']['37-00027_UOF-P_2014-2016_prepped.csv'], departments['Dept_37-00049']['37-00049_UOF-P_2016_prepped.csv']]


# In[ ]:


print(prepped_dfs[0].shape)
print(prepped_dfs[1].shape)
print(prepped_dfs[2].shape)


# Let's first try to build a model on the commmon fields. We can later consider other models.

# In[ ]:


from functools import reduce

columns = [list(zip(*pre.columns))[0] for pre in prepped_dfs]
common_columns = reduce(lambda x, y: list(set(x).intersection(y)), columns)


# In[ ]:


common_columns


# In[ ]:


# rearrange a little

common_columns = [
 'INCIDENT_DATE',
 'LOCATION_LONGITUDE',
 'LOCATION_LATITUDE',
 'LOCATION_FULL_STREET_ADDRESS_OR_INTERSECTION',
 'SUBJECT_GENDER',
 'SUBJECT_RACE',
 'SUBJECT_INJURY_TYPE',
]


# In[ ]:


prepped_dfs[1].head()


# In[ ]:


class TransformAndSelectForDept_35_00103(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
        
    def fit(self, X, y = None):
        #print("TransformAndSelectForDept_35_00103")
        return self
    
    def transform(self, X, y = None):
        columns = common_columns
        ret_df = X[columns].copy()
        ret_df.columns = columns
        ret_df['SUBJECT_GENDER'] = ret_df['SUBJECT_GENDER'].map({'Male': 'M', 'Female': 'F'})
        return ret_df


# In[ ]:


class TransformAndSelectForDept_37_00027(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
        
    def fit(self, X, y = None):
        #print("TransformAndSelectForDept_37_00027")
        return self
    
    def transform(self, X, y = None):
        columns1 = [
         'INCIDENT_DATE',
        ]
        ret_df = pd.DataFrame()
        ret_df[('Y_COORDINATE', 'Y-Coordinate')] = - X[('Y_COORDINATE', 'Y-Coordinate')] / 100000.0        
        ret_df[('Y_COORDINATE', 'X-Coordinate')] = X[('Y_COORDINATE', 'X-Coordinate')] / 100000.0
        columns2 = [
         'LOCATION_FULL_STREET_ADDRESS_OR_INTERSECTION',
         'SUBJECT_GENDER',
         'SUBJECT_RACE',
         'SUBJECT_INJURY_TYPE',
        ]
        ret_df = pd.concat([X[columns1], ret_df, X[columns2]], axis=1)
        ret_df.columns = common_columns
        return ret_df


# In[ ]:


class TransformAndSelectForDept_37_00049(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
        
    def fit(self, X, y = None):
        #print("TransformAndSelectForDept_37_00049")
        return self
    
    def transform(self, X, y = None):
        columns = common_columns
        ret_df = X[columns].copy()
        ret_df.columns = columns
        ret_df['SUBJECT_GENDER'] = ret_df['SUBJECT_GENDER'].map({'Male': 'M', 'Female': 'F'})
        return ret_df


# In[ ]:


transformations = [
    TransformAndSelectForDept_35_00103(),
    TransformAndSelectForDept_37_00027(),
    TransformAndSelectForDept_37_00049()
]

dfs = [trans.fit_transform(df1) for df1, trans in zip(prepped_dfs, transformations)]


# In[ ]:


for d in dfs:
    print(d.shape)


# In[ ]:


df = pd.concat(dfs)


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


df['INCIDENT_DATE'] = pd.to_datetime(df['INCIDENT_DATE'])
dti = pd.DatetimeIndex(df['INCIDENT_DATE'])
df['year'] = dti.year
df['month'] = dti.month
df['dayofweek'] = dti.dayofweek
df['dayofyear'] = dti.dayofyear


# In[ ]:


df = df.dropna()

lb_gender = LabelEncoder()
lb_race = LabelEncoder()
lb_injury_type = LabelEncoder()

df["subject_gender_code"] = lb_gender.fit_transform(df["SUBJECT_GENDER"])
df['subject_race_code'] = lb_race.fit_transform(df["SUBJECT_RACE"]) 
df['subject_injury_type_code'] = lb_injury_type.fit_transform(df['SUBJECT_INJURY_TYPE'])


# In[ ]:


df.info()


# In[ ]:


columns_for_prediction = [
    'LOCATION_LONGITUDE',
    'LOCATION_LATITUDE',
    'year',              
    'month',             
    'dayofweek',         
    'dayofyear',         
    'subject_gender_code',
    'subject_race_code',  
    'subject_injury_type_code',    
]


# In[ ]:


len(columns_for_prediction)


# In[ ]:


# display large dataframes in an html iframe
def ldf_display(df, lines=500):
    txt = ("<iframe " +
           "srcdoc='" + df.head(lines).to_html() + "' " +
           "width=1000 height=500>" +
           "</iframe>")

    return display.HTML(txt)


# In[ ]:


ldf_display(df, lines=20)


# In[ ]:


for col in df.columns:
    print(col)
    print('=' * 20)
    print(df[col].value_counts())


# In[ ]:


prepped_dfs[0]['SUBJECT_GENDER']['INDIVIDUAL_GENDER'].value_counts()


# In[ ]:


prepped_dfs[1]['SUBJECT_GENDER']['Subject Sex'].value_counts()


# In[ ]:


prepped_dfs[2]['SUBJECT_GENDER']['CitSex'].value_counts()


# In[ ]:


#from pandas.tools.plotting import scatter_matrix
from pandas.plotting import scatter_matrix

scatter_matrix(df[columns_for_prediction], figsize=(20, 20), alpha=0.2, diagonal='kde')
plt.show()


# Status as of this point:
# 
# * I note only few main "subject_injury_type"s. Maybe that can be used as a good candidate for "classification" (collect all other types to "OTHER").
# * I handled missing values very lightly (removed them). Maybe worth looking into those (re: LON/LAT). Also need to verify how I have obtained some of the LON/LAT, is it valid (divided by 10000 and took counter intuitively Y as LON, and X as LAT?).
# * Still did not take values from the statistic / demographics of the locations.
# * Another type of model that I can consider, is to ignore the "subject_injury_type" and just focus on what is the likelihood of the subject to be of a specific race/age, given the population there. Ex. if a region has 55% bright skin people and 45% dark skin people, why 80% of the cases are with dark skin subjects? TODO: think if direction interesting.
# ..

# In[ ]:




