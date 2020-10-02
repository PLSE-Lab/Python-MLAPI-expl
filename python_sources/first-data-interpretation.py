# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
from math import log10
sns.set_style('whitegrid')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# get data csv files as a DataFrame
database_df = pd.read_csv("../input/database.csv")
# database_df.columns.values to visualize the name of every columns

# Preview of data :

# Remove some categories 
data = (database_df.drop(['Record ID', 'Agency Code', 'Agency Name', 'Agency Type', 'Crime Solved',
       'Record Source'],axis=1))

ID = database_df['Record ID'].values

data.head()

# Definition of a small function to construct a dictionnary
def list_par(x,l):
    if x not in l :
        l[x]=len(l)
    return l
    
# Definition of all dictionnaries to allow an acp on data

dic_city = {}
for city in data['City']:
    (list_par(city,dic_city))

dic_state = {}
for state in data['State']:
    (list_par(state,dic_state))
    
dic_crime_type = {}
for crime_type in data['Crime Type']:
    (list_par(crime_type,dic_crime_type))
    
dic_victim_race = {}
for victim_race in data['Victim Race']:
    (list_par(victim_race,dic_victim_race))
    
dic_victim_ethnicity = {}
for victim_ethnicity in data['Victim Ethnicity']:
    (list_par(victim_ethnicity,dic_victim_ethnicity))
    
dic_perpetrator_race = {}
for perpetrator_race in data['Perpetrator Race']:
    (list_par(perpetrator_race,dic_perpetrator_race))
    
dic_perpetrator_sex = {}
for perpetrator_sex in data['Perpetrator Sex']:
    (list_par(perpetrator_sex,dic_perpetrator_sex))
    
dic_perpetrator_ethnicity = {}
for perpetrator_ethnicity in data['Perpetrator Ethnicity']:
    (list_par(perpetrator_ethnicity,dic_perpetrator_ethnicity))
    
dic_relationship = {}
for relationship in data['Relationship']:
    (list_par(relationship,dic_relationship))

dic_weapon = {}
for weapon in data['Weapon']:
    (list_par(weapon,dic_weapon))

dic_month = {}
for month in data['Month']:
    (list_par(month,dic_month))
    
dic_sexe = {}
for sexe in data['Victim Sex']:
    (list_par(sexe,dic_sexe))
    
# Data transformation with dictionnaries 

data['Month']=data['Month'].map(dic_month)
data['Victim Sex']=data['Victim Sex'].map(dic_sexe)
data['City']=data['City'].map(dic_city)
data['State']=data['State'].map(dic_state)
data['Crime Type']=data['Crime Type'].map(dic_crime_type)
data['Victim Race']=data['Victim Race'].map(dic_victim_race)
data['Victim Ethnicity']=data['Victim Ethnicity'].map(dic_victim_ethnicity)
data['Perpetrator Race']=data['Perpetrator Race'].map(dic_perpetrator_race)
data['Perpetrator Sex']=data['Perpetrator Sex'].map(dic_perpetrator_sex)
data['Perpetrator Ethnicity']=data['Perpetrator Ethnicity'].map(dic_perpetrator_ethnicity)
data['Relationship']=data['Relationship'].map(dic_relationship)
data['Weapon']=data['Weapon'].map(dic_weapon)

# New dataframes 
data.head()

# Correlation between all datas 
data.corr()

# Visualization of the correlations, code by nirajverma
correlation = data.corr()
plt.figure(figsize=(15,15))
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')

plt.title('Correlation between different fearures')