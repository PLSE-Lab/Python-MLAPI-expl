# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt;

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# Import .csv file
foodFactData = pd.read_csv('../input/FoodFacts.csv',low_memory=False);

# Get top 10 countries and update
foodFactData.countries=foodFactData.countries.str.lower();
foodFactData.loc[foodFactData["countries"] == "united kingdom", "countries"] = 'england';
foodFactData.loc[foodFactData["countries"] == "en:uk", "countries"] = 'england';
foodFactData.loc[foodFactData["countries"] == "us", "countries"] = 'usa';
foodFactData.loc[foodFactData["countries"] == "en:us", "countries"] = 'usa';
foodFactData.loc[foodFactData["countries"] == "united states", "countries"] = 'usa';

countries_for_mean = ['france','england','spain', 'usa', 'belgium'];    
    
'''
get mean of specified column for the listed counties
'''
def get_mean(col_name):
    
    mean_value_array=[];
        
    # Loop through all listed countries 
    for country in countries_for_mean:    
        mean_value = getattr(foodFactData[foodFactData.countries==country], col_name).mean();        
        mean_value_array.append(mean_value);
        
    return mean_value_array;


# get mean for fruits_vegetables_nuts_mean and sugars_100g_mean 
fruits_vegetables_nuts_mean = get_mean('fruits_vegetables_nuts_100g');
sugars_100g_mean = get_mean('sugars_100g');

# Iniitate Figure
fig = plt.figure(figsize=(20,8))
ax = fig.add_subplot(111)

width = 0.27       # the width of the bars

y_pos = np.arange(len(countries_for_mean))

# add bar 1
bar1 = ax.bar(y_pos, fruits_vegetables_nuts_mean, width, color='b')

# add bar2
bar2 = ax.bar(y_pos+width, sugars_100g_mean, width, color='g')

plt.title('Average total fruits vegetables nuts content per 100g vs Average sugar per 100g')
plt.xticks(y_pos, countries_for_mean)
plt.ylabel('fruits vegetables nuts / 100g & sugar per 100g') 
plt.legend((bar1[0], bar2[0]), ('Average total fruits vegetables nuts content per 100g', 'Average sugar per 100g'))

plt.show()
