
import pandas as pd
import os

#==============================================================================
def getCurrentDirectory():
	return os.path.dirname(os.path.abspath(__file__)) + '/'
path = getCurrentDirectory()

raw_data = pd.read_csv(path + 'USA_name_data.csv')

#==============================================================================
raw_data_females = raw_data[raw_data['gender'] == 'F']
raw_data_males = raw_data[raw_data['gender'] == 'M']
if len(raw_data_females) > len(raw_data_males):
    print("There are more female names than male")
else:
    print("There are more male names than female")
    
#==============================================================================
most_common_names = pd.crosstab(index=raw_data["name"], columns="count")  
most_common_names = most_common_names.sort_values('count', ascending=False)
most_common_names['name'] = most_common_names.index
most_common_names = most_common_names.reset_index(drop=True)
most_common_100_names = list(most_common_names[:100]['name'])

#==============================================================================
most_common_female_names = pd.crosstab(index=raw_data_females["name"], columns="count") 
most_common_female_names = most_common_female_names.sort_values('count', ascending=False)
most_common_female_names['name'] = most_common_female_names.index
most_common_female_names = most_common_female_names.reset_index(drop=True)
most_common_female_100_names = list(most_common_female_names[:100]['name'])

