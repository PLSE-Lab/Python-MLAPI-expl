#!/usr/bin/env python
# coding: utf-8

# In[31]:


# Import Packages Used for Analysi
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.


# In[ ]:


# List of CSV files we would like to import
data_frame_list = ['Resources','Schools', 'Donors', 'Donations', 'Teachers', 'Projects']

# Function that makes importing cleaner, there are some errors, skipping over with bad lines
def import_data(item: str)-> None:
    return pd.read_csv(r"../input/" + item + ".csv", error_bad_lines=False)

# importing all items from the above list
for item in data_frame_list:
    globals()[item] = import_data(item)
    


# In[23]:


# Create a function that will determine whether a donor has donated multiple times
def multi_donation_class(row):
    if row > 1:
        return 1
    else:
        return 0

# Apply Donor Multi Function
Donations['Multi_Donations'] = Donations['Donor Cart Sequence'].apply(multi_donation_class)


# In[24]:


# TODO: Join Donor Data with Whether They have donated multiple times
# TODO: Examine Multi Donors vs Non Donors
# TODO: Examine What Projects Receive Multi Donations Vs Single i.e. is someone more likely to make a second smaller loan or a large one?
# TODO: Are people more likely to donate to schools close to them or far away?
# TODO: Add nearness to donor 

# Sort Donations so that if a Donor has donated multiple times the join picks up that they have
Donations_Sorted = (Donations.sort_values(by = 'Multi_Donations', ascending = False)).copy

# Merge the Sorted Donations so that Donor information includes the fact that they have donated multiple times
Donors = Donors.merge(Donations[['Donor ID', 'Multi_Donations']], on = 'Donor ID', how = 'left')


# In[25]:


# Create dataset of just first time donations as we want to predict if the will donate again
first_time_donations = Donations[Donations['Donor Cart Sequence'] == 1]

#Drop Multi_Donations since all the labels would be 0
first_time_donations = first_time_donations.drop(columns = ['Multi_Donations'])

# adding labels to our dataset of whether donor actually donated again
first_time_donations = first_time_donations.merge(Donors[['Donor ID', 'Multi_Donations']], on = 'Donor ID', how = 'left')


# In[26]:


# Importing additional sklearn packages for SVM & Train Test Split
from sklearn import svm
from sklearn.model_selection import train_test_split

# Selecting features for analysis
X = first_time_donations['Donation Amount']

# Taking Average for filling NaN values
average_X = np.average(X)

# Fill NaN with the Average Loan Values
X = X.fillna(value = average_X)

# Reshaping the dataframe
X = X.reshape(-1,1)

# Selecting the labels
y = first_time_donations['Multi_Donations']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[30]:


clf = svm.SVC()
clf.fit(np.nan_to_num(X_train), y_train)


# In[ ]:




