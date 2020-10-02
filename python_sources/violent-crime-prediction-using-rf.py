#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import pylab as plt
from datetime import datetime
#Parse the dates as datetime instead of an object
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
training_data = pd.read_csv("../input/train.csv",   parse_dates=['Dates'], date_parser=dateparse)
training_data.head()


# In[ ]:


#Selects only the violent crimes from the list of crimes
training_data = (training_data.loc[training_data['Category'].isin(['ASSAULT','ROBBERY','SEX OFFENSES FORCIBLE'])])
training_data.head()


# In[ ]:


#Make new columns for each year, month, day, hour and minute since a datetime is too unique and
#is not useful for a randomforest classifier
training_data['Year'] = training_data['Dates'].dt.year
training_data['Month'] = training_data['Dates'].dt.month
training_data['Day'] = training_data['Dates'].dt.day
training_data['Hour'] = training_data['Dates'].dt.hour
training_data['Minute'] = training_data['Dates'].dt.minute
training_data.head()


# In[ ]:


#Check to see what are object types in the data so we can change them into an integer for
#the RF classifier
training_data.dtypes[training_data.dtypes.map(lambda x: x == 'object')]


# In[ ]:


#We are going to drop the columns that we will not be given in the testing data as well as the address
#The address is too specific so we will get rid of it.
training_data = training_data.drop('Descript',1)
training_data = training_data.drop('Resolution',1)
training_data = training_data.drop('Address',1)
training_data.head()


# In[ ]:


#Drop the 'Dates' column since we are using the split up version of it
training_data = training_data.drop('Dates',1)
training_data.head()


# In[ ]:


#Since the PdDistric is a object we will be switching it for an integer
#In order to do this we need to get all the uniqe PdDistrict's in that column
PdDistrict = sorted(training_data['PdDistrict'].unique())
#We now need to map then to a dict so we can switch them out in the next step
PdDistrict_mapping = dict(zip(PdDistrict, range(0, len(PdDistrict) + 1)))
PdDistrict_mapping


# In[ ]:


#We create a new column for which we use the mapped version of the PdDistrict and use the type int
training_data['PdDistrict_Val'] = training_data['PdDistrict'].map(PdDistrict_mapping).astype(int)
training_data.head()


# In[ ]:


#If we were to do the same thing for the DayofWeek we would get a strange looking list because
#if we were to sort if in alphbetical order it wouldn't look the same
#We need to create a variable that has the list sorted the way we want it to look, then sort the
#training data the way we want it to look.
s = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday","Sunday"]
DayOfWeek = sorted((training_data['DayOfWeek'].unique()), key=s.index)
print(DayOfWeek)
#Then we do the same thing that we did for the PdDistrict for the DayofWeek
DayOfWeek_mapping = dict(zip(DayOfWeek, range(1, len(DayOfWeek) + 1)))
print(DayOfWeek_mapping)
#We create a new column for which we use the mapped version of the DayOfWeek and use the type int
training_data['DayOfWeek_Val'] = training_data['DayOfWeek'].map(DayOfWeek_mapping).astype(int)
training_data.head()


# In[ ]:


#Check to see what objects are left, so we know what we need to change, and we see that Category is left
training_data.dtypes[training_data.dtypes.map(lambda x: x == 'object')]


# In[ ]:


#We are going to map Category the same way we mapped PdDistrict
Category = sorted(training_data['Category'].unique())
Category_mapping = dict(zip(Category, range(0, len(Category) + 1)))
##We create a new column for which we use the mapped version of the Category and use the type int
training_data['Category_val'] = training_data['Category'].map(Category_mapping).astype(int)
training_data.head()


# In[ ]:


#Now we will drop the columns that we changed since we no longer need them
training_data = training_data.drop('DayOfWeek',1)
training_data = training_data.drop('PdDistrict',1)
training_data = training_data.drop('Category',1)
training_data.head()


# In[ ]:


#It is easier to have the target variable at the beginning of the data, so we will do something like
#this to move it to the front and re-print the dataframe to see if it worked
cols = training_data.columns.tolist()
cols = cols[-1:] + cols[:-1]
training_data = training_data[cols]
training_data.head()


# In[ ]:


#We will now import the Random Forest classifier and set the number of estimators to 100
#and the number of processors that will be used to -1, -1 by default uses every processor that your
#computer has which makes the classifier much faster than just using one
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, n_jobs = -1)


# In[ ]:


#If you don't have the hardware to process the entire dataset with this part you can split it and use
#whatever percent you want to use by switching out the 1 for any decimal
msk = np.random.rand(len(training_data)) < 1 #I used 1 for 100% of the data
train = training_data[msk]
#Checks the size of the training set
train.shape


# In[ ]:


# Training data features, skip the first column 'Category'
train_features = train.ix[:, 1:]

# 'Category' column values
train_target = train.ix[:, 0]


# In[ ]:


#Run the model, score it, then print out the accuracy of the Random Forests,
#this may take some time depending on how good your hardware is
clf = clf.fit(train_features, train_target)
score = clf.score(train_features, train_target)
"Mean accuracy of Random Forest: {0}".format(score)


# In[ ]:


testing_data = pd.read_csv("../input/test.csv",   parse_dates=['Dates'], date_parser=dateparse)
testing_data.head()


# In[ ]:





# In[ ]:


#The next couple of steps will be the exact same as the training data so I will pick up my notes
#when we do something new
PdDistrict = sorted(testing_data['PdDistrict'].unique())
PdDistrict_mapping = dict(zip(PdDistrict, range(0, len(PdDistrict) + 1)))
PdDistrict_mapping


# In[ ]:


testing_data['PdDistrict_Val'] = testing_data['PdDistrict'].map(PdDistrict_mapping).astype(int)
testing_data.head()


# In[ ]:


s = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday","Sunday"]
DayOfWeek = sorted((testing_data['DayOfWeek'].unique()), key=s.index)
print(DayOfWeek)
DayOfWeek_mapping = dict(zip(DayOfWeek, range(1, len(DayOfWeek) + 1)))
testing_data['DayOfWeek_Val'] = testing_data['DayOfWeek'].map(DayOfWeek_mapping).astype(int)
testing_data.head()


# In[ ]:


testing_data = testing_data.drop('Dates',1)
testing_data = testing_data.drop('DayOfWeek',1)
testing_data = testing_data.drop('PdDistrict',1)
testing_data = testing_data.drop('Address',1)
testing_data.head()


# In[ ]:


#When I first started this my computer would freeze when I ran the testing dataset, I was given this script
#to split it out into chunks for the model
#Test_x is the information that we will be using, we are skipping the ID since that will not be useful
test_x = testing_data.ix[:, 1:]
def chunks(l, n):
    """ Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


# In[ ]:


#Create a new Pandas DataFrame for the results of our prediction to go and use the function Chucks
test_result = pd.DataFrame()
for chunk in chunks(test_x.index, 10000):
    test_data = test_x.ix[chunk]
    test_result = pd.concat([test_result, pd.DataFrame(clf.predict(test_data))])
    #This last line prints a period everyime the for loop starts over giving you a nice little progress
    #bar so you know your computer is actually doing something
    print(end= ". ")


# In[ ]:


#Send the dataframe to a set of values as it is easier to add to a dataframe than it is to
#add a dataframe to another dataframe
test_result_list = test_result.values
testing_data['Predicted Category Numeric'] = test_result_list
testing_data.head()


# In[ ]:


#Now we need to reverse the numeric transforations that we did so we ca get the categorical data back
#so we can make some sense of the prediction
inv_map_Category = {v: k for k, v in Category_mapping.items()}
inv_map_DayOfWeek = {v: k for k, v in DayOfWeek_mapping.items()}
inv_map_PdDistrict = {v: k for k, v in PdDistrict_mapping.items()}
testing_data['PdDistrict'] = testing_data['PdDistrict_Val'].map(inv_map_PdDistrict)
testing_data['DayOfWeek'] = testing_data['DayOfWeek_Val'].map(inv_map_DayOfWeek)
testing_data['Predicted Category'] = testing_data['Predicted Category Numeric'].map(inv_map_Category)
testing_data.head()


# In[ ]:


#Drop the numeric versions of the categorical data that we transformed
testing_data = testing_data.drop('Predicted Category Numeric',1)
testing_data = testing_data.drop('PdDistrict_Val',1)
testing_data = testing_data.drop('DayOfWeek_Val',1)
testing_data.head()


# In[ ]:


testing_data.to_csv('predicted.csv', index=False)

