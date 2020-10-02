#!/usr/bin/env python
# coding: utf-8

# ## Import the necessary libraries 

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load the data into dataframe

# In[3]:


interview_df = pd.read_csv('../input/Interview.csv', delimiter=',')


# ## Understanding the data

# In[4]:


interview_df


# * Drop the columns that are not required for data analysis.

# In[5]:


interview_df.drop(['Unnamed: 23', 'Unnamed: 24', 'Unnamed: 25', 'Unnamed: 26', 'Unnamed: 27', 'Name(Cand ID)'], 
                  axis=1, inplace=True)


# In[6]:


interview_df.info()


# Change the column names into something that is more readable.

# In[7]:


interview_df.columns = ['Date', 'Client', 'Industry', 'Location', 'Position', 'Skillset', 'Interview_Type', 
                     'Gender', 'Curr_Location', 'Job_Location', 'Venue', 'Native_Loc', 'Necc_Perm', 'No_random_meet', 
                      'Call_3_hours', 'Alternative_Number', 'Printout_resume', 'Details_Clear_Landmark', 
                      'Shared_Letter', 'Exp_Attendance', 'Ob_Attendance', 'Martial_Status']
interview_df.head()


# The last row has been dropped as it only contained NaNs.

# In[8]:


interview_df.tail()
interview_df.drop(1233, inplace=True)
interview_df.info()


# ## Cleaning the data

# In[9]:


for column in interview_df.columns:
    print(column, interview_df[column].unique())
    print('-'*40)


# The following columns were dropped as the data was too messy or seemed irrelevant.
# 
# * *Skillset was dropped as it had a lot of diverse data and felt that Position attribute would be sufficient.*
# 
# * *Native location was dropped and greater importance was given to current location.*

# In[10]:


interview_df.drop(['Date', 'Skillset', 'Native_Loc', 'Alternative_Number'], axis=1, inplace=True)


# Replaced the keywords that were similar across all columns.

# In[11]:


interview_df.Client.replace(['Aon Hewitt', 'Hewitt', 'Aon hewitt Gurgaon'], 'Hewitt', inplace=True)
interview_df.Client.replace(['Standard Chartered Bank', 'Standard Chartered Bank Chennai'], 
                            'Standard Chartered', inplace=True)

interview_df.Industry.replace(['IT Services', 'IT Products and Services', 'IT'], 
                              'IT', inplace=True)

interview_df.Location.replace(['chennai', 'Chennai', 'chennai ', 'CHENNAI'], 'Chennai', inplace=True)
interview_df.Location.replace('- Cochin- ', 'Cochin', inplace=True)
interview_df.Location.replace('Gurgaonr', 'Gurgaon', inplace=True)

interview_df.Curr_Location.replace(['chennai', 'Chennai', 'chennai ', 'CHENNAI'], 'Chennai', inplace=True)
interview_df.Curr_Location.replace('- Cochin- ', 'Cochin', inplace=True)

interview_df.Job_Location.replace('- Cochin- ', 'Cochin', inplace=True)

interview_df.Venue.replace('- Cochin- ', 'Cochin', inplace=True)

interview_df.Exp_Attendance.fillna('yes', inplace=True)
interview_df.Exp_Attendance.replace(['Yes', '10.30 Am', '11:00 AM'], 'yes', inplace=True)
interview_df.Exp_Attendance.replace(['No', 'NO'], 'no', inplace=True)

interview_df.Ob_Attendance.replace(['Yes', 'yes '], 'yes', inplace=True)
interview_df.Ob_Attendance.replace(['No', 'NO', 'No ', 'no '], 'no', inplace=True)

interview_df.Printout_resume.replace('Yes', 'yes', inplace=True)
interview_df.Printout_resume.replace(['No', 'Not yet', 'na', 'Na', 'Not Yet', 
                                      'No- will take it soon'], 'no',inplace=True)

interview_df.Necc_Perm.replace(['Yes', 'yes'], 'yes', inplace=True)
interview_df.Necc_Perm.replace(['No', 'Not yet', 'NO', 'Na', 'Yet to confirm'] ,'no', inplace=True)

interview_df.Details_Clear_Landmark.replace(['Yes', 'yes'], 'yes', inplace=True)
interview_df.Details_Clear_Landmark.replace(['No', 'na', 'no', 'Na', 'No- I need to check'] ,'no', inplace=True)

interview_df.Interview_Type.replace(['Scheduled Walk In', 'Scheduled ', 'Scheduled Walk In', 'Sceduled walkin'],
                                    'Scheduled', inplace=True)
interview_df.Interview_Type.replace(['Walkin '], 'Walkin', inplace=True)

interview_df.No_random_meet.replace(['Yes', 'yes'], 'yes', inplace=True)
interview_df.No_random_meet.replace(['Na', 'No', 'Not Sure', 'cant Say', 'Not sure'] ,'no', inplace=True)

interview_df.Call_3_hours.replace(['No', 'No Dont', 'Na'], 'no', inplace=True)
interview_df.Call_3_hours.replace(['Yes', 'yes'], 'yes', inplace=True)

interview_df.Shared_Letter.replace(['Yes', 'yes'], 'yes', inplace=True)
interview_df.Shared_Letter.replace(['Havent Checked', 'No', 'Need To Check', 'Not sure', 
                                   'Yet to Check','Not Sure', 'Not yet', 'no', 'na', 'Na'], 'no', inplace=True)


# In[12]:


interview_df.info()


# ## Handling the missing data

# ### Printout of Resume
# 
# Whether the candidate has taken a printout has been converted into a binary variable.

# In[13]:


sns.countplot(x=interview_df.Printout_resume, hue=interview_df.Exp_Attendance)


# Comparitively, people who chose not to attend the interview didn't take the printout

# In[14]:


index_nan_print = interview_df['Printout_resume'][interview_df['Printout_resume'].isnull()].index
for i in index_nan_print:
    
    if interview_df.iloc[i]['Exp_Attendance'] == 'no':
        interview_df.iloc[i]['Printout_resume'] = 'no'
    else:
        interview_df.iloc[i]['Printout_resume'] = 'yes'


# ### Necessary Permissions

# In[15]:


sns.countplot(x=interview_df.Necc_Perm, hue=interview_df.Exp_Attendance)
plt.show()
sns.countplot(x=interview_df.Necc_Perm, hue=interview_df.Ob_Attendance)


# * People who were certain of not attending the interview didn't take the necessary permissions. Hence, the same was chosen while filling the missing data.

# In[16]:


index_nan_perm = interview_df['Necc_Perm'][interview_df['Necc_Perm'].isnull()].index
for i in index_nan_perm:
    
    if interview_df.iloc[i]['Exp_Attendance'] == 'no':
        interview_df.iloc[i]['Necc_Perm'] = 'no'
    else:
        interview_df.iloc[i]['Necc_Perm'] = 'yes'


# ### Landmark Details

# In[17]:


sns.countplot(x=interview_df.Details_Clear_Landmark, hue=interview_df.Exp_Attendance)
plt.show()
sns.countplot(x=interview_df.Details_Clear_Landmark, hue=interview_df.Ob_Attendance)


# The attribute was set to 'no' if the candidate was not expected to attend the interview.

# In[18]:


index_nan_details = interview_df['Details_Clear_Landmark'][interview_df['Details_Clear_Landmark'].isnull()].index
for i in index_nan_details:
    
    if (interview_df.iloc[i]['Exp_Attendance'] == 'no'):
        interview_df.iloc[i]['Details_Clear_Landmark'] = 'no'
    else:
        interview_df.iloc[i]['Details_Clear_Landmark'] = 'yes'


# **Similarly, the other three attributes were filled based on Observed Attendance and Expected Attendance.**

# ### Random Meetings

# In[19]:


sns.countplot(x=interview_df.No_random_meet, hue=interview_df.Exp_Attendance)
plt.show()
sns.countplot(x=interview_df.No_random_meet, hue=interview_df.Ob_Attendance)


# In[20]:


index_nan_details = interview_df['No_random_meet'][interview_df['No_random_meet'].isnull()].index
for i in index_nan_details:
    
    if (interview_df.iloc[i]['Ob_Attendance'] == 'no' and interview_df.iloc[i]['Exp_Attendance'] == 'no'):
        interview_df.iloc[i]['No_random_meet'] = 'no'
    else:
        interview_df.iloc[i]['No_random_meet'] = 'yes'


# ### Shared the Letter

# In[21]:


sns.countplot(x=interview_df.Shared_Letter, hue=interview_df.Exp_Attendance)
plt.show()
sns.countplot(x=interview_df.Shared_Letter, hue=interview_df.Ob_Attendance)


# In[22]:


index_nan_details = interview_df['Shared_Letter'][interview_df['Shared_Letter'].isnull()].index
for i in index_nan_details:
    
    if (interview_df.iloc[i]['Ob_Attendance'] == 'no' or interview_df.iloc[i]['Exp_Attendance'] == 'no'):
        interview_df.iloc[i]['Shared_Letter'] = 'no'
    else:
        interview_df.iloc[i]['Shared_Letter'] = 'yes'


# ### Call within 3 hours

# In[23]:


sns.countplot(x=interview_df.Call_3_hours, hue=interview_df.Exp_Attendance)
plt.show()
sns.countplot(x=interview_df.Call_3_hours, hue=interview_df.Ob_Attendance)


# Exp_Attendence column was preferred over the Observed Attendance column (Need a better method to fill this data).

# In[24]:


index_nan_details = interview_df['Call_3_hours'][interview_df['Call_3_hours'].isnull()].index
for i in index_nan_details:
    
    if (interview_df.iloc[i]['Exp_Attendance'] == 'no'):
        interview_df.iloc[i]['Call_3_hours'] = 'no'
    else:
        interview_df.iloc[i]['Call_3_hours'] = 'yes'


# In[25]:


interview_df.info()


# ## Basic Visualization
# 
# The data was visualized to further understand the relationship between the different variables

# In[26]:


f, (ax1, ax2) = plt.subplots(1,2, figsize=(15,8))
sns.countplot(x=interview_df.Exp_Attendance, ax=ax1)
ax1.set_title('Expected attendance')
sns.countplot(x=interview_df.Ob_Attendance, ax=ax2)
ax2.set_title('Observed attendance')


# In[27]:


f, (ax1, ax2) = plt.subplots(2, figsize=(15,16))
sns.countplot(x=interview_df.Position, hue=interview_df.Exp_Attendance, ax=ax1)
sns.countplot(x=interview_df.Position, hue=interview_df.Ob_Attendance, ax=ax2)


# In[28]:


f, (ax1, ax2) = plt.subplots(1,2, figsize=(15,8))
sns.countplot(x=interview_df.Job_Location, hue=interview_df.Exp_Attendance, ax=ax1)
sns.countplot(x=interview_df.Job_Location, hue=interview_df.Ob_Attendance, ax=ax2)
f, (ax1, ax2) = plt.subplots(2, figsize=(15,16))
sns.countplot(x=interview_df.Location, hue=interview_df.Client, ax=ax1)
ax1.legend(loc='right')
sns.countplot(x=interview_df.Job_Location, hue=interview_df.Client, ax=ax2)


# * Traditional trends were seen when it came to Expected and Observed attendance.
#  * Lesser people attended the interview than the expected count.
# * In this dataset, more jobs were offered at Bangalore and Chennai.
# * Standard Chartered offered the highest number of jobs at these locations as it had operations at both Chennai and Bangalore.

# In[29]:


f, (ax1, ax2) = plt.subplots(1,2, figsize=(15,8))
sns.countplot(x=interview_df.Curr_Location, hue=interview_df.Exp_Attendance, ax=ax1)
sns.countplot(x=interview_df.Job_Location, hue=interview_df.Exp_Attendance, ax=ax2)
f, (ax1, ax2) = plt.subplots(1,2, figsize=(15,8))
sns.countplot(x=interview_df.Curr_Location, hue=interview_df.Ob_Attendance, ax=ax1)
sns.countplot(x=interview_df.Job_Location, hue=interview_df.Ob_Attendance, ax=ax2)


# In[30]:


interview_df.groupby(['Curr_Location', 'Job_Location', 'Exp_Attendance', 'Ob_Attendance']).size()


# Looking at the above graphs and the groupby data:
# * People who were unsure of attending the interview had a pretty even distribution in the Observed Attendance attribute.
# * People were offered higher percentage of jobs at locations that were same as their current location or were close to their current location.

# In[31]:


f, ax1 = plt.subplots(1, figsize=(10,6))
sns.countplot(x=interview_df.Industry, ax=ax1)


# In[32]:


interview_df.Industry.value_counts()


# * BFSI industry offered the highest number of jobs with 949 interviews.

# In[33]:


f, (ax1, ax2) = plt.subplots(2, figsize=(15,16))
sns.countplot(x=interview_df.Industry, hue=interview_df.Position, ax=ax1)
sns.countplot(x=interview_df.Job_Location, hue=interview_df.Industry, ax=ax2)


# In[34]:


interview_df.groupby(['Industry', 'Position']).size()


# * There were a lot of routine jobs offered across every industry.

# In[35]:


f, (ax1, ax2) = plt.subplots(2, figsize=(15,16))
sns.countplot(x=interview_df.Interview_Type, hue=interview_df.Client, ax=ax1)
sns.countplot(x=interview_df.Position, hue=interview_df.Client, ax=ax2)


# In[36]:


interview_df.groupby(['Industry', 'Client', 'Position']).size()


# * Standard Chartered conducted quite a lot of interviews and the *Routine* position took the lion share.
# * The same was seen across most companies.

# In[37]:


f, (ax1, ax2) = plt.subplots(2, figsize=(15,16))
sns.countplot(x=interview_df.Gender, hue=interview_df.Industry, ax=ax1)
sns.countplot(x=interview_df.Gender, hue=interview_df.Position, ax=ax2)


# In[38]:


interview_df.groupby(['Gender', 'Industry', 'Position']).size()


# * The dataset contains a higher number of male candidates than female candidates. However, the distibution of jobs offered across industry was pretty similar.

# In[39]:


f , ax = plt.subplots(1, figsize=(10,8)) 
sns.countplot(x=interview_df.Gender, hue=interview_df.Martial_Status, ax=ax)


# In[40]:


interview_df.groupby(['Gender', 'Martial_Status', 'Exp_Attendance', 'Ob_Attendance']).size()


# * A higher number of female candiates were married compared to male candidates.
# * However, when it came to expected attendance and observed attendance, the disribution was pretty similar.

# In[41]:


le = LabelEncoder()
interview_df = interview_df.apply(le.fit_transform)


# In[42]:


interview_df.corr()


# In[43]:


f , ax = plt.subplots(1, figsize=(15,10)) 
sns.heatmap(interview_df.corr(), ax=ax, annot=True)


# * The heatmap shows the relationships that were protrayed earlier:
#     * Current Location, Job Location and Venue were highly correlated.
#     * Industry and Client were also correlated.
# * Necc_Perm, Call_3_hours, Printout_resume, Details_Clear_Landmark, Shared_Letter were also highly correlated.

# 
# * *The dataset does not have an even distribution and for machine learning tasks, the training set, development set and the validation set need to distributed carefully, ensuring that all three sets have enough data of positive and negative examples.*
# * *Necc_Perm, Call_3_hours, Printout_resume, Details_Clear_Landmark, Shared_Letter columns could be reduced to a single column using dimensionality reduction techniques such as PCA.*
