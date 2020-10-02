#!/usr/bin/env python
# coding: utf-8

# ## INTRODUCTION
# * I am writing this kernel from a beginner perspective. Feel free to ask anything unclear to you.
# * You can write to me also  `kumarnikhil104@gmail.com`. I will respond at my earliest.
# 

# ### Loading the modules/dependencies we would be needing

# In[ ]:


import numpy as np #linear algebra
import pandas as pd # loading the csv files and data processing
import matplotlib.pyplot as plt # visualization library
import seaborn as sns # visualization library
import plotly.plotly as py #visualization ilbrary
from plotly.offline import init_notebook_mode, iplot # plotly offline mode
init_notebook_mode(connected = True)
import plotly.graph_objs as go # plotly graphical object

import os
print(os.listdir("../input"))
import warnings
warnings.filterwarnings("ignore")
#plt.style.use('sns') # style of plots. 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows',100)
get_ipython().run_line_magic('matplotlib', 'inline')


# ### From above output we can see we have following files in out current directory:
#   * Resources.csv
#   * Schools.csv
#   * Donors.csv
#   * Donations.csv
#   * Teachers.csv
#   * Projects.csv

# ### we are gonna load all the data and see what information they contains 

# In[ ]:


resources = pd.read_csv('../input/Resources.csv',error_bad_lines = False,warn_bad_lines = False)#index_col = "Project ID")
schools = pd.read_csv('../input/Schools.csv',error_bad_lines = False, warn_bad_lines = False)# index_col="School ID")
donors = pd.read_csv('../input/Donors.csv', low_memory= False)#,index_col= "Donor ID")
donations = pd.read_csv('../input/Donations.csv')#index_col="Donation ID")
teachers = pd.read_csv('../input/Teachers.csv')#index_col="Teacher ID")
projects =  pd.read_csv('../input/Projects.csv',error_bad_lines=False,warn_bad_lines = False ,parse_dates=["Project Posted Date","Project Fully Funded Date"])#index_col="Project ID")


# In[ ]:


print("Shape of the Donors.csv dataframe is: ",resources.shape)
resources.head(10)


# In[ ]:


print("Shape of the Donors.csv dataframe is: ",schools.shape)
schools.head(10)


# In[ ]:


print("Shape of the Donors.csv dataframe is: ",donors.shape)
donors.head(10)


# #### Donations.csv contains information about the transactions i.e the donation relevant informations 

# In[ ]:


print("Shape of the Donations.csv dataframe is: ",donations.shape)
donations.head(10)


# In[ ]:


print("Shape of the Teachers.csv dataframe is: ",teachers.shape)
teachers.head(10)


# In[ ]:


print("Shape of the Donors.csv dataframe is: ",projects.shape)
projects.head(5)


# ### Now we have seen that all the dataframes have some information regarding the project and donations. So, we are gonna merge it all into one single dataframe which would have all the information.<br>Before that we are gonna do some visualization

# In[ ]:


#dd -> donors + donations
dd = donations.merge(donors, on='Donor ID', how='inner')
dd.head()


# ## Let see some visualization. 

# In[ ]:


teachers.describe()


# In[ ]:


plt.rcParams["figure.figsize"] = [12,6]
#teachers['Teacher Prefix'].plot(kind = 'bar')
sns.countplot(x='Teacher Prefix', data=teachers);


# In[ ]:


plt.rcParams["figure.figsize"] = [20,6]
donors.groupby("Donor State")['Donor State'].count().plot.bar()


# In[ ]:


plt.rcParams["figure.figsize"] = [12,6]
sns.countplot(x='School Metro Type',data = schools)


# In[ ]:


plt.rcParams["figure.figsize"] = [20,6]
#schools.groupby("School State").count().plot.bar()
schools.groupby("School State")['School District'].count().plot.bar()


# In[ ]:


temp = dd["Donor State"].value_counts()
plt.title("Top donor states")
plt.xlabel("State Name")
plt.ylabel("Count")
temp.plot.bar()


# In[ ]:


#projects['Project Current Status'].value_counts().plot.bar() # both do the same thing I like sns because of legend
sns.countplot(x='Project Current Status',data = projects,hue='Project Current Status')
#sns.despine(left=True, bottom=True)


# In[ ]:


#temp_1= sns.countplot(x='Project Resource Category',data = projects)
temp_1= sns.factorplot('Project Resource Category',data = projects, aspect=1.5, kind="count")#, color="viridis")
plt.title("Project Resource")
plt.ylabel("Count");plt.xlabel("Project Resource Category")
temp_1.set_xticklabels(rotation=90)


# In[ ]:


#projects['Project Grade Level Category'].value_counts()
sns.countplot('Project Grade Level Category',data = projects)


# In[ ]:


#projects['Project Type'].value_counts().plot.bar()
sns.countplot('Project Type',data =projects)
print(projects['Project Type'].value_counts())


# - This is great there are students who are also leading projects. 

# In[ ]:


#X["DayOfMonth"] = X["DayofMonth"].apply(lambda x: x.replace("c-", ""))
projects['Project Cost'] = projects['Project Cost'].apply(lambda x: x.replace("$",""))
projects['Project Cost'].hist( figsize=(12, 4))
projects.head()
#projects['Project Cost'].plot(kind='density')#,sharex=False, figsize=(12, 4));


# In[ ]:


#sns.distplot(projects['Project Cost'])


# ### Practice some visualization on othre dataframes yourself.  I would addd few more in future. 
# 
# ### Now we are gonna do some anlysis of data and as what kind of data we have? Are there missing values? What is trend in the data?
# 
# - We will start with the Donations dataframe to see how much amount people donate in general.

# <h3>1) Analyzing Donations Dataframe

# In[ ]:


# we are gonna use pandas describe() function for this purpose also but that return output in exponential form so I prefer using `dataframe_name['column_name'].min()/max()/std() ,etc 
#donations['Donation Amount'].describe()
print("Donation:      Minimum  |   Maximum  | Mean/average     |  Median |  Std")
print("Money donated:  ",donations['Donation Amount'].min(),"   |",donations['Donation Amount'].max(),"  |",donations['Donation Amount'].mean(),"|",donations['Donation Amount'].median(),"   |",donations['Donation Amount'].std())


# From statistics when mean > median the normal distribution gets Right skewed.<br>
# We can see that the minimum amount of donation is '$0'. Let find out how many such donations are there? Are ther NaN values?

# In[ ]:


print("Shape of the donation is: ",donations.shape)
print("--"*30)
print(donations.count())
print("--"*30)
print("No.of rows in Donation Amount which are NaN:",donations.shape[0]-donations['Donation Amount'].count())
print("% .of rows in Donation Amount which are NaN:",( (donations.shape[0]-donations['Donation Amount'].count()) / donations.shape[0])*100,"%")


# We can see that there are **NaN** values in the column Donation Amount.<br>
# **Note:** Outliers never tell the true spread or distribution of the data because,  they are influenced by outlieres and get biased towards them. We can use median in place to get the understanding how much people donate in general.

# In[ ]:


#donations['Donation Amount'].plot(kind='density', subplots= False,sharex=False,figsize =(12,6));  #layout=(1, 2), sharex=False, figsize=(12, 4));


# In[ ]:


#plt.rcParams["figure.figsize"] = [12,6]
#sns.distplot(donations['Donation Amount'].dropna(axis=0));


# ### There is one more column regarding donations in the dataframe i.e  <i>"Donation Included Optional Donation"</i>
# This is the amount which helps the organization to keep running and includes money for their proper functioning, their resources and employees salary. we would see how many people like to do it.

# In[ ]:


plt.rcParams["figure.figsize"] = [12,6]
data = donations['Donation Included Optional Donation'].value_counts()[:]
data.plot.bar()


# In[ ]:


plt.rcParams["figure.figsize"] = [12,6]
sns.countplot(x='School Metro Type',data = schools)


# We can see majority of people who decide to donate also supports the organization.
# <h3>2)Analyzing Donors Dataframeaa

# In[ ]:


print("Size of the Donors Dataframe is: ",donors.shape)
print("--"*30)
donors.info(null_counts=True)


# **We can see there are many missing values in <i>"Donor City" </i> and<i> "Donar Zip"</i>.  May be donor want to be anonymised.**<br><hr>
# Let's see how many teacher are donors.

# In[ ]:


print(donors['Donor Is Teacher'].value_counts())
print("--"*30)
print("Percentage of teachers are not donor: %f" %(1910355/donors.shape[0] *100),"%")
sns.countplot(x=donors['Donor Is Teacher'],data = donors)


# - It is visible that only 10% teachers are donors also. That raises a question<br>
# **Q)** How many Donors are unique or there are some people who keeps donating?

# In[ ]:


print("Total no. of Donors in Donors Dataframe are:",donors.shape[0])
same_donor = donations['Donor ID'].value_counts().to_frame()
print('People Donating more than once are: ',same_donor[ same_donor['Donor ID']>1].shape[0])
print(" % of people donating again are :",  (same_donor[ same_donor['Donor ID']>1].shape[0]/same_donor.shape[0] *100) )


# **Q) I'm curious to know which city peoples are most generous?** (Try plotting it by yourself, I will do it in next update. ) You will find out something interesting .

# In[ ]:





# ### Notebook Under constant upgrade. Keep visiting and cast an <i>Upvote</i>
