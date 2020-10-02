#!/usr/bin/env python
# coding: utf-8

# # DonorsChoose EDA
# 
# The following is an exploratory data analysis of DonorsChoose dataset. It will consist of basic explroation and 10 questions looking for answers from data. It is always a work in progress...
# - <a href='#0'>Basic Exploration</a>
# - <a href='#Q1'>Question 1: How donations are distributed among donors?</a>
# - <a href='#Q2'>Question 2: What are the biggest projects that the largest donors donate?</a>
# - <a href='#Q3'>Question 3: Are there seasonality pattern for donations?</a>
# - <a href='#Q4'>Question 4: What are the geographical breakdown of donations?</a>
# - <a href='#Q5'>Question 5: Are there any 'home bias' on donations?</a>
# - <a href='#Q6'>Question 6: Are there any special characteristics in one-time donors?</a>
# - <a href='#Q7'>Question 7: What are the projects receiving donations from the most donors?</a>
# - <a href='#Q8'>Question 8: What kind of resources are easier/more difficult to get fully funded?</a>
# - <a href='#Q9'>Question 9: Apple vs Google vs Microsoft: Which computers are more popular among requests and donations?</a>
# - <a href='#Q10'>Question 10: What are the favorite project types for top 10 donors?</a>
# 
# Stay tuned for more updates!

# ## <a id='0'>Basic Exploration</a>

# In[1]:


# Load packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings("ignore")
print(os.listdir("../input"))


# In[2]:


resources = pd.read_csv('../input/Resources.csv', error_bad_lines=False, warn_bad_lines=False)
schools = pd.read_csv('../input/Schools.csv', error_bad_lines=False)
donors = pd.read_csv('../input/Donors.csv')
donations = pd.read_csv('../input/Donations.csv', parse_dates=['Donation Received Date'])
# teachers = pd.read_csv('../input/Teachers.csv', error_bad_lines=False, parse_dates=['Teacher First Project Posted Date'])
projects = pd.read_csv('../input/Projects.csv', error_bad_lines=False, warn_bad_lines=False, parse_dates=["Project Posted Date","Project Fully Funded Date"])
# print (resources.shape, schools.shape, donors.shape, donations.shape, teachers.shape, projects.shape)
print (resources.shape, schools.shape, donors.shape, donations.shape, projects.shape)


# In[3]:


resources.sample(2)


# In[4]:


schools.sample(2)


# In[5]:


donors.sample(2)


# In[6]:


donors['Donor ID'].nunique()


# In[7]:


donations.sample(2)


# In[8]:


donations['Donor ID'].nunique()


# There are total of about 2 million donors. Note that the number of unique donor ID is different in "donations" table (2.02M) and "donors" table (2.12M).

# In[9]:


# Top 3 donations
donations.sort_values(['Donation Amount'], ascending=False).head(3).loc[:,['Project ID','Donation Amount','Donation Received Date']]


# In[10]:


print ('Earliest donation: ', donations['Donation Received Date'].min())
print ('Latest donation: ', donations['Donation Received Date'].max())


# In[11]:


projects.sample(2)


# In[12]:


projects['Project Type'].value_counts()


# Most projects are teacher-led.

# In[13]:


projects['Project Current Status'].value_counts()


# In[14]:


projects['Project Grade Level Category'].value_counts()


# In[15]:


projects['Project Resource Category'].value_counts(ascending=True).plot(kind='barh', color='lawngreen', alpha=0.8, figsize=(8,5));


# In[16]:


projects['Project Subject Category Tree'].value_counts().reset_index().head(10)


# The "Project Subject Category Tree" column has an interesting pattern that each project may belong to 2 or more categories, separated by comma.

# ## <a id='Q1'>Question 1: How donations are distributed among donors?</a>

# In[17]:


by_donor = donations.groupby(['Donor ID'])['Donation Amount'].agg([np.sum, np.mean, np.size]).reset_index()

plt.figure(figsize=(10,15))
plt.subplot(311)
by_donor.sort_values(['sum'], ascending=False).head(50)['sum'].plot(kind='bar', color='c')
plt.title('Top 50 Donors by Total Donation Amount', fontsize=14)
plt.xticks([])
plt.subplot(312)
by_donor.sort_values(['mean'], ascending=False).head(50)['mean'].plot(kind='bar', color='darkcyan')
plt.title('Top 50 Donors by Mean Donation Amount', fontsize=14)
plt.xticks([])
plt.subplot(313)
by_donor.sort_values(['size'], ascending=False).head(50)['size'].plot(kind='bar', color='cyan')
plt.title('Top 50 Donors by No. of Donations', fontsize=14)
plt.xticks([])
plt.show()


# In[18]:


by_donor = by_donor.sort_values(['sum'], ascending=False)
print ('Largest donor by amount: ', by_donor.iloc[0,1].round(0))
print ('2nd largest donor: ', by_donor.iloc[1,1].round(0))
print ('3nd largest donor: ', by_donor.iloc[2,1].round(0))
print ('No. of donors donate above $1,000,000: ', sum(by_donor['sum']>1000000))
print ('No. of donors donate above $100,000: ', sum(by_donor['sum']>100000))
print ('No. of donors donate above $10,000: ', sum(by_donor['sum']>10000))
print ('No. of donors donate above $1,000: ', sum(by_donor['sum']>1000))
print ('No. of donors donate above $100: ', sum(by_donor['sum']>100))
print ('No. of donors donate $100 or below: ', sum(by_donor['sum']<=100))
print ('No. of donors donate $10 or below: ', sum(by_donor['sum']<=10))


# In[19]:


by_donor = by_donor.sort_values(['mean'],ascending=False)
print ('Largest donor by mean amount: ', by_donor.iloc[0,2].round(0))
print ('2nd largest donor: ', by_donor.iloc[1,2].round(0))
print ('3nd largest donor: ', by_donor.iloc[2,2].round(0))
print ('No. of donors donate above $10,000 on average: ', sum(by_donor['mean']>10000))
print ('No. of donors donate above $1,000 on average: ', sum(by_donor['mean']>1000))
print ('No. of donors donate above $500 on average: ', sum(by_donor['mean']>500))
print ('No. of donors donate above $100 on average: ', sum(by_donor['mean']>100))
print ('No. of donors donate above $50 on average: ', sum(by_donor['mean']>50))
print ('No. of donors donate above $10 on average: ', sum(by_donor['mean']>10))
print ('No. of donors donate $10 or below: ', sum(by_donor['mean']<=10))


# In[20]:


by_donor = by_donor.sort_values(['size'], ascending=False)
print ('Largest donor by no. of donations made: ', by_donor.iloc[0,3])
print ('2nd largest donor: ', by_donor.iloc[1,3])
print ('3nd largest donor: ', by_donor.iloc[2,3])
print ('No. of donors who donate more than 10000 times: ', sum(by_donor['size']>10000))
print ('No. of donors who donate more than 1000 times: ', sum(by_donor['size']>1000))
print ('No. of donors who donate more than 100 times: ', sum(by_donor['size']>100))
print ('No. of donors who donate more than 50 times: ', sum(by_donor['size']>50))
print ('No. of donors who donate more than 10 times: ', sum(by_donor['size']>10))
print ('No. of donors who donate 10 times or below: ', sum(by_donor['size']<=10))
print ('No. of donors who donate only once: ', sum(by_donor['size']==1))


# Findings:
# - 3 donors made more than \$1 million donation in total; 1.6 million donors donated \$100 or less;
# - 12 donors donated more than \$10,000 on average donation; 1.3 million donors donated \$10 to \$50;
# - 3 donors donated more than 10,000 times; 1.47 million donors donated only once.

# In[25]:


def top_donor(function, position):
    top_donor = donations[donations['Donor ID']==by_donor.sort_values([function], ascending=False).iloc[position-1,0]]
    return top_donor

top_donor('sum', 1)['Donation Amount'].describe()


# The donor who donated the largest amount in total made 10515 donations, with mean amount \$179 and median \$71.  His largest donation made is \$13,336

# In[22]:


top_donor('mean', 1)['Donation Amount'].describe()


# The donor with largest average donation amount donated only once with amount \$31,856.6.

# In[23]:


top_donor('size', 1)['Donation Amount'].describe()


# With 18,035 donations, the most frequent donor donated \$2.06 on average in each donation, and more than half of them are \$1.

# ## <a id='Q2'>Question 2: What are the biggest projects that the largest donors donate??</a>

# Firstly we look at the donor who donates the largest total amount:

# In[95]:


top1 = pd.merge(top_donor('sum', 1), projects, on='Project ID', how='left')
plt.figure(figsize=(10,12))
plt.subplot(211)
top1.groupby('Project Title').sum()['Donation Amount'].sort_values(ascending=False).head(15).plot(kind='barh', color='orange')
plt.title('Top 15 Projects by Largest Donor by Total Amount')
plt.xlabel('Donation Amount')

plt.subplot(212)
_ = top1.groupby('Project Resource Category').sum()['Donation Amount'].sort_values(ascending=False)
_.plot(kind='barh', color='purple')
plt.title('Resources Breakdown by Largest Donor by Total Amount')
plt.xlabel('Donation Amount')

plt.show()


# As the donor with largest average donation amount donates to only 1 project, we just locate what it is below:

# In[29]:


top1 = pd.merge(top_donor('mean', 1), projects, on='Project ID', how='left')
top1[['Project ID','Donation Amount','Project Title', 'Project Need Statement', 'Project Subject Category Tree','Project Resource Category','Project Cost']]


# The project is about Chromebooks. This donation made up of 42.5% of total project cost.
# 
# Then we look at the donor who makes donates most frequently:

# In[30]:


top1 = pd.merge(top_donor('size', 1), projects, on='Project ID', how='left')
plt.figure(figsize=(10,12))

plt.subplot(211)
top1.groupby('Project Title').sum()['Donation Amount'].sort_values(ascending=False).head(15).plot(kind='barh', color='orange')
plt.title('Top 15 Projects by the Most Frequent Donor')
plt.xlabel('Donation Amount')

plt.subplot(212)
_ = top1.groupby('Project Resource Category').sum()['Donation Amount'].sort_values(ascending=False)
_.plot(kind='barh', color='purple')
plt.title('Resources Breakdown by the Most Frequent Donor')
plt.xlabel('Donation Amount')
plt.show()


# Quite a lot is about art and music, while the largest one is donated related to Hurricane Harvey.

# ## <a id='Q3'>Question 3: Are there seasonality pattern for donations?</a>

# We look at monthly donation data from 2013-2017 which whole year data is available.

# In[31]:


ts = donations.loc[:,['Donation Received Date', 'Donation Amount']]
ts.set_index('Donation Received Date', inplace=True)
ts = ts[(ts.index>='2013-01-01') & (ts.index<'2018-01-01') ]
ts['month'] = ts.index.month
ts.groupby('month').sum()['Donation Amount'].plot(kind='barh', color='dodgerblue', figsize=(8,5))
plt.xlabel('Donation Amount')
plt.show()


# Total donation is the lowest in **June**, where summer vacation is approaching; and highest in **December** (as Christmas gift?) and **August** (when schools are about to open).

# ## <a id='Q4'>Question 4: What are the geographical breakdown of donations?</a>

# In[32]:


donors = pd.merge(donors, by_donor, on='Donor ID', how='outer')

by_state = donors.groupby(['Donor State'])['sum','size'].agg(np.sum).reset_index()
by_city = donors.groupby(['Donor City'])['sum','size'].agg(np.sum).reset_index()


# In[33]:


plt.figure(figsize=(16,6))
plt.subplot(121)
by_state = by_state.sort_values('sum', ascending=False)
sns.barplot(x=by_state['sum'].head(20)/1000000, y=by_state['Donor State'].head(20), palette = 'summer')
plt.title('Top 20 States by Donation Amount', fontsize=13)
plt.xlabel('Amount M')
plt.subplot(122)
by_city = by_city.sort_values('sum', ascending=False)
sns.barplot(x=by_city['sum'].head(20)/1000000, y=by_city['Donor City'].head(20), palette = 'spring')
plt.title('Top 20 Cities by Donation Amount', fontsize=13)
plt.xlabel('Amount M')
plt.show()


# In[34]:


plt.figure(figsize=(16,6))
plt.subplot(121)
by_state = by_state.sort_values('size', ascending=False)
sns.barplot(x=by_state['size'].head(20), y=by_state['Donor State'].head(20), palette = 'summer')
plt.title('Top 20 States by Donation Counts', fontsize=13)
plt.xlabel('Count')
plt.subplot(122)
by_city = by_city.sort_values('size', ascending=False)
sns.barplot(x=by_city['size'].head(20), y=by_city['Donor City'].head(20), palette = 'spring')
plt.title('Top 20 Cities by Donation Counts', fontsize=13)
plt.xlabel('Count')
plt.show()


# ## <a id='Q5'>Question 5: Are there any 'home bias' on donations?</a>
# Here we examine whether donors tend to donate to schools in their cities or states

# In[35]:


donations = pd.merge(donations, donors.loc[:,['Donor ID','Donor City', 'Donor State']], 
                on='Donor ID', how='left')
donations = donations.merge(projects.loc[:,['Project ID','School ID','Project Grade Level Category','Project Resource Category']], on='Project ID', how='left')
donations = donations.merge(schools.loc[:,['School ID','School State','School City', 'School Name']], on='School ID', how='left')
donations['same_state'] = (donations['Donor State'] == donations['School State'])*1
donations['same_city'] = (donations['Donor City'] == donations['School City'])*1


# In[36]:


pd.pivot_table(donations, values='Donation Amount', index='same_state', aggfunc='sum').div(sum(donations['Donation Amount']))*100


# The majority (71.5%) of the donations are towards schools in the same state as the donors.
# 
# For donations to the home state, how much donation is towards the home city?

# In[37]:


_ = donations.loc[donations['same_state']==1,:]
pd.pivot_table(_, values='Donation Amount', index='same_city', aggfunc='sum').div(sum(_['Donation Amount']))*100


# More than half (61.2%) of the donations are not to the same city as the donor.

# ## <a id='Q6'>Question 6: Are there any special characteristics in one-time donors?</a>
# What are the difference between donors who donate only once, and those who donate multiple times?

# In[38]:


pd.options.display.float_format = '{:.2f}'.format

donations = donations.merge(by_donor.loc[:,['Donor ID','size']], on='Donor ID', how='left')
donations = donations.merge(projects.loc[:,['Project ID','Project Cost']], on='Project ID', how='left')
donations['once'] = (donations['size']==1)*1
_ = pd.concat([donations.loc[donations['once']==0,'Donation Amount'].describe(),
           donations.loc[donations['once']==1,'Donation Amount'].describe()],axis=1)
_.columns=['multiple','single']
_


# Findings:
# - The median donation amount from one-time donors is the same as that from multiple-time donors. Both are \$25;
# - The upper quartile is also the same at \$50;
# - Mean donation amount from one-time donors is lower (\$53.41 vs \$63.99 for multiple-time donors)
# 
# For each donation, what is its percentage of project cost? How is it different between one-time donors and multiple-time donrs?

# In[39]:


pd.options.display.float_format = '{:.4f}'.format

donations['per_cost']=donations['Donation Amount']/donations['Project Cost']
_ = pd.concat([donations.loc[donations['once']==0,'per_cost'].describe(),
           donations.loc[donations['once']==1,'per_cost'].describe()],axis=1)
_.columns=['multiple','single']
_


# Findings:
# - On average, one-time donors donate 8.29% of project cost, vs 11.65% for multiple-time donors;
# - Median and upper quartile donation (as % of project cost) are lower for one-time donors;
# - It is interesting that some donations are even higher than total project cost.

# ## <a id='Q7'>Question 7: What are the projects receiving donations from the most donors?</a>
# We then look at the top 10 projects with the most number of donors:

# In[40]:


pd.options.display.float_format = None
_ = donations.groupby(['Project ID','Donor ID']).size().reset_index()
top10p = _.groupby(['Project ID']).size().sort_values(ascending=False).head(10).to_frame()
top10p_d = projects.merge(top10p, how='right', left_on='Project ID', right_index=True)
top10p_d = top10p_d.rename(columns={0:'donor_no'})
top10p_d = top10p_d.sort_values('donor_no',ascending=False)
top10p_d[['Project Title','Project Resource Category','Project Need Statement','donor_no','Project Cost']]


# Findings:
# - The top 3 projects attracted 846, 597 and 517 donors.
# - 5 of the top 10 projects are "Trips", while trips only made a tiny portion of all projects!
# - Two projects have the name "Living History by Making History". Looks like they are sequels
# - "Learning With Light" is a project which costed \$540.94 but had 209 donors

# ## <a id='Q8'>Question 8: What kind of resources are easier/more difficult to get fully funded?</a>
# We first look at 'Project Resource Category' in projects table:

# In[41]:


avg_success = (len(projects[projects['Project Current Status']=='Fully Funded']) - len(projects[projects['Project Current Status']=='Expired'])) / len(projects)

_ = pd.crosstab(projects['Project Resource Category'], projects['Project Current Status'], normalize='index')
_['success'] = _['Fully Funded']-_['Expired']
_['success'].sort_values(ascending=False).plot(kind='barh', color='darkorange', figsize=(8,6))
plt.axvline(x=avg_success)
plt.title('% Fully Funded minus % Expired', fontsize=14)
plt.show()


# We define the "success ratio" as % of projects for each resource category getting fully funded minus those expired. The vertical line is the overall success ratio of 52.7%. Findings:
# - "Food, Clothing & Hygiene" has the highest success ratio of 86.2%, followed by "Sports & Exercise Equipment" (76.9%) and Musical Instruments (76.5%);
# - "Technology" has the lowest success ratio of only 34.1%, followed by "Visitors" and "Trips", with sucess ratios of 45.4% and 52.1% respectively.
# 
# Then we look into more details in resources:

# In[42]:


resources = resources.dropna(axis=0, subset=['Resource Item Name', 'Resource Quantity'], how='any')
resources = resources.merge(projects.loc[:,['Project ID','Project Current Status']], how='left', on='Project ID')
resources['Resource Total Price'] = resources['Resource Quantity'] * resources['Resource Unit Price']


# We calculate a similar "success ratio" at a resource level, and compare it with resources with a higher unit price (above \$500), to see if a higher unit price of resource will be negative to the ability of the project to get fully funded.

# In[43]:


avg_success2 = (resources.groupby('Project Current Status').size()['Fully Funded'] - resources.groupby('Project Current Status').size()['Expired'])/len(resources)
_ = resources[resources['Resource Unit Price']>500]
high_success = (_.groupby('Project Current Status').size()['Fully Funded'] - _.groupby('Project Current Status').size()['Expired'])/len(_)
print ('Success ratio of all resources: {:.4f}'.format(avg_success2))
print ('Success ratio of high unit price resources: {:.4f}'.format(high_success))


# Compared with success ratio of 52.8% for overall resources, expensive items (above \$500) have a significantly lower success ratios of 14.35%.

# ## <a id='Q9'>Question 9: Apple vs Google vs Microsoft: Which computers are more popular among requests and donations?</a>
# 
# We are trying the compare three brands of notebooks or tablets:
# - Apple's iPad and Macbook;
# - Google's Chromebook, and;
# - Microsoft's Surface
# 
# To see which ones appear more in resource requests, and how much of them got successfully funded. We apply a screen of unit price above \$100 as some items are actually accessories (e.g. protective cases):

# In[44]:


d = {}
def create_keys(brand):
    d[brand] = {'No. of Requests': len(_), 
                'Quantity Requested': sum(_['Resource Quantity']), 
                'Total Amount Requested': sum(_['Resource Total Price']), 
                'Total Amount Funded': sum(_.loc[_['Project Current Status']=='Fully Funded','Resource Total Price'])}

cond_p = resources['Resource Unit Price']>100
# Apple
cond1 = resources['Resource Item Name'].str.contains('ipad')
cond2 = resources['Resource Item Name'].str.contains('macbook')
_ = resources[(cond1 | cond2) & cond_p]
create_keys('Apple')
# Google
cond1 = resources['Resource Item Name'].str.contains('chromebook')
_ = resources[cond1 & cond_p]
create_keys('Google')
# Microsoft
cond1 = resources['Resource Item Name'].str.contains('microsoft')
cond2 = resources['Resource Item Name'].str.contains('surface')
_ = resources[cond1 & cond2 & cond_p]
create_keys('Microsoft')


# In[45]:


pd.options.display.float_format = '{:.1f}'.format
df = pd.DataFrame.from_dict(d, orient='index')
df['Funded %'] = df['Total Amount Funded'] / df['Total Amount Requested'] * 100
df


# Findings:
# - iPad or Macbook appears in 127,246 requests, much higher than Chromebook (66,633) and Microsoft Surface (4,027)
# - However, the total number of computers requested are higher for Google (280K) than Apple (239K)
# - 50.9% of Chromebook requests are funded, higher than 47.0% for iPad and Macbook, and 43.6% for Microsoft Surface.

# ## <a id='Q10'>Question 10: What are the favorite project types for top 10 donors?</a>
# We intend to return a table that list the favorite Project Resource Category for each of the top 10 donors (by total amount donated). 

# In[56]:


def top_x_donor(function, position):
    x_donor_list = by_donor.sort_values([function], ascending=False).iloc[:position,0].tolist()
    top_x_donor = donations[donations['Donor ID'].isin(by_donor.sort_values([function], ascending=False).iloc[:position,0].tolist())]
    return top_x_donor


# In[89]:


def donor_table(columns):
    top_10 = top_x_donor('sum', 10)
    df = pd.DataFrame(columns=['Donor ID'])
    df['Donor ID'] = top_10['Donor ID'].unique()
    for column in columns:
        dp = top_10.groupby(['Donor ID', column])['Donation Amount'].sum().reset_index()
        max_col = dp.loc[dp.groupby(["Donor ID"])["Donation Amount"].idxmax(), ['Donor ID', column]]
        df = df.merge(max_col, how='left', on='Donor ID')
        df.rename(columns={column: 'Favorite '+column}, inplace=True)
    return df


# In[94]:


df = donor_table(['Project Grade Level Category', 'Project Resource Category', 'School State', 'School City', 'School Name'])
df


# This table can act as a very basic recommender that show donors new requests when it fits one or more of the criteria of his/her favorite in the past. For example, when new projects are from New York City or about technology or from 'Lafayette Eastside Elementary School', it can be shown to the first donor.

# That's it for now. Stay tuned!
