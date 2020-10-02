#!/usr/bin/env python
# coding: utf-8

# ##### Exploratory Data Analysis of:
# ## Microsoft, Apple, Amazon, Google, Netflix, and Facebook Workplace Reviews
# ___
# **Context:** We have over 67k workplace reviews for Google, Amazon, Facebook, Apple, Netflix, and Microsoft. The dataset is hosted on [Kaggle](https://www.kaggle.com/petersunga/google-amazon-facebook-employee-reviews) and is scraped from www.glassdoor.com.

# **Table of contents:**
# 
# Exploration
# 
# 1. [Libraries, Settings and Loading the Dataset](#Libraries)
# 1. [Variables Assessment, Data Cleaning, and Univariant Exploration](#Univariant)
# 1. [What insights and questions are we exploring?](#Context)
# 1. [Bivariate and Multivariate Explorations](#Multivariate)
# 1. [Feature Engineering](#Feature)
# 
# Explanatory Visualization
# 
# 1. [Main findings from this exploration](#findings)
# 1. [Story of our Data](#Story)
# ___

# ## Libraries, Settings and Loading the Dataset <a name="Libraries"></a>

# In[7]:


# Essential Data Analysis Ecosystem
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

# Visualizations
import matplotlib.pyplot as plt
import seaborn as sns

# Python Standard Libraries
import os  # For os file operations.
import re  # Used for data cleaning purposes.
import webbrowser  # Used to see sample reviews in glassdoor.com

# Ensures plots to be embedded inline.
get_ipython().run_line_magic('matplotlib', 'inline')

# Plot size frequently used.
two_in_row = (12, 4)  
# Style and Base color used for seaborn plots.
bcolor = sns.color_palette()[0]
sns.set(style='ticks', palette='pastel')

# Suppress warnings from final output.
import warnings
warnings.simplefilter("ignore")


# #### Load up the dataset and first look at the data

# In[8]:


dataset_path = '../input/'
df = pd.read_csv(os.path.join(dataset_path, 'employee_reviews.csv'), index_col=0)


# #### Let's look at shape of our raw data and  three  random  reviews:

# In[9]:


print('Number of rows (reviews) and columns:', df.shape)
df_samples = df.sample(3)
df_samples


# In[10]:


random_review = np.random.randint(0, df.shape[0]-1)
df.iloc[random_review] # A detailed look at a random review


# In[11]:


df.info()


# We don't see lots of missing data but that's most likely because missing data represented by string 'none'

# In[12]:


df.nunique()


# We have four specific categories of ratings and one overall rating. They should be based on a 1-star to 5-stars ratings, but there are up to 10 different vlues for those ratings that needs to be looked at and cleaned.

# #### Reviews on the glassdoor.com
# We can follow our sample scrapped reviews to www.glassdoor.com to see the actual reviews.

# In[13]:


links = df_samples['link']
print(links)
answer = input('Enter,  y  if you would like to open and see these sample reviews\` urls? ')
if answer.lower()=='y':
    [webbrowser.open(link) for link in links]


# Looking at the actual source of the data, we can see  two very likely useful peices of information in reviews are not scrapped. That is the length and type of employments. Here is 3 examples:
#  - I worked at Amazon full-time (More than 3 years)
#  - I have been working at Microsoft full-time (More than 10 years)
#  - I have been working at Google full-time (Less than a year)

# end of quick look
# ___

# ## Variables Assessment, Data Cleaning, and Univariant Exploration <a name="Univariant"></a>

# Make column names Python/Pandas friendly by changinh dashes to underscors

# In[14]:


df.columns


# In[15]:


df.columns = df.columns.str.replace('-', '_') 


# **1st Column** `company` is the company name
# 
# Companies in this data set are just names and have no order. To add more meaning for this column we will convert it to an ordinal categorical type sorted by date founded, from Microsoft to Facebook. 

# In[16]:


companies_by_founded_date = ['microsoft', 'apple', 'amazon', 'netflix', 'google', 'facebook']
company_cat = CategoricalDtype(ordered=True, categories=companies_by_founded_date)
df['company'] = df['company'].astype(company_cat)
# TEST
df['company'].values


# In[17]:


def plot_cat_counts(data=None, x=None):
    """Plot a categorical value with side by side horizantal bar and pie charts"""
    
    plt.figure(figsize=two_in_row)

    plt.subplot(1, 2, 1)
    sns.countplot(data=data, y=x, color=bcolor)
    plt.ylabel('')
    plt.xlabel('Review Counts')
    sns.despine() # remove the top and right borders


    plt.subplot(1, 2, 2)
    sorted_counts = data[x].value_counts()
    labels = sorted_counts.index

    plt.pie(sorted_counts, labels=None, 
            startangle=90, counterclock=False, wedgeprops = {'width' : 0.35})
    plt.axis('square')

    plt.legend(labels,
              title="Companies Proportions",
              loc="top left",
              bbox_to_anchor=(1, 0, .25, 1));


# In[18]:


plot_cat_counts(df, 'company')


# #### Notes:
# The disproportionate distributions of reviews between these 6 companies considering their scope of operations and year founded are consistent with our expectations. Microsoft and Apple founded in the mid-seventies, Amazon founded in 1994, Netflix with a DVD rental business model in 1997, Google in 1998 and most recently Facebook in 2004. Glassdoor, the review site itself, founded in 2007. 
# ___

# **2nd Column** `location`
# This dataset is global. As such, it may include the country's name in parenthesis, i.e., "Toronto, ON(Canada)"]. However, if the location is in the USA, then it only includes the city and state, i.e., "Los Angeles, CA" ]

# - `none` is used for missing values in this data set. This shows in this bar plot as more than 35% of locations are missing.

# In[19]:


# Replace string "none" with NaN in entire dataset.

df = df.replace('none', np.nan)
df = df.replace('None', np.nan)
df = df.replace('None.', np.nan)


# In[20]:


# Plot top 30 frequent locations
plt.figure(figsize=two_in_row)
(df['location'].value_counts().head(30) / len(df)).plot.bar();


# **Findings** 
# - **This column is not Tidy.** There are 3 values of  `city, state, country` in this one column that each should be in its own column.
# - Location values are in two different formats.
# - - For the US, country name is missing and we have City, State
# - - For other countries the format is City, State/Region if applies, and the Country name between parentheses.
# 
# ##### Functions to extract `city`, `state`, and `country` values from location column to 3 columns

# In[21]:


btween_parentheses = r'\(([^)]+)\)'  # Regular expression to get a string between parentheses

def get_country(location):
    """Extracts and returns country name from location string.
    Returns NaN if 'none'."""
    
    if pd.isnull(location):
        return np.nan
    
    not_usa = re.findall(btween_parentheses, location)
    if not_usa:
        return not_usa[0]
    else:
        return 'USA'
    

def get_state(location):
    """Extracts and returns state name (if aby) from location string.
    Returns Nan if 'none or not applicable."""
    
    if pd.isnull(location):
        return np.nan
    
    not_usa = re.findall(btween_parentheses, location)
    if not_usa:
        if ',' in location:
            return location.split(',')[1].split()[0]
        else:
            return np.nan
    else:
        return location.strip()[-2:]

    
def get_city(location):
    """Extracts and returns city name from location string.
    Returns Nan if 'none'."""
    
    if pd.isnull(location):
        return np.nan
    
    not_usa = re.findall(btween_parentheses, location)
    if not_usa:
        if ',' in location:
            return location.split(',')[0]
        else:
            return location.split()[0]
    else:
        return location.split(',')[0]    


# In[22]:


# Creating three new columns for location data
df['city'] = df['location'].apply(get_city)  # New column for the city.
df['state'] = df['location'].apply(get_state)  # New column for the State/Region.
df['country'] = df['location'].apply(get_country)  # New Column for the Country.

# Drop the untidy and no longer needed location column.
del df['location']


# In[23]:


# TEST location columns
df[['city', 'state', 'country']].sample(5)


# In[24]:


def plot_top_cats(col, top_percentage):
    """Plot members of a categorical variable that make up the top_percentage."""
    
    mask = df[col].value_counts(normalize=True).cumsum() < top_percentage
    top_items = mask[mask].index

    def group_top_itesm(x):
        if x in top_items:
            return x
        elif pd.isna(x):
            return np.nan
        else:
            return 'Other Countries'
    items = df[col].apply(group_top_itesm)

    plt.figure(figsize=two_in_row)
    sns.countplot(y=items, color=bcolor, order=items.value_counts().index)
    plt.ylabel(f'TOP {top_percentage*100}% in {col.upper()}')
    sns.despine()


# In[25]:


plot_top_cats('country', 0.90)


# In[26]:


plot_top_cats('country', 0.925)


# **Findings**
# - As expected the super majority of reviews are posted in the US and, the India is the runner up.
# - UK, Ireland, and Canada are following India.
# 
# **Notes**
# - Companies day to day operations can be very different in each country. We are focusing on employees in the USA, India, UK, Ireland, and Canada.

# In[27]:


countries_mask = (df.country == 'USA') | (df.country == 'UK') | (df.country == 'Ireland') | (df.country == 'Canada')
df = df[countries_mask]
del countries_mask
df.shape


# ___

# **3rd Column** `dates` Date review posted

# In[28]:


df['dates'] = pd.to_datetime(df['dates'], errors='coerce')  # Type Casting to date
df.sort_values(by='dates', ascending=False, inplace=True)  # Sort reviews by date
df.rename(columns={'dates': 'date_posted'}, inplace=True)


# #### Distribution of Yearly Number of Reviews 

# In[29]:


yearly = df.groupby(df['date_posted'].dt.year).size()
positions = yearly.index
plt.bar(positions, yearly.values)
plt.title(f'{df.date_posted.min()} to {df.date_posted.max()}');


# **Findings**
# - The majority of reviews posted are from 2015 to 2018.

# ___

# **4th Column** `job-title` This string includes whether the reviewer is a 'Current' or 'Former' Employee at the time of the review. If Employee posting review Anonymously There is no value for job title and it presented with Anonymous Employee.

# In[30]:


plt.figure(figsize=two_in_row)
(df['job_title'].value_counts().head(30) / len(df)).plot.bar();


# **Findings** 
# - **This column is not Tidy.** There are 3 values of  `Is reviewer is current or past employee` and `the Job Title` in this one column that each should be in its own column. If an employee is `anonymous`, Anonymous is listed instead of the job title.
# 
# ##### Functions to extract if reviewer is `current_employee`,  `anonymou`, and the employee's `job_title`.

# In[31]:


def clean_text(col):
    """Cleaning text from formatings."""
    col = col.str.strip()
    col = col.str.replace("(<br/>)", "")
    col = col.str.replace('(<a).*(>).*(</a>)', '')
    col = col.str.replace('(&amp)', '')
    col = col.str.replace('(&gt)', '')
    col = col.str.replace('(&lt)', '')
    col = col.str.replace('(\xa0)', ' ')  
    return col

df['job_title'] = clean_text(df['job_title'])


# In[32]:


df['current_emp'] = df['job_title'].apply(lambda x: True if x.split()[0] == 'Current' else False)
df['anonymous'] = df['job_title'].apply(lambda x: True if 'Anonymous' in str(x) else False)

df['job_title'] = df['job_title'].apply(lambda x: x.split('-')[1])
df['job_title'] = df['job_title'].apply(lambda x: np.nan if 'Anonymous' in str(x) else x)


# In[33]:


# Test
plt.figure(figsize=two_in_row)
(df['job_title'].value_counts().head(30) / len(df)).plot.bar();
df[['job_title', 'current_emp', 'anonymous']].sample(5)


# In[34]:


# Most popular Job Titles in entire dataset i.e. all companies combined.
df['job_title'].value_counts()[:20]


# In[35]:


## Most popular group of employee wrote review in each company
df.groupby(['company', 'job_title']).size().sort_values(ascending=False)[:25]


# In[36]:


ax = sns.countplot(df['current_emp'], hue=df['anonymous'], color=bcolor)
ax.set_xticklabels(['Past Employees', 'Current Employees'])
ax.set_xlabel('')

ax.legend(['Identified', 'Anonymous'], 
          title="Job-Title")

sns.despine();


# - Current Employees are more inclined to provide more information including their job title.
# - For both past and current employees, NOT a significant number of reviews are anonymous.
# ____

# **Columns 5th to 9th:** `summary`, `pros`, `cons`, and `advice_to_mgmt` are the actual text review content.
# 
# 
# - Pros: Some of the best reasons to work at the Company.
# - Cons: Some of the downsides of working at t the Company.
# - Summary in the dataset seems to match Review Headline in glassdoor.
# 
# - Advice to Management is optional

# In[37]:


# we leave summary (Review Headline) out of our analysis.
#text_cols = ['summary', 'pros', 'cons', 'advice_to_mgmt']
text_cols = [           'pros', 'cons', 'advice_to_mgmt']
for col in text_cols:
    df[col] = clean_text(df[col])

df[text_cols] = df[text_cols].replace('none', np.nan)
df[text_cols] = df[text_cols].replace('None', np.nan)
df['summary'][df['summary']=='.'] = np.nan # These are actually missing values


# In[38]:


df.sample(5)[text_cols]


# ___

# **1-Star to 5-Starts Rating Columns** `overall_ratings`, `work_balance_stars`, `culture_values_stars`, `senior_mangemnet_stars`, `carrer_opportunities_stars`, and `comp_benefit_stars` We have 1 overall rating value and 5 specific values.

# In[39]:


rating_cols = ['overall_ratings', 'work_balance_stars', 'culture_values_stars',
              'carrer_opportunities_stars', 'comp_benefit_stars', 'senior_mangemnet_stars']
df[rating_cols] = df[rating_cols].replace('none', np.nan)
df['overall_ratings'].nunique()


# In[40]:


# Rating values to Numeric
for col in rating_cols:
    df[col] = pd.to_numeric(df[col], downcast='unsigned')
    
for col in rating_cols:
    if df[col].nunique() > 5:
        print(df[col].value_counts())


# The rating is based on a widespread scale from 1 the worst to 5 the best. Some rating values in rating columns are in between levels, i.e. 1.5, 2.5, 3.5, or 4.5. We correct these rating by lowering them one level. E.g., 1.5 to 1, 2.5 to 2. Note There are no 0.5 nor 5.5 ratings to be corrected.
# 
# Our approach is conservative here, and all mid-stars ratings are truncating to lower star rating. The other approach could be using a binomial distribution with a 50% success rate, to divide the ratings between higher and lower stars levels.

# In[41]:


def five_ratings_only(col):
    for idx in col.value_counts().index:
        col[col==idx] = int(float(idx))
    return col

for col in rating_cols:
    df[col] = five_ratings_only(df[col])
    df[col].astype(np.unsignedinteger, errors='ignore')


# In[42]:


fig, ax = plt.subplots(figsize=two_in_row)

fig.suptitle('Employees\' Overall Ratings\' Distributions', fontsize=14, fontweight='bold')

color = sns.color_palette()[1]

sns.countplot(data=df, x='overall_ratings', hue='current_emp', color=color)

ax.set(title='1 Star to 5 Stars')
ax.legend(['Past Employees', 'Current Employees'])
ax.set_axis_off()

locs = ax.get_xticks()
labels = ax.get_xlabel()

counts = list(df['overall_ratings'].value_counts(normalize=True).iloc[::-1])
for loc, lable, count in zip(locs, labels, counts):

    text = '{:0.0f}%'.format(100*count)
    ax.text(loc, 0, text, color='black', va='top', ha='center', fontsize=14)


# **Findings:**
#  - This plot shows unhappy past employees were more inclined to leave a low review than current employees.
#  - We can see the current/past ratio increases as rating stars increeases. 
#  - Happier current employees are leaving higher rating reviews maybe to show their appreciation.
#  - Uphappier past employees are leaving lower rating reviews maybe to be finally heard!

# In[43]:


fig, ax = plt.subplots(figsize=two_in_row)

fig.suptitle('Employees\' Overall Ratings\' Distributions', fontsize=14, fontweight='bold')

color = sns.color_palette()[3]

sns.countplot(data=df, x='overall_ratings', hue=None, color=color)

ax.set(title='1 Star to 5 Stars')
ax.set_axis_off()

locs = ax.get_xticks()
labels = ax.get_xlabel()

counts = list(df['overall_ratings'].value_counts(normalize=True).iloc[::-1])
for loc, lable, count in zip(locs, labels, counts):

    text = '{:0.0f}%'.format(100*count)
    ax.text(loc, 0, text, color='black', va='top', ha='center', fontsize=14)


# **Findings**
# - This plot is a repeat of the previous plot without distinguishing between past and current employees.
# - Overall star rating shows that almost two-thirds of all employees are generally happy about companies in our data set. One-Fifth neutral and about 15% troubled. 
# ___

# **Helpful Count Column** A count of how many people found the review to be helpful. This variable is the only numeric feature that comes with our dataset. Let's dig into it.

# In[44]:


plt.hist(df['helpful_count']);


# Our first histogram shows evidence of `extreme outliers`.

# In[45]:


def hist_magnifier(df, x, xlim1, xlim2, binsize):
    plt.hist(data=df, x=x, bins=np.arange(xlim1, xlim2+binsize, binsize))
    plt.xlim(xlim1, xlim2);


# In[46]:


plt.figure(figsize=(18, 4))

plt.subplot(1, 3, 1)
hist_magnifier(df, df['helpful_count'], 0, 11, 1)

plt.subplot(1, 3, 2)
hist_magnifier(df, df['helpful_count'], 11, 100, 10)

plt.subplot(1, 3, 3)
hist_magnifier(df, df['helpful_count'], 100, df['helpful_count'].max()+100, 100)


# In[47]:


cum_hist = df['helpful_count'].value_counts(normalize=True).cumsum()
cum_hist[cum_hist<0.95]


# **Findings:**
# 
# - After some trial and error, we arrived with these 3 histograms. Please pay attention to count for each plot and x data range.
# - More than half of the reviews have no helpful_count, i.e. no one found those reviews helpful or worthy. 
# - Only 5% of reviews are found helpful by at least 5 other people
# - Any helpful_count of 50 or more to the highest count of about low two thousand are far in between.
# 
# **Note:**
# - We will work on this column again later when doing feature engineering.
# ___

# **Link Column:** Direct link to the page that contains the review. This column is not required for analysis.

# In[48]:


del df['link']


# ___

# ### Missing Values
#  - Cleaning Missing Values
#  - Plotting Missing Values' Counts

# In[49]:


def plot_missings(df, figsize=(12, 4)):
    """Plot missing values bar visualization for each column of a DataFrame."""
    
    print(f'The DataFrame has {df.shape[0]} rows and {df.shape[1]} columns.')
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.set(style='ticks', palette='pastel')
    color = sns.color_palette()[3]
    
    x = df.isna().sum().index.values
    y = df.isna().sum()
    sns.barplot(x, y, color=color, ax=ax)
    locs, labels = plt.xticks(rotation=90)
    for loc, label, missings, in zip(locs, labels, y):
        if not missings:
            ax.text(loc, 0, 'None', rotation=0, va='bottom', ha='center')
        else:
            ax.text(loc, missings, missings, rotation=0, va='bottom', ha='center')

    ax.set(title='Missing Value Counts in all Columns', xlabel='Columns', ylabel='Counts')
    sns.despine() # remove the top and right borders


# In[50]:


plot_missings(df)


# - Fortunately, there are no missing values in for overall stars rating while there are many missings for other ratings the most for Culture and Values.
# - We certainly want to be aware of what's missing, but in our analysis, it doesn't seem we have to drop any of these missings.

# In[51]:


# Test; Checking the review with missing cons comment.
df[df['cons'].isna()]


# In[52]:


# Test; Checking 4 reviews with missing date_posted.
df[df['date_posted'].isna()]


# **Findings:**
# - The first review with missing date has an extremely very high helpful count, i.e. 580. Reading the `pros` reveal this employee has only worked at Amazon for 1 month.
# - - How could someone working only 1 month write such strong positive review?
# - - The Same review job title shows a `management position`.
# - - There is no advice to management for this review either.
# - Based on the above I am very doubtful this is an honest review. There is reasonable chance employees of this manager or department were asked to mark the review helpful.
# 
# 
# - Having the length of employment as a variable (column) in this dataset could be used to filter similar reviews and also weigh more on reviews for those employees who stayed a minimum reasonable time with their companies.
# 
# 
# - Highest missing values are for those who didn't leave feedback/advice for their higher managers. This is a very valuable part of this dataset as companies can aggregate, summarize and learn a lot from this information.

# In[53]:


# Drop rows/reviews with no 'advice_to_managment`
df = df[df['advice_to_mgmt'].notna()]


# ##### Looking at our cleaned dataset

# In[54]:


df.sample()


# In[55]:


df.describe().transpose()  # Numeric Columns


# In[56]:


df.info(null_counts=False)


# **Notes:**
# - We had to do extensive data cleaning on job_title column, if we wanted to find answers to questions similar to "what positions in general and for each company ratings have need highest attention for workplace improvements?".
# 
# - We still have to figure out what to do with healful_count outliers.
# 
# ##### end of Variables Assessment, Data Cleaning, and Univariant Exploration.
# ___

# ## What insights and questions are we exploring?<a name="Context"></a>

# This dataset is a fraction of what can be easily found on glassdoor.com which already provides so much information about each company. We are not going to focus on answering trivial questions similar to what company has the highest ratings. We like to find the pattern between what matters to employees regardless in general.
# ___

# ## Bivariate and Multivariate Explorations <a name="Multivariate"></a>
# 
# We want to see the relationships between 6 star ratings variables. We need to understand if these variables interact with one another.

# In[57]:


fig, axes = plt.subplots(nrows=1, ncols=6, sharey=True, figsize=(20, 5))
fig.suptitle('Star-Ratings\' Distributions', fontsize=22, fontweight='bold')
xticks=[1, 2, 3, 4, 5]
for ax, col in zip(axes, rating_cols):
    
    if col=='overall_ratings':
        color = sns.color_palette()[2]
    else:
        color = sns.color_palette()[1]
        
    ax = sns.countplot(ax=ax, data=df, x=df[col], color=color, order=xticks, hue=None)
    # plt.ylim(0, 12000)
    mean = '{:0.2f}'.format(df[col].mean())
    ax.set(title=ax.get_xlabel(), xlabel=mean, ylabel='')

    # TODO: Print percentage of each bar on each bar on it.


# **Findings Here:**
# 
# A lot to be discussed here.
# 
# - Means of  the Overall-Rating and Culture-and-Values are closest together but the pattern or shape of distributions of  the Overall-Rating best matches with Compensations-and-Benefits.
# - As said above the Overall-Rating distribution pattern best follows Compensation-and-Benefits. 
# - Per these plots employees mostly are happy about their pay, or maybe this means tech employees only work where they feel happy about their pay.
# - The highest negativity is about Sr. management. Lowest mean and (relatively) the highest number of 1-star ratings.
# - Work-Life Balance pattern looks very similar to Sr. Management both in shape and means.

# ### 2 Overall Rating vs Other Ratings Trend and Averages
# 
# For each five level of the Overa Rating we have 5 other sub-ratings. That is 25 data points. We want to plot the mean of each sub-rating at every oerall rating star.

# In[58]:


def ratings_trend(df=df, rating_cols=rating_cols):
    plt.figure(figsize=(7, 7))

    colors = ['grey', 'blue', 'green', 'red', 'brown']
    ypos = 4.25
    for col, color in zip(rating_cols[1:], colors):
        sns.pointplot(data=df, x='overall_ratings', y=col, color=color)
        plt.text(0.5, ypos, str(col), color=color)
        ypos += 0.15

    plt.ylim(1, 5)
    plt.grid()
    plt.xlabel('Overall Rating Stars')
    plt.ylabel('Star-Level average ratings of each Sub-Ratings vs. Overall Rating');


# In[59]:


ratings_trend(df)


# **Findings:**
# 
# This plot is gold for our analysis.
# 
# - Employees who were given the highest Overall -Rating, i.e. 5 Stars, on average, are given the highest ratings to the Culture-and-Values.
# - Employees who were giving the highest overall rating, i.e. 5 Stars, on average, still are least happy with the Work-Life balance.
# - For unhappy 1 or 2 stars overall rating, Sr Management seems to be the most critical issue impacting it. The runner up affecting unhappy employees in the plot is the Culture-and-Values.
# - It's interesting to see the unhappier (lowest overall rating) employees are the least concern about Compensations-and-Benefits. As we can see the Compensation-Benefits has the highest mean from 1-Star to 4 Stars.

# end of bivariate explorations
# ___

# ## Feature Engineering<a name="Feature"></a>
# 
# Basedon what we have learned sofar we make and explore some new variables, i.e. columns.
# 
# 1. Detail Factor based on review length i.e. word count
# - Time Factor based on date_posted
# - Helpful Factor based on number of people found a review helpful i.e. agreeing with it.
# - Overall-Rating Factor, i.e. Mapping Values to the Overall-Rating Stars
# 
# - Anonymous reviews are less credible.
# 
# - Review Score = Detail-Factor * Time-Factor * Helpfulness * Overall_Rating * Anonymous-Factor

# ### Detail Factor based on review length i.e. word count
# 
# In real life we pay attention when someone cares and pays attention to a subject giving detailed and possibly passionate feed back on subject than short quick answers. We want ot take this fact in consideration in this data set. someone leaving a detailed review summary, pros and cons points and advise for management with about 1000 words deserve more weight than a partial quick review with 15 words. To measure this we make a new feature `detail_factor` based on the overall length of the different part s of the review. 
# 
# There 4 text columns for each review divided into `Summary`, `Pros`, `Cons`, and `Advice to Management`. 

# In[60]:


review_cols = ['pros', 'cons', 'advice_to_mgmt']
correction_dict = {r'-': '',
                   r'w/': 'with',
                   r' i ': ' I ',
                   r' & ': ' and '}

df[review_cols] = df[review_cols].replace(regex=correction_dict)


# In[61]:


# Counting Words
df['wordcount'] = 0
for col in review_cols:
    df['wordcount'] += df[col].astype(str).apply(lambda text: len(text.split()))


# In[62]:


df['wordcount'].describe()


# In[63]:


df = df[df['wordcount'] >= 39]


# In[64]:


def distplot_closelook(series, **kwarg):
    """"""
    
    fig, ax0 = plt.subplots(1, 1, figsize=(20, 2))
    sns.boxplot(series, color=bcolor)
    
    ax0.set_xlabel(f'All {len(series)} observations')
    
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 2))
    sns.boxplot(series, ax=ax0, **kwarg)
    ax0.set_xlim(0, np.percentile(series, 25))
    ax0.set_xlabel(f'Bottom (left) 25% Distribution')

    sns.boxplot(series, ax=ax1, **kwarg)
    ax1.set_xlim(np.percentile(series, 75), series.max())
    ax1.set_xlabel(f'Top (right) 25% Distribution')    


# In[65]:


distplot_closelook(df['wordcount'])
df['wordcount'].describe()


# I think boxen plot works great showing spread of exterem outliers in large distributions. The boxen plot here shows:
# - An review with more than 3500 words in its comments and a couple of reviews arounf 2250.
# - There are exterem outliers with more than 1000 words.

# In[66]:


# Assigns values outside 92.50% boundary to boundary value. 
# In other words capping word_count to a set ceiling value.
df['wordcount'] = df['wordcount'].clip(0, np.percentile(df['wordcount'], 90.0))


# In[67]:


distplot_closelook(df['wordcount'])


# - With trial and error, we find out that at 90.0% upper threshold, all word_count outliers included in our data without being an outlier. We didn't have to lose these reviews because of thier word_count outliers; Instead we capped them at a highest number.
# - We now can bin (group) word_count values give each a weight for furthur analysis.

# In[68]:


plt.figure(figsize=(16, 4))
bin_size = 10
bins = np.arange(5, np.max(df['wordcount'])+bin_size, bin_size)
plt.hist(df.wordcount, bins)
plt.xticks(np.arange(0, 220+10, 10));


# - Now looking at trimmed word_count distribution, we can see with proper trimming and binning, the word_cound values can be groupped (binned for number of word counts and given a weight for each review.
# - The last bar includes/caps all work_count outliers together.

# In[69]:


bins


# In[70]:


# assign/map each review to the bin it belongs
bin_id = pd.cut(df['wordcount'], bins=bins, right=False, include_lowest=True)

# We linearly assign a score for each bin
bv = 1 / bins.shape[0]
f'Number of bins: {bins.shape[0]} - Each bins\' value: {bv} (evenly distributed over all bins)'


# In[71]:


bins_table =  bin_id.value_counts().sort_index().to_frame().reset_index()
# calculate each bin wc_score increamentally from 0 to 1
bins_table['bin_score'] = (bins_table.index + 2) * bv  
bins_table


# In[72]:


mapping_series = pd.Series(data=bins_table['bin_score'].values, index=bins_table['index'])  # make a series with bins' names (edges) as index and bin_score as value
mapping_series


# In[73]:


# for each review, map the weight/score of the bin it beloges to
df['detail_factor'] = bin_id.map(mapping_series)


# In[74]:


# Test
df[['wordcount', 'detail_factor']].sample(7)


# - As seen above the detail_factor value is from (0 to 1).
# 
# end of engineering detail_factor
# ___

# ### Time Factor based on date_posted
# 
# 
# We are interested in the direction and how a workplace is evolving than just averaging historic data `evenly` over time. 
# 
# A recent review must have much higher weight than a review of 5 years ago. To take this into consideration we define a new feature `time_factor` that is from (0 to 1). 1 for a review left today and zero for 5 years (1826 days) or beyond.
# 
# This is a linear weight system we choose to use; Perhaps it is not  the most optimal but it should be practical enough for our dataset for initial experiemnts. 

# In[75]:


max_days = 5 * 364.25 
df['review_days'] = pd.to_numeric((pd.datetime.today() - df['date_posted']).dt.days)
df['time_factor'] = df['review_days'].apply(lambda x: 1 - x/max_days if x < max_days else 0.0)


# In[ ]:


# Test
sns.lineplot(data=df, x=df['date_posted'], y=df['time_factor'])  # reviews time_factor values for the last 5 year


# This plot confirms linear weightig of our reviews over the last 5 years.
# ____

# ### Helpful Factor based on number of people found a review helpful i.e. agreeing with it

# To take helpful_count into account for every 10 people finding a review helpful it doubles the values of that review (This calculation is now rounding to whole numbers. i.e. ints)

# In[ ]:


df['helpful_count'] = df['helpful_count'].clip(0, 100)
df['helful_factor'] = df['helpful_count'].apply(lambda x: 1 + x / 10)


# In[ ]:


plt.figure()

plt.subplot(2, 1, 1)
plt.hist(df['helful_factor'], bins=50)


plt.subplot(2, 1, 2)
plt.hist(df['helful_factor'], bins=100)
plt.xlim(1, 3);


# As seen earlier in this notebook the feature has lots of exterem outliers, but this outliers (as long as they are not errors) don't distort our analysis and shall not be discarded.

# In[ ]:


# Test
df[['helpful_count', 'helful_factor']].sample(7)


# ___

# ### Assigning Numeric Score Value to Overall-Rating Stars
# 
# We use a scoring mapping as follows:
# - 5 Stars: 1.00
# - 4 Stars: 0.75
# - 3 Stars: 0.50
# - 2 Stars: 0.25
# - 1 Star:  0.00

# In[ ]:


df['stars_score'] = df['overall_ratings'].apply(lambda x: (int(x) - 1) / 4)


# In[ ]:


# Test
df[['stars_score','overall_ratings']].sample(7)


# Overall-Rating Values also, are from 0 to 1 but not continuous.
# ___

# In[ ]:


plt.figure(figsize=(18,6))

plt.subplot(1, 3, 1)
sns.boxplot(data=df, x=df['overall_ratings'], y=df['wordcount'], hue='country')

plt.subplot(1, 3, 2)
sns.boxplot(data=df, x=df['overall_ratings'], y=df['detail_factor'], hue='anonymous')

plt.subplot(1, 3, 3)
sns.boxplot(data=df, x=df.overall_ratings, y=df.culture_values_stars)


# In[ ]:


sns.boxplot(data=df, x=df.overall_ratings, y=df.helpful_count)


# In[ ]:


plt.figure(figsize=(18,6))
sns.barplot(data=df, x='company', y='detail_factor', hue='country');


# In[ ]:


df.groupby(['company', 'country']).count()


# In[ ]:


plt.figure(figsize=(18,6))
sns.barplot(data=df, x='company', y='overall_ratings', hue=df.country);


# - This results and lots of other variations of it based on new features that we made doesn't seem to be compelling.
# - We will use all actual ratings values in the data set.

# In[ ]:


df.groupby('company').mean()[['detail_factor', 'overall_ratings']].sort_values('detail_factor', ascending=False)


# In[ ]:


plt.figure(figsize=(18,6))
sns.barplot(data=df, x=df.helpful_count, y=df.overall_ratings);
plt.xlim(-0.5, 10+0.5)


# In[ ]:


cum_hist = df['helpful_count'].value_counts(normalize=True).cumsum()
cum_hist[cum_hist < 0.975].index.max()


# Considering that 97.5% of reviews have a helpful_count of 10 or less, We can see that lower rating reviews get more helpful counts.

# In[ ]:


df.groupby('country').mean().sort_values(by='overall_ratings', ascending=False)


# ## Main Findings<a name="findings"></a>
# 
# We were interested to see what matters most to employees. This visulaliztion revealed very interesting answers (patterns) mostly based on the shape (distribution) of the data rather the statistical summaries.

# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=6, sharey=True, figsize=(20, 5))
fig.suptitle('Ratings\' Distributions', fontsize=22, fontweight='bold')
xticks=[1, 2, 3, 4, 5]
for ax, col in zip(axes, rating_cols):
    
    if col=='overall_ratings' or col=='comp_benefit_stars':
        color = sns.color_palette()[0]
    elif col=='work_balance_stars' or col=='senior_mangemnet_stars':
        color = sns.color_palette()[1]
    else:
        color = sns.color_palette()[2]
        
    ax = sns.countplot(ax=ax, data=df, x=df[col], color=color, order=xticks, hue=None)
    mean = '{:0.2f}'.format(df[col].mean())
    std = '{:0.2f}'.format(df[col].std())
    ax.set(title=ax.get_xlabel(), xlabel=f'Mean:{mean}\nSD:{std}', ylabel='')


# - The Overall-Stars distribution (pattern)  is very similar to the Compensation-Benefit-Stars distribution.
# - The 2nd pair of very similar patterns are for Work-Life-Balance and Management ratings. 
# 
# The employees and senior management relationship shows it is the most troubling area. This goes hand in had with work-life balance with very similar distribution pattern. One main insight to take home here is the better the relationship between teams and their management the work feels more like life or time spent in worthy way.

# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=6, sharey=True, figsize=(20, 5))
fig.suptitle('Ratings\' Distributions between Current and Past Employees', fontsize=22, fontweight='bold')
xticks=[1, 2, 3, 4, 5]
for ax, col in zip(axes, rating_cols):
    
    if col=='overall_ratings' or col=='comp_benefit_stars':
        color = sns.color_palette()[0]
    elif col=='work_balance_stars' or col=='senior_mangemnet_stars':
        color = sns.color_palette()[1]
    else:
        color = sns.color_palette()[2]
        
    ax = sns.countplot(ax=ax, data=df, x=df[col], color=color, order=xticks, hue='current_emp')
    mean = '{:0.2f}'.format(df[col].mean())
    std = '{:0.2f}'.format(df[col].std())
    ax.set(title=ax.get_xlabel(), xlabel=f'Mean:{mean}\nSD:{std}', ylabel='')


# We can see when it comes to 1-star reviews, i.e. employess with most negative experaince, in almost all rating categories number of `past employees` is greater than current employees. In other words, upset past employees are more vocal than those still working at the their current position. In general the higher the rating level from 1-start to 5-stars, the higher the proportion of `current employees over past employees`. The one exception here is the compensation-benefit at 1-start rating between past and current employees that the proprtion is showing the same between the two; That chould translate that pay was the least important issue for past employees. (We see exact numeric statistic averages for this fact in our exploration). The other observation in these charts is that the ratio of 4 or 5 stars reviews that is the ratio of Current Happy employees over Past Happy employees very noticeably increases.
# 
# - Not only those employees with a negative experience are more vocal in general, but past unhappy employees also seem to be more so than current employees.
# ___

# ## Story of our Data<a name="Story"></a>
# 
# Our work and workplace are a big part of our lives, and often they become a significant factor in our identities. In exploring our data set that is scraped off of popular job review site, glassdoor.com, we wanted to see how different things that matter to employees relate to each other. To do so, we only use reviews within the US, Canada, Irland, and the UK to have a reasonably similar workplace similarity yet diverse enough to consider as many people as possible. We could focus on comparing different regions of the world together for example how employees experience stack up against each other between the US and India. We could also separate employees based on their job titles into two general groups of Engineering and Production vs. Retail, and Administrative jobs. To answer our main question here, after some trial and error we realized it is best to consider all type of employees' ratings together.
# 
# Here we plot the mean of each one of the other five-stars rating group against each level of the Overal-Rating. We can see that the lower the Overall-Ratings, the wider of a gap between the Compensations-Benefit. In other words, For employees who were most negative about their workplace experience, money was the least important factor, and as the plot clearly shows below, those employees on average were relatively not troubled with their pay as to the other factors. This finding also holds even when we distinguish anonymous employees and or current or past employees.

# In[ ]:


def ratings_trend(df=df, x='overall_ratings', rating_cols=rating_cols, hue=None):
    plt.figure(figsize=(7, 7))

    colors = ['grey', 'blue', 'green', 'red', 'brown']
    ypos = 4.25
    for col, color in zip(rating_cols[1:], colors):
        sns.pointplot(data=df, x=x, y=col, color=color, hue=hue)
        plt.text(0.5, ypos, str(col), color=color)
        ypos += 0.15

    plt.ylim(1, 5)
    plt.grid()
    plt.xlabel('Overall Rating Stars')
    plt.ylabel('Star-Level average ratings of each Sub-Ratings vs. Overall Rating');


# In[ ]:


ratings_trend()


# In[ ]:


ratings_trend(hue='anonymous')


# In[ ]:


ratings_trend(hue='current_emp')


# ___
# **Other Questions:** These are other interesting questions not directly addressed in this exploration but seem very interesting for later analysis.
# - Based on this data what company seems the best choice for a specific position?
# - Is there a relationship base on a company's stock market momentum its employees' happiness?
# - What is each company best at making its employees happy?
# - What positions seem to be happiest in each company? (and overall)
