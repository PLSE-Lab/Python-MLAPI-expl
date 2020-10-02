#!/usr/bin/env python
# coding: utf-8

# # City of San Francisco Employee Compensation Data
# ## By Brice Pulley

# ## Preliminary Wrangling
# > This is a dataset of employees of the City of San Francisco, from 2013-2019, that includes nearly 330,000 entries. The data was gathered from [DataSF](https://datasf.org/), which is an opensource data site helping the city to thrive through the use of data. Most of the content is provided the local government of the City of San Francisco. An online data query web app was used to filter out an initial dataset and then exported to a local computer. This dataset explores total compensation which is made of up salaries and benefits, and contains information about the job, the year, department, etc. 

# In[ ]:


# import all packages and set plots to be embedded inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from matplotlib.ticker import FuncFormatter
from scipy.stats import norm
from scipy import stats
import warnings
warnings.simplefilter('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# intial formatting of the data before we read the file in
# helps reduce clutter and noise when displaying numbers
pd.set_option('float_format', '{:.02f}'.format)


# #### Load in Dataset

# In[ ]:


# load in dataset into a pandas dataframe
sf_data = pd.read_csv('../input/SALARIES_2.csv', low_memory = False)


# In[ ]:


sf_data.info()


# In[ ]:


sf_data.sample(5)


# In[ ]:


# cleaning up of column headers
sf_data.columns = sf_data.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('-', '_')


# In[ ]:


sf_data.columns


# In[ ]:


# adjust columns names for easier code later on and consistency
sf_data.rename(index=str, columns={"employee_identifier": "emp_id",
                                   "organization_group": "org_group",
                                   "department": "dept",
                                   "total_compensation": "total_comp",
                                   "salaries": 'base_salary',
                                   "other_salaries": "other_salary",
                                   "total_benefits": "total_benefit",
                                   "other_benefits": "other_benefit",
                                   "health_and_dental": "health_dental"}, inplace = True)


# In[ ]:


sf_data.columns


# In[ ]:


# make year categorical for analysis later
# make emp_id an object since the numerical representation cannot be evaluated statistically
sf_data['year'] = sf_data['year'].astype('object')
sf_data['emp_id'] = sf_data['emp_id'].astype('object')


# In[ ]:


# set continuous variables
variables = ['base_salary', 'overtime', 'other_salary', 'total_salary', 'retirement',
       'health_dental', 'other_benefit', 'total_benefit', 'total_comp']


# In[ ]:


sf_data.info()


# In[ ]:


# find descriptive statistics for the numeric variable in the dataset
sf_data.describe()


# > All the variables have a minimum of 0, in order to make sure we do not have entries with all zeros. We will do a query to find those and eliminate them. They will be viewed as if they are "NaN" entries.

# In[ ]:


# using pandas query function to filter out negative entries for total_salary and total_benefit
zero_comp = sf_data[(sf_data.total_benefit == 0) & (sf_data.total_salary == 0) &
                     (sf_data.other_benefit == 0) & (sf_data.other_salary == 0) &
                     (sf_data.retirement == 0) & (sf_data.overtime == 0) &
                     (sf_data.health_dental == 0) & (sf_data.base_salary == 0)]
zero_comp_index = zero_comp.index


# In[ ]:


# dropping negative entries
sf_data.drop(index = zero_comp_index, inplace = True)


# In[ ]:


# reset the index in order for it to be easier to work with later
sf_data.reset_index(inplace = True, drop = True)


# In[ ]:


sf_data.describe()


# > Also checked for duplicates in order to avoid distorted information. 

# In[ ]:


duplicates = sf_data[sf_data.duplicated(subset = variables,keep = 'first')]
duplicates.head()


# In[ ]:


dupe_index = duplicates.index


# In[ ]:


# dropped these in place
sf_data.drop(index = dupe_index, inplace = True)


# In[ ]:


sf_data.reset_index(inplace = True, drop = True)


# In[ ]:


sf_data.describe()


# ### Functions

# In[ ]:


# tick converter for x-axis
def thousands(x, pos):
    '''The two args are the value and tick position'''
    return '%1.fK' % (x * 1e-3)
format_x = FuncFormatter(thousands)
def x_format():
    x_tick_form = ax.xaxis.set_major_formatter(format_x)
    return x_tick_form


# In[ ]:


# tick converter for y-axis
def thousands(y, pos):
    '''The two args are the value and tick position'''
    return '%1.fK' % (y * 1e-3)
format_y = FuncFormatter(thousands)
def y_format():
    y_tick_form = ax.yaxis.set_major_formatter(format_y)
    return y_tick_form


# In[ ]:


# cuberoot converter
def cr_trans(x, inverse = False):
    if not inverse:
        return x ** (1/3)
    else:
        return x ** 3


# In[ ]:


# log converter
def log_trans(x, inverse = False):
    if not inverse:
        return np.log10(x)
    else:
        return np.power(10, x)


# ### What is the structure of your dataset?
# 
# > > There are 297,106 entries of city employees over a 7 year period (2013-2019) with 13 columns or variables. There are 9 numeric variables which are most likely highly correlated due to the fact that three of them are products of the other 7 variables. There are 4 variables that are a mix of categorical data, and object.
# 
# > `total_comp` = `total_salary` + `total_benefit`
#     
# > `total_benefit` = `retirement` + `other_benefit` + `health_dental`
#     
# > `total_salary` = `base_salary` + `overtime` + `other_salary`
# 
# ### What is/are the main feature(s) of interest in your dataset?
# 
# > I am most interested in which variables are the best predictors of  `total_comp` and whether benefits or salaries are more influential.
# 
# ### What features in the dataset do you think will help support your investigation into your feature(s) of interest?
# 
# > I suspect that salary will have the most influence because that is the basis of pay when hired. The other variables such as overtime and other salaries will have the most influence as well.

# In[ ]:


sf_data.describe()


# ## Univariate Exploration

# #### Initial `total_comp` Plot

# In[ ]:


# start with a histogram to see distribution
fig, ax = plt.subplots(figsize=[8, 5])
bins = np.arange(0, 797635.20 + 5000, 5000)
plt.hist(data = sf_data, x = 'total_comp', bins = bins)
plt.xlabel('Total Compensation ($)');
x_format();


# > According to the chart, the largest group is in the \\$0 to \\$5,000 bin. The minimum wage in San Francisco is \\$15/hr which comes out to \\$31,200 annually. Therefore, we want to filter out the part-time workers and only explore full-time workers to reduce skewness. We'll also apply a log-transformation to normalize the distribution.

# In[ ]:


# filtering out part-time work by total_comp and total_salary
part_time = sf_data[(sf_data.total_comp < 31200) | (sf_data.total_salary < 31200)].index


# In[ ]:


sf_data.drop(index = part_time, inplace = True)


# In[ ]:


# reset the index in order for it to be easier to work with later
sf_data.reset_index(inplace = True, drop = True)


# In[ ]:


# used to find the bins for total_comp log
log_trans(sf_data.total_comp.describe())


# ##### Log-scaled `total_comp` distribution

# In[ ]:


fig, ax = plt.subplots(figsize = [9, 6])
x_ticks = [50000, 70000, 90000, 110000, 150000, 200000, 300000]
bins = 10 ** np.arange(4.50, 5.90 + 0.01, 0.01)
plt.hist(data = sf_data, x = 'total_comp', bins = bins)
plt.xlabel('Total Compensation ($)');
plt.xscale('log');
x_format();
plt.xticks(x_ticks);
plt.xlim(32100, 400000);


# > Since, `total_comp` had a long-tailed distribution, that was right-skewed, a log-scale transformation was used in order to see a clearer picture. You can see that the it is mostly unimodal with a dip right after the peak which is around \\$110,000. This indicates most city employees total compensation lies around \\$100,000 which is the median wage in San Francisco according to the [US Census Bureau](https://www.census.gov/library/visualizations/2018/comm/acs-income.html). This indicates a fairly accurate labor market value. Next, we'll explore `total_benefit` and `total_salary`.

# In[ ]:


sf_data.describe()


# #### Initial Plot `total_salary`

# In[ ]:


# showing total salary with standard plot
fig, ax = plt.subplots(figsize = [8, 5])
bins = np.arange(31200, 641374.64 + 5000, 5000)
plt.hist(data = sf_data, x = "total_salary", bins = bins);
plt.xlabel('Total Salary ($)');
plt.xticks(rotation = 15);
x_format();


# > As with the `total_comp` plot, there is a long-tailed right-skewed distribution, so a log transformation is appropriate. We will also investigate extreme outliers as well.

# In[ ]:


# did a search for total_salary above 400K because that is where the bars above die off. 
high_outliers = sf_data[(sf_data.total_salary > 400000)]
high_outliers.sort_values(by = 'total_salary',ascending = False).head()


# In[ ]:


# investigating Police Officer 3 since that doesnt match high positions like the others
pol_3 = sf_data[sf_data.job == "Police Officer 3"]
pol_3.total_salary.sort_values(ascending = False).head()


# In[ ]:


# looking at the lower end as well
pol_3.total_salary.sort_values(ascending = False).tail()


# In[ ]:


# dropped high police officer 3 entry
irr_sal = sf_data[sf_data.index == 215652]


# In[ ]:


# found index of entry
irr_sal_index = irr_sal.index


# In[ ]:


# dropped in place
sf_data.drop(index = irr_sal_index, inplace = True)


# In[ ]:


# reset index
sf_data.reset_index(inplace = True, drop = True)


# > These job positions and salaries, follow a normal trend of money so we'll keep these. We'll also apply a log transformation to normalize the distribution.

# #### `total_salary` log scale

# In[ ]:


# used to find bins for total_salary log
np.log10(sf_data.total_salary.describe())


# In[ ]:


# both distributions are gonna take on a log-scale transformaton 
# added x limits in zoom in on distributions
fig, ax = plt.subplots(figsize = [8, 5])
x_ticks = [30000, 50000, 80000, 110000, 150000, 200000, 300000]
bins = 10 ** np.arange(4.49, 5.73 + 0.01, 0.01)
plt.hist(data = sf_data, x = "total_salary", bins = bins);
plt.xlabel('Total Salary ($)');
plt.xscale('log');
x_format();
plt.xticks(x_ticks);
plt.xlim(30000, 400000);


# > So for `total_salary`, the distribution on a log scale is unimodal and shows that a majority of the total salaries lies near the \\$80,000 mark. 

# #### `total_benefit` initial plot

# In[ ]:


sf_data.total_benefit.describe()


# In[ ]:


# showing total benefits with standard plot
fig, ax = plt.subplots(figsize = [8, 5])
bins = np.arange(0, 151681.38 + 2000, 2000)
plt.hist(data = sf_data, x = "total_benefit", bins = bins);
plt.xlabel('Total Benefits ($)');
x_format();


# > This histogram has a long tail that is right skewed, lets investigate some outliers to see if something should be eliminated.

# In[ ]:


high_outliers = sf_data[sf_data['total_benefit'] > 100000]
high_outliers.sort_values(by = 'total_benefit', ascending = False).head()


# > These are part of the same group that have higher end salaries so we'll keep these as well.

# In[ ]:


low_outliers = sf_data[sf_data['total_benefit'] == 0]
low_outliers.sample(5)


# > These people are exempt from payroll deductions, so the city does not contribute to their benefits. This is a very small portion of the group that might distort results so we'll safely remove these.

# #### `total_benefit` cube-root transformation

# In[ ]:


# used to find the bins
np.cbrt(sf_data.describe())


# In[ ]:


# needed to get the cube root of total_benefits isolated
tot_bene = sf_data.total_benefit
cube_tot_bene = np.cbrt(tot_bene)


# In[ ]:


fig, ax = plt.subplots(figsize = [8, 5])
x_ticks = [10, 15, 20, 25, 30, 35, 40, 45, 50]
tick_labels = ['1K', '3K', '8K', '15K', '27K', '43K', '64K', '91K', '125K']
bins = np.arange(0, 53.33 + 0.25, 0.25)
plt.hist(x = cube_tot_bene, bins = bins);
plt.xlabel('Total Benefits ($)');
plt.xticks(x_ticks, tick_labels);
plt.xlim(10, 50);


# > As you can see, `total_benefit` is bi-modal, with a lower peak around \\$20,000, and a second higher peak around \\$43,000. 

# #### compare `base_salary`, `overtime` and `other_salary`

# In[ ]:


sf_data.describe()


# In[ ]:


plt.figure(figsize=[15,5])

# base_salary
bins = np.arange(0, 537847.86 + 5000, 5000)
plt.subplot(1, 3, 1)
plt.hist(data = sf_data, x = "base_salary", bins = bins);
plt.xlabel('Salary ($)');
plt.xticks(rotation = 15)

# overtime
bins2 = np.arange(0, 304546.25 + 5000, 5000)
plt.subplot(1, 3, 2)
plt.hist(data = sf_data, x = "overtime", bins = bins2);
plt.xlabel('Overtime ($)');

# other_salary
bins3 = np.arange(0, 342802.63 + 5000, 5000)
plt.subplot(1, 3, 3)
plt.hist(data = sf_data, x = "other_salary", bins = bins3);
plt.xlabel('Other Salary ($)');
plt.xticks(rotation = 15);


# > Let's take a look at high and low outliers.

# In[ ]:


high_outliers = ((sf_data['base_salary'] > 250000) | (sf_data['overtime'] > 60000) | (sf_data['other_salary'] > 40000))


# In[ ]:


sf_data[high_outliers].sort_values(by = ['base_salary'], ascending = False).head(10)


# In[ ]:


sf_data[high_outliers].sort_values(by = 'overtime', ascending= False).head()


# In[ ]:


sf_data[high_outliers].sort_values(by = 'other_salary', ascending= False).head()


# In[ ]:


base_outliers = sf_data[sf_data.base_salary > 250000]
base_outliers.sort_values(ascending = False, by = ['job','base_salary']).head()


# In[ ]:


sf_data[sf_data.job == "Transit Operator"].base_salary.sort_values(ascending = False).head()


# In[ ]:


sf_data[sf_data.job == "Transit Operator"].base_salary.sort_values(ascending = False).tail()


# In[ ]:


irr_base_sal = sf_data[(sf_data.index == 215628)]


# > The salary range for a Transit Operator is between 6,000 and 102,000 dollars. Therefore, the operator with a salary over 300,000 is probably a typo. It'll be thrown out. 

# In[ ]:


irr_base_sal_index = irr_base_sal.index


# In[ ]:


# dropped these in place
sf_data.drop(index = irr_base_sal_index, inplace = True)


# In[ ]:


sf_data.reset_index(inplace = True, drop = True)


# In[ ]:


sf_data.describe()


# In[ ]:


low_outliers = (sf_data['base_salary'] < 20000)
sf_data[low_outliers].sample(5)


# > There are some entries where `salary` and `other_salary` could've been switched because according to the definitions. `Salary` is the base pay and the `other_salary` is premium, incentive pay, or irregular payments made to the individual. Therefore, another query will be performed to eliminate the oddly paid individuals because we cannot investigate the nature of their pay and is affecting the data.

# In[ ]:


irregular_incomes = sf_data[((sf_data['base_salary']) <= (sf_data['other_salary'])) 
                            & ((sf_data['base_salary'] == 0))]
irregular_incomes.head()


# In[ ]:


index_irr_income = irregular_incomes.index


# In[ ]:


# dropped these in place
sf_data.drop(index = index_irr_income, inplace = True)


# In[ ]:


sf_data.reset_index(inplace = True, drop = True)


# In[ ]:


sf_data.describe()


# In[ ]:


# log bins for base_salary
np.log10(sf_data.base_salary.describe())


# In[ ]:


np.cbrt(sf_data.describe())


# In[ ]:


# setting data for overtime and other_salary cube root
cube_ot = np.cbrt(sf_data['overtime'])
cube_other_salary = np.cbrt(sf_data['other_salary'])


# In[ ]:


plt.figure(figsize=[15,6])

# base_salary log transformation
x_ticks = [15000, 20000, 30000, 50000, 70000, 110000, 170000, 250000]
bins = 10 ** np.arange(4.07, 5.73 + 0.01, 0.01)
plt.subplot(1, 3, 1)
plt.hist(data = sf_data, x = "base_salary", bins = bins);
plt.xlabel('Base Salary ($)');
plt.xscale('log')
plt.xticks(rotation = 15)
plt.xticks(x_ticks, ['15K', '20K', '30K', '50K', '70K', '110K', '170K', '250K'])


# overtime cuberoot transformation
x_ticks = [0, 10, 20, 30, 40, 50, 60]
labels = [0, '1K', '8K', '27K', '64K', '125K', '216K']
bins2 = np.arange(0, 67.28 + 1, 1)
plt.subplot(1, 3, 2)
plt.hist(x = cube_ot, bins = bins2);
plt.xlabel('Overtime ($)');
plt.xticks(x_ticks, labels);



# other_salary cuberoot transformation
x_ticks = [0, 10, 20, 30, 40, 50, 60, 70]
labels = [0, '1K', '8K', '27K', '64K', '125K', '216K', '343K']
bins3 = np.arange(0, 69.99 + 1, 1)
plt.subplot(1, 3, 3)
plt.hist(x = cube_other_salary, bins = bins3);
plt.xlabel('Other Salary ($)');
plt.xticks(x_ticks, labels);


# > The `base_salary` distribution is unimodal and peaks around \\$60,000 dollars and follows a normal distribution. 
# 
# > The `overtime` distribution is bimodal with \\$500 being the strongest peak and around \\$5,000 being the second peak. The \\$500 peak is extreme because not many employees can claim overtime and is only available to certain jobs or fields, most notably policeman, firefighters, or nurses. Mostly, people who work erratic hours. Otherwise, this column would remain empty. 
# 
# > The `other_salary` is a bimodal distribution with highest peak at \\$500 and the second peak at \\$1,000. The other_salary category is described as premium pay, incentive pay, or one time payments. There is a high peak at \\$500 because most people do not receive other forms of salary other than their base pay.

# #### `retirement`, `health_dental`, and `other_benefits`

# In[ ]:


sf_data[['retirement', 'health_dental', 'other_benefit' ]].describe()


# #### Standard Histogram for `retirement`, `health_dental`, and `other_benefit`

# In[ ]:


plt.figure(figsize=[15,5])

# retirement
bins = np.arange(0, 105052.98 + 1000, 1000)
plt.subplot(1, 3, 1)
plt.hist(data = sf_data, x = "retirement", bins = bins);
plt.xlabel('Retirement ($)');


# health_dental
bins2 = np.arange(0, 36609.50 + 500, 500)
plt.subplot(1, 3, 2)
plt.hist(data = sf_data, x = "health_dental", bins = bins2);
plt.xlabel('Health & Dental ($)');


# other_benefit
bins3 = np.arange(0, 37198.60 + 500, 500)
plt.subplot(1, 3, 3)
plt.hist(data = sf_data, x = "other_benefit", bins = bins3);
plt.xlabel('Other Benefits ($)');
x_format();
plt.xticks(rotation = 0);


# > The `retirement` and `other_benefit` charts look like they are right-skewed so a cubic root transformation would be helpful. There is also a spike in the `retirement` chart at \\$0, which should be investigated further. 
# 
# > The `health_dental` chart is multimodal distribution with the first spike around \\$8,000 and the second at around \\$13,000. 

# In[ ]:


sf_data[sf_data['retirement'] == 0].sample(5)


# In[ ]:


# bins for retirement and other_benefit cube root
np.cbrt(sf_data[['retirement', 'other_benefit']].describe())


# In[ ]:


# setting data cube root transformation for both variables
cube_retire = np.cbrt(sf_data['retirement'])
cube_benefit = np.cbrt(sf_data['other_benefit'])


# In[ ]:


# applied transformations
plt.figure(figsize=[15,5])

# retirement
x_ticks = [0, 10, 20, 30, 40]
labels = [0, '1K', '8K', '27K', '64K']
bins = np.arange(0, 49.18 + 1, 1)
plt.subplot(1, 3, 1)
plt.hist(x = cube_retire, bins = bins);
plt.xlabel('Retirement ($)');
plt.xticks(x_ticks, labels);


# health_dental
bins2 = np.arange(0, 21715.08 + 500, 500)
plt.subplot(1, 3, 2)
plt.hist(data = sf_data, x = "health_dental", bins = bins2);
plt.xlabel('Health & Dental ($)');


# other_benefit
x_ticks = [0, 5, 10, 15, 20, 25, 30]
labels = [0, '125', '1K', '3K', '8K', '16K', '27K']
bins3 = np.arange(0, 33.38 + 1, 1)
plt.subplot(1, 3, 3)
plt.hist(x = cube_benefit, bins = bins3);
plt.xlabel('Other Benefits ($)');
plt.xticks(x_ticks, labels);


# > For the `retirement` variable, its bimodal with a significant peak around the \\$15,000 mark and one at the \\$0 mark. The retirement accounts are a portion of contribution by the city of San Francisco, so its a percentage of how much someone contributes based on their salary. The spike at \\$0 indicates those people who have chosen not to make contributions to their retirement or the people who have chosen an alternative plan that the city does not contribute to. 
# 
# > The `health_dental` has a bimodal distribution with one major peak at around \\$7,500 and another at \\$12,500. There are most likely two spikes because most employers have insurance coverage options on a tier basis. This would indicate 2 different health/dental plan tiers and is probably distributed based on a percentage of income as well. There is also a small spike at \\$0 which would indicate those employees opting to not participate in coverage or possibly not offered based on other criteria. 
# 
# > The `other_benefit` has a unimodal distribution that is normal and according to the data dictionary, this is the mandatory amount contributed such as Social Security, FICA, Medicare, unemployment insurance, etc. This is solely based on the percentage of income, therefore that is why it is varied with a peak most likely in connection with the median income of the total data set. Also, the peak is around \\$5,000. 

# #### Exploring non-numeric variables

# ##### `org_group`

# > An Organizational Group is a group of departments, such as the Public Protection Group comprises departments such as: Police, Fire, Adult Probation, District Attorney, and Sheriff. 

# In[ ]:


sf_data.org_group.value_counts()


# In[ ]:


# made these category names easier to work with
org_series = sf_data.org_group
org_series.replace({'Public Works, Transportation & Commerce': 'Pub Wrks, Tran & Comm.',
                    'Community Health': 'Comm. Health',
                    'Public Protection': 'Pub. Protect',
                    'Human Welfare & Neighborhood Development': 'Hmn Wlfr & Nbrhd Dev.',
                    'General Administration & Finance': 'Gen Admin & Fin.',
                    'Culture & Recreation': 'Culture & Rec.'}, inplace=True)


# In[ ]:


sf_data.org_group.value_counts()


# In[ ]:


group_cats = ['Pub Wrks, Tran & Comm.', 'Pub. Protect', 'Comm. Health',
              'Gen Admin & Fin.', 'Hmn Wlfr & Nbrhd Dev.', 'Culture & Rec.']
categories = pd.api.types.CategoricalDtype(ordered = True, categories = group_cats)
sf_data['org_group'] = sf_data['org_group'].astype(categories)


# In[ ]:


group_cats = sf_data.org_group.value_counts()
labels = group_cats.index


# In[ ]:


# Pie Chart shows distributions of Proportions of Employees in each Organizational Group
plt.figure(figsize= [6,5])
plt.pie(group_cats, autopct='%1.1f%%', labels = labels, pctdistance = 0.83);


# > The pie chart shows the largest employed group is Public Works, Transportation and Community, while the lowest is Culture and Recreation.

# ##### `job`

# In[ ]:


# EXPLORING JOB
top_five_job = sf_data.job.value_counts().sort_values(ascending = False).head(5).index
top_five_job


# In[ ]:


# getting top jobs
top_five_by_job = (sf_data[sf_data.job.isin(top_five_job)].job.value_counts())
top_five_by_job


# In[ ]:


# countplot top five jobs
top_five_by_job.plot(kind='barh');


# > There are 1,097 job descriptions, which is too high to view on a categorical basis, so this variable might be thrown out. The top 5 however were: Transit Operator, Registered Nurse, Firefighter, Police Officer 3, and Custodian. This information might be useful when exploring with other variables. 

# ##### `year`

# In[ ]:


# exploring year
base_color = sb.color_palette()[0]
sb.countplot(data= sf_data, x = 'year', color = base_color);


# > This countplot shows that the amount of employees each year is relatively the same. 

# ### Discuss the distribution(s) of your variable(s) of interest. Were there any unusual points? Did you need to perform any transformations?
# 
# > The variable `total_comp` had a long tail and right-skewed distribution. The highest peak was at the 5,000 dollars and below. Since, the minimum wage in San Francisco is \\$15/hour, which is \\$31,200 annually, those entries were filtered out to eliminate part-time workers. Also, do to the extreme range of values, a log transformation was applied to the variable.
# 
# ### Of the features you investigated, were there any unusual distributions? Did you perform any operations on the data to tidy, adjust, or change the form of the data? If so, why did you do this?
# 
# > There were several extreme outliers for the other variables that did not match the descriptions of job titles. Therefore, those entries were removed as well. Also, compensation where there were no base salaries, and only lump sum payments were thrown out as well.

# ### Feature Engineering

# In[ ]:


data_clean = sf_data.copy()


# In[ ]:


# log columns
data_clean['log_salary'] = data_clean['base_salary'].apply(log_trans)
data_clean['log_total_comp'] = data_clean['total_comp'].apply(log_trans)
data_clean['log_total_salary'] = data_clean['total_salary'].apply(log_trans)
# cuberoot columns
data_clean['cr_overtime'] = data_clean['overtime'].apply(cr_trans)
data_clean['cr_other_salary'] = data_clean['other_salary'].apply(cr_trans)
data_clean['cr_retirement'] = data_clean['retirement'].apply(cr_trans)
data_clean['cr_other_benefit'] = data_clean['other_benefit'].apply(cr_trans)
data_clean['cr_total_benefit'] = data_clean['total_benefit'].apply(cr_trans)


# In[ ]:


# saving cleaned copy for explanatory analysis
data_clean.to_csv(path_or_buf='master_sf_salary.csv', index = False)


# In[ ]:


# take a sample of the dataset so the scatterplot will process quicker
np.random.seed(2018)
sample = np.random.choice(data_clean.shape[0], 2000, replace = False)
sf_subset = data_clean.iloc[sample]


# ## Bivariate Exploration

# In[ ]:


# set continuous variables
variables = ['base_salary', 'overtime', 'other_salary', 'total_salary', 'retirement',
       'health_dental', 'other_benefit', 'total_benefit', 'total_comp']


# In[ ]:


#scatter plot
sb.set_context('talk')
sb.pairplot(data = sf_subset, vars = variables);


# > Due to the number of continuous variables, a heatmap might be able to show correlation in a more condensed version.

# In[ ]:


# heat map of continuous variables UNTRANSFORMED
sb.set_context('notebook')
plt.figure(figsize = [10,7])
sb.heatmap(data_clean[variables].corr(), annot = True, fmt = '.3f',
          cmap = 'vlag_r', center = 0);


# > Areas of Interest
# - `salary` v `overtime` 
# - `overtime` v  `other_benefit` 
# - `other_salary` v  `other_benefit` 

# ##### `overtime` v `other_benefit`

# In[ ]:


# relplot `overtime` vs `other_benefit`
sb.relplot(data = sf_subset, x = 'overtime', y = 'other_benefit');
plt.xticks(rotation = 15);


# In[ ]:


sf_subset[['other_benefit', 'overtime', 'cr_other_benefit',
             'cr_overtime']].corr()


# > According to the heatmap, there was a negative correlation between `other_benefit` and `overtime`. This seemed odd because `other_benefit` are the benefits towards medicare, social security, unemployment insurance, etc paid on behalf of the employer. Therefore, an increase in pay I would imagine there would be an increase in benefits paid because it is percentage based.

# ##### `salary` v `overtime`

# In[ ]:


# relplot `base_salary` vs `overtime`
sb.relplot(data = sf_subset, x = 'base_salary', y = 'overtime');
plt.xticks(rotation = 15);


# In[ ]:


sf_subset[['base_salary', 'overtime', 'log_salary',
             'cr_overtime']].corr()


# > Found only a tiny correlation between `salary` and `overtime`. I thought it would be higher because overtime is dependent on salary. Maybe, transforming the data might highlight more correlation. 

# ##### `other_salary` vs `other_benefit`

# In[ ]:


# relplot `other_salary` vs `other_benefit`
sb.relplot(data = sf_subset, x = 'other_salary', y = 'other_benefit');
plt.xticks(rotation = 15);


# In[ ]:


sf_subset[['other_salary', 'other_benefit', 'cr_other_salary',
             'cr_other_benefit']].corr()


# > Same approach as `overtime` and `other_benefit` but no conclusions have been made thus far. 

# #### `org_group` vs `total_comp`

# In[ ]:


fig, ax = plt.subplots(figsize = [8, 5])
x_ticks = log_trans(np.array([30000, 45000, 65000, 100000, 160000, 250000, 400000, 630000]))
sb.violinplot(data = sf_subset, y = 'org_group',
              x = 'log_total_comp', color = base_color)
ax.tick_params(axis = 'both', which = 'major',labelsize = 10);
plt.xlabel('Total Comp ($)')
plt.ylabel('Organizational Group');
plt.xticks(x_ticks, ['30K', '45K', '65K', '100K',
                     '160K', '250K', '400K', '630K' ]);


# > Public Protection has the highest median total compensation indicating the best overall pay for those employees in those departments. Culture and Recreation has the smallest range and highest concentration around the median indicating a lower variability of salary and benefits. Community Health had the least concentrated distribution which indicates a high variability for salary and benefits. It would be interesting to see the percentage distributions of the variables making up total benefits and total salaries.

# #### `job` vs `total_comp`

# In[ ]:


# to get job subset
job_subset = (sf_subset[sf_subset.job.isin(top_five_job)])


# In[ ]:


# BY JOB BY TOTAL_COMP
fig, ax = plt.subplots(figsize = [7, 5])
sb.violinplot(data = job_subset, y = 'job', x = 'log_total_comp', color = base_color)
ax.tick_params(axis = 'both', which = 'major',labelsize = 10)
plt.xticks(log_trans(np.array([30000, 40000, 60000, 100000, 150000, 250000, 400000])),
          ['30K', '40K', '60K', '100K', '150K', '250K', '400K'])
plt.xlabel('Total Comp ($)')
plt.ylabel('Top Five Jobs');


# > Custodian has the smallest distribution of total compensation and the lowest median. This follows conventions that a lower-skilled worker would be paid the least and have less room for advancement or flexibility in salary or benefits. 
# > Registered Nurse has the highest median total compensation and a wider range of the distribution of total compensation. This is understandable because Nurses are highly skilled and highly educated workers. It would be interesting to see the distribution of their pay. 

# In[ ]:


job_sub = data_clean[data_clean['job'].isin(top_five_job)]
g = sb.FacetGrid(job_sub, col = 'job', height = 3, aspect = 1.5,
                 col_wrap = 3, legend_out = True)
g.map(plt.hist, 'log_total_comp', alpha = 0.9);
g.add_legend();


# ### Talk about some of the relationships you observed in this part of the investigation. How did the feature(s) of interest vary with other features in the dataset?
# 
# > The heatmap showed positive strong correlations between `total_comp` and the other numerical features. This is logical considering `total_comp` is a collective product of the other varaiables combined. When compared with the `org_group` it was surprising that Public Protection had the highest median pay, considering in most places, compensation for firefighters, policemen, etc are poorly paid. When compared with the top five jobs, it followed the typical conventions of highly skilled workers are paid better.
# 
# ### Did you observe any interesting relationships between the other features (not the main feature(s) of interest)?
# 
# > The relationship between `other_benefit` compared to `other_salary` and `overtime` had a negative correlation. This seemed counterintuitive because `other_benefit` are the benefits paid on behalf of the employer towards medicare, social security, unemployment insurance, etc. Therefore, an increase in pay should result in an increase in benefits paid because it is based on percentage of income. 

# ## Multivariate Exploration

# #### `org_group` vs `base_salary`, `overtime`, `other_salary`

# In[ ]:


# salary variables
mean_org_sal = data_clean.groupby('org_group')[['base_salary', 'overtime', 'other_salary']].agg('mean')
mean_org_sal


# In[ ]:


mean_org_sal.plot(kind='bar');


# In[ ]:


# found percentage to see proportions
salary_percents = mean_org_sal.div(mean_org_sal.sum(1), axis = 0)
salary_percents


# In[ ]:


# charted proportions
salary_percents.plot(kind = 'barh', stacked = True);
plt.legend(bbox_to_anchor=(1.05, 1), loc = 'upper left', borderaxespad = 0.);
plt.ylabel('Organizational Group');


# > Over 95% of the first 3 groups have the `base_salary` that makes up the majority of their `total_salary`. The Public Protection group has the widest range with 12% `overtime` and 7% is `other_salary`, which is incentivized pay. It would be interesting to explore more in depth.

# #### `org_group` vs. `retirement`, `health_dental`, `other_benefits`

# In[ ]:


# benefits Variables
mean_org_bene = data_clean.groupby('org_group')[['retirement',
                                              'health_dental',
                                              'other_benefit']].agg('mean')
# found percentage to see proportions
benefit_percents = mean_org_bene.div(mean_org_bene.sum(1), axis = 0)
benefit_percents


# In[ ]:


# charted proportions
benefit_percents.plot(kind = 'barh', stacked = True);
plt.legend(bbox_to_anchor=(1.05, 1), loc = 'upper left', borderaxespad = 0.);
plt.ylabel('Organizational Group');


# > Public Protection again had the lowest contribution to `other_benefit`, which is odd because they had the highest median `total_comp` earlier. Investigating further the break down of all variables for Pub Protection would highlight some things.

# ##### `org_group` vs `all`

# In[ ]:


# ALL Variables
mean_org_total = data_clean.groupby('org_group')[['base_salary',
                                              'overtime',
                                              'other_salary',
                                              'retirement',
                                              'health_dental',
                                              'other_benefit']].agg('mean')
# found percentage to see proportions
total_percents = mean_org_total.div(mean_org_total.sum(1), axis = 0)
total_percents


# In[ ]:


# charted proportions
total_percents.plot(kind = 'barh', stacked = True);
plt.legend(bbox_to_anchor=(1.05, 1), loc = 'upper left', borderaxespad = 0.);
plt.ylabel('Organizational Group');


# > * Public Protection had the lowest `base_salary` as part of total_compensation at 61%, with Gen Admin & Fin having the highest at 68%.
# > * Public Protection had the lowest contribution to `other_benefit` and the highest for `overtime` and `other_salary`.

# ##### Top Five Jobs vs. Salary, Overtime, Other Salary

# In[ ]:


# salary variables
mean_job_sal = job_sub.groupby('job')[['base_salary',
                                       'overtime',
                                       'other_salary']].agg('mean')
# found percentage to see proportions
sal_percent_job = mean_job_sal.div(mean_job_sal.sum(1), axis = 0)
sal_percent_job


# In[ ]:


# charted proportions
sal_percent_job.plot(kind = 'barh', stacked = True);
plt.legend(bbox_to_anchor=(1.05, 1), loc = 'upper left', borderaxespad = 0.);
plt.ylabel('Top Five Jobs');


# > The majority of total salary starts with the `base_salary` which is to be expected. Transit Operator, Firefighter, and Police Officer 3 have the most variability because they have a higher percentage of total salary in `overtime` and `other_salary`. These tend to  fluctuate based on a number of factors such as hours worked, type of shift, etc. The Custodian has the most stability because their `base_salary` makes up the majority of their salary. 

# ##### Top Jobs vs Retirement, Health/Dental, Other_Benefits

# In[ ]:


# salary variables
mean_job_bene = job_sub.groupby('job')[['retirement',
                                        'health_dental',
                                        'other_benefit']].agg('mean')
# found percentage to see proportions
bene_percent_job = mean_job_bene.div(mean_job_bene.sum(1), axis = 0)
bene_percent_job


# In[ ]:


# charted proportions
bene_percent_job.plot(kind = 'barh', stacked = True);
plt.legend(bbox_to_anchor=(1.05, 1), loc = 'upper left', borderaxespad = 0.);
plt.ylabel('Top Five Jobs');


# > This shows that Registered Nurses, Firefighters, and Police Officers 3 have more contributions to retirement proportionally than Transit Operator and Custodian. This is probably because they have more disposible income to put more towards retirement than the others. Also note that Police Officer 3 and Firefighter have the lowest contribution to `other_benefit` because some can "opt" out of those certain taxes. This might skew the results of the others in comparison. 

# In[ ]:


# ALL variables
mean_job_total = job_sub.groupby('job')[['base_salary', 
                                         'overtime', 
                                         'other_salary',
                                         'retirement',
                                         'health_dental',
                                         'other_benefit']].agg('mean')
# found percentage to see proportions
sal_percent_total = mean_job_total.div(mean_job_total.sum(1), axis = 0)
sal_percent_total


# In[ ]:


# charted proportions
sal_percent_total.plot(kind = 'barh', stacked = True);
plt.legend(bbox_to_anchor=(1.05, 1), loc = 'upper left', borderaxespad = 0.);
plt.ylabel('Top Five Jobs');


# > This overview of all parts of the breakdown of total compensation can help highlight the biggest contributors.

# #### Plot by `job` , `total_salary` vs `total_benefit`

# In[ ]:


g = sb.FacetGrid(data = job_sub, hue = 'job', hue_order = top_five_job,
                height = 10, aspect = 1.5)
g.map(sb.scatterplot, 'total_benefit',
      'log_total_salary').set(yticks=log_trans(np.array([30000, 40000, 60000,
                                                         100000, 150000, 
                                                         250000, 400000])));
g.add_legend(title='Jobs');
g.set_ylabels('Total Salary ($)');
g.set_yticklabels(['30K', '40K', '60K', '100K', '150K', '250K', '400K'])
g.set_xticklabels([0, '0k', '10K', '20K', '30K', '40K', '50K', '60K', '70K'])
g.set_xlabels('Total Benefits ($)');


# > The scattor plot of `log_total_salary` v `total_benefit` shows a positive linear trend. A color code of the top 5 jobs was applied to see which ones are more correlated.  Police Officer 3 is the least linear while Registered Nurse is the most linear. Firefighter looks the same as Police Officer 3. This might have something to do with the `other_benefits` having the option to opt out. A faceted grid might have a better view.

# In[ ]:


g = sb.FacetGrid(data = job_sub, col = 'job',
                 hue = 'job', height = 4, aspect = 1.25, hue_order = top_five_job)
g.map(sb.scatterplot, 'total_benefit', 'log_total_salary');
g.add_legend();


# > A faceted view shows that Registered Nurse is the most strongly correlated between `total_salary` and `total_benefit` because the points are more tightly clustered. Police Office 3 is the weakest along with Firefighter, which makes sense because those jobs had the most variability in `total_salary` and `total_benefit`. Transit Operator and Custodian follow similiar trends because they are the least skilled workers and earlier were shown to have the least variability in pay. 

# In[ ]:


sb.pairplot(job_sub, hue = 'job', vars = ['log_total_salary', 'total_benefit', 'log_total_comp'],
            hue_order = top_five_job);


# ### Talk about some of the relationships you observed in this part of the investigation. Were there features that strengthened each other in terms of looking at your feature(s) of interest?
# 
# > The `job` and the `org_group` seemed to change the interaction between the salary and benefit variables in regards to `total_comp`. The nature of the job itself and the organization that the job was clustered in seemed to have the biggest strength on the interaction between `total_benefit` and `total_salary`. 
# 
# ### Were there any interesting or surprising interactions between features?
# 
# > The interaction between `total_benefit` and `total_salary` was lower than previously thought but it was mostly do to the type of job. This highly affected whether `total_benefit` or `total_salary` contributed the most to `total_comp`.
