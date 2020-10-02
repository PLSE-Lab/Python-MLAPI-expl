#!/usr/bin/env python
# coding: utf-8

# ## What impacts the percentage of students who go to college?
# 
# ### 1. Economic disadvantage is the most significant predictor of college attendance rates
# 
# Massachusetts DOE classifies students as 'economically disadvantaged' if they participate in one of the following programs:
# * Supplemental Nutrition Assistance Program (SNAP)
# * The Transitional Assistance for Families with Dependent Children (TAFDC)
# * The Department of Children and Families' (DCF) foster care program
# * MassHealth (Medicaid)
# 
# ### 2. Charter and pilot schools outperform public schools with similar levels of economic disadvantage
# 
# Both Charter schools and Pilot schools operate autonomously. Pilot schools are part of the Boston Public School system and have to comply with a subset of the Union rules which apply to Public schools.

# | |NOTEBOOK SECTION  | ANALYSIS   |
# | :---| :--- |:---|
# | 1.| PREPROCESSING | 1.1 Loading packages <br> 1.2 Reading the data <br> 1.3 Calculating the dependent variable <br> 1.4 Cleaning up the data <br> 1.5 Selecting variables|
# | 2.| INITIAL <br> REGRESSION <br> MODEL | 2.1 Running the regression <br> 2.2 Testing linearity <br> 2.3 Testing homoskedasticity <br> 2.4 Removing outliers|
# | 3.| REFINED <br> REGRESSION <br> MODEL | 3.1 Re-running the regression <br> 3.2 Refining the model <br> 3.3 Testing OLS assumptions  |

# # 1. PREPROCESSING 
# ## 1.1 Loading packages 

# In[ ]:


# Load packages
import warnings
warnings.simplefilter(action='ignore', 
                      category=FutureWarning)      # suppress warnings
import numpy as np                                 # linear algebra
import pandas as pd                                # data analysis
import matplotlib.pyplot as plt                    # visualization
import seaborn as sns                              # visualization
import scipy.stats as scipystats                   # statistics  
import statsmodels.formula.api as smf              # statistics
from statsmodels.api import add_constant           # statistics
from sklearn.feature_selection import SelectKBest  # feature selection
from sklearn.feature_selection import f_regression # feature selection

pd.set_option('display.float_format', lambda x: '%.1f' % x) # format decimals
sns.set(font_scale=1.5) # increse font size for seaborn charts
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 1.2 Reading the data
# The dataset includes elementary, middle, and high schools. We'll filter the data to only include schools with students in 12th grade:

# In[ ]:


SCHOOLS = pd.read_csv('../input/MA_Public_Schools_2017.csv', dtype={'School Code':'str',
                                                                    'District Code':'str'} )
HS = SCHOOLS.loc[SCHOOLS['12_Enrollment'] > 0].reset_index(drop=True)
print ("Rows: ",HS.shape[0],"   Variables: ", HS.shape[1])


# ## 1.3 Calculating the dependent variable
# The first task is to calculate the percentage of students who attend 4-year colleges. The dataset contains:
# 1. The percentage of high school graduates who attend public 4-year colleges and 
# 2. The percentage of high school graduates who attend private 4-year colleges. 
# 
# We'll need to multiply these two variables together to find the percentage of students attending 4-year colleges as a percentage of their high-school cohort. 
# 
# We'll drop any schools where we have a missing value for the target variable, and further restrict the set of schools to those with at least 30 students in 12th grade:

# In[ ]:


HS['4YR_College_%'] = HS['% Private Four-Year'] + HS['% Public Four-Year']
HS['4YR_College_%']= HS['4YR_College_%'] * (HS['% Graduated']/100)
HS = HS.drop(['% Private Four-Year',
              '% Public Four-Year',
              '% Graduated',
              '% Public Two-Year',
              '% MA Community College',
              '% Attending College'],axis=1)
HS_30 = HS.loc[HS['12_Enrollment'] >= 30].dropna(subset=['4YR_College_%']).reset_index(drop=True)

plt.figure(figsize=(15,5))
plt.hist(HS_30['4YR_College_%']); # distribution of 4-year college attendance
plt.title('Distribution of Percentage of students attending 4-year College by School');


# ## 1.3 Cleaning up the data
# As a first step to cleaning up the data, let's look at fields that have a lot of null values:

# In[ ]:


HS_30.isnull().sum()[HS_30.isnull().sum()>100].head()


# Clearly, there are variables that only apply to elementary and middle school grades. Let's exclude any variables where more than half the values are null.

# In[ ]:


HS_30 = HS_30[HS_30.columns[ ( (pd.isnull(HS_30).sum()) / (HS_30.shape[0]) < 0.5 ).values ]]
HS_30.head()


# Most of the non-numeric fields (e.g., address fields) won't be useful for the analysis. There are a handful of non-numeric varibles that are interesting (e.g., Accountability and Assistance Level), so we should consider encoding these later. However, for now, let's just drop the non-numeric fields.

# In[ ]:


ST = HS_30['School Type'] # store School Type before dropping non-numeric fields
HS_30.set_index('School Name', inplace=True) # set the index to the School Name
HS_30 = HS_30.select_dtypes(include=[np.number]) # drop on-numeric fields
print ("Rows: ",HS_30.shape[0],"   Variables: ", HS_30.shape[1])


# Now we're down to 109 variables - still plenty to work with! For variables that still have missing values, let's replace them with the median for that variable.

# In[ ]:


HS_30 = HS_30.fillna(HS_30.median()) # fill in missing values


# ## 1.4 Selecting variables
# To narrow down the list of variables, we'll use SelectKBest, a SciKitLearn function.<br>(Note: To use this function, we'll need to convert our DataFrame to a NumPy array.)

# In[ ]:


y = HS_30['4YR_College_%'].values
X_df = HS_30.select_dtypes(include=[np.number]).drop(['4YR_College_%'],axis=1)
X = X_df.values
row_names = HS_30.index.values # store row names
col_names = X_df.columns.values # store column names

selector = SelectKBest(f_regression, k=20)
HS_30 = pd.DataFrame(selector.fit_transform(X, y))

vars =  (np.array(col_names)[selector.get_support()]) # variable names
fstats = (selector.scores_)[selector.get_support()] # F-statistics
for a,b in zip(vars,fstats):
   print ( "{0:40} {1:.0f}".format(a, b) )


# We can see that the percentage of economically disadvantaged students has the highest F-statistic. Before we look more closely at the other variables, let's convert the data back to a DataFrame:

# In[ ]:


HS_30.columns = (np.array(col_names)[selector.get_support()]) # restore column names
HS_30['School Name'] = row_names # restore row names
HS_30.set_index('School Name', inplace=True) # restore index
HS_30['4YR_College_%'] = y # restore dependent variable
HS_30['School Type']=ST.values # add back 'School Type'


# To avoid issues with collinearity, we need regression varibles that aren't highly correlated. Let's take a look at the correlation matrix:

# In[ ]:


fig, ax = plt.subplots(figsize=(12,12)) 
sns.heatmap(HS_30.corr(), linewidths=0.1,cbar=True, annot=True, square=True, fmt='.1f')
plt.title('Correlation between Variables');


# As almost all the variables are highly correlated, we'll start off with a simple linear regression using the variable with the strongest relationship with college attendance - the percentage of economically disadvantaged students.

# # 2. INITIAL MODEL
# ## 2.1 Running the regression

# In[ ]:


X = add_constant(HS_30[['% Economically Disadvantaged']])
Y = HS_30['4YR_College_%']
regr = smf.OLS(Y,X).fit()
regr.summary()


# This model is a reasonable fit with the data - the R-squared is high and the independent variable has a zero p-value. Let's check the standard assumptions for ordinary least squares regression...

# ## 2.2 Testing Linearity
# The scatter plot suggests that the relationship is linear:

# In[ ]:


sns_plot = sns.lmplot(x='% Economically Disadvantaged', y='4YR_College_%',data=HS_30,size = 10)
plt.title('Relationship between Economic Disadvantage & College Attendance');


# ## 2.3 Testing Homoscedasticity
# To check for homoscedasticity (homogeneity in the variance of the residuals), let's plot the residuals plotted against the fitted values...

# In[ ]:


plt.figure(figsize=(10,10))
plt.scatter(regr.predict(), regr.resid)
plt.title('Residuals versus Predicted Values of 4-Year College Attendance');


# The spread of data points on the left of the chart is clearly wider than on the right. To better understand what's going on, we'll take a look at the observations with the largest residuals...

# In[ ]:


HS_30 = HS_30[['School Type','% Economically Disadvantaged','4YR_College_%']]
HS_30  = pd.concat([HS_30, pd.Series(regr.resid, name = 'resid')], axis = 1)
HS_30  = HS_30.sort_values(ascending=False,by=['resid'])
HS_30.loc[HS_30['resid'] > 20]


# There are exam schools and pilot schools on this list. To make it easier to interpret, let's tag Boston's there exame schools and nine pilot schools, then re-run the list of schools with positive residuals...

# ## 2.4 Removing outliers

# In[ ]:


HS_30.loc[(HS_30.index == "Boston Latin Academy") | 
            (HS_30.index == "O'Bryant School Math/Science") |
            (HS_30.index == "Boston Latin"), 'School Type'] = 'Exam School'
                   
HS_30.loc[(HS_30.index == "Another Course To College") | 
            (HS_30.index == "Boston Arts Academy") | 
            (HS_30.index == "Boston Community Leadership Academy") |
            (HS_30.index == "Fenway High School") |
            (HS_30.index == "Greater Egleston Community High School") |
            (HS_30.index == "Lyon Upper 9-12") |
            (HS_30.index == "New Mission High School") |
            (HS_30.index == "Quincy Upper School") |
            (HS_30.index == "TechBoston Academy"), 'School Type'] = 'Pilot School'

HS_30.loc[HS_30['resid'] > 20].sort_values(ascending=True,by=['School Type'])


# Almost all the schools with high positive residuals come from the relatively small population of exam, charter and pilot schools. 
# Let's also take a look at the schools with high negative residuals:

# In[ ]:


HS_30.loc[HS_30['resid'] < -20].sort_values(ascending=True,by=['resid'])


# The pattern here is a little less obvious, but a quick search reveals that a number of these schools are specialist schools:
# 
# * **Boston Adult Academy** is 'for mature, motivated students between the ages of 19-22'*
# 
# * **Edison Academy** 'offers instructional support and intervention strategies that reconnect students who are over aged and under credited for grade level, and are either at-risk of, or have already dropped out of school' 
# 
# * **The Gateway to College** 'serves students between the ages of 16 and 21 who have left high school, or are struggling to finish school.' 
# 
# * **Greater Eggleston Communnity** has a program for 'overage students who need to work to support their families, and/or students who are parents or students who have medically related circumstances'.
# 
# * **Boston Day and Evening Academy Charter School** 'was created to serve any Boston Public School student who is overage for high school, who has had trouble with attendance issues, has been held back in 8th grade, who feels they are not getting the attention in class that they need to succeed, or who has dropped out but is eager to come back to school to earn their diploma.'
# 
# Exam schools, charter schools, pilot schools and specialist schools operate very differently than traditional high schools. So, for now, let's exclude them from our analysis.

# In[ ]:


HS_30.loc[(HS_30.index == "Boston Adult Academy") | 
          (HS_30.index == "Edison Academy") |
          (HS_30.index == "The Gateway to College") |
          (HS_30.index == "Greater Egleston Community High School") |
          (HS_30.index == "Boston Day and Evening Academy Charter School"), 
          'School Type'] = 'Specialist School'           
            
HS_TRAD =  HS_30.loc[(HS_30['School Type'] == 'Public School')].reset_index(drop=True) 


# # 3. REFINED MODEL
# ## 3.1 Re-running the regression
# Now that we've elimiated the outliers, let's rerun the regression:

# In[ ]:


X = add_constant(HS_TRAD[['% Economically Disadvantaged']])
Y = HS_TRAD['4YR_College_%']
regr = smf.OLS(Y,X).fit()
HS_TRAD  = pd.concat([HS_TRAD, pd.Series(regr.resid, name = 'resid')], axis = 1)
regr.summary()


# The fit of the model has improved significantly, with the R-squared going up to 0.754.

# ## 3.2 Refining the model
# We had previously excluded non-traditional schools. Let's take another look at how these schools perform compared to traditional schools:

# In[ ]:


sns_scatter = sns.pairplot(HS_30,
                 x_vars=['% Economically Disadvantaged'],
                 y_vars=['4YR_College_%'],
                 hue='School Type',
                 markers=["x", "o",'D','x','o'],
                 size = 10)
plt.title('Relationship between Economic Disadvantage and College Attendance');


# We can see that exam, charter and pilot schools seem to perform better than traditional schools, especially in schools with a more economically-disadvantaged student body. This is even more obvious if we look at the top 10 schools in terms of 4-college attendance, among schools with more than 30% economically disadvantaged students:

# In[ ]:


HS_30.loc[HS_30['% Economically Disadvantaged'] > 30].sort_values(
    ascending=False,by=['4YR_College_%']).head(10)


# Let's try to quantify the impact of being a charter or pilot school. We'll add these schools back into the dataset and add another variable. Because being a charter or pilot schools seems to matter more for greater levels of economic disadvantage, the new variable we'll add is (dummy variable for Charter or Pilot) * (% Economically Disadvantaged).

# In[ ]:


mask = ( (HS_30['School Type'] == 'Public School') | 
         (HS_30['School Type'] == 'Charter School') | 
         (HS_30['School Type'] == 'Pilot School') )
HS_TCP =  HS_30.loc[mask].copy() 

mask = (HS_TCP['School Type'] == "Charter School") | (HS_TCP['School Type'] == "Pilot School")
HS_TCP['Indep_School'] = np.where(mask, 1, 0)
HS_TCP['Indep_School_Mult_Econ'] = HS_TCP['Indep_School'] * HS_TCP['% Economically Disadvantaged']


# Now, let's rerun the regression including the new variable:

# In[ ]:


X = add_constant(HS_TCP[['% Economically Disadvantaged','Indep_School_Mult_Econ']])
Y = HS_TCP['4YR_College_%']
regr = smf.OLS(Y,X).fit()
regr.summary()


# We can see that this model is a very good fit with the data, and the p-value for both variables is very low. Being a charter or pilot schools effectively halves the impact of economic disadvantage. 

# ## 3.3 Testing OLS Assumptions

# In[ ]:


plt.figure(figsize=(10,10))
plt.scatter(regr.predict(), regr.resid)
plt.title('Residuals versus Predicted Values of 4-Year College Attendance');


# The scatter plot suggests that we don't have any signicant bias in the residuals.

# In[ ]:


HS_TCP = HS_TCP[['School Type','% Economically Disadvantaged','4YR_College_%','Indep_School_Mult_Econ']]
HS_TCP  = pd.concat([HS_TCP, pd.Series(regr.resid, name = 'resid')], axis = 1)
HS_TCP = HS_TCP.sort_values(ascending=False,by=['resid'])
HS_TCP.loc[HS_TCP['resid'] > 30]


# The regression shows that the relationship between economic disadvantage is a very strong predictor of how many kids will go on to attend college. To end on a more positive note, it's encouraging to see one school that bucks the trend - the University Park Campus school in Worcester achieves strong 4-year college attendance rates despite drawing students from a poor neighborhood. Perhaps there are some lessons to be learned from their experience...
