#!/usr/bin/env python
# coding: utf-8

# # The Challenge
# PASSNYC aims to increase the diversity of students taking the Specialized High School Admissions Test (SHSAT).
# They've asked the Kaggle communiy to help "*identify schools where minority and underserved students stand to gain the most from services like after school programs, test preparation, mentoring, or resources for parents.*"

# # What I Did
# While going through SHSAT resources, particuarly [the Pathways to an Elite Education report](https://steinhardt.nyu.edu/research_alliance/publications/pathways_to_an_elite_education) and  [Data from the New School Center for NYC Affairs](http://www.centernyc.org/high-school-diversity-data/), I was struck by the numbers of minority and economically disadvantaged students who score highly on Common Core tests, yet do not apply to or receive offers to New York City's elite specialized high schools. Since a major part of PASSNYC's mission is to identify such "diamonds in the rough," I decided to bring a data-oriented approach to this topic.
# 
# My hypothesis is that students with strong Common Core test scores will be easiest to convert into SHSAT test takers and to receive offers. Their existing academic ability means they'll be especially receptive to awareness outreach and convert to Elite High School offers with minimal mentoring or SHSAT-specific test prep from PASSNYC's partners.
# 
# Apart from the school explorer data, I also used the following data to build my solution:
# * *nyc-shsat-test-results-2017* to measure SHSAT offers
# * *ny-school-demographics-and-accountability-snapshot* from Socrata NYC Open Data  for enrollment information
# * *ny-2010-2016-school-safety-report* from Socrata NYC Open Data for crime statistics and to measure distances from the middle schools to the elite high schools.

# # Results
# I find that Common Core results and attendance are the best ways to identify schools that should recieve many SHSAT high school offers. 
# 
# And I found dozens of schools with these positive characteristics yet did not receive a single offer to a specialized high school in 2017.  These 30 schools are generally made up of the underrepresented populations that PASSNYC wants to help.
# 
# The top 30 most promising schools are displayed in a table at the very bottom of this report, with the full detailed rankings of all 580 schools saved to a csv file.
# 
# Surprisingly, neither crime statistics nor distance to the specialized schools had any impact on whether a school's students received offers.
# 

# 
# 
# ### My solution has two steps:   
# 
# **Part I** takes this forum post from Max B of PASSNYC as a starting point:
# 
# *"The hypothesis is that using what we know about students/schools who do take the test, we can find similar students/schools and rank them on their likelihood/opportunity of converting into test-takers"*
# 
# I use machine learning to find the relationship between school characteristics and SHSAT offers per student. Comparing  actual SHSAT offers per student with the model's expectations gives a  over or underperformance, and schools that are doing worse than they should be are ripe for outreach.  
# 
# **Part II** specifically addresses this portion of the PASSNYC Challenge problem statement: 
# 
# *"The best solutions will enable PASSNYC to identify the schools where minority and underserved students stand to gain the most from services"*
# 
# I created an *attractiveness score* for each school in the data set, ** _that PASSNYC can adapt to match their goals_ ** based on each school's share of the groups in PASSNYC's mission (for example the economically disadvanted, minorities, or SHSAT nonfeeder schools). The score can be changed with user input from PASSNYC, depending on which groups they want to emphasize for a particulat type of outreach.
# 
# ** Putting it all together** by using over / underperform rankings in conjuction with the adaptable attractiveness score, schools can be quickly sorted and filtered by ** _both_ untapped academic potential _and_ alignment with PASSNYC's mission**.  
# 

# # Data Prep
# The following couple sections of data prep can be skipped if you're here for high level results only.  The story picks up again in the section entitled *Data Prep Results*. 

# In[ ]:


# Loading python libraries
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
cm = sns.light_palette("grey", as_cmap=True)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
import shap
import os
import math

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter('ignore')


# ### SCHOOL Explorer Data Prep

# In[ ]:


############################################################
# Load the school explorer data and clean it
SCHOOL = pd.read_csv('../input/data-science-for-good/2016 School Explorer.csv', index_col=["Location Code"])
SCHOOL.columns = [x.upper() for x in SCHOOL.columns]
SCHOOL.index.name = 'DBN'

# Convert to a numeric value (i.e. 3174.99 instead of $3,174.99)
dollars = ["SCHOOL INCOME ESTIMATE"]
for col in dollars:
    SCHOOL[col] = SCHOOL[col].str.replace(',', '')
    SCHOOL[col] = SCHOOL[col].str[1:].astype(float)
# Convert these to numerics
percents = ["PERCENT ELL", "PERCENT ASIAN", "PERCENT BLACK", "PERCENT HISPANIC", 
            "PERCENT BLACK / HISPANIC", "PERCENT WHITE", "STUDENT ATTENDANCE RATE",
            "PERCENT OF STUDENTS CHRONICALLY ABSENT", "RIGOROUS INSTRUCTION %", 
            "COLLABORATIVE TEACHERS %", "SUPPORTIVE ENVIRONMENT %", 
            "EFFECTIVE SCHOOL LEADERSHIP %", "STRONG FAMILY-COMMUNITY TIES %", "TRUST %"]
for col in percents:
    SCHOOL[col] = SCHOOL[col].str[:-1].astype(float) / 100
# I'm just going to drop these ratings since we already have a % version of each
ratings = ["RIGOROUS INSTRUCTION RATING", "COLLABORATIVE TEACHERS RATING",
          "SUPPORTIVE ENVIRONMENT RATING", "EFFECTIVE SCHOOL LEADERSHIP RATING",
          "STRONG FAMILY-COMMUNITY TIES RATING", "TRUST RATING", "STUDENT ACHIEVEMENT RATING"]
SCHOOL = SCHOOL.drop(ratings,axis=1)

# Save school name and address for later use
SCHOOL_NAMES = SCHOOL[["SCHOOL NAME", "ADDRESS (FULL)"]]

# Save latitude and longitude to calculate distance from closest elite school
DISTANCE = SCHOOL[['LATITUDE', 'LONGITUDE']]

# I didn't attempt to use these variables
other_data_not_used = ["SCHOOL INCOME ESTIMATE", "OTHER LOCATION CODE IN LCGMS", "SCHOOL NAME", "SED CODE", "ADDRESS (FULL)", "GRADES", 'CITY', 'LATITUDE', 'LONGITUDE', 'ZIP']
SCHOOL = SCHOOL.drop(other_data_not_used, axis=1)

# Some more simple data cleaning / preprocessing
SCHOOL[["ADJUSTED GRADE", "NEW?"]] = SCHOOL[["ADJUSTED GRADE", "NEW?"]].replace("x", 1).fillna(0)
SCHOOL["GRADE LOW"] = SCHOOL["GRADE LOW"].replace("0K", 0).replace("PK", -1).astype(float)
SCHOOL["GRADE HIGH"] = SCHOOL["GRADE HIGH"].replace("0K", 0).astype(float)
SCHOOL["COMMUNITY SCHOOL?"] = SCHOOL["COMMUNITY SCHOOL?"].replace("Yes", 1).replace("No", 0).astype(float)

# There are a massive amount of common core Result variable.  I treat these in the following way:
# 1. Sum up all the 4s scored in ethnic, economic need and ELL sub-categories across all grade levels
#    separately for both ELA and MATH
# 2. Divide by the total number of students tested in all these grade levels
# 3. The result is the total fraction of the student body that both received a 4 and belonged to
#    that sub-category.  So for example, if my new variable MATH ELL% was 0.1 for a school,
#    then that 10% of that school's student body was an ELL student who also received a 4 in MATH.
# 4. Finally, I average ELA and MATH together
SCHOOL["ELA TESTED"] = SCHOOL[["GRADE 3 ELA - ALL STUDENTS TESTED", "GRADE 4 ELA - ALL STUDENTS TESTED", "GRADE 5 ELA - ALL STUDENTS TESTED", "GRADE 6 ELA - ALL STUDENTS TESTED", "GRADE 7 ELA - ALL STUDENTS TESTED", "GRADE 8 ELA - ALL STUDENTS TESTED"]].sum(1)
SCHOOL["ELA ALL 4%"] = SCHOOL[["GRADE 3 ELA 4S - ALL STUDENTS", "GRADE 4 ELA 4S - ALL STUDENTS", "GRADE 5 ELA 4S - ALL STUDENTS", "GRADE 6 ELA 4S - ALL STUDENTS", "GRADE 7 ELA 4S - ALL STUDENTS", "GRADE 8 ELA 4S - ALL STUDENTS"]].sum(1)
SCHOOL["ELA AAALN 4%"] = SCHOOL[["GRADE 3 ELA 4S - AMERICAN INDIAN OR ALASKA NATIVE", "GRADE 4 ELA 4S - AMERICAN INDIAN OR ALASKA NATIVE", "GRADE 5 ELA 4S - AMERICAN INDIAN OR ALASKA NATIVE", "GRADE 6 ELA 4S - AMERICAN INDIAN OR ALASKA NATIVE", "GRADE 7 ELA 4S - AMERICAN INDIAN OR ALASKA NATIVE", "GRADE 8 ELA 4S - AMERICAN INDIAN OR ALASKA NATIVE"]].sum(1)
SCHOOL["ELA BLACK 4%"] = SCHOOL[["GRADE 3 ELA 4S - BLACK OR AFRICAN AMERICAN", "GRADE 4 ELA 4S - BLACK OR AFRICAN AMERICAN", "GRADE 5 ELA 4S - BLACK OR AFRICAN AMERICAN", "GRADE 6 ELA 4S - BLACK OR AFRICAN AMERICAN", "GRADE 7 ELA 4S - BLACK OR AFRICAN AMERICAN", "GRADE 8 ELA 4S - BLACK OR AFRICAN AMERICAN"]].sum(1)
SCHOOL["ELA LATINO 4%"] = SCHOOL[["GRADE 3 ELA 4S - HISPANIC OR LATINO", "GRADE 4 ELA 4S - HISPANIC OR LATINO", "GRADE 5 ELA 4S - HISPANIC OR LATINO", "GRADE 6 ELA 4S - HISPANIC OR LATINO", "GRADE 7 ELA 4S - HISPANIC OR LATINO", "GRADE 8 ELA 4S - HISPANIC OR LATINO"]].sum(1)
SCHOOL["ELA ASIAN 4%"] = SCHOOL[["GRADE 3 ELA 4S - ASIAN OR PACIFIC ISLANDER", "GRADE 4 ELA 4S - ASIAN OR PACIFIC ISLANDER", "GRADE 5 ELA 4S - ASIAN OR PACIFIC ISLANDER", "GRADE 6 ELA 4S - ASIAN OR PACIFIC ISLANDER", "GRADE 7 ELA 4S - ASIAN OR PACIFIC ISLANDER", "GRADE 8 ELA 4S - ASIAN OR PACIFIC ISLANDER"]].sum(1)
SCHOOL["ELA WHITE 4%"] = SCHOOL[["GRADE 3 ELA 4S - WHITE", "GRADE 4 ELA 4S - WHITE", "GRADE 5 ELA 4S - WHITE", "GRADE 6 ELA 4S - WHITE", "GRADE 7 ELA 4S - WHITE", "GRADE 8 ELA 4S - WHITE"]].sum(1)
SCHOOL["ELA MULTIRACIAL 4%"] = SCHOOL[["GRADE 3 ELA 4S - MULTIRACIAL", "GRADE 4 ELA 4S - MULTIRACIAL", "GRADE 5 ELA 4S - MULTIRACIAL", "GRADE 6 ELA 4S - MULTIRACIAL", "GRADE 7 ELA 4S - MULTIRACIAL", "GRADE 8 ELA 4S - MULTIRACIAL"]].sum(1)
SCHOOL["ELA ECON 4%"] = SCHOOL[["GRADE 3 ELA 4S - ECONOMICALLY DISADVANTAGED", "GRADE 4 ELA 4S - ECONOMICALLY DISADVANTAGED", "GRADE 5 ELA 4S - ECONOMICALLY DISADVANTAGED", "GRADE 6 ELA 4S - ECONOMICALLY DISADVANTAGED", "GRADE 7 ELA 4S - ECONOMICALLY DISADVANTAGED", "GRADE 8 ELA 4S - ECONOMICALLY DISADVANTAGED"]].sum(1)
SCHOOL["ELA ELL 4%"] = SCHOOL[["GRADE 3 ELA 4S - LIMITED ENGLISH PROFICIENT", "GRADE 4 ELA 4S - LIMITED ENGLISH PROFICIENT", "GRADE 5 ELA 4S - LIMITED ENGLISH PROFICIENT", "GRADE 6 ELA 4S - LIMITED ENGLISH PROFICIENT", "GRADE 7 ELA 4S - LIMITED ENGLISH PROFICIENT", "GRADE 8 ELA 4S - LIMITED ENGLISH PROFICIENT"]].sum(1)

SCHOOL["MATH TESTED"] = SCHOOL[["GRADE 3 MATH - ALL STUDENTS TESTED", "GRADE 4 MATH - ALL STUDENTS TESTED", "GRADE 5 MATH - ALL STUDENTS TESTED", "GRADE 6 MATH - ALL STUDENTS TESTED", "GRADE 7 MATH - ALL STUDENTS TESTED", "GRADE 8 MATH - ALL STUDENTS TESTED"]].sum(1)
SCHOOL["MATH ALL 4%"] = SCHOOL[["GRADE 3 MATH 4S - ALL STUDENTS", "GRADE 4 MATH 4S - ALL STUDENTS", "GRADE 5 MATH 4S - ALL STUDENTS", "GRADE 6 MATH 4S - ALL STUDENTS", "GRADE 7 MATH 4S - ALL STUDENTS", "GRADE 8 MATH 4S - ALL STUDENTS"]].sum(1)
SCHOOL["MATH AAALN 4%"] = SCHOOL[["GRADE 3 MATH 4S - AMERICAN INDIAN OR ALASKA NATIVE", "GRADE 4 MATH 4S - AMERICAN INDIAN OR ALASKA NATIVE", "GRADE 5 MATH 4S - AMERICAN INDIAN OR ALASKA NATIVE", "GRADE 6 MATH 4S - AMERICAN INDIAN OR ALASKA NATIVE", "GRADE 7 MATH 4S - AMERICAN INDIAN OR ALASKA NATIVE", "GRADE 8 MATH 4S - AMERICAN INDIAN OR ALASKA NATIVE"]].sum(1)
SCHOOL["MATH BLACK 4%"] = SCHOOL[["GRADE 3 MATH 4S - BLACK OR AFRICAN AMERICAN", "GRADE 4 MATH 4S - BLACK OR AFRICAN AMERICAN", "GRADE 5 MATH 4S - BLACK OR AFRICAN AMERICAN", "GRADE 6 MATH 4S - BLACK OR AFRICAN AMERICAN", "GRADE 7 MATH 4S - BLACK OR AFRICAN AMERICAN", "GRADE 8 MATH 4S - BLACK OR AFRICAN AMERICAN"]].sum(1)
SCHOOL["MATH LATINO 4%"] = SCHOOL[["GRADE 3 MATH 4S - HISPANIC OR LATINO", "GRADE 4 MATH 4S - HISPANIC OR LATINO", "GRADE 5 MATH 4S - HISPANIC OR LATINO", "GRADE 6 MATH 4S - HISPANIC OR LATINO", "GRADE 7 MATH 4S - HISPANIC OR LATINO", "GRADE 8 MATH 4S - HISPANIC OR LATINO"]].sum(1)
SCHOOL["MATH ASIAN 4%"] = SCHOOL[["GRADE 3 MATH 4S - ASIAN OR PACIFIC ISLANDER", "GRADE 4 MATH 4S - ASIAN OR PACIFIC ISLANDER", "GRADE 5 MATH 4S - ASIAN OR PACIFIC ISLANDER", "GRADE 6 MATH 4S - ASIAN OR PACIFIC ISLANDER", "GRADE 7 MATH 4S - ASIAN OR PACIFIC ISLANDER", "GRADE 8 MATH 4S - ASIAN OR PACIFIC ISLANDER"]].sum(1)
SCHOOL["MATH WHITE 4%"] = SCHOOL[["GRADE 3 MATH 4S - WHITE", "GRADE 4 MATH 4S - WHITE", "GRADE 5 MATH 4S - WHITE", "GRADE 6 MATH 4S - WHITE", "GRADE 7 MATH 4S - WHITE", "GRADE 8 MATH 4S - WHITE"]].sum(1)
SCHOOL["MATH MULTIRACIAL 4%"] = SCHOOL[["GRADE 3 MATH 4S - MULTIRACIAL", "GRADE 4 MATH 4S - MULTIRACIAL", "GRADE 5 MATH 4S - MULTIRACIAL", "GRADE 6 MATH 4S - MULTIRACIAL", "GRADE 7 MATH 4S - MULTIRACIAL", "GRADE 8 MATH 4S - MULTIRACIAL"]].sum(1)
SCHOOL["MATH ECON 4%"] = SCHOOL[["GRADE 3 MATH 4S - ECONOMICALLY DISADVANTAGED", "GRADE 4 MATH 4S - ECONOMICALLY DISADVANTAGED", "GRADE 5 MATH 4S - ECONOMICALLY DISADVANTAGED", "GRADE 6 MATH 4S - ECONOMICALLY DISADVANTAGED", "GRADE 7 MATH 4S - ECONOMICALLY DISADVANTAGED", "GRADE 8 MATH 4S - ECONOMICALLY DISADVANTAGED"]].sum(1)
SCHOOL["MATH ELL 4%"] = SCHOOL[["GRADE 3 MATH 4S - LIMITED ENGLISH PROFICIENT", "GRADE 4 MATH 4S - LIMITED ENGLISH PROFICIENT", "GRADE 5 MATH 4S - LIMITED ENGLISH PROFICIENT", "GRADE 6 MATH 4S - LIMITED ENGLISH PROFICIENT", "GRADE 7 MATH 4S - LIMITED ENGLISH PROFICIENT", "GRADE 8 MATH 4S - LIMITED ENGLISH PROFICIENT"]].sum(1)

# I also save the total number of Grade 8 students tested as a rough proxy for stduent enrollment
number_of_ela_grades = (SCHOOL[["GRADE 3 ELA - ALL STUDENTS TESTED", "GRADE 4 ELA - ALL STUDENTS TESTED", "GRADE 5 ELA - ALL STUDENTS TESTED", "GRADE 6 ELA - ALL STUDENTS TESTED", "GRADE 7 ELA - ALL STUDENTS TESTED", "GRADE 8 ELA - ALL STUDENTS TESTED"]]>1).sum(1)
number_of_math_grades = (SCHOOL[["GRADE 3 MATH - ALL STUDENTS TESTED", "GRADE 4 MATH - ALL STUDENTS TESTED", "GRADE 5 MATH - ALL STUDENTS TESTED", "GRADE 6 MATH - ALL STUDENTS TESTED", "GRADE 7 MATH - ALL STUDENTS TESTED", "GRADE 8 MATH - ALL STUDENTS TESTED"]]>1).sum(1)
SCHOOL = SCHOOL.drop(SCHOOL.iloc[:,23:143].columns, axis=1)

# Here they are converted from total number, to fraction of student-body
elaCC = ["ELA TESTED","ELA ALL 4%","ELA AAALN 4%","ELA BLACK 4%","ELA LATINO 4%","ELA ASIAN 4%",
    "ELA WHITE 4%","ELA MULTIRACIAL 4%","ELA ECON 4%","ELA ELL 4%"]
mathCC = ["MATH TESTED","MATH ALL 4%","MATH AAALN 4%","MATH BLACK 4%","MATH LATINO 4%",
        "MATH ASIAN 4%","MATH WHITE 4%","MATH MULTIRACIAL 4%","MATH ECON 4%","MATH ELL 4%"]
for col in mathCC[1:]:
    SCHOOL[col] = SCHOOL[col] / SCHOOL["MATH TESTED"].values
for col in elaCC[1:]:
    SCHOOL[col] = SCHOOL[col] / SCHOOL["ELA TESTED"].values

COMMON_CORE4S = pd.DataFrame(data=(SCHOOL[elaCC[1:]].values + SCHOOL[mathCC[1:]].values)/ 2.0, index=SCHOOL.index, columns=["%Students scored 4", "%AAALN and scored 4", "%BLACK and scored 4", "%LATINO and scored 4", "%ASIAN and scored 4", "%WHITE and scored 4", "%MULTIRACIAL and scored 4", "%ECON NEED and scored 4", "%ELL and scored 4"])
COMMON_CORE4S = COMMON_CORE4S.round(2)
SCHOOL = SCHOOL.drop(elaCC, axis=1)
SCHOOL = SCHOOL.drop(mathCC, axis=1)

# We would like to identify schools with these traits, that are either doing well or potentially could do well
demographics = ["ECONOMIC NEED INDEX", "PERCENT ELL", "PERCENT ASIAN",
"PERCENT BLACK","PERCENT HISPANIC","PERCENT BLACK / HISPANIC","PERCENT WHITE"]
SCHOOL_DEMOGRAPHICS = SCHOOL[demographics]
SCHOOL_DEMOGRAPHICS.fillna(SCHOOL_DEMOGRAPHICS.mean(), inplace=True) 
OTHER_SCHOOL_DATA = SCHOOL.drop(demographics, axis=1)


# ### SHSAT Test Results Data

# In[ ]:


SHSAT = pd.read_csv("../input/nyc-shsat-test-results-2017/nytdf.csv", index_col="DBN")
SHSAT.columns = [x.upper() for x in SHSAT.columns]
SHSAT = SHSAT[["OFFERSPERSTUDENT"]]
SHSAT.columns = ["OffersPerStudent"]

# Join with school explorer on the DBN.  
SHSAT = SHSAT.join(OTHER_SCHOOL_DATA, how='inner').iloc[:,:1]

# convert from sting xx% to numeric 0.xx
SHSAT["OffersPerStudent"].fillna("0%", inplace=True)
SHSAT["OffersPerStudent"].replace("0", "0%", inplace=True)
SHSAT["OffersPerStudent"] = SHSAT["OffersPerStudent"].str[:-1].astype(float) / 100


# ### Socrata NY 2010-2016 School Safety Report
# * Used for crime statistics
# * Also used for the specialized high school Latitudes and Longitudes to calculate distances from them to the middle schools.

# In[ ]:


SAFETY = pd.read_csv("../input/ny-2010-2016-school-safety-report/2010-2016-school-safety-report.csv", index_col="DBN")
SAFETY = SAFETY.loc[SAFETY.index.dropna()]
SAFETY.columns = [x.upper() for x in SAFETY.columns]
EliteSchools = ["Brooklyn Latin School, The", "Brooklyn Technical High School", "Bronx High School of Science", "High School for Mathematics, Science and Engineeri", "High School of American Studies at Lehman College", "Queens High School for the Sciences at York Colleg", "Staten Island Technical High School", "Stuyvesant High School"]
ELITES = SAFETY.loc[SAFETY['LOCATION NAME'].isin(EliteSchools), ['LOCATION NAME','LATITUDE','LONGITUDE','REGISTER']]
ELITES['REGISTER'] = ELITES['REGISTER'].str.replace(',', '')
ELITES['REGISTER'] = ELITES['REGISTER'].astype(int)
ELITES = ELITES.groupby(ELITES['LOCATION NAME']).mean()
SAFETY = SAFETY[['MAJOR N','OTH N','NOCRIM N','PROP N','VIO N']]
SAFETY=SAFETY.groupby(['DBN']).mean()
SAFETY.columns = ["MAJOR CRIMES", "OTHER CRIMES", "NONCRIMINAL CRIMES", "PROPERTY CRIMES", "VIOLENT CRIMES"]


# In[ ]:


#This code lifted from last example on this stackoverflow:
#https://stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude
def distance(origin, destination):
    """
    Calculate the Haversine distance.

    Parameters
    ----------
    origin : tuple of float
        (lat, long)
    destination : tuple of float
        (lat, long)

    Returns
    -------
    distance_in_km : float

    Examples
    --------
    >>> origin = (48.1372, 11.5756)  # Munich
    >>> destination = (52.5186, 13.4083)  # Berlin
    >>> round(distance(origin, destination), 1)
    504.2
    """
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371  # km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c

    return d


# In[ ]:


for school in EliteSchools:
    DISTANCE[school] = 0.0
    EliteTuple = (ELITES.loc[school, "LATITUDE"], ELITES.loc[school, "LONGITUDE"])
    for row in DISTANCE.index:
        MidTuple = (DISTANCE.loc[row, "LATITUDE"], DISTANCE.loc[row, "LONGITUDE"])
        DISTANCE.loc[row, school] = distance(MidTuple, EliteTuple)
DISTANCE = DISTANCE.round(1)
del DISTANCE['LATITUDE']
del DISTANCE['LONGITUDE']
DISTANCE['DISTANCE_AVERAGE_SPECIALIZED_SCHOOL'] = DISTANCE.mean(1)
DISTANCE['DISTANCE_NEAREST_SPECIALIZED_SCHOOL'] = DISTANCE.min(1)
DISTANCE = DISTANCE[['DISTANCE_NEAREST_SPECIALIZED_SCHOOL', 'DISTANCE_AVERAGE_SPECIALIZED_SCHOOL']]


# ### Socrata NYC Demographics and Accountability Snapshot
# * Used to get total grade 7 enrollment to find out if school size impacts SHSATs.

# In[ ]:


ENROLLMENT = pd.read_csv("../input/ny-school-demographics-and-accountability-snapshot/2006-2012-school-demographics-and-accountability-snapshot.csv", index_col=["DBN"])
ENROLLMENT = ENROLLMENT.loc[ENROLLMENT["schoolyear"]==20112012, "grade7"]
ENROLLMENT.replace('    ', "0", inplace=True)
ENROLLMENT = ENROLLMENT.astype('int')
ENROLLMENT.name = "7thGradeEnrollment"
OTHER_SCHOOL_DATA = OTHER_SCHOOL_DATA.join(ENROLLMENT, how='left')


# # Data Prep Results
# The data is now cleaned and split into a half-dozen portions.

# ### School names and addresses

# In[ ]:


SCHOOL_NAMES.head()


# ### School percentages of SHSAT offers per Student
# This is the data that the machine-learning model will try to explain.

# In[ ]:


SHSAT.head().style.background_gradient(cmap=cm)


# ### School demographics
# All scaled from 0 to 1. I'll use this data to calculate the "attractiveness score"

# In[ ]:


SCHOOL_DEMOGRAPHICS.head().style.background_gradient(cmap=cm)


# ### Common Core Test Results.
# Each value is the percentage of school's student body, within each subcategory, *that also* scored a 4.  ELA and MATH are simply averaged together and treated equally.
# 
# For example, imagine a school enrollment of 100, inclding 10 Latino students, and 3 of those Latino students scored a 4 on their Common Core (both MATH and ELA). In this example, * %LATINO and scored 4 * would be set to 0.03
# 
# This data will also be used to make an "attractiveness score".

# In[ ]:


COMMON_CORE4S.head().style.background_gradient(cmap=cm)


# ### School Safety Report of Crimes in each School

# In[ ]:


SAFETY.head()


# ### Distance from the Specialized Schools
# Calculated from the Latitude and Longitudes.  Used distance from the nearest specialized school and average distance from all 8 specialized schools. Presumably, if the specialized school was right down the street, you might be more likely to seek an offer.

# In[ ]:


DISTANCE.head()


# ### And finally the rest of the school data
# We will feed this as inputs to the machine learning model.
# Some notes on specific columns:
# * Added 7th grade enrollment from SOCRATA to see if school size had any impact.
# * Converted GRADE LOW of kindergarden to 0 and preschool to -1.
# * Converted "x" values for NEW, ADJUSTED GRADE and COMMUNITY SCHOOL to 1.

# In[ ]:


OTHER_SCHOOL_DATA.head().style.background_gradient(cmap=cm)


# # Solution Part 1
# # Which Schools Are Under / Overperforming Expectations?
# With the data ready, we can build a model to relate school characteristics to percentages of students receiving offers from the specialized schools. 
# 
# Max B again:
# 
# *"The hypothesis is that using what we know about students/schools who do take the test, we can find similar students/schools and rank them on their likelihood/opportunity of converting into test-takers*"
# 
# The machine learning model below will do exactly that.  It will find the relationships in the data between school characteristics and SHSAT offers.  With that relationship in hand, we can compare school's expected results to actual results, and rank or identify underperforming schools.

# ### Building the machine learning model.
# Clicking the black "code" button below will show how I built a relatively simple machine learning model known as a "random forest."  For non-technical people, the key thing to understand is that a well-built machine learning model will learn the patterns that exist in the data between the input variables and the output variable (offers per student).
# 
# **Technical Details** I follow the very standard approach of using cross-validation to find model parameters. Then fit and score using all of the data points - normally this would be a big no-no, due to overfitting, but with this simple of a model and an explanation objective (rather than a model accuracy objective), I believe we can get away with it.  The alternative would be carving our the small 580 row data set into an even smaller train and test set.

# In[ ]:


# First identify a target for the model to work on, offers per student.
target = "OffersPerStudent"
# Next assemble input data: common core 4 percentages, school demographics and other school information
model_input_variables = OTHER_SCHOOL_DATA.join(COMMON_CORE4S["%Students scored 4"]).join(SAFETY, how='left').join(DISTANCE, how='left')
model_input_variables.fillna(-.01, inplace=True) # fill in missing data with an arbitary value.
model_input_variables = SHSAT[[target]].join(model_input_variables, how='inner').iloc[:,1:]

RF = RandomForestRegressor(min_samples_leaf=10, n_jobs=8, n_estimators=100, random_state=0)
# A simple grid for parameter tuning
RF_params = {"max_depth": [3,6,None],
              "max_features": [0.33,0.67,1.0],
              "min_samples_leaf": [4,9,16]}
RF_GRID = GridSearchCV(RF, RF_params, n_jobs=2, cv=2)
RF_GRID.fit(model_input_variables, SHSAT[target])
RF = RF.set_params(**RF_GRID.best_params_)
RF.fit(model_input_variables, SHSAT[target])
# delete variables which are not used or almost unused to keep the model on the simpler side
model_input_variables = model_input_variables.loc[:, RF.feature_importances_>0.01]
RF.fit(model_input_variables, SHSAT[target])
# Save the model's predictions as a new variable
SHSAT["PREDICTED"] = RF.predict(model_input_variables)
SHSAT["PREDICTED"] = SHSAT["PREDICTED"].round(2)


# A Random Forest machine learning model comes with some metrics to understand how it worked.  As the model runs, it keeps track of which variables it used to explain the patterns between the inputs and output.  The total amount  each variable was used is called the "Importance" and is shown in the chart below.
# 
# The charts shows that Common Core variables dominate the relationship, with attendance variables coming next, and all other variables rest contributing a very small amount.  We can interpret this to mean that Common Core variables best explain why schools receive or do not receive offers per student.

# In[ ]:


importances = pd.Series(index=model_input_variables.columns, data=RF.feature_importances_).sort_values(ascending=True)
importances.plot(kind='barh', figsize=(11,7), color="orange");


# ## Expected vs. actual percentages of SHSAT offers for each school.
# Now we can compare the model's expected percentages with the actual percentages and identify underperforming schools.  A plot will help us visualize and understand our results.
# 
# The plot's X-axis is the expected percentage of offers while the Y-axis is the actual SHSAT percentages.  The schools below the line of best fit are underperforming and may be especially receptive to outreach.

# In[ ]:


SHSAT["UNDERPERFORM"] = SHSAT["PREDICTED"] - SHSAT[target]
# sort all of our datasets by UNDERPERFORM
SHSAT = SHSAT.sort_values("UNDERPERFORM", ascending=False)
SCHOOL_NAMES = SCHOOL_NAMES.loc[SHSAT.index]
SCHOOL_DEMOGRAPHICS = SCHOOL_DEMOGRAPHICS.loc[SHSAT.index]
COMMON_CORE4S = COMMON_CORE4S.loc[SHSAT.index]
OTHER_SCHOOL_DATA = OTHER_SCHOOL_DATA.loc[SHSAT.index]
SAFETY = SAFETY.loc[SHSAT.index]
DISTANCE = DISTANCE.loc[SHSAT.index]
model_input_variables = model_input_variables.loc[SHSAT.index]
# Calculate shapley explanations to display later
explainer = shap.TreeExplainer(RF)
shap_values = explainer.shap_values(model_input_variables)[:,:-1]
# recombine all the variables so we can use them all at once if want to.
all_variables = SCHOOL_NAMES.join(SHSAT, how="inner").join(SCHOOL_DEMOGRAPHICS, how="inner").join(COMMON_CORE4S, how="inner").join(OTHER_SCHOOL_DATA, how="inner").join(SAFETY, how="left").join(DISTANCE, how="left")

# Plot expected vs actual
sns.lmplot(x="PREDICTED", y=target, data=SHSAT, fit_reg=True, markers='.', size = 6,
           palette="coolwarm", );
#           scatter_kws={'s':OTHER_SCHOOL_DATA['APPROXIMATE_ENROLLMENT_PER_GRADE']}); 


# ## Let's dig into a few specific underperforming schools
# 
# The model can tell us exactly why it made any school's prediction.
# 
# For reference, here are the top 5 underperforming schools.

# In[ ]:


SCHOOL_NAMES.join(SHSAT, how='inner').head().style.background_gradient(cmap=cm)


# ### Let's compare two of  the schools on this list: Special Music School vs. Medgar Evers College Prep
# The model expected 26% of SPECIAL MUSIC SCHOOL's enrollment to receive an offer (although none did, a 26% underperformance!).  
# 
# This following chart shows how much each variable contributed to the model's 26% prediction.
#  (You can see the raw data values in paranthesis next to each column's name)
#  
#  We can see that the Common Core scores were very good on average, with a school average of 3.67 ELA and 3.85 for Math.  The model liked those scores a lot, which is why they contributed so much.  Attendance was also very good.

# In[ ]:


row = SCHOOL_NAMES.index.get_loc("03M859")
print(SCHOOL_NAMES.iloc[row])
index = model_input_variables.columns + " (" + model_input_variables.iloc[row].astype(str) + ")"
pd.Series(index=index, data=shap_values[row]).sort_values(ascending=True).plot(kind='barh', figsize=(11,7));


# ### And now, here is why the model expected MEDGAR EVERS COLLEGE PREP to receive SHSAT offers
# Again, very good Common Core and expecially good attendance rates.

# In[ ]:


row = SCHOOL_NAMES.index.get_loc("17K590")
print(SCHOOL_NAMES.iloc[row])
index = model_input_variables.columns + " (" + model_input_variables.iloc[row].astype(str) + ")"
pd.Series(index=index, data=shap_values[row]).sort_values(ascending=True).plot(kind='barh', figsize=(11,7));


# ### How are these previous two schools alike? How are they different?
# These two schools share 3 things in common: good Common Core scores, good attendance, but SHSAT specialized high school offers.  Consequantly, they lead in underperforming expectations.
# 
# However, when we look at demographics, we see they are very different in ethnic composition and economic need.

# In[ ]:


SCHOOL_NAMES.join(SCHOOL_DEMOGRAPHICS).loc[["03M859", "17K590"]].style.background_gradient(cmap=cm, axis=1)


# The difference in demographics suggests several things:
# * Quite different outreach may be needed in these different school.  Economic levels are different, not to mention the fact that the students at Special Music School may be interested in a specialized Music high school, rather than SHSAT-entry schools
# * The second school may be a better fit to PASSNYC's stated mission, as the economic conditions are lower and underrepresented minorities much higher.  Which leads to Part 2 of my solution... 

# # Solution Part 2
# ### School Attractiveness to PASSNYC's  Mission
# PASSNYC's mission is to increase diversity, serve the economically disadvantaged and counter the feeder school trend.  To that end, I've created parameters to weight these factors relative to one another as the organizers wish.
# 
# I've set some initial weights as a starting point to demonstrate my approach, but I want to emphasize that my goal is to provide a *flexible and dynamic* way for PASSNYC to balance these objectives themselves. They know their own mission better than anyone, and their mission and goals are likely to evolve and expand over time. A good solution should be able to adapt to keep up with those changes.

# In[ ]:


# First define a function to conveniantly calculate attractiveness.
def calculate_ATTRACTIVENESS():
    attract = ECON_NEED_WEIGHT * all_variables["ECONOMIC NEED INDEX"] 
    attract = attract + ELL_WEIGHT * all_variables["PERCENT ELL"]
    attract = attract + ASIAN_WEIGHT * all_variables["PERCENT ASIAN"]
    attract = attract + BLACK_WEIGHT * all_variables["PERCENT BLACK"]
    attract = attract + WHITE_WEIGHT * all_variables["PERCENT WHITE"]
    attract = attract + HISPANIC_WEIGHT * all_variables["PERCENT HISPANIC"]
    attract = attract + NONFEEDER_WEIGHT*(all_variables[target].mean()-all_variables[target])
    attract = attract + AAALN_4_WEIGHT * all_variables["%AAALN and scored 4"]
    attract = attract + BLACK_4_WEIGHT * all_variables["%BLACK and scored 4"]
    attract = attract + LATINO_4_WEIGHT * all_variables["%LATINO and scored 4"]
    attract = attract + ASIAN_4_WEIGHT * all_variables["%ASIAN and scored 4"]
    attract = attract + WHITE_4_WEIGHT * all_variables["%WHITE and scored 4"]
    attract = attract + MULTIRACIAL_4_WEIGHT * all_variables["%MULTIRACIAL and scored 4"] 
    attract = attract + ECON_4_WEIGHT * all_variables["%ECON NEED and scored 4"]
    attract = attract + ELL_4_WEIGHT * all_variables["%ELL and scored 4"]
    attract = attract / (ECON_NEED_WEIGHT + ELL_WEIGHT + ASIAN_WEIGHT + BLACK_WEIGHT + WHITE_WEIGHT + HISPANIC_WEIGHT + NONFEEDER_WEIGHT  + AAALN_4_WEIGHT + BLACK_4_WEIGHT + LATINO_4_WEIGHT + ASIAN_4_WEIGHT + WHITE_4_WEIGHT + MULTIRACIAL_4_WEIGHT + ECON_4_WEIGHT + ELL_4_WEIGHT)
    attract = attract.clip(lower=0.0)
    return attract


# ## Setting weights for school attractivness score
# Here are some weights I've initially set with a mix of PASSNYC's target groups as an example. They can be changed to any number from 0 to as high as you wish.  The variables with the highest weights will have the most influence on the school *attractiveness score*.  These weights represent my best guess on how PASSNYC is balancing their mission objectives, with some special emphasis on populations who are already successful on Common Core.
# 
# 

# In[ ]:


# First, we can assign ATTRACTIVENESS weightings to overall school demographics
ECON_NEED_WEIGHT = 1.0 # How much to weight school's Economic Need index
ELL_WEIGHT = 0.5       # How much to weight school's ELL student percentage
ASIAN_WEIGHT = 0.0     # How much to weight school's Asian student percentage
BLACK_WEIGHT = 1.0     # How much to weight school's Black student percentage
WHITE_WEIGHT = 0.0     # How much to weight school's White student percentage
HISPANIC_WEIGHT = 1.0  # How much to weight school's Hispanic student percentage
NONFEEDER_WEIGHT = 1.0 # How much to weight % of students who do not receive SHSAT offers

# We can put extra-empahsis on target groups who are already performing well on common core. 
AAALN_4_WEIGHT = 1.0       # How much to weight school's AAALN students with 4s percentage
BLACK_4_WEIGHT = 2.0       # How much to weight school's Black students with 4s percentage
LATINO_4_WEIGHT = 2.0      # How much to weight school's Latino students with 4s percentage
ASIAN_4_WEIGHT = 0.0       # How much to weight school's Asian students with 4s percentage
WHITE_4_WEIGHT = 0.0       # How much to weight school's White students with 4s percentage
MULTIRACIAL_4_WEIGHT = 1.0 # How much to weight school's Multiracial students with 4s percentage
ECON_4_WEIGHT = 2.0        # How much to weight school's Econ. disadvantaged students with 4s percentage
ELL_4_WEIGHT = 0.5         # How much to weight school's ESL students with 4s percentage
# Now create the ATTRACTIVENESS score
all_variables["ATTRACTIVENESS"] = calculate_ATTRACTIVENESS()


# ## Re-plot Expected vs. Actual SHSAT percentages, adding ATTRACTIVENESS as the color.
# To help understand what this looks like, I've replotted the expected vs actual amount of offers.  This time I also added a color to show the new attractiveness score.
# 
# More attractive (economically disadvantaged, underrepresented minorities and non-feeder schools) are in a bluer color.  Many of the attractive schools are also underperfomring . Putting those two together can build a recommendation for PASSNYC .

# In[ ]:


sns.lmplot(x="PREDICTED", y=target, data=all_variables, 
           markers='.', size = 6, fit_reg=False,
           hue="ATTRACTIVENESS",
           palette="coolwarm_r",
           legend=False,);


# ### Changing the weights to emphasize a different target population
# You can change my initial weights to anything you want.  For example, let's suppose someone at PASSNYC is especially focused on economically disadvantaged students who need help getting offers.  Simply set other weights to 0 and the economic need weights high, as I demonstrate here.  Now a different pattern emerges, as the bluer dots identify those schools that are more economically challenged.  There seem to be mix of both under and overperforming schools when attractiveness is defined in this way.

# In[ ]:


# First, we can assign ATTRACTIVENESS weightings to overall school demographics
ECON_NEED_WEIGHT = 1.0 # How much to weight school's Economic Need index
ELL_WEIGHT = 0.0       # How much to weight school's ELL student percentage
ASIAN_WEIGHT = 0.0     # How much to weight school's Asian student percentage
BLACK_WEIGHT = 0.0     # How much to weight school's Black student percentage
WHITE_WEIGHT = 0.0     # How much to weight school's White student percentage
HISPANIC_WEIGHT = 0.0  # How much to weight school's Hispanic student percentage
NONFEEDER_WEIGHT = 0.0 # How much to weight percentage of students who do not receive SHSAT offeres

# We can get extra-empahsis to target groups who are already performing well on common core. 
AAALN_4_WEIGHT = 0.0       # How much to weight school's AAALN students with 4s percentage
BLACK_4_WEIGHT = 0.0       # How much to weight school's Black students with 4s percentage
LATINO_4_WEIGHT = 0.0      # How much to weight school's Latino students with 4s percentage
ASIAN_4_WEIGHT = 0.0       # How much to weight school's Asian students with 4s percentage
WHITE_4_WEIGHT = 0.0       # How much to weight school's White students with 4s percentage
MULTIRACIAL_4_WEIGHT = 0.0 # How much to weight school's Multiracial students with 4s percentage
ECON_4_WEIGHT = 1.0        # How much to weight school's Economically disadvantaged students with 4s percentage
ELL_4_WEIGHT = 0.0         # How much to weight school's ESL students with 4s percentage

# recalculate the ATTRACTIVENESS score
all_variables["ATTRACTIVENESS"] = calculate_ATTRACTIVENESS()

# Plot the new ATTRACTIVENESS
sns.lmplot(x="PREDICTED", y=target, data=all_variables, 
           markers='.', size = 6, fit_reg=False,
           hue="ATTRACTIVENESS",
           palette="coolwarm_r",
           legend=False,); 


# ### Finally, I define attractiveness as the schools where targeted populations from non-feeder schools are already scoring high on Common Core.
# This is the population that I hypothesize would be most receptive to SHSAT awareness initatives, and most likely to convert to offers with minimal test-prep.

# In[ ]:


# First, we can assign ATTRACTIVENESS weightings to overall school demographics
ECON_NEED_WEIGHT = 0.0 # How much to weight school's Economic Need index
ELL_WEIGHT = 0.0       # How much to weight school's ELL student percentage
ASIAN_WEIGHT = 0.0     # How much to weight school's Asian student percentage
BLACK_WEIGHT = 0.0     # How much to weight school's Black student percentage
WHITE_WEIGHT = 0.0     # How much to weight school's White student percentage
HISPANIC_WEIGHT = 0.0  # How much to weight school's Hispanic student percentage
NONFEEDER_WEIGHT = 2.0 # How much to weight % of students who do not receive SHSAT offeres

# We can get extra-empahsis to target groups who are already performing well on common core. 
AAALN_4_WEIGHT = 1.0       # How much to weight school's AAALN students with 4s percentage
BLACK_4_WEIGHT = 1.0       # How much to weight school's Black students with 4s percentage
LATINO_4_WEIGHT = 1.0      # How much to weight school's Latino students with 4s percentage
ASIAN_4_WEIGHT = 0.0       # How much to weight school's Asian students with 4s percentage
WHITE_4_WEIGHT = 0.0       # How much to weight school's White students with 4s percentage
MULTIRACIAL_4_WEIGHT = 0.0 # How much to weight school's Multiracial students with 4s percentage
ECON_4_WEIGHT = 1.0        # How much to weight school's Econ. disadvantaged students with 4s percentage
ELL_4_WEIGHT = 1.0         # How much to weight school's ESL students with 4s percentage

# Now create the ATTRACTIVENESS score
all_variables["ATTRACTIVENESS"] = calculate_ATTRACTIVENESS()

# Plot the new ATTRACTIVENESS
sns.lmplot(x="PREDICTED", y=target, data=all_variables, 
           markers='.', size = 6, fit_reg=False,
           hue="ATTRACTIVENESS",
           palette="coolwarm_r",
           legend=False,);


# # Putting it All Together: Recommended Outreach Schools
# Let's put this all together give a final recommendation.  
# 
# I use the attractiveness as just defined: non-feeder schools with underrepresented groups who score high on Common Core. They fit with my original hypothesis that awareness outreach and test services will be especially effective in those schools.
# 
# Displayed below are the top 30 recommended schools, sorted by the combined rank of attractiveness and underperformance. Scrolling to the right will show more data from each school.
# 
# **_None_** of these top 30 recommended schools received a single offer during 2017 (they are not feeder schools), and demographically nearly all of them align with PSSNYC's mission very well.
# 
# The complete list of schools is saved in the file: BenS_PASSNYC_Recommendations.csv
# 
# 

# In[ ]:


recommended_schools = all_variables[["SCHOOL NAME", "ADDRESS (FULL)",target,"PREDICTED", "UNDERPERFORM", "ATTRACTIVENESS"]+list(SCHOOL_DEMOGRAPHICS.columns)+list(COMMON_CORE4S.columns[1:] )]
recommended_schools["ATTRACTIVENESS_UNDERPERFORM_COMBINED"] = recommended_schools["ATTRACTIVENESS"].rank() + recommended_schools["UNDERPERFORM"].rank()
recommended_schools["ATTRACTIVENESS"] = recommended_schools["ATTRACTIVENESS"].round(2)
recommended_schools = recommended_schools.sort_values(["ATTRACTIVENESS_UNDERPERFORM_COMBINED"], ascending=False)
del recommended_schools["ATTRACTIVENESS_UNDERPERFORM_COMBINED"]
recommended_schools.to_csv("BenS_PASSNYC_Recommendations.csv")
recommended_schools.head(30).style.background_gradient(cmap=cm)


# # Thank You!

# Thanks to PASSNYC for making this possible and answering my qustions, everyone on the Kaggle forums who commented and gave feedback or inspired me through their work, and Kaggle for taking competitions in an exciting new direction.
