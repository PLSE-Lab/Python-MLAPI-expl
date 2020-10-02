#!/usr/bin/env python
# coding: utf-8

# ### Understanding the overall objective:
# 
# > Which schools have students that would benefit from outreach services and lead to a more diverse group of students taking the SHSAT and being accepted into New York City's Specialized High Schools. <br>
# 
# <div style="text-align: right"> Chris Crawford, Kaggle Team </div> 
# 
# > Only a third (roughly) of eligible students take the SHSAT, and our goal is to drive more test takers (you can't get in if you don't sit for the test!). The education space is full of non-profits like ours with limited resources. So the research question is, given limited resources, where (at which schools) can you target your intervention efforts to make an impact on those participation numbers. The hypothesis is that using what we know about students/schools who do take the test, we can find similar students/schools and rank them on their likelihood/opportunity of converting into test-takers.
# 
# <div style="text-align: right">  Max B, Dataset Creator </div>
# 
# 
# 
# *Submissions for the Main Prize Track will be judged based on the following general criteria:*
# * Performance - How well does the solution match schools and the needs of students to PASSNYC services? PASSNYC will not be able to live test every submission, so a strong entry will clearly articulate why it is effective at tackling the problem.
# * Influential - The PASSNYC team wants to put the winning submissions to work quickly. Therefore a good entry will be easy to understand and will enable PASSNYC to convince stakeholders where services are needed the most.
# * Shareable - PASSNYC works with over 60 partner organizations to offer services such as test preparation, tutoring, mentoring, extracurricular programs, educational consultants, community and student groups, trade associations, and more. Winning submissions will be able to provide convincing insights to a wide subset of these organizations. 
# 
# ![ds for good](https://raw.githubusercontent.com/InFoCusp/kaggle_images/master/image1.png)

# ### Organisation of this notebook:
# 
# We address the above requirements with the following analysis/ outcomes:
# * [Data gathering/ preprocessing (cleaning & aligning together)](#1)
# * [Data Augmentation with relevant fields derived from raw data](#2)
# * [Identifying schools with low number of test takers and large number of high performing underrepresented (Black and Hispanic) students in grades 7 and 8 ](#3)
# * [Linking schools to test prep centers](#4)
# * [Analysing schools with high Economic Need Index](#5)
# * [Analysing and mapping Schools based on School Performance Index](#6)
# * [Analysis of school safety](#7)
# * [Classification of schools based on Percentage of English Language Learners](#8)
# * [Summary](#9)

# <h3><a id="1">Data gathering/ preprocessing </a></h3>
# 
# <h4> Datasets we have used</h4>
# 1. ***School Explorer*** (internal): provided by PASSNYC.
# 1. ***NY 2010 - 2016 School Safety Report*** (Socrata's NYC Open Data): reporting crimes in localities of different schools.
# 1. ***NY 2010-2011 Class Size - School-level detail*** (Socrata's NYC Open Data): School wise data of number of students and teachers per grade.
# 1. ***Admission data ***(external): Loading this information is very crucial to understand SHSAT admissions, registrations and offers in schools across NYC State.
# 1. ***PASSNYC resource centers data*** (external): Resource centers information we scraped from PASSNYC site with address, and they have also been categorized based on the services they provided. 
# 1.  ***8 Elite high schools data*** (external): school names and locations
#  
# <h4> Pre-processing steps </h4>
# 1. Convert the columns having values in percentage to integer.
# * Remove the schools which only has 0K as the highest grade. 
# * Remove the schools which has highest grade 5 or below.
# * Make data  "NaN-free".
# * Merge schools information with admission information.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter
import folium
import sklearn
import seaborn as sns
from IPython.core.display import display, HTML
import ipywidgets as widgets
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from IPython.display import clear_output
import itertools
import warnings
import base64

import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot

warnings.filterwarnings('ignore')
init_notebook_mode(connected=True)


# In[ ]:


# Function to convert Percent fields to integers
def percent_to_int(df_in):
    for col in df_in.columns.values:
        if col.startswith("Percent") or col.endswith("%") or col.endswith("Rate"):
            df_in[col] = df_in[col].astype(np.object).str.replace('%', '').astype(float)
    return df_in


# In[ ]:


# Dataset 1: School Information
df_schools_raw = pd.read_csv('../input/data-science-for-good/2016 School Explorer.csv')

# Preprocessing
# If school just has 0k, might as well drop
df_schools_raw = df_schools_raw[df_schools_raw['Grade High'] != '0K'] 
df_schools_raw = percent_to_int(df_schools_raw)

# Convert dollars to int 
df_schools_raw['School Income Estimate'] = df_schools_raw['School Income Estimate'].astype(np.object).str.replace('$', '').str.replace(',', '').str.replace('.', '').astype(float)
df_schools_raw.replace(np.NaN,0, inplace=True)

df_schools = df_schools_raw.copy()

# Dataset 2: SHSAT offers information
df_shsat_offers = pd.read_csv('../input/2017-2018-shsat-admissions-test-offers-by-schools/2017-2018 SHSAT Admissions Test Offers By Sending School.csv')

# Preprocessing - convert to int, remove NaNs
df_shsat_offers = percent_to_int(df_shsat_offers)
df_shsat_offers.replace(np.NaN,0, inplace=True)

print("After the pre-processing process, ")
print("We have school information data for {} schools".format(df_schools_raw.shape[0]))
print("We have admission information for {} schools".format(df_shsat_offers.shape[0]))

# Dataset 3: Information regarding PASSNYC resource centers
df_passnyc_centers = pd.read_csv('../input/passnyc-resource-centers/passnyc-resource-centers.csv')


# In[ ]:


# Drop list for merge - fields which are common in both
df_shsat_offers.drop(['Borough','School Category','School Name'], axis=1, inplace=True)

df_merged = pd.merge(df_schools_raw, df_shsat_offers, how='outer', left_on='Location Code' ,right_on='School DBN')
df_merged.dropna(inplace=True)
print("We have {} schools in the merged dataset out of the {} schools in the admissions dataset, implying that we do not have school information for {} schools."      .format(df_merged.shape[0], df_shsat_offers.shape[0], df_shsat_offers.shape[0] - df_merged.shape[0]))

# For SHSAT, if school has just grades below 5, no point in using those 
df_incorrect = df_merged[df_merged['Grade High'].astype(int) <= 5]
print("We have %d schools with Grade High field 5 and SHSAT results"%(df_incorrect.shape[0]))

# 4 types aids available in PASSNYC resource centers 
employment_centers = df_passnyc_centers[df_passnyc_centers['Crime']==1]
test_prep_centers = df_passnyc_centers[df_passnyc_centers['Test Prep']==1]
after_school_centers = df_passnyc_centers[df_passnyc_centers['After School Program']==1]
economic_help_centers = df_passnyc_centers[df_passnyc_centers['Economic Help']==1]

# print(len(set(df_schools_raw[df_schools_raw['Grade High'].astype(int)>7]['Location Code'].values)))
# print(len(set(df['School DBN'].values) - set(df_schools_raw[df_schools_raw['Grade High'].astype(int)>7]['Location Code'].values))) 


# In[ ]:


# Create a trace
layout = go.Layout(
        title='Number of Students taking test v/s Percent Black Hispanic',
        xaxis=dict(
            title='Percentage of Black/Hispanic students',
            titlefont=dict(size=18),
            showticklabels=True,
            tickangle=0,
            tickfont=dict(size=10)
        ),  
        yaxis=dict(
        title='Number of Students taking test',
        titlefont=dict(size=18)
        )         
   )

trace = go.Scatter(
    x = df_merged['Percentage of Black/Hispanic students'],
    y =  df_merged['Number of students who took test'],
    mode = 'markers'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
# Plot and embed in ipython notebook!
iplot(fig)


# From the above plot, we can observe that, in the schools where percentage black / hispanic student is very high, the number of students taking the test is mostly low.

# <h3> <a id='2'>Data Augmentation</a></h3>
# Let's add some more derived fields from the existing fields. We'll add a per grade (grades 5 through 9) per subject (ELA/Maths) percentage of Black/ Hispanic students who scored a 4.  
# We name that field: ** Grade grade_num Total Minority 4s **

# In[ ]:


def grade_minority_total_4s_append(df, grade):
    out_field = ('Grade %d Total Minority 4s')%(grade)
    num1 = ('Grade %d ELA 4s - Black or African American')%(grade)
    num2 = ('Grade %d ELA 4s - Hispanic or Latino')%(grade)
    num3 = ('Grade %d Math 4s - Black or African American')%(grade)
    num4 = ('Grade %d Math 4s - Hispanic or Latino')%(grade)
    df[out_field] = df[num1] + df[num2] + df[num3] + df[num4]

    return df


# In[ ]:


def grade_minority_percent_4s_append(df, grade, subject):
    out_field = ('Grade %d %s Minority 4s')%(grade, subject)
    num1 = ('Grade %d %s 4s - Black or African American')%(grade, subject)
    num2 = ('Grade %d %s 4s - Hispanic or Latino')%(grade, subject)
    den = ('Grade %d %s 4s - All Students')%(grade, subject)
    
    df[out_field] = (df[num1] + df[num2])/(df[den])*100
    df.fillna(0, inplace = True)
    return df


# In[ ]:


grades = range(5,9)
subjects = ['ELA','Math']

for grade in grades:
    for subject in subjects:
        df_merged = grade_minority_percent_4s_append(df_merged, grade, subject) 
    df_merged = grade_minority_total_4s_append(df_merged, grade) 

df_merged['5_6_minority_added'] = df_merged['Grade 5 Total Minority 4s'] + df_merged['Grade 6 Total Minority 4s'] 
df_merged['7_8_minority_added'] = df_merged['Grade 7 Total Minority 4s'] + df_merged['Grade 8 Total Minority 4s']


# In[ ]:


def download_link(df, filename = "data.csv"):
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    title = "Download CSV file"
    html = '<button type="button" style="font-size: larger;  background-color: #FFFFFF; border: 0pt;"><a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a></button>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)


# <h3> <a id="3"> Identifying schools with low number of test takers and large number of high performing underrepresented  students in grades 7 and 8 </a></h3>
# 
# ![Finished diamonds](https://raw.githubusercontent.com/InFoCusp/kaggle_images/master/blob.png)
# 
# <p  style="text-align: left"> Image Courtesy:  [Our Pastimes](https://ourpastimes.com/clean-rough-gem-mine-5869043.html) </p>
# <br/>
# 
# Let's figure out the schools on the basis of a few thresholds and call this low_registration_high_minority data frame: 
# 
# * Schools where  Number of students who took test is less than 50 <br>
# * After this filter is applied, we choose schools where the percentage black / hispanic students are greater than 80%. <br><br>
# 
# Note : These numbers are chosen after careful scrutiny of the data. The thresholds might be relaxed a little if we wish to find/ target more schools to help out. 

# In[ ]:


registered_students_thresh = 50
min_minority_students_thresh = 80


# In[ ]:


# Keep the schools where number of registrations are low and the percentage of Blank and Hispanic students is high
condition_registration = df_merged['Number of students who took test'] < registered_students_thresh
condition_minority = df_merged['Percentage of Black/Hispanic students'] > min_minority_students_thresh
condition = np.logical_and(condition_registration, condition_minority)

df_low_reg_high_minority = df_merged[condition]

# display(download_link(df_low_reg_high_minority, "schools_low_registrations_high_minority.csv"))
df_low_reg_high_minority.head()


# In[ ]:


print("There are {} schools with  Number of students who took test is less than {} and  percentage black / hispanic students are greater than {}%".format(
    df_low_reg_high_minority.shape[0], registered_students_thresh, min_minority_students_thresh))


# Next, we filter out schools where 
# * Percentage of minority students scoring 4s is greater than 70%
# * Number of minority students scoring 4s is greater than 7
# 
# This is done for grades 5 through 9 and ELA/ Maths.

# In[ ]:


percent_minority_4s_thresh = 70
total_4s_thresh = 7


# In[ ]:


def find_minority_high_score_schools(df, percent_minority_4s_thresh, total_4s_thresh, grade='7'):
    df_high_minority = df[condition_minority]
    condition_high_scoring_math = np.logical_and(df_high_minority["Grade {} Math Minority 4s".format(grade)] > percent_minority_4s_thresh,
                                        df_high_minority["Grade {} Math 4s - All Students".format(grade)] > total_4s_thresh)
    condition_high_scoring_ELA = np.logical_and(df_high_minority["Grade {} ELA Minority 4s".format(grade)] > percent_minority_4s_thresh,
                                                   df_high_minority["Grade {} ELA 4s - All Students".format(grade)] > total_4s_thresh)
    condition = np.logical_or(condition_high_scoring_math, condition_high_scoring_ELA)
    df_minority_highscore = df_high_minority[condition]
    return df_minority_highscore

for i in range(5,9):
    globals()['df_minority_highscore_grade'+str(i)] = find_minority_high_score_schools(df_merged, percent_minority_4s_thresh, total_4s_thresh, str(i)).sort_values(by='Grade ' + str(i) +' Total Minority 4s', ascending = False)
    globals()['relevant_fields_grade'+str(i)] =     ['School Name',  'Grade {} Math Minority 4s'.format(i), 'Grade {} Math 4s - All Students'.format(i), 
     'Grade {} ELA Minority 4s'.format(i), 'Grade {} ELA 4s - All Students'.format(i)] 

print("Below we show the schools with maximum number of students scoring 4s in grades 5 through 8: ")
display(HTML(df_minority_highscore_grade5[relevant_fields_grade5].head().to_html()))
display(HTML(df_minority_highscore_grade6[relevant_fields_grade6].head().to_html()))
display(HTML(df_minority_highscore_grade7[relevant_fields_grade7].head().to_html()))
display(HTML(df_minority_highscore_grade8[relevant_fields_grade8].head().to_html()))


# <h3><a id="4"> Mapping the PASSNYC resource Centers </a> </h3>
# 
# ![](https://raw.githubusercontent.com/InFoCusp/kaggle_images/master/Artboard.png)
# 
# As defined in the overview of the problem,
# > PASSNYC and its partners provide outreach services that improve the chances of students taking the SHSAT and receiving placements in these specialized high schools.
# 
# In order to map the schools that require help with the resource centers, let's load the PASSNYC resource centers data. Based on the description of each of the center's work, we have categorized PASSNYC centers into four categories: 
# * Test preparation centers
# * After School centers
# * Economic help centers
# * Employment assistance centers (Also prevent students from engaging in criminal activities)
# 
# Next, we'll map schools with low number of registrations and high percentage of minority students with 4s for Grade 7 and 8 dataframe with nearest test preparation centers.

# In[ ]:


# Function to find the nearest PASSNYC center of required type to the schools in need

def find_nearest_test_center(row, center_type):
    school_latitude = row['Latitude']
    school_longitude = row['Longitude']
    
    passnyc_centers = df_passnyc_centers[df_passnyc_centers[center_type]==1].copy(deep=True)
    test_center_lat = passnyc_centers['Lat'].astype(float)
    test_center_long = passnyc_centers['Long'].astype(float)
    test_prep_centers_nearest = passnyc_centers.copy(deep=True)
    
    test_prep_centers_nearest["Distance to Test Center"] = np.sqrt(np.square(test_center_lat-school_latitude)+np.square(test_center_long-school_longitude))
    nearest_test_center_argmin = test_prep_centers_nearest["Distance to Test Center"].argmin()
    nearest_test_center_series = (test_prep_centers_nearest.loc[nearest_test_center_argmin,["Resource Center Name","Address","Distance to Test Center","Lat","Long"]])
    nearest_test_center_series.rename({"Resource Center Name":"Nearest Test Center Name","Address":"Nearest Test Center Address","Distance to Test Center":"Nearest Test Center Distance","Lat":"Nearest Test Center Latitude","Long":"Nearest Test Center Longitude"}, inplace=True)
    return nearest_test_center_series


# In[ ]:


colors = [
    'red',
    'blue',
    'gray',
    'darkred',
    'lightred',
    'orange',
    'beige',
    'green',
    'darkgreen',
    'lightgreen',
    'darkblue',
    'lightblue',
    'purple',
    'darkpurple',
    'pink',
    'cadetblue',
    'lightgray',
    'black', 'darkgray','darkpink','lightpink','lightpurple','lightred'
]

# Function to plot schools and the PASSNYC help centers
def plot_school_test_centers_map(grouped_df):
    school_map = folium.Map([test_prep_centers['Lat'].mean(), test_prep_centers['Long'].mean()], 
                        zoom_start=10.50,
                        tiles='Stamen Terrain')
    for i, group in enumerate(grouped_df):
        group = group[1]
        for index in group.index:
            row = group.loc[index]
            folium.Marker([row['Latitude'], row['Longitude']], icon=folium.Icon(color=colors[i], icon='university', prefix="fa")).add_to(school_map)

        for index in group.index:
            row = group.loc[index]
            folium.Marker([row['Nearest Test Center Latitude'], row['Nearest Test Center Longitude']], icon=folium.Icon(color=colors[i],  icon ='info-sign')).add_to(school_map)
    return school_map


# In[ ]:


# Merged Grade 7 and 8 for analysis
df_minority_highscore_grade7_grade8 = pd.concat([df_minority_highscore_grade7, df_minority_highscore_grade8])
print("The number of schools where either grade 7 or grade 8 minority students are performing well is "+ str(len(df_minority_highscore_grade7_grade8)))

test_prep_nearest_test_centers = (df_minority_highscore_grade7_grade8[["Latitude","Longitude"]].apply(find_nearest_test_center, center_type="Test Prep", axis=1))
df_minority_highscore_grade7_grade8_test_centers_merged = pd.merge(df_minority_highscore_grade7_grade8, test_prep_nearest_test_centers, left_index=True, right_index=True)
required_fields = ['School Name', 'Nearest Test Center Name', 'Number of students who took test'] + relevant_fields_grade7 + relevant_fields_grade8 + ['7_8_minority_added', 'Percent of Students Chronically Absent']

df_minority_highscore_grade7_grade8_test_centers_merged.drop_duplicates(inplace=True)
df_minority_highscore_grade7_grade8_test_centers_merged_relevant_fields = df_minority_highscore_grade7_grade8_test_centers_merged[required_fields].sort_values(['7_8_minority_added'], ascending = False)
display(download_link(df_minority_highscore_grade7_grade8_test_centers_merged_relevant_fields, "link_schools_after_school_centers.csv"))
print ("List of the schools with high minority students performing well in Grade 7 and Grade 8")
df_minority_highscore_grade7_grade8_test_centers_merged_relevant_fields.head()


# In[ ]:


nearest_after_school_centers_df_groupby = df_minority_highscore_grade7_grade8_test_centers_merged.groupby(by='Nearest Test Center Address', as_index=False)

# AFTER SCHOOL CENTERS - GRADE 7,8
print("Scatter map of schools with high minority and good performance of Grade-7 and 8 Students")
print ("")
print ("The university symbol on the marker represents a school belonging to a cluster (defined by a particular color) and the marker with the same color having 'i' symbol represents passnyc help center")
school_map = plot_school_test_centers_map(nearest_after_school_centers_df_groupby)
school_map


# Next, we'll map schools with low number of registrations and high percentage of minority students with 4s in Grade 5 and 6 dataframe with nearest after school centers. These can also be mapped to the test preparation centers as above if needed. 

# In[ ]:


# Merged Grade 5 and 6 for analysis
required_fields = ['School Name', 'Nearest Test Center Name', 'Number of students who took test'] + relevant_fields_grade5 + relevant_fields_grade6 + ['5_6_minority_added', 'Percent of Students Chronically Absent']
df_minority_highscore_grade5_grade6 = pd.concat([df_minority_highscore_grade5, df_minority_highscore_grade6])
print("The number of schools where both grade 5 and grade 6 minority students are performing well is "+ str(len(df_minority_highscore_grade5_grade6)))

after_school_nearest_test_centers = (df_minority_highscore_grade5_grade6[["Latitude","Longitude"]].apply(find_nearest_test_center, center_type="After School Program", axis=1))
df_minority_highscore_grade5_grade6_test_centers_merged = pd.merge(df_minority_highscore_grade5_grade6, after_school_nearest_test_centers, left_index=True, right_index=True)

df_minority_highscore_grade5_grade6_test_centers_merged.drop_duplicates(inplace=True)
df_minority_highscore_grade5_grade6_test_centers_merged_relevant_fields = df_minority_highscore_grade5_grade6_test_centers_merged[required_fields].sort_values(['Grade 5 Math 4s - All Students', 'Grade 5 ELA 4s - All Students'], ascending = False)
display(download_link(df_minority_highscore_grade5_grade6_test_centers_merged_relevant_fields, "link_schools_after_school_centers.csv"))
print ("List of the schools with high minority students performing well in Grade 5 and Grade 6")
df_minority_highscore_grade5_grade6_test_centers_merged_relevant_fields.head()


# In[ ]:


nearest_after_school_centers_df_groupby = df_minority_highscore_grade5_grade6_test_centers_merged.groupby(by='Nearest Test Center Address', as_index=False)

# AFTER SCHOOL CENTERS - GRADE 5,6
print("Scatter map of schools with high minority and good performance of Grade-5 and 6 Students")
print ("")
print ("The university symbol on the marker represents a school belonging to a cluster (defined by a particular color) and the marker with the same color having 'i' symbol represents passnyc help center")
school_map = plot_school_test_centers_map(nearest_after_school_centers_df_groupby)
school_map


# <h3> RECOMMENDATIONS to PASSNYC </h3>
# 
# From the above mapping, below are our recommendations : 
# 
# * As the students of grade 8 & 9 are eligible for the exam, we have mapped the schools with high percentage of minority students whose performance is good (based on MATH and ELA) in Grade 7 and 8 with **test preparation** centers of PASSNYC. These are the schools which require immediate help, as they are going to appear for SHSAT exam in the coming year, practising for the test might be more helpful to the students.
# 
# * Similarly, for schools where students of Grade 5 & 6 who perform good (based on MATH and ELA) and have high percentage of minority students we have mapped these schools to centers of PASSNYC offering **after school** programs. The after school programs might help the students to clear the basic concepts and make them ready for SHSAT. 
# 
# If worked on these schools and the minority students, we claim that this will increase the number of registrations for SHSAT, as they have higher percentage of minority students performing well and in turn will also increase the probablity of their selection, ultimately leading to the admissions which serves the purpose of increasing the diversity in the elite high schools.

# ### Looking at the correlation between students performing well in school and SHSAT offers they receive
# 
# The following section is to provide mathematical justification of why targetting schools with high number of 4s would in turn lead to higher number of selections in SHSAT. <br><br>
# Here, we are considering Class 6 and 7 students in the School Explorer dataset which was taken in 2016, as the admissions data for which we are validating is for 2017-18. So, the performance in the SHSAT would be the reflection of Class 6 and 7 students of the school explorer dataset. 

# In[ ]:


# Adding a field called 'Selection Ratio' which is the ratio of students who received an offer to the number of students who took SHSAT
df_merged['Selection_ratio'] = df_merged['Number of students who received offer']/df_merged['Number of students who took test']


# In[ ]:


# Filter out schools based on threshold of number of students who received an offer
df_high_offers = df_merged[df_merged['Number of students who received offer']>10]


# In[ ]:


# Fit a regression on Percentage of students who received a 4 and percent of students who received an offer
num_students_4s = df_high_offers['Grade 7 Math 4s - All Students']+df_high_offers['Grade 7 ELA 4s - All Students']+df_high_offers['Grade 6 Math 4s - All Students']+df_high_offers['Grade 6 ELA 4s - All Students']
num_studetns_tested = df_high_offers['Grade 7 Math - All Students Tested']+df_high_offers['Grade 7 ELA - All Students Tested']+df_high_offers['Grade 6 Math - All Students Tested']+df_high_offers['Grade 6 ELA - All Students Tested']
x = (num_students_4s/num_studetns_tested)*100
y =  df_high_offers['Selection_ratio']*100

sns.set(rc={'figure.figsize':(12,10)})
regression = sns.regplot(x, y)
regression = regression.set(xlabel='Percentage of students scoring 4s', ylabel='SHSAT Selection Percentage')


# In[ ]:


model  = sklearn.linear_model.LinearRegression()
model.fit(x.values.reshape(-1, 1),y)
print ("The coefficient of the regression model is %f."%model.coef_)


# We can see the trend of selection ratio v/s the ratio of number of students scoring 4s in grades 6&7, in the schools which had some minimum number of selections (10).
# * It can be observed from the above plot that, the regression line in the scatter plot is has an upward slope, which suggests that these two features highly correlate. 
# * This result provides evidence that targetting schools with high number of 4s would in turn lead to higher number of selections in SHSAT.  <br><br>
# Note : The reason for applying the threshold on minimum number of selections is so that the numbers in schools with very few selections do not skew the overall scatter plot. For example, 4 selections out of 5 who took the test --> 80% selection ratio, but this 80% does not matter as the number of registrations itself is very low.

# <h3><a id="5">Analysing schools with high Economic Need Index</a></h3>
# 
# This metric reflects the economic condition of the school population. ENI is computed as:
# 
# <b>ENI</b> = (% temp housing) + (% HRA eligible x 0.5) + (% free lunch eligible x 0.5).
# 
# The higher this index, the higher is the economic need of the students

# In[ ]:


layout = go.Layout(
        title='Histogram of Economic Need Index (ENI)',
        xaxis=dict(
            title='Economic Need Index (ENI)',
            titlefont=dict(size=18),
            showticklabels=True,
            tickangle=0,
            tickfont=dict(size=10)
        ),  
        yaxis=dict(
        title='Number of Schools',
        titlefont=dict(size=18)
        )         
   )

data = [go.Histogram(x = df_low_reg_high_minority['Economic Need Index'].values)]
fig = go.Figure(data=data, layout=layout);
iplot(fig);


# Below is the visualization of **Economic Need Index** and **School Income Estimate** of schools.
# * In the plot, each data-point (circle) corresponds to a school.
# * Color of the circle represents ENI of the school
# * Size of the circle represents school income.

# In[ ]:


layout = go.Layout(
        autosize=False,
        width=800,
        height=600,
        title='ENI Heatmap over the geographic layout',
        xaxis=dict(
            title='Longitude',
            titlefont=dict(size=18)
        ),  
        yaxis=dict(
        title='Latitude',
        titlefont=dict(size=18)
        )         
   )

trace1 = go.Scatter(
    x = df_low_reg_high_minority['Longitude'],
    y = df_low_reg_high_minority['Latitude'],
    mode='markers',
    marker=dict(
        symbol='circle',
        sizemode='area',
        sizeref = 2.*max(df_low_reg_high_minority['School Income Estimate'])/(100**1.75),
        size = df_low_reg_high_minority['School Income Estimate'].values,
        color = df_low_reg_high_minority['Economic Need Index'].values, #set color equal to a variable
        colorscale='Viridis',
        showscale=True
        
    )
)
data = [trace1]

fig = go.Figure(data=data, layout=layout);
iplot(fig);


# In[ ]:


#Divide schools based on Economic Need Index
df_schools_high_eni = df_low_reg_high_minority[df_low_reg_high_minority['Economic Need Index'] > df_low_reg_high_minority['Economic Need Index'].quantile(0.75)]
df_schools_low_eni = df_low_reg_high_minority[df_low_reg_high_minority['Economic Need Index'] < df_low_reg_high_minority['Economic Need Index'].quantile(0.25)]


# In[ ]:


df_schools_high_eni_relevant_fields = df_schools_high_eni[['School Name', 'Economic Need Index', 'Percent Black / Hispanic']].sort_values('Economic Need Index', ascending = False)
display(download_link(df_schools_high_eni_relevant_fields, 'schools_high_eni.csv'))
print ("There are %d schools with high ENI (where students' economic need is high)"%df_schools_high_eni_relevant_fields.shape[0])
df_schools_high_eni_relevant_fields.head()


# * We have divided the schools into two segments based on the Economic Need Index. 
# * We observed that the students of schools with higher ENI have performed poorly in both the subjects by a significant margin when compared to students of schools with lower ENI.

# In[ ]:


def display_barplot(comparison_dict, axis = 0):
        
    list_traces = []
    for _trace in comparison_dict.keys():
        _data_trace = comparison_dict[_trace]
        _data_trace_name = _data_trace['name']
        del(_data_trace['name'])
    
        _trace_features = []
        _names = []
#         for ix in _data_trace.keys():
#             _trace_features.append(_data_trace[ix])
        _features = _data_trace['features']

        for i, cols in enumerate(_features):
            _names.append(_features.index[i])
            _trace_features.append(cols)
        
        if not axis:
            _trace = go.Bar(
                y = _names,
                x = _trace_features,
                name = _data_trace_name,
                orientation = 'h'
            )
        else:
            _trace = go.Bar(
                x = _names,
                y = _trace_features,
                name = _data_trace_name
            )
        list_traces.append(_trace)

    #print (list_traces)
    data = list_traces
    layout = go.Layout(
        height = 500,
        width = 800,
        barmode='group',
#         yaxis=dict(
#             tickangle=270
#         ),
#         xaxis = dict(
#             tickangle = 350
#         )
    )
    
    fig = go.Figure(data=data, layout=layout)
    iplot(fig, filename='grouped-bar')


# In[ ]:


print ("Average ELA / Math performance for the high ENI schools")
print (str(df_schools_high_eni[['Average ELA Proficiency','Average Math Proficiency']].mean()))
print ("")
print ("Average ELA / Math performance for the low ENI schools")
print (str(df_schools_low_eni[['Average ELA Proficiency','Average Math Proficiency']].mean()))

comparison_dict = {
    'trace1':
    {
        'features': df_schools_high_eni[['Average ELA Proficiency','Average Math Proficiency']].mean(),
        'name':'High ENI schools'
    },
    
    'trace2':
    {
        'features': df_schools_low_eni[['Average ELA Proficiency','Average Math Proficiency']].mean(),
        'name':'Low ENI schools'
    }
}


display_barplot(comparison_dict, axis=1)


# In[ ]:


print ("Click the link below to get details of the economic help centers")
display(download_link(economic_help_centers, 'economic_help_centers_details.csv'))


# ### Analysis/ Suggestions:
# From the above analysis, we observe that the schools with high ENI also have high percentage of black/ hispanic students and a lower school income. Their performance is much lower compared to low ENI schools. This could originate from the lack of financial/ educational resources among the students. Resource centers providing study materials, after school guidance/ mentoring could help boost the performance of the students from these schools. 
# 
# * We have not mapped these schools with the 28 **finance help centers** of PASSNYC because, these schools can be helped from anywhere. The geographical distance does not matter while providing financial aid. However if the question is of resources distribution across places, then we can map these 28 resource centers to the 85 needy schools as we have done previously for **test preparation** centers.

# <h3><a id="6">Analysing and mapping Schools based on School Performance Index</a></h3>
# 
# ![School Performance Indicator](https://raw.githubusercontent.com/InFoCusp/kaggle_images/master/School%20%20Performance%20%20Indicator.png)
# 
# 
# Next, we explore the 6 fields in the dataset which are indicative of the overall standing of the school. These features include:
# * <b>Rigorous Instruction %</b> - Degree to which school curriculum and instructions engage students by fostering critical thinking.
# * <b>Collaborative Teachers %</b> - Degree to which teachers participate in development, growth and contribution towards students and school.
# * <b>Supportive Environment %</b> - Degree to which school establishes a culture where students feel safe and challenged to grow.
# * <b>Effective School Leadership %</b> - Degree to which there is an instructional vision and the leadership is distributed for realising this vision.
# * <b>Strong Family-Community Ties %</b> - Degree to which a school works in partnerships with families and other organizations to improve the school.
# * <b>Trust %</b> - Degree of relationships of a school with students, teachers, families and adminstrators. 
# 
# [comment]:![SPI.jpg](attachment:SPI.jpg)
# 
# We show some typical values of these measures below and then study crrelation between these features:

# In[ ]:


features_list = ['Rigorous Instruction %',
'Collaborative Teachers %',
'Supportive Environment %',
'Effective School Leadership %',
'Strong Family-Community Ties %',
'Trust %']

df_schools[['School Name'] + features_list ].head()


# In[ ]:


# fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 12))
# for i in range(2):
#     for j in range(3):
#         ax[i, j].set_title(features_list[i*3 + j])
#         sns.distplot(a=df_schools[features_list[i*3 + j]].dropna().values, kde_kws={"color": "red"}, color='darkblue', ax=ax[i, j])

# # fig.tight_layout()
# temp = fig.suptitle('School Performance features', fontsize=15)


# In[ ]:


df_schools[features_list].corr()


# In[ ]:


corr = df_schools[features_list].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, cmap='YlGnBu')
temp = plt.xticks(rotation=75, fontsize=8) 
temp = plt.yticks(fontsize=8) 


# There is a high correlation was found between three of the features: 
# * Effective School Leadership 
# * Collaborative Teachers and
# * Trust <br/>
# 
# 'Principal Component Anaysis (PCA)' has been applied on the 3 features to get a single combined feature capturing the key characteristics of all those three features.

# In[ ]:


correlated_features_list = ["Effective School Leadership %","Collaborative Teachers %","Trust %"]
corr_features_values = df_schools[correlated_features_list].values

pca = PCA(n_components=1)
combined_feature_value = pca.fit_transform(corr_features_values)
df_schools['PCA Combined Feature'] = combined_feature_value
#df_schools[correlated_features_list + ['PCA Combined Feature']].corr()


# In[ ]:


scaler = sklearn.preprocessing.MinMaxScaler()
scale_factor = 2*(df_schools['PCA Combined Feature'].corr(df_schools["Effective School Leadership %"])>0) -1 
df_schools['PCA Combined Feature'] =  scaler.fit_transform(scale_factor * df_schools['PCA Combined Feature'].values.reshape(-1,1))*100

print ("The correlation between the three correlated features and their PCA is shown below:")
df_schools[correlated_features_list + ['PCA Combined Feature']].corr()


# NOTE: PCA combined features incorporates these three features - Effective School Leadership %, Collaborative Teachers %, Trust %. As can be seen from the correlation values, it captures the essence of these 3 features quite well.
# 
# #### Weights to each factor for calculating School Performace Indicator:
# 
# * 1.0 := Supportive Environment % <br>
# * 0.8 := Rigorous Instruction % <br>
# * 0.7 := PCA combined feature % <br>
# * 0.5 := Strong Family-Community Ties % <br>

# In[ ]:


weights = [ 1, 0.8, 0.7, 0.5]


# In[ ]:


features = ['Supportive Environment %','Rigorous Instruction %','PCA Combined Feature',
            'Strong Family-Community Ties %']


df_schools['SPI'] = df_schools[features].dot(weights)


print ("A glimpse of the School Performance Index (SPI) :")
df_schools[features+['SPI']].head(5)


# In[ ]:


df_low_spi_schools = df_schools[df_schools['SPI'] < df_schools['SPI'].quantile(.25)]
df_high_spi_schools = df_schools[df_schools['SPI'] > df_schools['SPI'].quantile(.25)]


# In[ ]:


print ("Average ELA / Math performance for the high SPI schools")
print(df_high_spi_schools[['Average ELA Proficiency','Average Math Proficiency']].mean())

print ("Average ELA / Math performance for the low SPI schools")
print(df_low_spi_schools[['Average ELA Proficiency','Average Math Proficiency']].mean())

comparison_dict = {
    'trace1':
    {
        'features': df_high_spi_schools[['Average ELA Proficiency','Average Math Proficiency']].mean(),
        'name':'High SPI schools'
    },
    
    'trace2':
    {
        'features': df_low_spi_schools[['Average ELA Proficiency','Average Math Proficiency']].mean(),
        'name':'Low SPI schools'
    }
}


display_barplot(comparison_dict, axis=1)


# <h3> <a id="8">SPI / ENI based clustering</a></h3>
# 
# As per the analysis till now, we observed that the two important features that can help in improvement of schools and students are SPI and ENI respectively. As a result, the schools with low SPI and high ENI should most definitely be targetted for aid by PASSNYC <br>
# However, we understand the perils of demographics and it is impossible for PASSNYC to cater to all the schools and their students independently or visit these schools to motivate the students to appear / prepare for SHSAT. Hence, we recommend that :
# 
# * PASSNYC could help the school students by setting up centrally located resource centers which are common across schools. 
# * In order to serve maximum schools using least resources, we have clustered the schools based on low SPI and high ENI.

# In[ ]:


df_schools_clustering = df_schools.copy()
df_schools_clustering = df_schools_clustering.dropna(subset=['Longitude', 'Latitude'])
df_schools_clustering = df_schools_clustering[df_schools_clustering['SPI'] < df_schools_clustering['SPI'].quantile(0.15)]
df_schools_clustering = df_schools_clustering[df_schools_clustering['Economic Need Index'] > 0.8]


# In[ ]:


model = KMeans(n_clusters=7)
model.fit(df_schools_clustering[['Longitude', 'Latitude']].values)
model.cluster_centers_
color = 'blue'


# In[ ]:


school_map = folium.Map([model.cluster_centers_[:, 1].mean(), model.cluster_centers_[:, 0].mean()], 
                        zoom_start=11,
                        tiles='Stamen Terrain') 

for index in df_schools_clustering.index:
    row = df_schools_clustering.loc[index]
    popup_text = "Economic Need Index : " + str(round(row['Economic Need Index'], 3)) + ' , SPI : ' + str(round(row['SPI'], 3))
    folium.Marker([row['Latitude'], row['Longitude']], popup=popup_text, icon=folium.Icon(color='blue', icon="university", prefix="fa")).add_to(school_map)
    
for row in model.cluster_centers_:
    folium.Marker([row[1], row[0]], icon=folium.Icon(color='red')).add_to(school_map)

print ("Scatter plot of schools with high ENI, low SPI with recommended cluster(help) center")
school_map


# In[ ]:


school_map = folium.Map([model.cluster_centers_[:, 1].mean(), model.cluster_centers_[:, 0].mean()], 
                        zoom_start=11,
                        tiles='Stamen Terrain') 
for row in model.cluster_centers_:
    folium.Marker([row[1], row[0]], icon=folium.Icon(color='red')).add_to(school_map)
    
for index in df_passnyc_centers.index:
        row = df_passnyc_centers.loc[index]
        folium.Marker([row['Lat'], row['Long']], icon=folium.Icon(color=colors[i],  icon ='info-sign')).add_to(school_map)

print ("Scatter plot of PASSNYC help centers and recommended help centers")
school_map


# As we can see from the  scatter map, most of the recommended cluster(help) centers are overlapping with PASSNYC help centers. PASSNYC can immediately target the schools based on these clusters and can assign a list of schools to a particular center based on the two maps shown above.

# <h3> <a id="7">Student to Teacher Ratio (STR) </a></h3>

# We have used a dataset with student-teacher ratio (STR) and merged it with the original data of schools to analyse the impact of student-teacher ratio on the performance of students and schools indeed.
# 
# <b>Student-Teacher Ratio (STR) = Number of Students per Teacher for a particular school.</b>

# In[ ]:


field_name = "SCHOOLWIDE PUPIL-TEACHER RATIO"
df_school_detail = pd.read_csv('../input/ny-2010-2011-class-size-school-level-detail/2010-2011-class-size-school-level-detail.csv')
df_school_detail["CSD"] =  df_school_detail['CSD'].astype('str').astype(np.object_).str.zfill(2)
df_school_detail["DBN_manual"] = df_school_detail["CSD"] + df_school_detail["SCHOOL CODE"] 
df_school_detail.dropna(subset=[field_name], inplace=True)


# In[ ]:


merged_str_df = pd.merge(df_school_detail, df_schools, how='inner', left_on=['DBN_manual'], right_on=['Location Code'])


# In[ ]:


data = [go.Histogram(x=merged_str_df['SCHOOLWIDE PUPIL-TEACHER RATIO'])]
iplot(data,filename='Number of Students per teacher')


# We decided thresholds (high and low) of 18 and 12.5 to split the dataset into schools with high STR and low STR to analyse the average performace of these schools.

# In[ ]:


higher_ratio_str_df = merged_str_df[merged_str_df[field_name].astype(float)>18]
lower_ratio_str_df = merged_str_df[merged_str_df[field_name].astype(float)<12.5]


# In[ ]:


feature_columns1 = ['Average ELA Proficiency','Average Math Proficiency', 'Economic Need Index']
feature_columns2 = ['Collaborative Teachers %', 'SPI','Percent Black / Hispanic']

print ("Average statistics for the schools with high STR")
print (higher_ratio_str_df[feature_columns1 + feature_columns2].mean())
print ("")
print ("Average statistics for the schools with low STR")
print (lower_ratio_str_df[feature_columns1 + feature_columns2].mean())


# * Counter-intuitively, it was observed that the schools with high STR are performing better when compared to the schools with low STR. The performance was observed by the average ELA proficiency and average Math proficiency. 
# 
# * In order to understand the anamoly, we tried looking into how teachers work with students in **collaboration**, **percent of black/hispanic students** and **Economic Need Index** of students.
# 
# * We observe that where the STR is high, the Collaborative teachers and school leadership indexes are high, due to which the SPI is high and might be the reason for better performace of students. 
# 

# <h3><a id="7"> Analysis of school safety</a></h3>

# In[ ]:


# Dataset 4: Information regarding criminal activities in school
df_school_crimes = pd.read_csv('../input/ny-2010-2016-school-safety-report/2010-2016-school-safety-report.csv')


# In[ ]:


df_school_crimes = df_school_crimes[df_school_crimes["School Year"]=="2015-16"]


# In[ ]:


crimes_col = ['Major N', 'Oth N', 'NoCrim N', 'Prop N', 'Vio N']
crimes = df_school_crimes.groupby(['DBN'], as_index=False)[crimes_col].sum()

merged_safety_df = pd.merge(crimes[crimes_col + ['DBN']], df_merged, how='inner', left_on=['DBN'], right_on=['Location Code'])
merged_safety_df.dropna(subset=crimes_col, inplace=True,how='all')


# In[ ]:


features_pca = merged_safety_df[crimes_col].values

from sklearn.decomposition import PCA

school_crime_pca = PCA(n_components=1)
school_crime_pca.fit(features_pca)
reduced_crime_features = school_crime_pca.transform(features_pca)
# print school_crime_pca.explained_variance_ratio_


# In[ ]:


import sklearn.preprocessing
scaler = sklearn.preprocessing.MinMaxScaler()
adjusted_reduced_crime_features = scaler.fit_transform(reduced_crime_features)

merged_safety_df['SRI'] = adjusted_reduced_crime_features


# In[ ]:


data = [go.Histogram(x=merged_safety_df['SRI'], nbinsx=20)]
print ("Histogram of School Risk Index")
iplot(data, filename='SRI histogram')


# In[ ]:


merged_safety_df.sort_values("SRI", inplace=True)

low_risk_schools_df = merged_safety_df[:30]
high_risk_schools_df = merged_safety_df[-30:]


# In[ ]:


feature_columns1 = ['Average ELA Proficiency','Average Math Proficiency', 'Economic Need Index']
feature_columns2 = ['Percent of Students Chronically Absent','Student Attendance Rate','Percent Black / Hispanic']
feature_columns = feature_columns1 + feature_columns2

print ("Average statistics for the schools with low risk index")
print (low_risk_schools_df[feature_columns].mean())
print ("")
print ("Average statistics for the schools with high risk index")
print (high_risk_schools_df[feature_columns].mean())


comparison_dict1 = {
    'trace1':
    {
        'features': low_risk_schools_df[feature_columns1].mean(),
        'name':'Low Risk schools'
    },
    
    'trace2':
    {
        'features': high_risk_schools_df[feature_columns1].mean(),
        'name':'High Risk schools'
    }
}

comparison_dict2 = {
    'trace1':
    {
        'features': low_risk_schools_df[feature_columns2].mean(),
        'name':'Low Risk schools'
    },
    
    'trace2':
    {
        'features': high_risk_schools_df[feature_columns2].mean(),
        'name':'High Risk schools'
    }
}

display_barplot(comparison_dict1, axis=1)
display_barplot(comparison_dict2, axis=1)


# On the basis of the **School Risk Report** analysed above, <br>
# * We can observe from that where the schools are categorized as high risk (unsafe), the **percent of students chronically absent** is 8% more than usually absent. 
# * The percent of **Black / Hispanic students** and **ENI**  is lower in Low Risk Schools when compared to High Risk Schools.
# * Other features like **Average Math and ELA Proficiency** are comparitively higher in Low Risk Schools.

# **Elite Eight Schools**  <br>
# 
# **Now, let's understand why we were focusing on the black / hispanic people and why passnyc needs to look into schools with high black / hispanic ratio. <br><br>
# **First of all, <br>
# Let's load the data for the eight elite schools which gives admission on the basis of SHSAT and look at the geograhics of these schools. 

# In[ ]:


elite_schools_df = pd.read_csv('../input/elite-8-school-data/elite_eight_data.csv')
elite_schools_df = elite_schools_df.iloc[:8]


# In[ ]:


more_registered_students_df = df_merged[df_merged['Number of students who took test'] > 50]
less_registered_students_df = df_merged[df_merged['Number of students who took test'] < 10]


# In[ ]:


fig, ax = plt.subplots(figsize=(16,9))
less_registered_students_df.plot(kind="scatter", x="Longitude", y="Latitude", 
                   c=less_registered_students_df['Number of students who took test'], s=200, cmap=plt.get_cmap("jet"), 
                   label='Schools', title='SHSAT Registrations', 
                   colorbar=True, alpha=0.6, ax=ax)

elite_schools_df.plot(kind="scatter", x=" Long", y="Lat", 
                   c=elite_schools_df['Enrollment'], s=1000, cmap=plt.get_cmap("jet"), 
                   label='Elite School', title='SHSAT Schools', 
                   colorbar=False, alpha=0.6, marker='^', ax=ax)

ax.legend(markerscale=0.5)

#change the marker size manually for both lines
# legend = ax.legend(frameon=True)
# for legend_handle in legend.legendHandles:
#     legend_handle.set_markersize(9)

f = ax.set_ylabel('Latitude')
f = ax.set_xlabel('Longitude')
f = ax.set_xlim(-74.2, -73.75)
f = ax.set_ylim(40.5, 40.95)


# In[ ]:


print ("Schools with Low test takers")
print ("Mean Percent Black / Hispanic ratio : " + str(less_registered_students_df['Percent Black / Hispanic'].mean()))
print ("Standard Deviation Percent Black / Hispanic ratio : " + str(less_registered_students_df['Percent Black / Hispanic'].std()))
print ("")
print ("Schools with High test takers")
print ("Mean Percent Black / Hispanic ratio : " + str(more_registered_students_df['Percent Black / Hispanic'].mean()))
print ("Standard Deviation Percent Black / Hispanic ratio : " + str(more_registered_students_df['Percent Black / Hispanic'].std()))


# We can observe two things from the above analysis. 
# *  One is that, the schools which have higher number of students not taking the test are located very near to the elite schools. 
# * The second one being, that the students from the schools having 90% black / hispanic population on an average are the ones that are not taking the test. 

# <h3> <a id="8">Classification of schools based on Percentage of English Language Learners </a></h3>
# 
# We have analysed the impact of number of ELL on the average proficiency of students and the ratio of offers received by the students in that school. From the results it can be observed that, the schools where number of ELL is high, their performcance in the basic subjects needed in SHSAT (i.e. English and Math) is poor which is in turn reflected by the selection ratio.

# In[ ]:


df_schools_high_ell = df_merged[df_merged['Percent ELL'] > df_merged['Percent ELL'].quantile(0.90)]
df_schools_low_ell = df_merged[df_merged['Percent ELL'] < df_merged['Percent ELL'].quantile(0.10)]

print("Average performance of schools with high ELL")
print(df_schools_high_ell[['Average ELA Proficiency','Average Math Proficiency']].mean())
print()
print("Average performance of schools with low ELL")
print(df_schools_low_ell[['Average ELA Proficiency','Average Math Proficiency']].mean())
print()
print("Difference between average performance of schools with high and low number of ELL")
print(abs(df_schools_high_ell[['Average ELA Proficiency','Average Math Proficiency']].mean() - df_schools_low_ell[['Average ELA Proficiency','Average Math Proficiency']].mean()))

comparison_dict = {
    'trace1':
    {
        'features': df_schools_high_ell[['Average ELA Proficiency','Average Math Proficiency']].mean(),
        'name':'Schools with High ELL'
    },
    
    'trace2':
    {
        'features': df_schools_low_ell[['Average ELA Proficiency','Average Math Proficiency']].mean(),
        'name':'Schools with Low ELL'
    }
    
}


display_barplot(comparison_dict, axis=1)


# In[ ]:


print("The mean selection ratio of students in schools with high ELL is %f."%(df_schools_high_ell['Selection_ratio'].mean()*100))
print("The mean selection ratio of students in schools with low ELL is %f."%(df_schools_low_ell['Selection_ratio'].mean()*100))


comparison_dict = {
    'trace1':
    {
        'features': df_schools_high_ell[['Selection_ratio']].mean()*100,
        'name':'Mean Selection Ratio of Schools with High ELL'
    },
    
    'trace2':
    {
        'features': df_schools_low_ell[['Selection_ratio']].mean()*100,
        'name':'Mean Selection Ratio of Schools with Low ELL'
    }
    
}


display_barplot(comparison_dict, axis=1)


# <h3> <a id="9"> Summary </a> </h3>
# 
# We have done analysis on 2016 school dataset (provided in the competition). However, we have registrations/offers data for 2017-18. So, the grade 6-7 students in the school dataset would be reflected in grade 7-8 of the registrations dataset. We searched for the 2016-17 registrations/offers data in order to do the analysis in sync with the 2016 school dataset. However, we were unable to find any such dataset online. So, we moved forward with the data at hand. The schools dataset, if availale for current years, can be updated in the analysis. 
# 
# Below, we present the summary of our findings: 
# 
# 1. **Recommendations for Schools with high performing students** [schools with (low number of registrations + high minority students + high minority students performing well (based on ELA and MATH 4s))] - which should be targetted for SHSAT preparations/ increasing number of registrations.
# We also present proof that if these schools are targetted, it would lead to increase in the number of offers, thus increasing diversity of students enrolled in the elite schools.
# 
# 2. **Mapping of the above schools to PASSNYC help centers.**
#     
#    The help centers are categorized into 4 major categories : 
#         Test preparation centers
#         After School centers
#         Economic help centers
#         Employment assistance centers (Also prevent students from engaging in criminal activities)
# 
#     As the students of grade 8 & 9 are eligible for the exam, we have mapped the schools with high percentage of minority students whose performance is good (based on MATH and ELA in Grade 7 & 8) to test preparation centers of PASSNYC partners.  
# 
#     Similarly, for schools where students of Grade 5 & 6 who perform well and have high percentage of minority students are mapped to PASSNYC offering after school program centers. 
#     
# 3. **Recommendations for Low performing schools on the basis of SPI - a combination of school performance parameters and Economic Need Index (ENI)**. <br>
# These schools should be targetted an overall performance improvement over a longer run. If these schools are targetted, their performance could be improved which will result into increased number of registrations for SHSAT. PASSNYC should also setup SHSAT awareness programs in order to motivate the students and school in general, which definitely leads to increase in registrations. We have not mapped these schools with the 28 economic help centers of PASSNYC because financial aid can be provided from anywhere.
# 
# 4. **Mapping of the PASSNYC help centers has been done in two ways **: 
#     
#     We mapped schools in need of PASSNYC services to the closest PASSNYC resource center and made clusters (based on colors as seen above). Using this technique, they can reach out to majority of schools quickly and with less burden. The goal was to reach out to majority of schools with minimum resources.
# 
#     During EDA, we had found ideal locations for centrally located help centers which could serve multiple schools. However, we came across the already present help centers of PASSNYC partners and found that most of them were overlapping with our suggested locations. One additional location has been suggested by us, which can be setup as per the requirements of PASSNYC contributors.

# In[ ]:




