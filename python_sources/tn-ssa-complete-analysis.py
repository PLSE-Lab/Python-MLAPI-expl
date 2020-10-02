#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install descartes pandas geopandas numpy')


# In[ ]:


import pandas as pd


# In[ ]:


import geopandas as gpd


# In[ ]:


import os

input_dir = '../input/tamil-nadu-school-data-ssa/'
print(input_dir)


# In[ ]:


os.listdir('../input')


# In[ ]:


# 2011_Dist.Json is a geoJSON file obtained from https://github.com/datameet/maps
# Unless otherwise states, the map dataset is shared under Creative Commons Attribution-ShareAlike 2.5 India license.


# In[ ]:


enroll_school_df = pd.read_csv(f'{input_dir}/enrollement_schoolmanagement_2.csv')


# In[ ]:


enroll_school_df.info()


# In[ ]:


enroll_school_df.head()


# In[ ]:


# Convert the District Names to title case


# In[ ]:


enroll_school_df['District'] = enroll_school_df['District'].apply(lambda x: x.title())


# In[ ]:


enroll_school_df.head()


# In[ ]:


# Show all District Names


# In[ ]:


enroll_school_df['District'].unique()


# In[ ]:


len(enroll_school_df['District'].unique())


# In[ ]:


# What's the district State Total?


# In[ ]:


enroll_school_df[enroll_school_df['District'] == 'State Total']


# In[ ]:


# Is it sum of other districts?

state_total_df = enroll_school_df[enroll_school_df['District'] == 'State Total']
district_df = enroll_school_df[enroll_school_df['District'] != 'State Total']


# In[ ]:


enroll_school_df.columns


# In[ ]:


cols = enroll_school_df.columns[2:]
for col in cols:
    assert state_total_df[col].sum() == district_df[col].sum()


# In[ ]:


# Yes, drop the 'State Total' Row


# In[ ]:


enroll_school_df.drop(enroll_school_df[enroll_school_df['District'] == 'State Total'].index, inplace=True)


# In[ ]:


enroll_school_df.tail()


# In[ ]:


# Grand Total column name has a space at the end. Rename the Column.
enroll_school_df.rename(columns={'Grand Total ': 'Grand Total'}, inplace=True)


# In[ ]:


# Check the renamed column name
enroll_school_df.columns


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
def draw_bar(df, x_column, y_column, vertical=True, to_sort=True, figsize=(5, 5), title='Bar Graph', 
             stacked=False):
    if to_sort:
        df = df.sort_values(y_column)
    df.plot.bar(x=x_column, y=y_column, figsize=figsize, title=title, stacked=stacked)
        
    


# In[ ]:


# Show district Enrollment


# In[ ]:


draw_bar(enroll_school_df, y_column='Grand Total', x_column='District', figsize=(15, 7), 
         title='Enrollment Grand Total')


# In[ ]:


### Top 3 Enrollment in the districts: Kancheepuram, Chennai, Thiruvallur
### Least 3 Enrollment in the districts: Perambulur, The Nilgris, Ariyalur


# In[ ]:


enroll_school_df[enroll_school_df['District'].isin(['Kancheepuram', 'Chennai', 'Thiruvallur'])][['District', 'Grand Total']]


# In[ ]:


enroll_school_df[enroll_school_df['District'].isin(['Perambalur', 'The Nilgiris', 'Ariyalur'])][['District', 'Grand Total']]


# In[ ]:


enroll_school_df['Grand Total'].describe()


# In[ ]:


# On an average every district has 3,85,360 enrollments.


# In[ ]:


enroll_school_df['Grand Total'].sum()


# In[ ]:


# Total Enrollments in schools are 1,23,31,525.


# In[ ]:


# Now let's map the districts and these values in the Tamil Nadu Map


# In[ ]:


districts_gdf = gpd.read_file(f'../input/2011-dist/2011_Dist.json')


# In[ ]:


districts_gdf.head()


# In[ ]:


districts_gdf['ST_NM'].unique()


# In[ ]:


# filter TN Districts
tn_districts_gdf = districts_gdf[districts_gdf['ST_NM'] == 'Tamil Nadu']


# In[ ]:


tn_districts_gdf.head()


# In[ ]:


len(tn_districts_gdf)


# In[ ]:


len(enroll_school_df)


# In[ ]:


# Rename the DISTRICT column to 'District'


# In[ ]:


tn_districts_gdf.rename(columns={'DISTRICT': 'District'}, inplace=True)


# In[ ]:


tn_districts_gdf.columns


# In[ ]:


tn_districts_gdf['District'].values


# In[ ]:


enroll_school_df['District'].values


# In[ ]:


# There is a typo in Krishnagiri in enroll_gender_df
enroll_school_df.replace(['Krishanagiri'], 'Krishnagiri', inplace=True)


# In[ ]:


# There is difference in spelling for Nagappatinam
enroll_school_df.replace(['Nagapattinam'], 'Nagappattinam', inplace=True)


# In[ ]:


tn_districts_gdf.replace(['Virudunagar'], 'Virudhunagar', inplace=True)


# In[ ]:


# Merge the tn_districts geometry information into enrollement df.
enroll_school_df = enroll_school_df.merge(tn_districts_gdf[['District', 'geometry']], on='District')


# In[ ]:


len(enroll_school_df)


# In[ ]:


enroll_school_df.columns


# In[ ]:


import numpy as np

def draw_map(df, column, annotation_column='District', figsize=(20, 20)):
    ax = df.plot(column=column, legend=True, figsize=figsize)
    if df[column].dtype == np.float:
        _ = df.apply(lambda x: ax.annotate(s=f"{x[annotation_column]}: {x[column]:.3f}", 
                                           xy=x.geometry.centroid.coords[0], ha='center'),axis=1)
    else:
        _ = df.apply(lambda x: ax.annotate(s=f"{x[annotation_column]}: {x[column]}", 
                                           xy=x.geometry.centroid.coords[0], ha='center'),axis=1)


# In[ ]:


from geopandas import GeoDataFrame
# We need geodataframe to plot, so maintain a copy of the dataframe
enroll_school_gdf = GeoDataFrame(enroll_school_df)
draw_map(enroll_school_gdf, column='Grand Total')


# In[ ]:


# Compute Grand Total Girls Vs Grand Total Boys Ratio


# In[ ]:


grand_gender_ratio = enroll_school_df['Grand Total Girls'] / enroll_school_df['Grand Total Boys'].astype(float)
enroll_school_df['Grand Gender Ratio'] = grand_gender_ratio
enroll_school_gdf['Grand Gender Ratio'] = grand_gender_ratio


# In[ ]:


# How many Girls are enrolled?

girls = enroll_school_df['Grand Total Girls'].sum()
print(girls)


# In[ ]:


# How many boys are enrolled?
boys = enroll_school_df['Grand Total Boys'].sum()
print(boys)


# In[ ]:


total = enroll_school_df['Grand Total'].sum()
print(f'Total % of girls enrolled is {(girls/total) * 100}')
print(f'Total % of boys enrolled is {(boys/total) * 100}')


# In[ ]:


draw_bar(enroll_school_df, x_column='District', y_column='Grand Gender Ratio', 
         title='District Wise Enrollment Gender Ratio', figsize=(16, 7))


# In[ ]:


# All the districts grand gender ratio is > 0.8


# In[ ]:


def print_district_info(df, districts, cols_to_print):
    print(df[df['District'].isin(districts)][cols_to_print])


# In[ ]:


# Top 3 Districts: Thiruvarur, Thoothukkudi, Chennai
# Least 3 Districts: Nammakkal, Perambalur, Dharmapuri


# In[ ]:


print_district_info(enroll_school_df, districts=['Thiruvarur', 'Thoothukkudi', 'Chennai'], 
                    cols_to_print=['District', 'Grand Gender Ratio'])


# In[ ]:


print_district_info(enroll_school_df, districts=['Namakkal', 'Perambalur', 'Dharmapuri'], 
                    cols_to_print=['District', 'Grand Gender Ratio'])


# In[ ]:


# Let's do Histogram of the ratio


# In[ ]:


def hist(df, column, title="Histogram", figsize=(5, 5)):
    df[column].plot.hist(figsize=figsize, title=title)


# In[ ]:


hist(enroll_school_df, 'Grand Gender Ratio', figsize=(7, 7), title='Enrollment Grand Gender Ratio')


# In[ ]:


enroll_school_df['Grand Gender Ratio'].describe()


# In[ ]:


# Average Gender ratio among the districts is 0.96
# Lowest Gender ration among the districts is 0.90


# In[ ]:


# What are the districts with greater than or equal to 1 as gender ratio


# In[ ]:


enroll_school_df[enroll_school_df['Grand Gender Ratio'] >= 1]['District']


# In[ ]:


# What are the districts with greater than or equal to 0.95 as gender ratio
enroll_school_df[enroll_school_df['Grand Gender Ratio'] >= 0.95]['District']


# In[ ]:


# Plot the grand gender ratio in Map


# In[ ]:


draw_map(enroll_school_gdf, column='Grand Gender Ratio')


# In[ ]:


enroll_school_df.columns


# In[ ]:


# Govt Girls vs Govt Boys Ratio
govt_gender_ratio = enroll_school_df['Govt Girls'] / enroll_school_df['Govt Boys'].astype(float)
enroll_school_df['Govt Gender Ratio'] = govt_gender_ratio
enroll_school_gdf['Govt Gender Ratio'] = govt_gender_ratio


# In[ ]:


# Describe the govt gender ratio
enroll_school_df['Govt Gender Ratio'].describe()


# In[ ]:


draw_bar(enroll_school_df, x_column='District', y_column='Govt Gender Ratio', 
         title='District Wise Enrollment Govt School Gender Ratio', figsize=(16, 7))


# In[ ]:


# All the districts govt gender ratio is > 0.8
enroll_school_df[enroll_school_df['Govt Gender Ratio'] > 1]['District']


# In[ ]:


# Highest three districts: Tirunelveli, Thoothukudi, Chennai
# Lowest three districts: The Nilgris, Kanniyakumari, Ariyalur


# In[ ]:


print_district_info(enroll_school_df, districts=['Tirunelveli', 'Thoothukkudi', 'Chennai'], 
                    cols_to_print=['District', 'Govt Gender Ratio'])


# In[ ]:


print_district_info(enroll_school_df, districts=['The Nilgiris', 'Kanniyakumari', 'Ariyalur'], 
                    cols_to_print=['District', 'Grand Gender Ratio'])


# In[ ]:


draw_map(enroll_school_gdf, column='Govt Gender Ratio')


# In[ ]:


# Private Aided Boys vs Private Aided Girls Ratio
gender_ratio = enroll_school_df['Private Aided Girls'] / enroll_school_df['Private Aided Boys'].astype(float)
enroll_school_df['Private Aided Gender Ratio'] = gender_ratio
enroll_school_gdf['Private Aided Gender Ratio'] = gender_ratio


# In[ ]:


# Describe the private aided gender ratio
enroll_school_df['Private Aided Gender Ratio'].describe()


# In[ ]:


draw_bar(enroll_school_df, x_column='District', y_column='Private Aided Gender Ratio', 
         title='District Wise Enrollment Private Aided School Gender Ratio', figsize=(16, 7))


# In[ ]:


# Highest 3 districts: The Nilgris, Ariyalur, Karur
# Lowest 3 districts: Dharmapuri, Tiruvannamalai, Pudukkotai


# In[ ]:


print_district_info(enroll_school_df, districts=['The Nilgiris', 'Ariyalur', 'Karur'], 
                    cols_to_print=['District', 'Private Aided Gender Ratio'])


# In[ ]:


print_district_info(enroll_school_df, districts=['Dharmapuri', 'Tiruvannamalai', 'Pudukkottai'], 
                    cols_to_print=['District', 'Private Aided Gender Ratio'])


# In[ ]:


draw_map(enroll_school_gdf, column='Private Aided Gender Ratio')


# In[ ]:


# Private UnAided Boys vs Private UnAided Girls Ratio
gender_ratio = enroll_school_df['Private Unaided Girls'] / enroll_school_df['Private Unaided Boys'].astype(float)
enroll_school_df['Private Unaided Gender Ratio'] = gender_ratio
enroll_school_gdf['Private Unaided Gender Ratio'] = gender_ratio


# In[ ]:


# Describe the private aided gender ratio
enroll_school_df['Private Unaided Gender Ratio'].describe()


# In[ ]:


draw_bar(enroll_school_df, x_column='District', y_column='Private Unaided Gender Ratio', 
         title='District Wise Enrollment Private Unaided School Gender Ratio', figsize=(16, 7))


# In[ ]:


# Highest Ratio: Chennai, Coimbatore, Kanniyakumari
# Lowest Ratio: Ariyalur, Dharmapuri, Perambalur


# In[ ]:


print_district_info(enroll_school_df, districts=['Chennai', 'Coimbatore', 'Kanniyakumari'], 
                    cols_to_print=['District', 'Private Unaided Gender Ratio'])


# In[ ]:


print_district_info(enroll_school_df, districts=['Ariyalur', 'Dharmapuri', 'Perambalur'], 
                    cols_to_print=['District', 'Private Unaided Gender Ratio'])


# In[ ]:


# No district has ratio > 1.


# In[ ]:


enroll_school_df[enroll_school_df['Private Unaided Gender Ratio'] > 1]['District']


# In[ ]:


draw_map(enroll_school_gdf, column='Private Unaided Gender Ratio')


# In[ ]:


### Describe on all the gender ratio
gender_cols = ['Govt Gender Ratio', 'Private Unaided Gender Ratio', 'Private Aided Gender Ratio']
enroll_school_df[gender_cols].describe()


# In[ ]:


# Govt Gender Ratio, Private Aided Gender Ratio has mean ratio greater than 1.


# In[ ]:


# Let's draw district wise line chart for ratio


# In[ ]:


def draw_line_chart(df, x, y, title='Line Chart', figsize=(10, 10), subplots=False):
    df.plot.line(x=x, y=y, title=title, figsize=figsize, subplots=subplots)


# In[ ]:


draw_line_chart(enroll_school_df, x='District', y=gender_cols, title='District wise gender ratio',
               figsize=(15, 7))


# In[ ]:


def draw_scatter_chart(df, x, y, title='Line Chart', figsize=(10, 10), subplots=False):
    df.plot.scatter(x=x, y=y, title=title, figsize=figsize, subplots=subplots)


# In[ ]:


# Let's draw bar graph with all four gender cols in 5 different bar charts
draw_bar(enroll_school_df[:6], x_column='District', y_column=gender_cols, title='District wise gender ratio',
               figsize=(15, 7), stacked=False)


# In[ ]:


draw_bar(enroll_school_df[6:12], x_column='District', y_column=gender_cols, title='District wise gender ratio',
               figsize=(15, 7), stacked=False)


# In[ ]:


draw_bar(enroll_school_df[12:18], x_column='District', y_column=gender_cols, title='District wise gender ratio',
               figsize=(15, 7), stacked=False)


# In[ ]:


draw_bar(enroll_school_df[18:24], x_column='District', y_column=gender_cols, title='District wise gender ratio',
               figsize=(15, 7), stacked=False)


# In[ ]:


draw_bar(enroll_school_df[24:], x_column='District', y_column=gender_cols, title='District wise gender ratio',
               figsize=(15, 7), stacked=False)


# In[ ]:


# Let's calculate the variance for each district on gender columns
gender_variance = {}
for idx, row in enroll_school_df.iterrows():
    dist = row['District']
    gender_variance[dist] = row[gender_cols].values.var()


# In[ ]:


gender_variance


# In[ ]:


# Lower the variance, uniform the ratio


# In[ ]:


var_df = pd.DataFrame({'District': list(gender_variance.keys()), 'Variance': list(gender_variance.values())})


# In[ ]:


var_df.head()


# In[ ]:


draw_bar(var_df, x_column='District', y_column='Variance', figsize=(16, 7))


# In[ ]:


# Add variance to each district
enroll_school_gdf['Variance'] = [np.nan] * len(enroll_school_gdf)
for idx, row in var_df.iterrows():
    enroll_school_gdf.loc[enroll_school_gdf['District'] == row['District'], 'Variance'] = row['Variance']


# In[ ]:


enroll_school_gdf.columns


# In[ ]:


enroll_school_gdf['Variance'] = enroll_school_gdf['Variance'].astype(float)
draw_map(enroll_school_gdf, column='Variance')


# In[ ]:


enroll_school_gdf['Variance'].describe()


# In[ ]:


# The Nilgris has highest gender ratio difference in different school types.


# In[ ]:


enroll_school_df[enroll_school_df['District'] == 'The Nilgiris'][gender_cols]


# In[ ]:


### How much is the grand enrollment percentage numbers?
govt_total = enroll_school_df['Govt Total'].sum()
private_aided_total = enroll_school_df['Private Aided Total'].sum()
private_unaided_total = enroll_school_df['Private Unaided Total'].sum()


# In[ ]:


labels = ['Govt Total', 'Private Aided Total', 'Private Unaided Total']
x = pd.DataFrame({'labels': labels,
                  'data': [govt_total, private_aided_total, private_unaided_total]}, index=labels)


# In[ ]:


x.head()


# In[ ]:


x.plot.pie(y='data', figsize=(5, 10), autopct='%.2f', title='Enrollment ratio across different of schools')


# In[ ]:


# 1. 43.09% enrollments in Private Unaided Total
# 2. 19.8% enrollments in Private Aided Total
# 3. 62.89% enrollments in Private School
# 4. 37.80% enrollment in Govt School


# In[ ]:


### How much is the girls enrollment percentage numbers?
govt_girls_total = enroll_school_df['Govt Girls'].sum()
private_aided_girls_total = enroll_school_df['Private Aided Girls'].sum()
private_unaided_girls_total = enroll_school_df['Private Unaided Girls'].sum()


# In[ ]:


labels = ['Govt Girls', 'Private Aided Girls', 'Private Unaided Girls']
y_girls = pd.DataFrame({'labels': labels,
                  'data': [govt_girls_total, private_aided_girls_total, private_unaided_girls_total]},
                 index=labels)
y_girls.head()


# In[ ]:


y_girls.plot.pie(y='data', figsize=(5, 10), autopct='%.2f', 
               title='Enrollment girls ratio across different of type of schools')


# In[ ]:


# 1. 21.09% girls enrolled in Private Aided School
# 2. 39.70% girls enrolled in Private Unaided School
# 3. 60.79% girls enrolled in Private School
# 4. 39.11% girls enrolled in Govt School


# In[ ]:


### How much is the boys enrollment percentage numbers?
govt_boys_total = enroll_school_df['Govt Boys'].sum()
private_aided_boys_total = enroll_school_df['Private Aided Boys'].sum()
private_unaided_boys_total = enroll_school_df['Private Unaided Boys'].sum()


# In[ ]:


labels = ['Govt Boys', 'Private Aided Boys', 'Private Unaided Boys']
y_boys = pd.DataFrame({'labels': labels,
                  'data': [govt_boys_total, private_aided_boys_total, private_unaided_boys_total]},
                 index=labels)
y_boys.head()


# In[ ]:


y_boys.plot.pie(y='data', figsize=(5, 10), autopct='%.2f', 
           title='Enrollment boys ratio across different of type of schools')


# In[ ]:


# 1. 18.54% boys enrolled in Private Aided School
# 2. 46.38% boys enrolled in Private Unaided School
# 3. 64.92% boys enrolled in Private School
# 4. 35.08% boys enrolled in Govt School


# In[ ]:


girls = y_girls['data'].sum()
boys = y_boys['data'].sum()
labels = ['Girls', 'Boys']
data = [girls, boys]
y = pd.DataFrame({'labels': labels, 'data': data}, index=labels)
y.head()


# In[ ]:


y.plot.pie(y='data', figsize=(5, 10), autopct='%.2f', 
           title='Enrollment girls vs boys')


# In[ ]:


# Overall Girls and boys enrollment are same


# In[ ]:


y_girls.columns


# In[ ]:


girls = enroll_school_df['Private Aided Girls'].sum()
boys = enroll_school_df['Private Aided Boys'].sum()
labels = ['Girls', 'Boys']
data = [girls, boys]
y = pd.DataFrame({'labels': labels, 'data': data}, index=labels)
y.head()


# In[ ]:


y.plot.pie(y='data', figsize=(5, 10), autopct='%.2f', 
           title='Private Aided Enrollment girls vs boys')


# In[ ]:


# In private Aided girls enrollement is higher than boys


# In[ ]:


girls = enroll_school_df['Private Unaided Girls'].sum()
boys = enroll_school_df['Private Unaided Boys'].sum()
labels = ['Girls', 'Boys']
data = [girls, boys]
y = pd.DataFrame({'labels': labels, 'data': data}, index=labels)
y.head()
y.plot.pie(y='data', figsize=(5, 10), autopct='%.2f', 
           title='Private Unaided Enrollment girls vs boys')


# In[ ]:


# In private unaided, there is significant difference in girls vs boys enrollment. Difference is ~10%


# In[ ]:


# Let's see the enrollment genderwise data see if any other data is available

enroll_gender_df = pd.read_csv(f'{input_dir}/enrollment_genderwise_0.csv')


# In[ ]:


enroll_gender_df.info()


# In[ ]:


# this dataset has primary school, middle school, high school, higher secondary school data district wise, 
# but missing private/govt split.


# In[ ]:


enroll_gender_df['District'].values


# In[ ]:


# remove district name with `nan` or select the district from previous dataframe


# In[ ]:


len(enroll_gender_df)


# In[ ]:


enroll_gender_df.replace(['Krishanagiri'], 'Krishnagiri', inplace=True)
enroll_gender_df.replace(['Nagapattinam'], 'Nagappattinam', inplace=True)


# In[ ]:


len(enroll_gender_df)


# In[ ]:


districts = enroll_school_df['District'].values
enroll_gender_df = enroll_gender_df[enroll_gender_df['District'].isin(districts)]


# In[ ]:


len(enroll_school_df), len(enroll_gender_df)


# In[ ]:


enroll_gender_df.head()


# In[ ]:


# We had already seen District wise Grand Total students enrollment. Let's skip


# In[ ]:


enroll_gender_df.columns


# In[ ]:


# See District Primary School Total students enrollment 
draw_bar(enroll_gender_df, x_column='District', y_column='Primary School Total', 
         title='District wise Primary School Enrollment', figsize=(16, 7))


# In[ ]:


# Highest Primary School Enrollment in Vellore, Vilippuram, Tirunnelveli
# Lowest Primary School Enrollment in The Nilgris, Perambalur, Karur


# In[ ]:


print_district_info(enroll_gender_df, districts=['Vellore', 'Viluppuram', 'Tirunelveli'], 
                    cols_to_print=['District', 'Primary School Total'])


# In[ ]:


print_district_info(enroll_gender_df, districts=['The Nilgiris', 'Perambalur', 'Karur'], 
                    cols_to_print=['District', 'Primary School Total'])


# In[ ]:


enroll_gender_df = enroll_gender_df.merge(tn_districts_gdf[['District', 'geometry']], on='District')
enroll_gender_gdf =  GeoDataFrame(enroll_gender_df)


# In[ ]:


len(enroll_gender_df), len(enroll_gender_gdf)


# In[ ]:


draw_map(enroll_gender_gdf, 'Primary School Total')


# In[ ]:


# Girls Enrollment


# In[ ]:


draw_bar(enroll_gender_df, x_column='District', y_column='Primary School Girls', 
         title='District wise Primary School Girls Enrollment', figsize=(16, 7))


# In[ ]:


# Highest Girls Enrollment: Vellore, Tirunnelveli, Viluppuram
# Lowest Girls Enrollment: The Nilgris, Perambalur, Ariyalur 


# In[ ]:


print_district_info(enroll_gender_df, districts=['Vellore', 'Viluppuram', 'Tirunelveli'], 
                    cols_to_print=['District', 'Primary School Girls'])


# In[ ]:


print_district_info(enroll_gender_df, districts=['The Nilgiris', 'Perambalur', 'Ariyalur'], 
                    cols_to_print=['District', 'Primary School Total'])


# In[ ]:


draw_map(enroll_gender_gdf, 'Primary School Girls')


# In[ ]:


# Boys Enrollment


# In[ ]:


draw_bar(enroll_gender_df, x_column='District', y_column='Primary School Boys', 
         title='District wise Primary School Boys Enrollment', figsize=(16, 7))


# In[ ]:


# Highest boys enrollment: Vellore, Viluppuram, Tirunelveli
# Lowest boys enrollment: The Nilgris, Perambalur, Karur


# In[ ]:


print_district_info(enroll_gender_df, districts=['Vellore', 'Viluppuram', 'Tirunelveli'], 
                    cols_to_print=['District', 'Primary School Boys'])


# In[ ]:


print_district_info(enroll_gender_df, districts=['The Nilgiris', 'Perambalur', 'Karur'], 
                    cols_to_print=['District', 'Primary School Boys'])


# In[ ]:


draw_map(enroll_gender_gdf, 'Primary School Boys')


# In[ ]:


# Pie chart: See Primary School Girls vs Boys enrollment 


# In[ ]:


enroll_gender_df[['Primary School Boys', 'Primary School Girls']].sum().plot.pie(autopct="%.2f")


# In[ ]:


# See Grand Total Girls vs Grand Total Boys enrollment Ratio
gender_ratio = enroll_gender_df['Primary School Girls'] / enroll_gender_df['Primary School Boys'].astype(float)
enroll_gender_df['Primary School Gender Ratio'] = gender_ratio
enroll_gender_gdf['Primary School Gender Ratio'] = gender_ratio


# In[ ]:


enroll_gender_df['Primary School Gender Ratio'].describe()


# In[ ]:


# Districts with gender >= 1
enroll_gender_df[enroll_gender_df['Primary School Gender Ratio'] >= 1]['District']


# In[ ]:


# 9 districts ratio is greater than or equal  to 1


# In[ ]:


draw_bar(enroll_gender_df, 'District', 'Primary School Gender Ratio', 
         title='District wise Primary School Gender Ratio', figsize=(16, 7))


# In[ ]:


# highest gender ratio: Kanniyakumari, the nilgris, Kancheepuram
# lowest gender ratio: Ariyalur, Cuddalore, Perambalur


# In[ ]:


print_district_info(enroll_gender_df, districts=['Kanniyakumari', 'The Nilgiris', 'Kancheepuram'], 
                    cols_to_print=['District', 'Primary School Gender Ratio'])


# In[ ]:


print_district_info(enroll_gender_df, districts=['Ariyalur', 'Cuddalore', 'Perambalur'], 
                    cols_to_print=['District', 'Primary School Gender Ratio'])


# In[ ]:


draw_map(enroll_gender_gdf, 'Primary School Gender Ratio')


# In[ ]:


# See District Middle School Total students enrollment 


# In[ ]:


draw_bar(enroll_gender_df, x_column='District', y_column='Middle School Total', 
         title='District wise Middle School Enrollment', figsize=(16, 7))


# In[ ]:


# Highest Primary School Enrollment in Vilippuram, Kancheepuram, Tirunnelveli
# Lowest Primary School Enrollment in The Nilgris, Perambalur, Ariyalur


# In[ ]:


print_district_info(enroll_gender_df, districts=['Kancheepuram', 'Viluppuram', 'Tirunelveli'], 
                    cols_to_print=['District', 'Middle School Total'])


# In[ ]:


print_district_info(enroll_gender_df, districts=['The Nilgiris', 'Perambalur', 'Ariyalur'], 
                    cols_to_print=['District', 'Middle School Total'])


# In[ ]:


draw_map(enroll_gender_gdf, 'Middle School Total')


# In[ ]:


# Girls Enrollment


# In[ ]:


draw_bar(enroll_gender_df, x_column='District', y_column='Middle School Girls', 
         title='District wise Middle School Girls Enrollment', figsize=(16, 7))


# In[ ]:


# Highest girls enrollment: Viluppuram, Kancheepuram, Vellore
# Lowest girls enrollment: The Nilgris, Perambualur, Ariyalur


# In[ ]:


print_district_info(enroll_gender_df, districts=['Vellore', 'Viluppuram', 'Kancheepuram'], 
                    cols_to_print=['District', 'Middle School Girls'])


# In[ ]:


print_district_info(enroll_gender_df, districts=['The Nilgiris', 'Perambalur', 'Ariyalur'], 
                    cols_to_print=['District', 'Middle School Girls'])


# In[ ]:


draw_map(enroll_gender_gdf, 'Middle School Girls')


# In[ ]:


# boys Enrollment


# In[ ]:


draw_bar(enroll_gender_df, x_column='District', y_column='Middle School Boys', 
         title='District wise Middle School Boys Enrollment', figsize=(16, 7))


# In[ ]:


# Highest boys enrollment: Viluppuram, Kancheepuram, Tirunelveli
# Lowest boys enrollment: The Nilgris, Perambualur, Ariyalur


# In[ ]:


print_district_info(enroll_gender_df, districts=['Kancheepuram', 'Viluppuram', 'Tirunelveli'], 
                    cols_to_print=['District', 'Middle School Boys'])


# In[ ]:


print_district_info(enroll_gender_df, districts=['The Nilgiris', 'Perambalur', 'Ariyalur'], 
                    cols_to_print=['District', 'Middle School Boys'])


# In[ ]:


draw_map(enroll_gender_gdf, 'Middle School Boys')


# In[ ]:


# Pie chart: See Middle School Girls vs Boys enrollment 


# In[ ]:


enroll_gender_df[['Middle School Girls', 'Middle School Boys']].sum().plot.pie(autopct="%.2f")


# In[ ]:


# See Grand Total Girls vs Grand Total Boys enrollment Ratio
gender_ratio = enroll_gender_df['Middle School Girls'] / enroll_gender_df['Middle School Boys'].astype(float)
enroll_gender_df['Middle School Gender Ratio'] = gender_ratio
enroll_gender_gdf['Middle School Gender Ratio'] = gender_ratio


# In[ ]:


enroll_gender_df['Middle School Gender Ratio'].describe()


# In[ ]:


# Districts with gender >= 1
enroll_gender_df[enroll_gender_df['Middle School Gender Ratio'] >= 1]['District']


# In[ ]:


# 12 districts ratio is greater than or equal  to 1


# In[ ]:


draw_bar(enroll_gender_df, 'District', 'Middle School Gender Ratio', 
         title='District wise Middle School Gender Ratio', figsize=(16, 7))


# In[ ]:


# highest gender ratio: Viluppuram, Dindigul, Thiruvarur
# lowest gender ratio: Chennai, Kanniyakumari, Tiruppur


# In[ ]:


print_district_info(enroll_gender_df, districts=['Dindigul', 'Viluppuram', 'Tirunelveli'], 
                    cols_to_print=['District', 'Middle School Gender Ratio'])


# In[ ]:


print_district_info(enroll_gender_df, districts=['Chennai', 'Kanniyakumari', 'Tiruppur'], 
                    cols_to_print=['District', 'Middle School Gender Ratio'])


# In[ ]:


draw_map(enroll_gender_gdf, 'Middle School Gender Ratio')


# In[ ]:


# See High School enrollment


# In[ ]:


draw_bar(enroll_gender_df, x_column='District', y_column='High School Total', 
         title='District wise High School Enrollment', figsize=(16, 7))


# In[ ]:


# Highest Primary School Enrollment in Kancheepuram, Thiruvallur, Vellore
# Lowest Primary School Enrollment in Perambalur, Theni, Ariyalur


# In[ ]:


print_district_info(enroll_gender_df, districts=['Kancheepuram', 'Thiruvallur', 'Vellore'], 
                    cols_to_print=['District', 'High School Total'])


# In[ ]:


print_district_info(enroll_gender_df, districts=['Perambalur', 'Theni', 'Ariyalur'], 
                    cols_to_print=['District', 'High School Total'])


# In[ ]:


draw_map(enroll_gender_gdf, 'High School Total')


# In[ ]:


# Girls Enrollment


# In[ ]:


draw_bar(enroll_gender_df, x_column='District', y_column='High School Girls', 
         title='District wise High School Girls Enrollment', figsize=(16, 7))


# In[ ]:


# Highest Primary School Enrollment in Kancheepuram, Thiruvallur, Vellore
# Lowest Primary School Enrollment in Perambalur, Theni, Ariyalur


# In[ ]:


print_district_info(enroll_gender_df, districts=['Kancheepuram', 'Thiruvallur', 'Vellore'], 
                    cols_to_print=['District', 'High School Girls'])


# In[ ]:


print_district_info(enroll_gender_df, districts=['Perambalur', 'Theni', 'Ariyalur'], 
                    cols_to_print=['District', 'High School Girls'])


# In[ ]:


draw_map(enroll_gender_gdf, 'High School Girls')


# In[ ]:


# Boys Enrollment


# In[ ]:


draw_bar(enroll_gender_df, x_column='District', y_column='High School Boys', 
         title='District wise High School Boys Enrollment', figsize=(16, 7))


# In[ ]:


# Highest Primary School Enrollment in Kancheepuram, Thiruvallur, Vellore
# Lowest Primary School Enrollment in Perambalur, theni, Ariyalur


# In[ ]:


print_district_info(enroll_gender_df, districts=['Kancheepuram', 'Vellore', 'Thiruvallur'], 
                    cols_to_print=['District', 'High School Boys'])


# In[ ]:


print_district_info(enroll_gender_df, districts=['Perambalur', 'Theni', 'Ariyalur'], 
                    cols_to_print=['District', 'High School Boys'])


# In[ ]:


draw_map(enroll_gender_gdf, 'High School Boys')


# In[ ]:


# Pie chart: See High School Girls vs Boys enrollment 


# In[ ]:


enroll_gender_df[['High School Girls', 'High School Boys']].sum().plot.pie(autopct="%.2f")


# In[ ]:


# See Grand Total Girls vs Grand Total Boys enrollment Ratio
gender_ratio = enroll_gender_df['High School Girls'] / enroll_gender_df['High School Boys'].astype(float)
enroll_gender_df['High School Gender Ratio'] = gender_ratio
enroll_gender_gdf['High School Gender Ratio'] = gender_ratio


# In[ ]:


enroll_gender_df['High School Gender Ratio'].describe()


# In[ ]:


# Districts with gender >= 1
enroll_gender_df[enroll_gender_df['High School Gender Ratio'] >= 1]['District']


# In[ ]:


# There is no district in high school enrollment women outnumber men


# In[ ]:


draw_bar(enroll_gender_df, 'District', 'High School Gender Ratio', 
         title='District wise High School Gender Ratio', figsize=(16, 7))


# In[ ]:


# highest gender ratio: Nagapattinam, Erode, Pudukkotai
# lowest gender ratio: Madurai, Theni, Thiruvarur


# In[ ]:


print_district_info(enroll_gender_df, districts=['Nagappattinam', 'Erode', 'Pudukkottai'], 
                    cols_to_print=['District', 'High School Gender Ratio'])


# In[ ]:


print_district_info(enroll_gender_df, districts=['Madurai', 'Theni', 'Thiruvarur'], 
                    cols_to_print=['District', 'High School Gender Ratio'])


# In[ ]:


draw_map(enroll_gender_gdf, 'High School Gender Ratio')


# In[ ]:


enroll_gender_df.columns


# In[ ]:


# See Hr.Secondary School enrollment


# In[ ]:


draw_bar(enroll_gender_df, x_column='District', y_column='Hr.Secondary School Total', 
         title='District wise Hr.Secondary School Enrollment', figsize=(16, 7))


# In[ ]:


# Highest Primary School Enrollment in Chennai, Kancheepuram, Thiruvallur 
# Lowest Primary School Enrollment in Ariyalur, Perambalur, The Nilgris,


# In[ ]:


print_district_info(enroll_gender_df, districts=['Chennai', 'Kancheepuram', 'Thiruvallur'], 
                    cols_to_print=['District', 'Hr.Secondary School Total'])


# In[ ]:


print_district_info(enroll_gender_df, districts=['Ariyalur', 'Perambalur', 'The Nilgiris'], 
                    cols_to_print=['District', 'Hr.Secondary School Total'])


# In[ ]:


draw_map(enroll_gender_gdf, 'Hr.Secondary School Total')


# In[ ]:


# Girls Enrollment


# In[ ]:


draw_bar(enroll_gender_df, x_column='District', y_column='Hr.Secondary School Girls', 
         title='District wise Hr.Secondary School Girls Enrollment', figsize=(16, 7))


# In[ ]:


# Highest Primary School Enrollment in Chennai, Kancheepuram, Thiruvallur
# Lowest Primary School Enrollment in Perambalur, Ariyalur, The Nilgris


# In[ ]:


print_district_info(enroll_gender_df, districts=['Chennai', 'Kancheepuram', 'Thiruvallur'], 
                    cols_to_print=['District', 'Hr.Secondary School Girls'])


# In[ ]:


print_district_info(enroll_gender_df, districts=['Perambalur', 'Ariyalur', 'The Nilgiris'], 
                    cols_to_print=['District', 'Hr.Secondary School Girls'])


# In[ ]:


draw_map(enroll_gender_gdf, 'Hr.Secondary School Girls')


# In[ ]:


# Boys Enrollment


# In[ ]:


draw_bar(enroll_gender_df, x_column='District', y_column='Hr.Secondary School Boys', 
         title='District wise Hr.Secondary School Boys Enrollment', figsize=(16, 7))


# In[ ]:


# Highest Primary School Enrollment in Chennai, Kancheepuram, Thiruvallur
# Lowest Primary School Enrollment in Ariyalur, The Nilgris, Perambalur


# In[ ]:


print_district_info(enroll_gender_df, districts=['Chennai', 'Kancheepuram', 'Thiruvallur'], 
                    cols_to_print=['District', 'Hr.Secondary School Boys'])


# In[ ]:


print_district_info(enroll_gender_df, districts=['Ariyalur', 'The Nilgiris', 'Perambalur'], 
                    cols_to_print=['District', 'Hr.Secondary School Boys'])


# In[ ]:


draw_map(enroll_gender_gdf, 'Hr.Secondary School Boys')


# In[ ]:


# Pie chart: See Hr.Secondary School Girls vs Boys enrollment 


# In[ ]:


enroll_gender_df[['Hr.Secondary School Girls', 'Hr.Secondary School Boys']].sum().plot.pie(autopct="%.2f")


# In[ ]:


# See Grand Total Girls vs Grand Total Boys enrollment Ratio
gender_ratio = enroll_gender_df['Hr.Secondary School Girls'] / enroll_gender_df['Hr.Secondary School Boys'].astype(float)
enroll_gender_df['Hr.Secondary School Gender Ratio'] = gender_ratio
enroll_gender_gdf['Hr.Secondary School Gender Ratio'] = gender_ratio


# In[ ]:


enroll_gender_df['Hr.Secondary School Gender Ratio'].describe()


# In[ ]:


# Districts with gender >= 1
enroll_gender_df[enroll_gender_df['Hr.Secondary School Gender Ratio'] >= 1]['District']


# In[ ]:


# 13 districts ratio is greater than or equal  to 1


# In[ ]:


draw_bar(enroll_gender_df, 'District', 'Hr.Secondary School Gender Ratio', 
         title='District wise Hr.Secondary School Gender Ratio', figsize=(16, 7))


# In[ ]:


# highest gender ratio: Ariyalur, Thothukkudi, Thiruvarur
# lowest gender ratio: Dharmapuri, Permabalur, Nammakkal


# In[ ]:


print_district_info(enroll_gender_df, districts=['Ariyalur', 'Thoothukkudi', 'Thiruvarur'], 
                    cols_to_print=['District', 'Hr.Secondary School Gender Ratio'])


# In[ ]:


print_district_info(enroll_gender_df, districts=['Dharmapuri', 'Perambalur', 'Namakkal'], 
                    cols_to_print=['District', 'Hr.Secondary School Gender Ratio'])


# In[ ]:


draw_map(enroll_gender_gdf, 'Hr.Secondary School Gender Ratio')


# In[ ]:


## Describe on all gender ratio columns
enroll_gender_df[['Primary School Gender Ratio', 'Middle School Gender Ratio',
                  'High School Gender Ratio', 'Hr.Secondary School Gender Ratio']].describe()


# In[ ]:


# 1. Mean is lowest in High School Gender ratio, the difference is significant. 
# But picks up in Higher Secondary school. 
# So lot of people drop in high school, those survive continue in higher secondary. 
# Is the data for high school corruput?

# 2. Std deviation for High School ratio is low, this pattern is common in all districts in TN.

# 3. there is not even a single district in TN, where high school enrollment ratio is greater than or equal to 1.

# Hr.Secondary Ratio
# highest gender ratio: Ariyalur, Thothukkudi, Thiruvarur
# lowest gender ratio: Dharmapuri, Permabalur, Nammakkal

# High School Ratio
# highest gender ratio: Nagapattinam, Erode, Pudukkotai
# lowest gender ratio: Madurai, Theni, Thiruvarur

# Middle School Ratio
# highest gender ratio: Viluppuram, Dindigul, Thiruvarur
# lowest gender ratio: Chennai, Kanniyakumari, Tiruppur

# Primary School Ratio
# highest gender ratio: Kanniyakumari, the nilgris, Kancheepuram
# lowest gender ratio: Ariyalur, Cuddalore, Perambalur

# Ariyalur which has low primary school gender ratio, has highest hr secondary school gender ratio
# Chennai has lowest middle school gender ratio
# Madurai, Theni shows up  lowest gender ratio in middle school
# thiruvarur which is one of the highest gender ratio in Hr.Secondary, Middle School has lowest gender ratio 
# in High School


# In[ ]:


### Teacher information


# In[ ]:


teacher_df = pd.read_csv(f'{input_dir}/no.ofteachers_0.csv')


# In[ ]:


teacher_df.info()


# In[ ]:


teacher_df['District'].values


# In[ ]:


teacher_df['District'].values[:32]


# In[ ]:


teacher_df.replace(['Krishanagiri'], 'Krishnagiri', inplace=True)


# In[ ]:


teacher_df.replace(['Nagapattinam'], 'Nagappattinam', inplace=True)


# In[ ]:


len(teacher_df)


# In[ ]:


teacher_df = teacher_df[teacher_df['District'].isin(enroll_school_df['District'])]


# In[ ]:


len(teacher_df)


# In[ ]:


teacher_df.head()


# In[ ]:


# Rename the column, and merge it with enroll_school_df, and enroll_school_gdf
teacher_df.rename(columns={'Govt': 'Govt Teachers', 'Pvt Aided': 'Private Aided Teachers',
                           'Pvt Unaided': 'Private Unaided Teachers'}, inplace=True)


# In[ ]:


teacher_df.columns


# In[ ]:


teacher_df.head()


# In[ ]:


len(enroll_school_df), len(enroll_school_df.columns)


# In[ ]:


enroll_school_df = enroll_school_df.merge(teacher_df[['District', 'Govt Teachers', 'Private Aided Teachers',
                                   'Private Unaided Teachers', 'Total Teachers']], on='District')


# In[ ]:


len(enroll_school_df), len(enroll_school_df.columns)


# In[ ]:


enroll_school_df.head()


# In[ ]:


enroll_school_gdf = enroll_school_gdf.merge(teacher_df[['District', 'Govt Teachers', 'Private Aided Teachers',
                                                        'Private Unaided Teachers', 'Total Teachers']],
                                            on='District')


# In[ ]:


enroll_school_df.columns


# In[ ]:


# Calculate the Students vs Govt Teachers Ratio
ratio = enroll_school_df['Govt Total'] / enroll_school_df['Govt Teachers']
enroll_school_df['Govt Student Teacher Ratio'] = ratio
enroll_school_gdf['Govt Student Teacher Ratio'] = ratio

ratio = enroll_school_df['Private Aided Total'] / enroll_school_df['Private Aided Teachers']
enroll_school_df['Private Aided Student Teacher Ratio'] = ratio
enroll_school_gdf['Private Aided Student Teacher Ratio'] = ratio

ratio = enroll_school_df['Private Unaided Total'] / enroll_school_df['Private Unaided Teachers']
enroll_school_df['Private Unaided Student Teacher Ratio'] = ratio
enroll_school_gdf['Private Unaided Student Teacher Ratio'] = ratio

ratio = enroll_school_df['Grand Total'] / enroll_school_df['Total Teachers']
enroll_school_df['Grand Student Teacher Ratio'] = ratio
enroll_school_gdf['Grand Student Teacher Ratio'] = ratio


# In[ ]:


# Govt Student Teacher Ratio

enroll_school_df['Govt Student Teacher Ratio'].describe()


# In[ ]:


draw_bar(enroll_school_df, 'District', y_column='Govt Student Teacher Ratio',
         title='Govt Student Teacher Ratio', figsize=(16, 7))


# In[ ]:


# Lower the ratio better it is
# Better ratio: The Nilgiris, Sivaganga, Ramnathapuram
# Higher ratio: Villuppuram, Krishnagiri, Tiruvannamalai


# In[ ]:


print_district_info(enroll_school_df, districts=['The Nilgiris', 'Sivaganga', 'Ramanathapuram'], 
                    cols_to_print=['District', 'Govt Student Teacher Ratio'])


# In[ ]:


print_district_info(enroll_school_df, districts=['Viluppuram', 'Krishnagiri', 'Tiruvannamalai'], 
                    cols_to_print=['District', 'Govt Student Teacher Ratio'])


# In[ ]:


draw_map(enroll_school_gdf, 'Govt Student Teacher Ratio')


# In[ ]:


# Districts with ratio less than mean.

enroll_school_df[enroll_school_df['Govt Student Teacher Ratio'] <= 19.3]['District']


# In[ ]:


# 16 districts


# In[ ]:


# see 'Private Aided Student Teacher Ratio'
col = 'Private Aided Student Teacher Ratio'


# In[ ]:


enroll_school_df[col].describe()


# In[ ]:


# Mean/Std is worse.


# In[ ]:


draw_bar(enroll_school_df, 'District', y_column=col, title=col, figsize=(16, 7))


# In[ ]:


# Lower the better
# Better: KanniyaKumari, The Nilgris, Namakkal
# Highest: Thiruvallur, Vellore, Krishnagiri


# In[ ]:


print_district_info(enroll_school_df, districts=['Kanniyakumari', 'The Nilgiris', 'Namakkal'], 
                    cols_to_print=['District', 'Private Aided Student Teacher Ratio'])


# In[ ]:


print_district_info(enroll_school_df, districts=['Thiruvallur', 'Vellore', 'Krishnagiri'], 
                    cols_to_print=['District', 'Private Aided Student Teacher Ratio'])


# In[ ]:


draw_map(enroll_school_gdf, col)


# In[ ]:


enroll_school_df[enroll_school_df[col] <= 30.31]['District']


# In[ ]:


#18 districts less than or equal to mean


# In[ ]:


# see 'Private Unaided Student Teacher Ratio'
col = 'Private Unaided Student Teacher Ratio'


# In[ ]:


enroll_school_df[col].describe()


# In[ ]:


enroll_school_df[enroll_school_df[col] <= 19.5]['District']


# In[ ]:


#20 Districts


# In[ ]:


draw_bar(enroll_school_df, 'District', y_column=col, title=col, figsize=(16, 7))


# In[ ]:


# Lower the better
# Better: Theni, Karur, Erode
# Highest: Kancheepuram, thiruvallur, Viluppuram


# In[ ]:


print_district_info(enroll_school_df, districts=['Theni', 'Karur', 'Erode'], 
                    cols_to_print=['District', 'Private Unaided Student Teacher Ratio'])


# In[ ]:


print_district_info(enroll_school_df, districts=['Kancheepuram', 'Thiruvallur', 'Viluppuram'], 
                    cols_to_print=['District', 'Private Unaided Student Teacher Ratio'])


# In[ ]:


draw_map(enroll_school_gdf, col)


# In[ ]:


# See 'Grand Student Teacher Ratio'
col = 'Grand Student Teacher Ratio'


# In[ ]:


enroll_school_df[col].describe()


# In[ ]:


enroll_school_df[enroll_school_df[col] <= 20.99]['District']


# In[ ]:


#14 districts


# In[ ]:


draw_bar(enroll_school_df, 'District', y_column=col, title=col, figsize=(16, 7))


# In[ ]:


# Lower is better
# Better: The nilgris, Karur, KanniyaKumari
# Highest: Kancheepuram, Viluppuram, Chennai


# In[ ]:


print_district_info(enroll_school_df, districts=['The Nilgiris', 'Karur', 'Kanniyakumari'], 
                    cols_to_print=['District', 'Grand Student Teacher Ratio'])


# In[ ]:


print_district_info(enroll_school_df, districts=['Kancheepuram', 'Viluppuram', 'Chennai'], 
                    cols_to_print=['District', 'Grand Student Teacher Ratio'])


# In[ ]:


cols = ['Govt Student Teacher Ratio', 'Private Aided Student Teacher Ratio',
       'Private Unaided Student Teacher Ratio', 'Grand Student Teacher Ratio']
enroll_school_df[cols].describe()


# In[ ]:


# Govt school student teacher ratio is lowest 19.3
# Private Aided Student teacher ratio is highest 30.3
# Private Unaided student teacher ratio is close to govt school
# Lowest std deviation is Private Unaided school.
# TN Student Teacher Ratio is 21.

# Govt School

# Better ratio: The Nilgiris, Sivaganga, Ramnathapuram
# Higher ratio: Villuppuram, Krishnagiri, Tiruvannamalai

# Private Aided

# Better: KanniyaKumari, The Nilgris, Namakkal
# Highest: Thiruvallur, Vellore, Krishnagiri

# Private Unaided 

# Better: Theni, Karur, Erode
# Highest: Kancheepuram, thiruvallur, Viluppuram

# Overall
# Better: The nilgris, Karur, KanniyaKumari
# Highest: Kancheepuram, Viluppuram, Chennai


# In[ ]:


management_df = pd.read_csv(f'{input_dir}/managementwise_schools_0.csv')


# In[ ]:


management_df.columns


# In[ ]:


management_df.head()


# In[ ]:


management_df['District'] = management_df['District'].apply(lambda x: x.title())


# In[ ]:


management_df['District'].values


# In[ ]:


# Rename values
management_df.replace(['Krishanagiri'], 'Krishnagiri', inplace=True)
management_df.replace(['Nagapattinam'], 'Nagappattinam', inplace=True)


# In[ ]:


len(management_df)


# In[ ]:


# Filter the valid districts


# In[ ]:


management_df = management_df[management_df['District'].isin(enroll_school_df['District'])]


# In[ ]:


len(management_df)


# In[ ]:


management_df.columns


# In[ ]:


# Rename columns
management_df.rename(columns={'Govt': 'Govt Schools',
                              'Pvt Aided': 'Private Aided Schools',
                              'Pvt Unaided': 'Private Unaided Schools',
                              'Grand Total': 'Grand Total Schools'}, inplace=True)


# In[ ]:


management_df.columns


# In[ ]:


len(enroll_school_df.columns)


# In[ ]:


# Merge with scholl enrollment df and gdf
enroll_school_df = enroll_school_df.merge(management_df[['District', 'Govt Schools',
                                                         'Private Aided Schools',
                                                         'Private Unaided Schools',
                                                         'Grand Total Schools']], on='District')


# In[ ]:


len(enroll_school_df.columns)


# In[ ]:


len(enroll_school_df)


# In[ ]:


# Merge with scholl enrollment df and gdf
enroll_school_gdf = enroll_school_gdf.merge(management_df[['District', 'Govt Schools',
                                                         'Private Aided Schools',
                                                         'Private Unaided Schools',
                                                         'Grand Total Schools']], on='District')


# In[ ]:


# See districts with more schools


# In[ ]:


draw_bar(enroll_school_df, 'District', 'Grand Total Schools', 'Districtwise Total Schools', figsize=(16, 7))


# In[ ]:


# Highest Schools in: Vellore, Villupuram, Tirunelveli
# Lowest Schools in: Perambalur, The Nilgris, Ariyalur


# In[ ]:


print_district_info(enroll_school_df, districts=['Vellore', 'Viluppuram', 'Tirunelveli'], 
                    cols_to_print=['District', 'Grand Total Schools'])


# In[ ]:


print_district_info(enroll_school_df, districts=['Perambalur', 'The Nilgiris', 'Ariyalur'], 
                    cols_to_print=['District', 'Grand Total Schools'])


# In[ ]:


# this information is not that useful because, let's take student school ratio, teacher school ratio


# In[ ]:


enroll_school_df.columns


# In[ ]:


# Calculate the Students vs School Ratio
ratio = enroll_school_df['Govt Total'] / enroll_school_df['Govt Schools']
enroll_school_df['Govt Student School Ratio'] = ratio
enroll_school_gdf['Govt Student School Ratio'] = ratio

ratio = enroll_school_df['Private Aided Total'] / enroll_school_df['Private Aided Schools']
enroll_school_df['Private Aided Student School Ratio'] = ratio
enroll_school_gdf['Private Aided Student School Ratio'] = ratio

ratio = enroll_school_df['Private Unaided Total'] / enroll_school_df['Private Unaided Schools']
enroll_school_df['Private Unaided Student School Ratio'] = ratio
enroll_school_gdf['Private Unaided Student School Ratio'] = ratio

ratio = enroll_school_df['Grand Total'] / enroll_school_df['Grand Total Schools']
enroll_school_df['Grand Student School Ratio'] = ratio
enroll_school_gdf['Grand Student School Ratio'] = ratio


# In[ ]:


# See 'Govt Student School Ratio'
col = 'Govt Student School Ratio'
enroll_school_df[col].describe()


# In[ ]:


draw_bar(enroll_school_df, 'District', y_column=col, title=col, figsize=(16, 7))


# In[ ]:


# Chennai, Kancheepuram, Viluppuram school size is large
# Ramanathapuram, Sivaganga, The nulgris school size is small


# In[ ]:


print_district_info(enroll_school_df, districts=['Chennai', 'Kancheepuram', 'Viluppuram'], 
                    cols_to_print=['District', 'Govt Student School Ratio'])


# In[ ]:


print_district_info(enroll_school_df, districts=['Ramanathapuram', 'Sivaganga', 'The Nilgiris'], 
                    cols_to_print=['District', 'Govt Student School Ratio'])


# In[ ]:


draw_map(enroll_school_gdf, col)


# In[ ]:


# See 'Private Aided Student School Ratio'
col = 'Private Aided Student School Ratio'
enroll_school_df[col].describe()


# In[ ]:


draw_bar(enroll_school_df, 'District', y_column=col, title=col, figsize=(16, 7))


# In[ ]:


# Thoothukkudi, The Nilgris, Tirunelvi school size is smaller
# Krishnagirir, Salem, Chennai school size is higher


# In[ ]:


draw_map(enroll_school_gdf, col)


# In[ ]:


# See 'Private Unaided Student School Ratio'
col = 'Private Unaided Student School Ratio'
enroll_school_df[col].describe()


# In[ ]:


draw_bar(enroll_school_df, 'District', y_column=col, title=col, figsize=(16, 7))


# In[ ]:


# Pudukkotai, Thiruvarur, Ariyalur has least school size
# Chennai, Thiruvallur, Coimbatore has highest school size


# In[ ]:


col = 'Private Unaided Student School Ratio'
print_district_info(enroll_school_df, districts=['Chennai', 'Thiruvallur', 'Coimbatore'], 
                    cols_to_print=['District', col])


# In[ ]:


print_district_info(enroll_school_df, districts=['Pudukkottai', 'Thiruvarur', 'Ariyalur'], 
                    cols_to_print=['District', col])


# In[ ]:


draw_map(enroll_school_gdf, col)


# In[ ]:


# See 'Grand Student School Ratio'
col = 'Grand Student School Ratio'
enroll_school_df[col].describe()


# In[ ]:


draw_bar(enroll_school_df, 'District', y_column=col, title=col, figsize=(16, 7))


# In[ ]:


# Sivaganga, Ramnathapuram, Pudukkotai has small school size in general
# Chennai, Kancheepuram, Thiruvallur has largest school size in general


# In[ ]:


col = 'Grand Student School Ratio'
print_district_info(enroll_school_df, districts=['Ramanathapuram', 'Pudukkottai', 'Sivaganga'], 
                    cols_to_print=['District', col])


# In[ ]:


print_district_info(enroll_school_df, districts=['Chennai', 'Kancheepuram', 'Thiruvallur'], 
                    cols_to_print=['District', col])


# In[ ]:


draw_map(enroll_school_gdf, col)


# In[ ]:


cols = ['Private Unaided Student School Ratio', 'Private Aided Student School Ratio', 'Govt Student School Ratio']
enroll_school_df[cols].describe()


# In[ ]:


# Private unaided schools have largest school student ratio
# govt school have largest school student ratio
# private aided schools have largest standard deviation


# In[ ]:


enroll_school_df.columns


# In[ ]:


draw_line_chart(x='District', df=enroll_school_df, y=cols, figsize=(16, 7))


# In[ ]:





# In[ ]:




