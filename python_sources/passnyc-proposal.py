#!/usr/bin/env python
# coding: utf-8

# # Overview
# ## Objective
# "[Identify] which schools have students that would benefit from outreach services and lead to a more diverse group of students taking the SHSAT and being accepted into New York City's Specialized High Schools." - Chris Crawford
# 
# 
# "Part of this challenge is to assess the needs of students by using publicly available data to quantify the challenges they face in taking the SHSAT. The best solutions will enable PASSNYC to identify the schools where minority and underserved students stand to gain the most from services like after school programs, test preparation, mentoring, or resources for parents." -Problem Statement
# 
# ## Methodology
# To determine how to increase participation, we want to identify schools with existing high and low participation. Then identify what traits groups of schools with high or low participation have in common in attempt to find causal features. <br> <br>
# Additionally, we want to identify schools with notable increases or decreases in participation over time, and determine other features that trend in a similar fashion in an attempt to establish a casual link. <br> <br>
# However, since granular school based data is only available for one school district, we use The New School Center for NYC Affairs' dataset that quantifies specialized highschool attendence by neighbourhood tabulation area (NTA). We can use this to identify correlates of participation on an NTA level and identify neighbourhoods that need more resources.

# # Exploratory Data Analysis
# Start analysis on the PASSNYC Explorer data set to examine middle and elementary public school characteristics

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import geopandas as gpd
import seaborn as sns
from shapely.geometry import Point


# In[ ]:


schools = pd.read_csv('Data/2016 School Explorer.csv')
schools['School Name'] = schools['School Name'].str.lower()
percentages = [c for c in schools.columns if '%' in c or 'Percent' in c or 'Rate' in c]
for c in percentages:
    schools[c] = schools[c].apply(lambda x: x if pd.isnull(x) else int(x[:-1]))
schools['School Income Estimate'] = schools['School Income Estimate'].apply(lambda x: x if pd.isnull(x) else float(x[1:].replace(',','')))
schools['Community School?'] = schools['Community School?'].apply(lambda x: 1 if x == 'Yes' else 0)
schools['Location'] = schools.apply(lambda x: Point(x['Longitude'], x['Latitude']), axis=1)


# ## Schools

# In[ ]:


geo_schools = gpd.GeoDataFrame(schools).set_geometry('Location')
school_types = geo_schools.groupby('Community School?')
reg_schools = school_types.get_group(0)
com_schools = school_types.get_group(1)
nyc = gpd.read_file('Data/nynta_18b/nynta.shp').set_geometry('geometry').to_crs({'init' :'epsg:4326'})


# In[ ]:


fig, ax = plt.subplots(figsize=(10,10))
nyc.plot(ax=ax, color='white',linewidth=1,edgecolor='black',alpha=.5)
reg_schools.plot(ax=ax, color='blue', markersize=2, label='Regular Schools')
com_schools.plot(ax=ax, color='red', markersize=2, label='Community Schools')
fig.suptitle('Public and Community Elementary and Middle Schools in NYC', size=16)
plt.legend(loc='upper left',prop={'size': 16});


# ## School Demographics

# In[ ]:


xaxis_labels = ['Percent Asian','Percent White','Percent Hispanic', 'Percent Black',
                'Percent Black / Hispanic', 'Percent ELL']
f, axs = plt.subplots(2,3,figsize=(24, 18))
for i in range(1, 7):
    plt.subplot(2, 3, i)
    plt.hist(schools[xaxis_labels[i-1]], bins=30)
    plt.xlabel(xaxis_labels[i-1],size=14)
    plt.ylabel('Number of Schools');


# In[ ]:


def point_to_nta(pt, ntas):
    for ix, row in ntas.iterrows():
        if pt.within(row['geometry']):
            return row['NTACode']
schools['nta'] = schools.apply(lambda x: point_to_nta(Point(x['Longitude'], x['Latitude']), nyc), axis=1)
nta_school_demogs = schools.groupby('nta').mean()[xaxis_labels]
nyc = nyc.set_index('NTACode')
nta_school_demogs['geometry'] = nyc['geometry']
nta_school = gpd.GeoDataFrame(nta_school_demogs).set_geometry('geometry')


# In[ ]:


schools[xaxis_labels].describe()


# In[ ]:


fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2, figsize=(24, 20))
fig.subplots_adjust(hspace=.05, wspace=.05)
ax1 = nta_school.plot(ax=ax1,cmap='gist_gray_r', linewidth=2,edgecolor='black')
nta_school.plot(ax=ax1, column= 'Percent Black', cmap='Reds', vmin=0, vmax=100)
ax1.set_axis_off()
ax1.set_title('Percent Black',size=16)
ax1 = nta_school.plot(ax=ax2,cmap='gist_gray_r', linewidth=2,edgecolor='black')
nta_school.plot(ax=ax2, column= 'Percent Hispanic', cmap='Reds', vmin=0, vmax=100)
ax1.set_title('Percent Hispanic',size=16)
ax2.set_axis_off()
ax1 = nta_school.plot(ax=ax3,cmap='gist_gray_r', linewidth=2,edgecolor='black')
nta_school.plot(ax=ax3, column= 'Percent Black / Hispanic', cmap='Reds', vmin=0, vmax=100)
ax1.set_title('Percent Black/Hispanic',size=16)
ax3.set_axis_off()
ax1 = nta_school.plot(ax=ax4,cmap='gist_gray_r', linewidth=2,edgecolor='black')
nta_school.plot(ax=ax4, column= 'Percent ELL', cmap='Reds', vmin=0, vmax=100)
ax1.set_title('Percent ELL',size=16)
ax4.set_axis_off()
ax1 = nta_school.plot(ax=ax5,cmap='gist_gray_r', linewidth=2,edgecolor='black')
nta_school.plot(ax=ax5, column= 'Percent White', cmap='Reds', vmin=0, vmax=100)
ax1.set_title('Percent White',size=16)
ax5.set_axis_off()
ax1 = nta_school.plot(ax=ax6,cmap='gist_gray_r', linewidth=2,edgecolor='black')
nta_school.plot(ax=ax6, column= 'Percent Asian', cmap='Reds', vmin=0, vmax=100)
ax1.set_title('Percent Asian',size=16)
ax6.set_axis_off()
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=0, vmax=100))
sm._A = []
fig.colorbar(sm, cax=cbar_ax);


# In[ ]:





# ## Socio-Economics

# In[ ]:


econ_labels = ['Economic Need Index', 'School Income Estimate']
nta_school_econs = schools.groupby('nta').mean()[econ_labels]
nta_school_econs['geometry'] = nyc['geometry']
nta_school_econ = gpd.GeoDataFrame(nta_school_econs).set_geometry('geometry')


# In[ ]:


f, axs = plt.subplots(1,2,figsize=(24, 18))
for i in range(1, 3):
    plt.subplot(1, 2, i)
    plt.hist(schools[econ_labels[i-1]].dropna(), bins=30)
    plt.xlabel(econ_labels[i-1],size=14)
    plt.ylabel('Number of Schools');


# In[ ]:


schools[econ_labels].describe()


# In[ ]:


fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(24, 16))
fig.subplots_adjust(hspace=.05, wspace=.05)
ax1 = nta_school_econ.plot(ax=ax1,cmap='gist_gray_r', linewidth=2,edgecolor='black')
nta_school_econ.plot(ax=ax1, column= 'Economic Need Index', cmap='Reds', vmin=0., vmax=1.)
ax1.set_axis_off()
ax1.set_title('Economic Need Index',size=16)
sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=0, vmax=1.))
sm._A = []
fig.colorbar(sm, ax=ax1, orientation="horizontal", pad=0.05)
ax2 = nta_school_econ.plot(ax=ax2,cmap='gist_gray_r', linewidth=2,edgecolor='black')
nta_school_econ.plot(ax=ax2, column= 'School Income Estimate', cmap='Greens', vmin=0, vmax=100000)
ax2.set_title('School Income Estimate',size=16)
ax2.set_axis_off()
sm = plt.cm.ScalarMappable(cmap='Greens', norm=plt.Normalize(vmin=0, vmax=100000))
sm._A = []
fig.colorbar(sm, ax=ax2, orientation="horizontal", pad=0.05);


# Notice how similar the Economic Need index looks to the combinded Black/Hispanic graph

# ## School Performance

# In[ ]:


proficiency = ['Average ELA Proficiency', 'Average Math Proficiency']
nta_school_perf = schools.groupby('nta').mean()[proficiency]
nta_school_perf['geometry'] = nyc['geometry']
nta_school_performance = gpd.GeoDataFrame(nta_school_perf).set_geometry('geometry')


# In[ ]:


f, axs = plt.subplots(1,2,figsize=(24, 18))
for i in range(1, 3):
    plt.subplot(1, 2, i)
    plt.hist(schools[proficiency[i-1]].dropna(), bins=30)
    plt.xlabel(proficiency[i-1],size=14)
    plt.ylabel('Number of Schools');


# In[ ]:


schools[proficiency].describe()


# In[ ]:


fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(24, 16))
fig.subplots_adjust(hspace=.05, wspace=.05)
ax1 = nta_school_performance.plot(ax=ax1,cmap='gist_gray_r', linewidth=2,edgecolor='black')
nta_school_performance.plot(ax=ax1, column= 'Average ELA Proficiency', cmap='Blues', vmin=1.5, vmax=4)
ax1.set_axis_off()
ax1.set_title('Average ELA Proficiency',size=16)
sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=1.5, vmax=4))
sm._A = []
fig.colorbar(sm, ax=ax1, orientation="horizontal", pad=0.05)
ax2 = nta_school_performance.plot(ax=ax2,cmap='gist_gray_r', linewidth=2,edgecolor='black')
nta_school_performance.plot(ax=ax2, column= 'Average Math Proficiency', cmap='Blues', vmin=1.5, vmax=4)
ax2.set_title('Average Math Proficiency',size=16)
ax2.set_axis_off()
sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=1.5, vmax=4))
sm._A = []
fig.colorbar(sm, ax=ax2, orientation="horizontal", pad=0.05);


# ## Correlates

# In[ ]:


def plot_corr(df,size=10):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    cax = ax.matshow(corr, cmap='RdYlGn', vmin=-1, vmax =1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90);
    plt.yticks(range(len(corr.columns)), corr.columns);
    fig.colorbar(cax);


# In[ ]:


other_metrics = ['Student Attendance Rate', 'Percent of Students Chronically Absent',
                 'Rigorous Instruction %','Collaborative Teachers %','Supportive Environment %',
                 'Effective School Leadership %','Strong Family-Community Ties %','Trust %',
                 'Community School?']
correlates = other_metrics + proficiency + econ_labels + xaxis_labels
plot_corr(schools[correlates])


# # SHSAT EDA

# In[ ]:


shsat = pd.read_csv('Data/SHSAT_reg.csv')
shsat['School name'] = shsat['School name'].str.lower()
shsat['take_reg_ratio'] = shsat.apply(lambda x: 
                                      x['Number of students who took the SHSAT']/
                                      (x['Number of students who registered for the SHSAT']+.000001),axis=1)
shsat['take_ratio'] = shsat.apply(lambda x: 
                                   x['Number of students who took the SHSAT']/
                                   float(x['Enrollment on 10/31']), axis=1)
shsat['reg_ratio'] = shsat.apply(lambda x:
                                x['Number of students who registered for the SHSAT']/
                                   float(x['Enrollment on 10/31']), axis=1)
shsat['charter'] = shsat['DBN'].apply(lambda x: 'charter' if x[:2] == '84' else 'public')
ratios = ['take_reg_ratio','take_ratio','reg_ratio']
shsat[ratios].corr()


# The two things to note here are: <br>
# 1.) The strong positive relationship between the ratio of students that register and the ratio of students that take the test <br>
# 2.) The near statistically insignifcant relationship between the ratio of students that register and the ratio of students that take the test of those that register <br>
# Lets take closer look at these relationships

# In[ ]:


color_map = {'charter':'red', 'public':'blue'}
fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(10,5))
plt.subplot(1, 2, 1)
school_types = shsat.groupby('charter')
for name, group in school_types:
    plt.plot(group['reg_ratio'], group['take_ratio'], marker='o',
             linestyle='', label=name, color=color_map[name])
plt.xlabel('Ratio of students Registered')
plt.ylabel('Ratio of students that take test')
plt.legend()
plt.subplot(1, 2, 2)
for name, group in school_types:
    plt.plot(group['reg_ratio'], group['take_reg_ratio'], marker='o', 
             linestyle='', label=name, color=color_map[name])
plt.xlabel('Ratio of students Registered')
plt.ylabel('Ratio of students that take test of those registered');


# This includes every data point, one for every school for every year. Let us average these schools over the years data is available to get a more concise view of performance.

# In[ ]:


shsat_schools = shsat.groupby('DBN').mean()
shsat_schools['charter'] = shsat_schools.apply(lambda x: 'charter' if x.name[:2] == '84' else 'public',axis=1)
color_map = {'charter':'red', 'public':'blue'}
fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(10,5))
plt.subplot(1, 2, 1)
#plt.scatter(shsat['reg_ratio'], shsat['take_ratio'])
school_types = shsat_schools.groupby('charter')
for name, group in school_types:
    plt.plot(group['reg_ratio'], group['take_ratio'], marker='o',
             linestyle='', label=name, color=color_map[name])
plt.xlabel('Ratio of students Registered')
plt.ylabel('Ratio of students that take test')
plt.legend()
plt.subplot(1, 2, 2)
for name, group in school_types:
    plt.plot(group['reg_ratio'], group['take_reg_ratio'], marker='o', 
             linestyle='', label=name, color=color_map[name])
plt.xlabel('Ratio of students Registered')
plt.ylabel('Ratio of students that take test of those registered');


# Charters appear to perform marginally better than regular public schools in this district. 
# 
# Lets identify the high/low performing schools and the schools with strong up or down trends in participation.

# In[ ]:


shsat_schools = shsat.groupby('DBN')
avg_particip = shsat_schools.apply(lambda x: sum(x['take_ratio'])) /shsat_schools.size()
most_partic = avg_particip[np.argsort(avg_particip.values)[-3:]][::-1]
least_partic = avg_particip[np.argsort(avg_particip.values)[:3]]


# In[ ]:


most_partic


# In[ ]:


least_partic


# In[ ]:


shsat.corr()
from scipy.stats import linregress
shsat['years_from_start'] = shsat.apply(lambda x: x['Year of SHST'] - 2013, axis=1)
shsat['take%'] = shsat.apply(lambda x: x['take_ratio']*100, axis=1)
take_trend = shsat_schools.apply(lambda x: linregress(list(x['take%']),list(x['years_from_start'])).slope)
#Indexing to get rid of outliers or schools with only one year
trending_up = take_trend[np.argsort(take_trend.values)[-5:-2]][::-1]
trending_down = take_trend[np.argsort(take_trend.values)[2:5]]


# In[ ]:


trending_up


# In[ ]:


trending_down


# In[ ]:


shsat_stats = shsat_schools.mean().drop(columns=['Year of SHST', 'Grade level',
                                   'years_from_start', 'take%'])
shsat_stats['trend'] = take_trend.fillna(0)
shsat_stats['trend'] = shsat_stats['trend'].apply(lambda x: x if (x>-.5) and (x<.5) else 0)


# In[ ]:


dbn_to_name = {}
for x in shsat['DBN'].unique():
    for index, row in shsat.iterrows():
        if row['DBN'] == x:
            dbn_to_name[x] = row['School name']
            break
shsat_stats['name'] = shsat_stats.apply(lambda x: dbn_to_name[x.name], axis=1)


# In[ ]:


shsat_stats


# In[ ]:


shsat_school_stats = shsat_stats.merge(schools, left_on='DBN', right_on='Location Code')


# In[ ]:


pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 100)
shsat_school_stats


# In[ ]:


nat_am = [x for x in shsat_school_stats.columns if 'Indian' in x]
corrs = shsat_school_stats.drop(columns=['SED Code','District','Latitude','Longitude','Zip'] + nat_am)
plot_corr(corrs.fillna(0), size=25)


# In[ ]:


from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
def create_hyperparam_grid():
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 250, stop = 2250, num = 250)]
    # Number of features to consider at every split
    max_features = [.1,.2,.3,.4,.5,.6,.7,.8,.9,.95]
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 8, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 3, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    return random_grid


# In[ ]:


def rf_classifier(labels, features, pct_train=.8, num_iter=50):
    rfclf = RandomForestClassifier()
    random_grid = create_hyperparam_grid()
    rf_random = RandomizedSearchCV(estimator = rfclf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    rf_random.fit(features, labels)
    best_params = rf_random.best_params_ 
    rfclf = RandomForestClassifier(**best_params)
    i = 0
    cumu_acc = 0
    while(i < num_iter):
        msk = np.random.rand(len(labels)) < pct_train
        train_data = features[msk]
        train_labels = labels[msk]
        test_data = features[~msk]
        test_labels = labels[~msk]
        rfclf.fit(train_data, train_labels) 
        cumu_acc += rfclf.score(test_data, test_labels)
        i += 1
    print cumu_acc/num_iter
    return rfclf


# In[ ]:


take_labels = (shsat_school_stats['take_ratio'] > .135).astype(np.int32)
reg_labels = (shsat_school_stats['reg_ratio'] > .28).astype(np.int32)
trend_labels = (shsat_school_stats['trend'] > 0).astype(np.int32)
labels = ['take_ratio', 'reg_ratio','trend']
non_numerics = ['Adjusted Grade','New?','Other Location Code in LCGMS',
                'School Name','Location Code','Address (Full)','City',
                'Grades','Grade High', 'Location', 'nta', 'name', 'Grade Low']
ratings = [c for c in corrs.columns if 'Rating' in c]
cheats = ['Enrollment on 10/31', 'Number of students who registered for the SHSAT',
         'Number of students who took the SHSAT']
features = corrs.drop(columns=labels+non_numerics+ratings+cheats)
features.fillna((features['School Income Estimate'].mean()), inplace=True)


# In[ ]:


take_clf = rf_classifier(take_labels, features,pct_train=.6, num_iter=20)


# In[ ]:


important_features = np.argsort(-take_clf.feature_importances_)[:10]
features.columns[important_features]


# In[ ]:


reg_clf = rf_classifier(reg_labels, features,pct_train=.6, num_iter=20)
important_reg_features = np.argsort(-reg_clf.feature_importances_)[:10]
features.columns[important_reg_features]


# In[ ]:


trend_clf = rf_classifier(trend_labels, features,pct_train=.6, num_iter=20)
important_trend_features = np.argsort(-trend_clf.feature_importances_)[:10]
features.columns[important_trend_features]


# The low accuracy indicates a lack of predictive power in the data given (or a methodological short coming) but I believe this has to do with the incredibly small size of available data. The little bit of signal that is evident shows that performance and participation in the middle school ELA and Math exams are indicators of SHSAT registration and participation. Notably, school and community factors (the scores) didn't seem to carry any predictive power. 

# Due to the lack of granular SHSAT data, it is perhaps more appropriate to identify areas of a nieghborhood granularity that could benefit from additional resources and then drill down to find the schools most in need. Fortunately we have access to The New School Center for NYC Affairs dataset for enrollment in specialized schools by neigborhood tabulation area (NTA) which is a proxy for SHSAT participation and performance. https://nimader.carto.com/datasets

# In[ ]:


special_hs = {
    'Stuyvesant High School': Point(-74.013762, 40.718222),
    'The Bronx High School of Science': Point(-73.890843, 40.878435),
    'Brooklyn Tech HS': Point(-73.976526, 40.688987),
    'James Madison HS': Point(-73.948227, 40.610011),
    'Hillcrest High School': Point(-73.802482, 40.709588),
    'Long Island City High School': Point(-73.933048, 40.765480),
    'Staten Island Technical High School': Point(-74.118124, 40.567703)
}
spechs = gpd.GeoSeries(special_hs)
spec_hs_distr = gpd.read_file('Data/special_hs/spec_hs_distr.shp')


# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(14,10)
spec_hs_distr.plot(ax=ax, column='speciali_1',cmap='Reds')
spechs.plot(ax=ax, color='blue')
#stations.plot(ax=ax, color='green', markersize=1)
sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=0, vmax=32))
sm._A = []
plt.colorbar(sm)
ax.legend(['Specialized High School'], loc='upper left')
plt.suptitle('Percent of NTA enrolled in specialized HS', size=18);


# In[ ]:


from geopy.distance import great_circle
spechs_lat_lon = [(x.y, x.x) for x in special_hs.values()]
def closest_spec_hs(pt, specs):
    return min([great_circle((pt.y, pt.x), hs).meters for hs in specs])
special_hs_map = {
    'M': Point(-74.013762, 40.718222),
    'X': Point(-73.890843, 40.878435),
    'B1': Point(-73.976526, 40.688987),
    'B2': Point(-73.948227, 40.610011),
    'Q1': Point(-73.802482, 40.709588),
    'Q2': Point(-73.933048, 40.765480),
    'R': Point(-74.118124, 40.567703)
}

def dist_to_test(row, spec_hs_locs, spec_hs_map):
    district = row['District']
    bor = row['Location Code'][2]
    pt = None
    if bor == 'M' or bor == 'X' or bor == 'R':
        pt = special_hs_map[bor]
    elif district in [13, 14, 15, 16, 19, 20, 32]:
        pt = special_hs_map['B1']
    elif district in [17, 18, 21, 22, 23]:
        pt = special_hs_map['B2']
    elif district in [26, 27, 28, 29]:
        pt = special_hs_map['Q1']
    elif district in [24, 25, 30]:
        pt = special_hs_map['Q2']
    return great_circle()
schools['dist_to_spec'] = schools['Location'].apply(lambda x: closest_spec_hs(x, spechs_lat_lon))


# In[ ]:


schools


# In[ ]:


district_map = {}


# In[ ]:





# In[ ]:





# # Recommendations

# # Conclusions

# In[ ]:





# In[ ]:




