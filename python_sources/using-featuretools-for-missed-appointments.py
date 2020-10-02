#!/usr/bin/env python
# coding: utf-8

# # Using Featuretools to Predict Missed Appointments
# In this notebook, we use [Featuretools](https://github.com/Featuretools/featuretools) to automatically generate features relating to when patients don't show up for doctor appointments. We quickly reconstruct the features that were made by hand in the most popular [kernel](https://www.kaggle.com/somrikbanerjee/predicting-show-up-no-show) and make some other interesting features automatically.

# In[1]:


import numpy as np
import pandas as pd
import featuretools as ft
print('Featuretools version {}'.format(ft.__version__))

# Data Wrangling
# After loading the data with pandas, we /have/ to fix typos in some column names
# but we change others as well to suit personal preference.
data = pd.read_csv("../input/KaggleV2-May-2016.csv", parse_dates=['AppointmentDay', 'ScheduledDay'])
data.index = data['AppointmentID']
data.rename(columns = {'Hipertension': 'hypertension',
                       'Handcap': 'handicap',
                       'PatientId': 'patient_id',
                       'AppointmentID': 'appointment_id',
                       'ScheduledDay': 'scheduled_time',
                       'AppointmentDay': 'appointment_day',
                       'Neighbourhood': 'neighborhood',
                       'No-show': 'no_show'}, inplace = True)
for column in data.columns:
    data.rename(columns = {column: column.lower()}, inplace = True)
data['appointment_day'] = data['appointment_day'] + pd.Timedelta('1d') - pd.Timedelta('1s')

data['no_show'] = data['no_show'].map({'No': False, 'Yes': True})

# Show the size of the data in a print statement
print('{} Appointments, {} Columns'.format(data.shape[0], data.shape[1]))
print('Appointments: {}'.format(data.shape[0]))
print('Schedule times: {}'.format(data.scheduled_time.nunique()))
print('Patients: {}'.format(data.patient_id.nunique()))
print('Neighborhoods: {}'.format(data.neighborhood.nunique()))
pd.options.display.max_columns=100 
pd.options.display.float_format = '{:.2f}'.format


data.head(3)


# This dataset is a single table of appointments with more than sixty thousand unique patients. Each row represents a scheduled appointment and our goal is to predict if the patient actually shows up for that appointment. From that table, we use Featuretools to automatically generate the features below.
# 
# *Note: For convenience, the next cell has all of the code necessary to create the feature matrix. We'll go through the content step-by-step in the next section.*

# In[2]:


import featuretools.variable_types as vtypes
# This is all of the code from the notebook
# No need to run/read this cell if you're running everything else

# List the semantic type for each column
variable_types = {'gender': vtypes.Categorical,
                  'patient_id': vtypes.Categorical,
                  'age': vtypes.Ordinal,
                  'scholarship': vtypes.Boolean,
                  'hypertension': vtypes.Boolean,
                  'diabetes': vtypes.Boolean,
                  'alcoholism': vtypes.Boolean,
                  'handicap': vtypes.Boolean,
                  'no_show': vtypes.Boolean,
                  'sms_received': vtypes.Boolean}

# Use those variable types to make an EntitySet and Entity from that table
es = ft.EntitySet('Appointments')
es = es.entity_from_dataframe(entity_id="appointments",
                              dataframe=data,
                              index='appointment_id',
                              time_index='scheduled_time',
                              secondary_time_index={'appointment_day': ['no_show', 'sms_received']},
                              variable_types=variable_types)

# Add a patients entity with patient-specific variables
es.normalize_entity('appointments', 'patients', 'patient_id',
                    additional_variables=['scholarship',
                                          'hypertension',
                                          'diabetes',
                                          'alcoholism',
                                          'handicap'])

# Make locations, ages and genders
es.normalize_entity('appointments', 'locations', 'neighborhood',
                    make_time_index=False)
es.normalize_entity('appointments', 'ages', 'age',
                    make_time_index=False)
es.normalize_entity('appointments', 'genders', 'gender',
                    make_time_index=False)

# Take the index and the appointment time to use as a cutoff time
cutoff_times = es['appointments'].df[['appointment_id', 'scheduled_time', 'no_show']].sort_values(by='scheduled_time')

# Rename cutoff time columns to avoid confusion
cutoff_times.rename(columns = {'scheduled_time': 'cutoff_time', 
                               'no_show': 'label'},
                    inplace = True)

# Make feature matrix from entityset/cutoff time pair
fm_final, _ = ft.dfs(entityset=es,
                      target_entity='appointments',
                      agg_primitives=['count', 'percent_true'],
                      trans_primitives=['is_weekend', 'weekday', 'day', 'month', 'year'],
                      approximate='3h',
                      max_depth=3,
                      cutoff_time=cutoff_times[20000:],
                      verbose=False)

print('Features: {}, Rows: {}'.format(fm_final.shape[1], fm_final.shape[0]))
fm_final.tail(3)


# This feature matrix has features like `MONTH` and `WEEKDAY` of the scheduled time and also more complicated features like "how often do patients not show up to this location" (`locations.PERCENT_TRUE(appointments.no_show)`). It takes roughly 20 minutes of work to structure any data and make your first feature matrix using Featuretools. We'll walk through the steps now.
# 
# ## Structuring the Data
# We are given a single table of data. Feature engineering requires that we use what we understand about the data to build numeric rows (feature vectors) which we can use as input into machine learning algorithms. The primary benefit of Featuretools is that it does not require you to make those features by hand. Instead, the requirement is that you pass in *what you know* about the data.
# 
# That knowledge is stored in a Featuretools [EntitySet](https://docs.featuretools.com/loading_data/using_entitysets.html). `EntitySets` are a collection of tables with  information about relationships between tables and semantic typing for every column. We're going to show how to
# + pass in information about semantic types of columns,
# + load in a dataframe to an `EntitySet` and
# + tell the `EntitySet` about reasonable new `Entities` to make from that dataframe.

# In[ ]:


# List the semantic type for each column

import featuretools.variable_types as vtypes
variable_types = {'gender': vtypes.Categorical,
                  'patient_id': vtypes.Categorical,
                  'age': vtypes.Ordinal,
                  'scholarship': vtypes.Boolean,
                  'hypertension': vtypes.Boolean,
                  'diabetes': vtypes.Boolean,
                  'alcoholism': vtypes.Boolean,
                  'handicap': vtypes.Boolean,
                  'no_show': vtypes.Boolean,
                  'sms_received': vtypes.Boolean}


# The `variable_types` dictionary is a place to store information about the semantic type of each column. While many types can be detected automatically, some are necessarily tricky. As an example, computers tend to read `age` as numeric. Even though ages are numbers, it's can be useful to think of them as `Categorical` or `Ordinal`. Changing the variable type will change which functions are automatically applied to generate features.
# 
# Next, we make an entity `appointments`:

# In[ ]:


# Make an entity named 'appointments' which stores dataset metadata with the dataframe
es = ft.EntitySet('Appointments')
es = es.entity_from_dataframe(entity_id="appointments",
                              dataframe=data,
                              index='appointment_id',
                              time_index='scheduled_time',
                              secondary_time_index={'appointment_day': ['no_show', 'sms_received']},
                              variable_types=variable_types)
es['appointments']


# We have turned the dataframe into an entity by calling the function `entity_from_dataframe`. Notice that we specified an index, a time index, a secondary time index and the `variable_types` from the last cell as keyword arguments. 
# 
# The time index and secondary time index notate when certain columns are valid for use. By explicitly marking those, we can avoid using data from the future while creating features. Supposing that we don't want to use our label to predict itself, we either need to specify that it is valid for use **after** the other columns or drop it out of the dataframe entirely.
# 
# Finally, we build new entities from our existing one using `normalize_entity`. We take unique values from `patient`, `age`, `neighborhood` and `gender` and make a new `Entity` for each whose rows are the unique values. To do that we only need to specify where we start (`appointments`), the name of the new entity (e.g. `patients`) and what the index should be (e.g. `patient_id`). Having those additional `Entities` and `Relationships` tells the algorithm about reasonable groupings which allows for some neat aggregations.

# In[ ]:


# Make a patients entity with patient-specific variables
es.normalize_entity('appointments', 'patients', 'patient_id',
                    additional_variables=['scholarship',
                                          'hypertension',
                                          'diabetes',
                                          'alcoholism',
                                          'handicap'])

# Make locations, ages and genders
es.normalize_entity('appointments', 'locations', 'neighborhood',
                    make_time_index=False)
es.normalize_entity('appointments', 'ages', 'age',
                    make_time_index=False)
es.normalize_entity('appointments', 'genders', 'gender',
                    make_time_index=False)


# In[ ]:


# Show the patients entity
es['patients'].df.head(2)


# ## Generating Features with Deep Feature Synthesis
# With our data  structued in an `EntitySet`, we can immediately build features across our entity and relationships with Deep Feature Synthesis ([DFS](https://docs.featuretools.com/automated_feature_engineering/afe.html)). As an example, the feature `locations.PERCENT_TRUE(no_show)` will calculate percentage of patients of at this location that haven't shown up in the past.
# 
# This is where the time indices get used. In order to make predictions at the time the appointment is scheduled, we set the `cutoff_time` for each row to be the `scheduled_time`. That means that DFS, while building features, will only use the data that is known as the appointment is made. In particular, because of our secondary time index, it won't use the label to create features.

# In[ ]:


# Take the index and the appointment time to use as a cutoff time
cutoff_times = es['appointments'].df[['appointment_id', 'scheduled_time', 'no_show']].sort_values(by='scheduled_time')

# Rename columns to avoid confusion
cutoff_times.rename(columns = {'scheduled_time': 'cutoff_time', 
                               'no_show': 'label'},
                    inplace = True)


# In[ ]:


# Generate features using the constructed entityset
fm, features = ft.dfs(entityset=es,
                      target_entity='appointments',
                      agg_primitives=['count', 'percent_true'],
                      trans_primitives=['is_weekend', 'weekday', 'day', 'month', 'year'],
                      max_depth=3, 
                      approximate='3h',
                      cutoff_time=cutoff_times[20000:],
                      verbose=True)
fm.tail()


# We have applied and stacked simple functions (called **primitives**) such as `MONTH`, `WEEKDAY` and `PERCENT_TRUE` to build features across all the `Entities` in our `EntitySet`.
# 
# Feel free to fork this kernel and modify the parameters. By doing so, you can get very different feature matrices. Here's a short overview of the keywords used:
# + `target_entity` is the entity for which we're building features. It would be equally easy to make a feature matrix for the `locations` entity
# + `agg_primitives` and `trans_primitives` are lists of which primitives will be used while constructing features. The full list can be found by running `ft.list_primitives()`
# + `max_depth=3` says to stack up to 3 primitives deep.
# + `approximate='3h'` rounds cutoff times into blocks that are 3 hours long for faster computation
# + `cutoff_time` is a dataframe that says when to calculate each row
# + `verbose=True` makes the progress bar
# 
# For more information, see the [documentation](https://docs.featuretools.com/automated_feature_engineering/afe.html) of dfs.
# ## Machine Learning
# We can put the created feature matrix directly into sklearn. Similar to the other kernels, we do not do a good job predicting no-shows. With one unshuffled train test split, our `roc_auc_score` is roughly .5 with similar scores for F1 and K-first. 

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

X = fm.copy().fillna(0)
label = X.pop('label')
X = X.drop(['patient_id', 'neighborhood', 'gender'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.30, shuffle=False)
clf = RandomForestClassifier(n_estimators=150)
clf.fit(X_train, y_train)
probs = clf.predict_proba(X_test)
print('AUC score of {:.2f}'.format(roc_auc_score(y_test, probs[:,1])))


# In[ ]:


feature_imps = [(imp, X.columns[i]) for i, imp in enumerate(clf.feature_importances_)]
feature_imps.sort()
feature_imps.reverse()
print('Random Forest Feature Importances:')
for i, f in enumerate(feature_imps[0:8]):
    print('{}: {} [{:.3f}]'.format(i + 1, f[1], f[0]/feature_imps[0][0]))


# In[ ]:


from bokeh.models import HoverTool
from bokeh.models.sources import ColumnDataSource
from bokeh.plotting import figure
from bokeh.io import show, output_notebook
from bokeh.layouts import gridplot
from sklearn.metrics import precision_score, recall_score, f1_score

def plot_roc_auc(y_test, probs, pos_label=1):
    fpr, tpr, thresholds = roc_curve(y_test, 
                                     probs[:, 1], 
                                     pos_label=pos_label)


    output_notebook()
    p = figure(height=400, width=400)
    p.line(x=fpr, y=tpr)
    p.title.text ='Receiver operating characteristic'
    p.xaxis.axis_label = 'False Positive Rate'
    p.yaxis.axis_label = 'True Positive Rate'

    p.line(x=fpr, y=fpr, color='red', line_dash='dashed')
    return(p)

def plot_f1(y_test, probs, nprecs):
    threshes = [x/1000. for x in range(50, nprecs)]
    precisions = [precision_score(y_test, probs[:,1] > t) for t in threshes]
    recalls = [recall_score(y_test, probs[:,1] > t) for t in threshes]
    fones = [f1_score(y_test, probs[:,1] > t) for t in threshes]
    
    output_notebook()
    p = figure(height=400, width=400)
    p.line(x=threshes, y=precisions, color='green', legend='precision')
    p.line(x=threshes, y=recalls, color='blue', legend='recall')
    p.line(x=threshes, y=fones, color='red', legend='f1')
    p.xaxis.axis_label = 'Threshold'
    p.title.text = 'Precision, Recall, and F1 by Threshold'
    return(p)

def plot_kfirst(ytest, probs, firstk=500):
    A = pd.DataFrame(probs)
    A['y_test'] = y_test.values
    krange = range(firstk)
    firstk = []
    for K in krange:
        a = A[1][:K]
        a = [1 for prob in a]
        b = A['y_test'][:K]
        firstk.append(precision_score(b, a))
    
    output_notebook()
    p = figure(height=400, width=400)
    p.step(x=krange, y=firstk)
    p.xaxis.axis_label = 'Predictions sorted by most likely'
    p.yaxis.axis_label = 'Precision'
    p.title.text = 'K-first'
    p.yaxis[0].formatter.use_scientific = False
    return p

p1 = plot_roc_auc(y_test, probs)
p2 = plot_f1(y_test, probs, 1000)
p3 = plot_kfirst(y_test, probs, 300)


# In[ ]:


show(gridplot([p1, p2, p3], ncols=1))


# # Some Plots
# An interesting workflow with this dataset is to plot generated features to learn about the data. Here, we'll show the number of visits by neighborhood, and the likelihood to show up by neighborhood and age as created by DFS.

# In[ ]:


tmp = fm.groupby('neighborhood').apply(lambda df: df.tail(1))['locations.COUNT(appointments)'].sort_values().reset_index().reset_index()
hover = HoverTool(tooltips=[
    ("Count", "@{locations.COUNT(appointments)}"),
    ("Place", "@neighborhood"),
])
source = ColumnDataSource(tmp)
p4 = figure(width=400, 
           height=400,
           tools=[hover, 'box_zoom', 'reset', 'save'])
p4.scatter('index', 'locations.COUNT(appointments)', alpha=.7, source=source, color='teal')
p4.title.text = 'Appointments by Neighborhood'
p4.xaxis.axis_label = 'Neighborhoods (hover to view)'
p4.yaxis.axis_label = 'Count'

tmp = fm.groupby('neighborhood').apply(lambda df: df.tail(1))[['locations.COUNT(appointments)', 
                                                               'locations.PERCENT_TRUE(appointments.no_show)']].sort_values(
    by='locations.COUNT(appointments)').reset_index().reset_index()
hover = HoverTool(tooltips=[
    ("Prob", "@{locations.PERCENT_TRUE(appointments.no_show)}"),
    ("Place", "@neighborhood"),
])
source = ColumnDataSource(tmp)
p5 = figure(width=400, 
           height=400,
           tools=[hover, 'box_zoom', 'reset', 'save'])
p5.scatter('index', 'locations.PERCENT_TRUE(appointments.no_show)', alpha=.7, source=source, color='maroon')
p5.title.text = 'Probability of no-show by Neighborhood'
p5.xaxis.axis_label = 'Neighborhoods (hover to view)'
p5.yaxis.axis_label = 'Probability of no-show'

tmp = fm.tail(5000).groupby('age').apply(lambda df: df.tail(1))[['ages.COUNT(appointments)']].sort_values(
    by='ages.COUNT(appointments)').reset_index().reset_index()
hover = HoverTool(tooltips=[
    ("Count", "@{ages.COUNT(appointments)}"),
    ("Age", "@age"),
])
source = ColumnDataSource(tmp)
p6 = figure(width=400, 
           height=400,
           tools=[hover, 'box_zoom', 'reset', 'save'])
p6.scatter('age', 'ages.COUNT(appointments)', alpha=.7, source=source, color='magenta')
p6.title.text = 'Appointments by Age'
p6.xaxis.axis_label = 'Age'
p6.yaxis.axis_label = 'Count'

source = ColumnDataSource(X.tail(5000).groupby('age').apply(lambda x: x.tail(1)))

hover = HoverTool(tooltips=[
    ("Prob", "@{ages.PERCENT_TRUE(appointments.no_show)}"),
    ("Age", "@age"),
])

p7 = figure(title="Probability no-show by Age", 
           x_axis_label='Age', 
           y_axis_label='Probability of no-show',
           width=400,
           height=400,
           tools=[hover, 'box_zoom', 'reset', 'save']
)

p7.scatter('age', 'ages.PERCENT_TRUE(appointments.no_show)', 
          alpha=.7, 
          source=source)


# In[ ]:


show(gridplot([p4, p6, p5, p7], ncols=2))

