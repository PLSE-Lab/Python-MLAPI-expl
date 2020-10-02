#!/usr/bin/env python
# coding: utf-8

# # Graph based Decision Making and Causal Inference
# ## This is kernel 2 for PASSNYC
# This kernel is kernel 2 for PASSNYC analysis and uses kernel 1's results so please go through kernel 1 for better understanding how the labeling is assigned using label propagation. Link - **https://www.kaggle.com/maheshdadhich/passnyc-label-propagation-algorithm**
# 
# In this kernel, we will talk about a supervised learning model to make a complete workflow process to behind any school being assigned as underperforming. This model is a tree-based model and can also be used when the data for all other districts (SHSAT registration data) is available. The other reason behind using this model is to get the proxy measures. The proxy measures are necessary to give an explanation about under-performing schools, they are necessary to convince stakeholders and partners to agree with the model's results. At last, this will try to draw a causal inference from the available data using the Bayesian network. This network is a deep learning model and with more and more data, its accuracy will increase but even with less amount of data, data preparation is done is the intelligible fashion to get it to work and give us the driving factors behind multiple proxy-measures. 
# 
# ### This kernel will have three parts - 
# -  **Decision tree to get the decision workflow to assign a school as underperforming** - This can be used to explain stakeholders and partners the results we are getting from kernel 1. This should be used if partners or stakeholder wants to know the way decisions are taken. 
# -  **Proxy measures** - It analyses the Label propagation results using a decision tree model and gives us the new and revised proxy measures.
# - **Driving factor analysis using Bayesian networks** - We create a discrete Bayesian network to get the underlying driving factors for different proxy measures.

# In[ ]:


# Loading the packages 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly 
import time
import os 
import folium 
import scipy.stats
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from scipy.spatial import distance
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff
from sklearn import preprocessing
from plotly import tools
import plotly.plotly as py
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import sklearn.semi_supervised 
from sklearn import datasets
from sklearn.semi_supervised import LabelPropagation
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


# Loading files from last pre-processing from kernel 1
path_to_save_dfs = '../input/passnyc-label-propagation-algorithm/'
bayes_df = pd.read_csv(path_to_save_dfs + "Bayes_df.csv")
SAT_summary = pd.read_csv(path_to_save_dfs + "SAT_summary.csv")
class_size = pd.read_csv(path_to_save_dfs + "Class_size.csv")
safety_df = pd.read_csv(path_to_save_dfs + "crime_df.csv")
label_df = pd.read_csv(path_to_save_dfs + "Labels.csv")
print("Files Loading done!!")


# ## Data preparation

# In[ ]:


# 1. Function for creating bins for bayesian network for causal inference 
def binning(col, labels=None):
    """Function to define bins for bayesian network analysis"""
    #Define min and max values
    minval = col.min()
    maxval = col.max()

    #create list by adding min and max to cut_points
    delta = (maxval - minval)/5.0
    cut_points = []
    for j in [1,2,3,4]:
        cut_points.append((minval+j*delta))
    # print(cut_points)   
    break_points =  [minval]+ cut_points + [maxval]
    # print(break_points)
    len_ = list(set(list(break_points))).__len__()

        
    #print(break_points)
    #if no labels provided, use default labels 0 ... (n-1)
    if not labels:
        labels = list(range(len(cut_points)+1))

    #Binning using cut function of pandas
    colBin = pd.cut(col,bins=break_points,labels=labels,include_lowest=True)
    return(colBin)

def to_upper(row):
    """Function to convert a string to upper case """
    return(row.upper())

# 2. prepare data for bayesian network 
def prepare_bayesian_network_input(new_df, cols_for_binning):
    """Function to prepare the input for bayesian network"""
    df = new_df.copy()
    new_cols = []
    for col in cols_for_binning:
        # print(cols_for_binning)
        new_cols.append("bin_"+col)
        df["bin_"+col] = binning(df[col])
    return(df[new_cols])


def Assign_k_means_cluster(df, n):
    """function to assign k-means clusters 
        n = number of clusters
    """
    columns = ['Latitude','Longitude']
    df_new = df[columns]
    kmeans = MiniBatchKMeans(n_clusters=n).fit(df_new)
    df.loc[:, 'location_cluster'] = kmeans.predict(df[columns])
    #test_meta.loc[:, 'k_means_cluster'] = kmeans.predict(test_meta[columns])
    return(df)

# bayes_df = Assign_k_means_cluster(bayes_df, 10)


# In[ ]:


# Merging data for making master dataframe 
bayes_df.head()
print("Shape of bayes df is {}".format(bayes_df.shape[0]))


# Merging SAT score and bayes data 
SAT_summary['School name'] = list(map(to_upper, SAT_summary['School name']))
master_df = bayes_df.merge(SAT_summary, left_on = 'Location Code', right_on = 'DBN', how = 'left')
print("Shape of master df is {}".format(master_df.shape[0]))



# Crime data pre-processing before merging to get district wise crime data
aggregate_crime_stats = {'Major N':np.mean,
                         'Oth N':np.mean,
                         'NoCrim N':np.mean,
                         'Prop N':np.mean,
                         'Vio N':np.mean}
crime_summary = pd.DataFrame(safety_df.groupby(['Geographical District Code']).agg(aggregate_crime_stats))
crime_summary.reset_index(inplace = True)
crime_summary['Geographical District Code'] = list(map(int, crime_summary['Geographical District Code']))
print("The shape of locatio wise crime data is {}".format(crime_summary.shape))
crime_summary.head()



# Merging class df for class size 
master_df = master_df.merge(crime_summary, left_on = 'District', right_on = 'Geographical District Code', how = 'left')
print("Shape of master_df is {}".format(master_df.shape[0]))
master_df.head()

A = master_df.copy()
# Merging class_size data somehow - Later -if time permits 
# Add perdicted_underperforming by label_df
label = label_df[['School Name','Predicted_underperforming']]
label.drop_duplicates(subset = ['School Name'],keep = 'first',inplace = True)
master_df = master_df.merge(label, left_on = 'School Name', right_on = 'School Name', how = "left")
master_df.head()
print(master_df.shape)
master_df.head()


# In[ ]:


# Creating economic disadvantage features 
master_df.head()
# Grade 8 Math 4s - Limited English Proficient
# Grade 8 Math 4s - Economically Disadvantaged
# Grade 8 Math - All Students Tested

created_perc_col = []
grades_name = [6,7,8]
for grade in grades_name:
    for subj in ['ELA', 'Math']:
        for suffix in [' 4s - Limited English Proficient', ' 4s - Economically Disadvantaged']:
            col_name = subj+'_'+str(grade)+suffix
            created_perc_col.append(col_name)
            All_4s = 'Grade '+str(grade)+' '+subj+suffix
            All = 'Grade '+str(grade)+' '+subj+' - All Students Tested'
            master_df.loc[:,col_name] = (master_df[All_4s].values*100.0)/master_df[All].values
created_perc_col


# ## 1. Decision trees and providing decision workflow

# In[ ]:


# Cols for decision tree model 
cols_for_binning = ['Economic Need Index',
                    'Percent Black / Hispanic', 
                    'Student Attendance Rate',
                    'Percent of Students Chronically Absent',
                    'Rigorous Instruction Rating',
                    'Collaborative Teachers Rating',
                    'Supportive Environment Rating',
                    'Effective School Leadership Rating',
                    'Trust Rating',
                    'Student Achievement Rating',
                    'Average Math Proficiency',
                    'Average ELA Proficiency',
                    'location_cluster',
                    'ELA_6_4s',
                    'Math_6_4s',
                    'ELA_7_4s',
                    'Math_7_4s',
                    'ELA_8_4s',
                    'Math_8_4s', 
                    'Oth N',
                    'NoCrim N',
                    'Prop N',
                    'Vio N',
                    'Major N']
                    #'Predicted_underperforming'
                   
cols_for_binning_crime = ['Major N',
                          #'Predicted_underperforming',
                          #'location_cluster',
                          'Average Math Proficiency',
                          'Average ELA Proficiency'
                         ]

# renaming columns for better visualization 
rename_dict = {'bin_Economic Need Index':'ENI',
               'bin_Percent Black / Hispanic':'P_black',
               'bin_Student Attendance Rate':'Atd',
               'bin_Percent of Students Chronically Absent':'chr_abt',
               'bin_Rigorous Instruction Rating':'Rig_inst',
               'bin_Collaborative Teachers Rating':'coll_teach',
               'bin_Supportive Environment Rating':'SUpp_env',
               'bin_Effective School Leadership Rating':'Leadership',
               'bin_Trust Rating':'Trust',
               'bin_Student Achievement Rating':"Achmt",
               'bin_Average Math Proficiency':'Math',
               'bin_Average ELA Proficiency':'ELA',
               'bin_location_cluster':'Loc',
               'bin_ELA_6 4s - Limited English Proficient':'E6_LEP',
               'bin_ELA_6 4s - Economically Disadvantaged':'E6_disadv',
               'bin_Math_6 4s - Limited English Proficient':'M6_LEP',
               'bin_Math_6 4s - Economically Disadvantaged':'M6_disadv',
               'bin_ELA_7 4s - Limited English Proficient':'E7_LEP',
               'bin_ELA_7 4s - Economically Disadvantaged':'E7_disadv',
               'bin_Math_7 4s - Limited English Proficient':'M7_LEP',
               'bin_Math_7 4s - Economically Disadvantaged':'M7_disadv',
               'bin_ELA_8 4s - Limited English Proficient':'E8_LEP',
               'bin_ELA_8 4s - Economically Disadvantaged':'E8_disadv',
               'bin_Math_8 4s - Limited English Proficient':'M8_LEP',
               'bin_Math_8 4s - Economically Disadvantaged':'M8_disadv',
               'bin_ELA_7_4s':'E7_4s',
               'bin_Math_7_4s':'M7_4s',
               'bin_ELA_8_4s':'E8_4s',
               'bin_Math_8_4s':'M8_4s',
               'Predicted_underperforming':'Low_perf'}

created_cols = ['ELA_8 4s - Limited English Proficient',
                 'ELA_8 4s - Economically Disadvantaged',
                 'Math_8 4s - Limited English Proficient',
                 'Math_8 4s - Economically Disadvantaged']  

class_columns = ['ELA_6_4s',
                 'Math_6_4s',
                 'ELA_7_4s',
                 'Math_7_4s',
                 'ELA_8_4s',
                 'Math_8_4s']


# In[ ]:


## Decision tree workflow for an easy executive level explanation
## But it can also be used as primary model as PASSNYC gets more data
master_df.head()
master_df['community_school']  = np.where(master_df['Community School?']=='No', 0,1)
cols_rf = list(set(['community_school'] + cols_for_binning + created_cols +class_columns))
y = master_df.Predicted_underperforming.values
     
    
train = master_df[cols_rf].copy()
rf = RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=4, max_features=0.2, n_jobs=-1, random_state=0)
rf.fit(train, y)
features = train.columns.values
print("----- Training Done -----")


# In[ ]:


from sklearn import tree
from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont
import re
decision_tree = tree.DecisionTreeClassifier(max_depth = 8,splitter = "random", random_state=0)
decision_tree.fit(train, y)

# Export our trained model as a .dot fil


with open("tree1.dot", 'w') as f:
     f = tree.export_graphviz(decision_tree,
                              out_file=f,
                              max_depth = 8,
                              impurity = False,
                              feature_names = train.columns.values,
                              class_names = ['No', 'Yes'],
                              rounded = True,
                              filled= True)


# In[ ]:


import pydot
import graphviz
import sys
from subprocess import check_call

from subprocess import check_call
check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])

#! dot -Tpng tree1.dot -o tree1.png
# Annotating chart with PIL
img = Image.open("tree1.png")
draw = ImageDraw.Draw(img)
img.save('sample-out.png')
PImage("sample-out.png")


# ### Why decision tree workflow ? To explain the results business people and partner organizations
# **Whenever one has to explain a difficult statistical model to stakeholders, one can use decision trees. Decision trees basically unfolds the complete process of decision making and series of decision to reach to the conclusion. Here we will have to focus on "yes" class, as it means yes that particular school is underperforming.** This is one of the simplest methods and most convinient methods when it comes to explain the statistical model to partners (or even to layman population) 

# ## 2. Revised Proxy Measures 

# In[ ]:


# Scatter plot 
trace = go.Scatter(
    y = rf.feature_importances_,
    x = features,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 13,
        #size= rf.feature_importances_,
        #color = np.random.randn(500), #set color equal to a variable
        color = rf.feature_importances_,
        colorscale='Portland',
        showscale=True
    ),
    text = features
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Proxy measure to determine underperforming school',
    hovermode= 'closest',
     xaxis= dict(
         ticklen= 5,
         showgrid=False,
        zeroline=False,
        showline=False
     ),
    yaxis=dict(
        title= 'Feature Importance',
        showgrid=False,
        zeroline=False,
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
iplot(fig,filename='scatter2010')


# #### List of revised proxy measures which can be used to explain it to stckholders and partners - 
# This list is in decreasing order of importance (first one being highest important and last one being of lowest importance) -
# - **Math_7_4s** ( Math 7 score can be considered as a base for Math_8 and further SAT registration)
# - **Math 8_4s**
# - **ELA_6_4s**
# - **Math_6_4s**
# - **ELA_8_4s**
# - **Avg ELA proficiency**
# - **ELA_7_4s**
# - **Avg Math Proficiency**
# - **% of black/hispanic people**
# - **Economic need index**
# - **ELA_8_4s ecomonically disadvantage**
# - **Student attendance rate**

# ## 3. Analyzing driving factors using Bayesian Network
# Assumptions - 
# - Bayesian network being a deep learning model, its hard to fit it so we are using at max 5 levels for each variables
# - Sometimes the results look opposite in Hill climb algorithm (used here), so sujective decision should be made in such cases.****
# - A helper kernel is made for easy implementation of bayesian network. Results are taken from that helper kernel only. Link - https://www.kaggle.com/maheshdadhich/helper-notebook-for-bayesian-network
# - This helper kernel is written with keeping in mind that if PASSNYC has to run a causal inference in future they can directly use this and run it, Another reason is to not make technicalities if bayesian network a part of this analysis, so it's kept seperate.

# In[ ]:


cols_for_binning = ['Economic Need Index',
                    #'Percent Black / Hispanic', 
                    'Student Attendance Rate',
                    'Percent of Students Chronically Absent',
                    #'Rigorous Instruction Rating',
                    #'Collaborative Teachers Rating',
                    #'Supportive Environment Rating',
                    #'Effective School Leadership Rating',
                    #'Trust Rating',
                    'Student Achievement Rating',
                    'Average Math Proficiency',
                    'Average ELA Proficiency',
                    #'location_cluster',
                    #'ELA_6_4s',
                    #'Math_6_4s',
                    #'ELA_7_4s',
                    #'Math_7_4s',
                    'ELA_8_4s',
                    'Math_8_4s', 
                    #'Oth N',
                    #'NoCrim N',
                    #'Prop N',
                    #'Vio N',
                    #'Major N']
                    #'Predicted_underperforming'] 
                   ]
cols_for_binning_crime = ['Major N',
                          #'Predicted_underperforming',
                          #'location_cluster',
                          'Average Math Proficiency',
                          'Average ELA Proficiency'
                         ]

# renaming columns for better visualization 
rename_dict = {'bin_Economic Need Index':'ENI',
               'bin_Percent Black / Hispanic':'P_black',
               'bin_Student Attendance Rate':'Atd',
               'bin_Percent of Students Chronically Absent':'chr_abt',
               'bin_Rigorous Instruction Rating':'Rig_inst',
               'bin_Collaborative Teachers Rating':'coll_teach',
               'bin_Supportive Environment Rating':'SUpp_env',
               'bin_Effective School Leadership Rating':'Leadership',
               'bin_Trust Rating':'Trust',
               'bin_Student Achievement Rating':"Achmt",
               'bin_Average Math Proficiency':'Math',
               'bin_Average ELA Proficiency':'ELA',
               'bin_location_cluster':'Loc',
               'bin_ELA_6 4s - Limited English Proficient':'E6_LEP',
               'bin_ELA_6 4s - Economically Disadvantaged':'E6_disadv',
               'bin_Math_6 4s - Limited English Proficient':'M6_LEP',
               'bin_Math_6 4s - Economically Disadvantaged':'M6_disadv',
               'bin_ELA_7 4s - Limited English Proficient':'E7_LEP',
               'bin_ELA_7 4s - Economically Disadvantaged':'E7_disadv',
               'bin_Math_7 4s - Limited English Proficient':'M7_LEP',
               'bin_Math_7 4s - Economically Disadvantaged':'M7_disadv',
               'bin_ELA_8 4s - Limited English Proficient':'E8_LEP',
               'bin_ELA_8 4s - Economically Disadvantaged':'E8_disadv',
               'bin_Math_8 4s - Limited English Proficient':'M8_LEP',
               'bin_Math_8 4s - Economically Disadvantaged':'M8_disadv',
               'bin_ELA_7_4s':'E7_4s',
               'bin_Math_7_4s':'M7_4s',
               'bin_ELA_8_4s':'E8_4s',
               'bin_Math_8_4s':'M8_4s',
               'Predicted_underperforming':'Low_perf'}

created_cols = ['ELA_8 4s - Limited English Proficient',
                 'ELA_8 4s - Economically Disadvantaged',
                 'Math_8 4s - Limited English Proficient',
                 'Math_8 4s - Economically Disadvantaged']  

class_columns = ['ELA_6_4s',
                 'Math_6_4s',
                 'ELA_7_4s',
                 'Math_7_4s',
                 'ELA_8_4s',
                 'Math_8_4s']


# In[ ]:


# Making dataframes for bayesian networks 
bn_All = prepare_bayesian_network_input(master_df, (cols_for_binning + created_cols))
bn_crime = prepare_bayesian_network_input(master_df, cols_for_binning_crime+created_cols)
bn_class = prepare_bayesian_network_input(master_df, class_columns)
bn_all_ = pd.concat([bn_All, master_df[['Predicted_underperforming']]], axis =1)
bn_crime_ = pd.concat([bn_crime, master_df[['Predicted_underperforming']]], axis =1)
bn_class_ = pd.concat([bn_class, master_df[['Predicted_underperforming']]], axis =1)
print(bn_all_.shape)
print(bn_crime_.shape)
print(bn_class_.shape)
# Talk and clear how 

bn_all_.rename(columns = rename_dict, inplace = True)
bn_crime_.rename(columns = rename_dict, inplace = True)
bn_class_.rename(columns = rename_dict, inplace = True)
bn_all_.to_csv("Bayesian_network_all_data.csv", index = False)
bn_crime_.to_csv("Bayesian_network_crime_data.csv", index = False)
bn_class_.to_csv("Bayesian_network_class_data.csv", index = False)


# In[ ]:


from IPython.display import Image
Image(filename = "../input/helper-notebook-for-bayesian-network/Basyesian_network_all_data.jpg", width = 700, height =500)


# In[ ]:


from IPython.display import Image
Image(filename = "../input/helper-notebook-for-bayesian-network/Basyesian_network_crime_data.jpg", width = 700, height =500)


# ### Findings from driving factor network analysis - 
# - Math proficiency is the most important measure for a school and is driving most of the other proxy measures, and ist main driving factor behind low performing schools. (N1)
# - The Math proficiency is shown to drive ELA proficiency, means making students good with maths will increase their logical and analytical capabilities and which are necessaru for having good ELA proficiency. (N2)
# - If a school perorming low, students of that school are more likely to get involved in some sort of crimes. (N2)
# - Chronic absent from the schools is major driving factor behind low math proficiency
# 

# 
# ## FInal results are provided in Performance_of_schools.csv file
# 
# **Note**- When **predicted_underperfoming** variable takes the value 1, the school is underperforming else it's performing well.

# In[ ]:


master_df.to_csv("Performance_of_schools.csv", index = False)


# ## Thanks for reading.. 
# I hope this unique analysis helps PASSNYC in helping others
