#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

mpl.rcParams['figure.figsize'] = [6, 4]
mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['font.size'] = 14

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import variation
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
        


# In[ ]:


get_ipython().system('pip install numpy-stl')
from stl import mesh


# ## Overview
# 
# In this note book we will do the following: 
# 
# 1. Explore the data from **ANSUR-II Anthropometric Data** which is a public dataset drawn from [2012 U.S. Army Anthropometric Survey](http://mreed.umtri.umich.edu/mreed/downloads.html), for some interesting observations. 
# 
# 3. Focus on the **wrist circumference measurements** from ANSUR-II data, and see how they compare with **data used for sizing Apple Watch bands**. 
# 
# 4. **Import and visualize 3-D full body models (in the form of STL files)** generated based on similar anthropometric data, with two different Body Mass Index values. These files were generated from University of Michigan's [humanshape.org](http://humanshape.org/) website. They were also down sampled to reduce the size using [Autodesk meshmixer](http://www.meshmixer.com/).
#    
# 5. **Extract wrist-circumferene from the imported STL files**, by first extracting the points around the wrist and computing approximate circumference from them, and see how those compare with the distributions we looked at in previous steps. 
# 
# Let us get started..
# 
# ## Exploratory Analysis
# 
# #### Import ANSUR-II data, fix some column name issues and combine male and female data

# In[ ]:


df_ansur2_female = pd.read_csv("../input/ansur-ii/ANSUR II FEMALE Public.csv", 
                               encoding='latin-1') 
df_ansur2_male = pd.read_csv("../input/ansur-ii/ANSUR II MALE Public.csv",
                             encoding='latin-1') 
df_ansur2_female = df_ansur2_female.rename(
                columns = {"SubjectId":"subjectid"}) # Fixing a column name
df_ansur2_all = pd.concat([df_ansur2_female,df_ansur2_male])
print("Shapes of the dataframes (Female,Male,All): " + 
      str((df_ansur2_female.shape,df_ansur2_male.shape,df_ansur2_all.shape)))
df_ansur2_all.head()


# A detailed report about the ANSUR-II survey methodology (including measurement definitions), and summary statistics is available at the [US military website](https://apps.dtic.mil/dtic/tr/fulltext/u2/a611869.pdf).

# Let us look at a couple of sample distributions before moving forward. 

# In[ ]:


for gender in ['Male','Female']:
    subset = df_ansur2_all[df_ansur2_all['Gender'] == gender]
    
    # Draw the density plot
    sns.distplot(subset['stature'], hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3}, 
                  label = gender)
    
# Plot formatting
plt.legend(prop={'size': 14}, title = 'Gender')
plt.title('Density Plot comparing Height Distributions')
plt.xlabel('Height (mm)')
plt.ylabel('Density')
plt.show()


# We can see the expected differences between the difference in heights for men and women. 
# 
# If we look at something like the distance between two pupils, we expect those values to be closer between men and women, as we expect the eye spacing to not vary so drastically. 
# 
# This can be observed in the figure below. 

# In[ ]:


for gender in ['Male','Female']:
    subset = df_ansur2_all[df_ansur2_all['Gender'] == gender]
    
    # Draw the density plot
    sns.distplot(subset['interpupillarybreadth'], hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3}, 
                  label = gender)
    
# Plot formatting
plt.legend(prop={'size': 14}, title = 'Gender')
plt.title('Density Plot comparing Inter-Pupilary Distance')
plt.xlabel('Inter-pupillary Breadth (mm)')
plt.ylabel('Density')
plt.show()


# ## Which measurements show most variation compared to mean ? 
# Let us identify which measurements show most variation (compared to mean) among the subjects measured, separately for each Gender

# In[ ]:


# Identify the numeric columns.
numeric_cols = list(df_ansur2_all.select_dtypes([np.number]).columns)
numeric_cols = ([ele for ele in numeric_cols if ele 
                 not in ['subjectid','SubjectNumericRace', 'DODRace', 'Age']]) 

# Function to compute coefficient of variation
# defined as ratio of standard deviation to mean
def cov(x):
    return(np.std(x) / np.mean(x))

# Generate a data frame with coeffecient variation for each column

df_cov = (df_ansur2_all[['Gender'] + numeric_cols]
.groupby('Gender')
.apply(cov)
.reset_index()
.melt(id_vars = ['Gender'], value_name = 'Coeff_of_Variation', 
      var_name = 'Measurement')
 .sort_values(by = ['Coeff_of_Variation'], ascending = False))


# In[ ]:


# Plot showing top-coefficients of variation
clrs = ['black' if ('earprotrusion' in x) else 'grey' for x in df_cov.head(24).Measurement ]
clrs = [clrs[i] for i in range(len(clrs)) if i % 2 != 0] 

g = sns.catplot(x="Coeff_of_Variation", y="Measurement",
                 col="Gender",
                data=df_cov.head(24), kind="bar", palette = clrs,
                height=6, aspect=1);


plt.subplots_adjust(top=0.85)
g.fig.suptitle('Coefficeint of Variation for Various Measurements')
plt.show()


# While we expect measurements such as height and weight to have highest variation, **some interesting measurements such as the "Ear Protrusion" show up as having a high variation** both in men and women. 
# 
# <font color='blue'>This may have implications for someone designing over-the-ear headphones, for example.</font> 
# 
# 

# ## Wrist Circumference Data
# 
# Let us look at the wrist circumference data from the ANSUR-II dataset, and see how it compares to what Apple is referencing at this link. 

# In[ ]:


for gender in ['Male','Female']:
    subset = df_ansur2_all[df_ansur2_all['Gender'] == gender]
    
    # Draw the density plot
    sns.distplot(subset['wristcircumference'], hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3}, 
                  label = gender)
    
# Plot formatting
plt.legend(prop={'size': 12}, title = 'Gender')
plt.title('Density Plot comparing Wrist Circumference')
plt.xlabel('Wrist Circumference (mm)')
plt.ylabel('Density')
plt.show()


# <font color='blue'>**From the above we can see that wrist circumference for the women seems to vary between about 130 mm to 180 mm, and for men it seems to vary between 150 mm to 210 mm or so.**</font>
# 
# Below is the guidance from Apple as indicated at this [sizing guide](https://store.apple.com/Catalog/regional/amr/pdf/static/pdf/content/Watch-Sizing-Guide.pdf): 
# 
# <img src="https://i.imgur.com/QK3BlQf.png" width="800px">
# 
# 

# **From the above we can see that Apple Watch bands are made to fit wrist sizes from about 135 mm to 210 mm, which covers most of the range for Women and Men**. However, it looks like if one's wrists are either too small or too big, they may have to choose a band more carefully, as all bands may not fit them. 
# 
# ## Import and visualize 3-D full body models (STL files)
# 
# In this section, we will import 3-D STL files generated based on similar anthropometric data, with two different Body Mass Index values (16 and 37). 
# 
# Notes:
# 
# * These files were generated from University of Michigan's [humanshape.org](http://humanshape.org/) website. 
# * They were also down sampled to reduce the size using [Autodesk meshmixer](http://www.meshmixer.com/), as the initial files downloaded from the above website were too big. 
# * The code to import STL files into plotly was borrowed from: https://plot.ly/~empet/15434/mesh3d-with-intensitymodecell/

# In[ ]:


def stl2mesh3d(stl_mesh):
    # stl_mesh is read by nympy-stl from a stl file; it is  an array of faces/triangles (i.e. three 3d points) 
    # this function extracts the unique vertices and the lists I, J, K to define a Plotly mesh3d
    p, q, r = stl_mesh.vectors.shape #(p, 3, 3)
    # the array stl_mesh.vectors.reshape(p*q, r) can contain multiple copies of the same vertex;
    # extract unique vertices from all mesh triangles
    vertices, ixr = np.unique(stl_mesh.vectors.reshape(p*q, r), return_inverse=True, axis=0)
    I = np.take(ixr, [3*k for k in range(p)])
    J = np.take(ixr, [3*k+1 for k in range(p)])
    K = np.take(ixr, [3*k+2 for k in range(p)])
    return vertices, I, J, K


# In[ ]:


mymesh = [
    mesh.Mesh.from_file('/kaggle/input/humanshapestlfiles/stature_1773_shs_0p52_age_38_bmi_16.stl'), 
    mesh.Mesh.from_file('/kaggle/input/humanshapestlfiles/stature_1773_shs_0p52_age_38_bmi_37.stl'),]


# In[ ]:


fig = make_subplots(rows=1, cols=2,
                    specs=[[{'is_3d': True}, {'is_3d': True}]],
                    subplot_titles=("BMI: 16", "BMI: 37"),
                    print_grid=False)

for i in [1,2]:
        vertices, I, J, K = stl2mesh3d(mymesh[i-1])
        triangles = np.stack((I,J,K)).T
        x, y, z = vertices.T

        fig.append_trace(
                          go.Mesh3d(x=x, y=y, z=z, 
                          i=I, j=J, k=K, 
                          showscale=False,
                          flatshading=False, 
                          lighting = dict(ambient=0.5,
                                          diffuse=1,
                                          fresnel=4,        
                                          specular=0.5,
                                          roughness=0.05,
                                          facenormalsepsilon=0),
                            ),
            row=1, col=i)
        
fig.update_layout(width=800, height=700,
                  template='plotly_dark', 
                 )


# fix the ratio in the top left subplot to be a cube
camera = dict(eye=dict(x=-1.25, y=-0.25, z=-0.25))
fig.update_layout(scene_aspectmode='manual',scene_aspectratio=dict(x=0.2, y=0.6, z=1), scene_camera = camera)
# manually force the z-axis to appear twice as big as the other two
fig.update_layout(scene2_aspectmode='manual',scene2_aspectratio=dict(x=0.25, y=0.6, z=1), scene2_camera = camera)


for i in fig['layout']['annotations']:
    i['font'] = dict(size=25,color='#ffffff')
    

fig.show()


# As shown above, we were able to import two different STL files with different Body Mass Index values
# ## Extracting wrist circumference from the STL files
# 
# Assuming we got these STL files from a 3-D body scan, we can try extracting some parameters from the STL files. 
# 
# As an exercise, we will try to extract the wrist circumference from these STL files and see how it compares with our expectation. 
# 
# <font color='blue'>From the above pictures, we can see that the subject wrists were located around Z = 920, and Y = +/- 400</font> We use this information to extract points corresponding to the wrists. 

# In[ ]:


# Extract all points from the STL files

points = [[],[]]
for i in range(2):
    points_ = []
    for triangle in list(mymesh[i].vectors):
        for point in triangle:
            points_.append(point)
    points[i] = np.array(points_)

# Extract the points corresponding to the wrist circumference
wrist_points = []
for wrist in points:
    wrist_points_ = pd.DataFrame(
        np.array(list(set(
            [tuple(item) for item in wrist if (
                (np.abs(item[2]-920) < 5) and item[1] > 300)]
        ))), columns = ['X','Y','Z'])
    wrist_points.append(wrist_points_)
    
# Plot the wrist circumferene cross-sections
f, (ax1, ax2) = plt.subplots(1, 2, sharex='all', sharey='all')
ax1.scatter(wrist_points[0]['X'], wrist_points[0]['Y'])
ax1.set_title('BMI: 16 Wrist Size')
ax2.scatter(wrist_points[1]['X'], wrist_points[1]['Y'])
ax2.set_title('BMI: 37 Wrist Size')
f.show()


# **As shown above, we were able to extract points along wrist circumference from STL files, and can see that the wrist-size for the person with BMI 37 is much bigger than that of the person with BMI 16 as expected.**
# 
# However, calculating the circumference from the points is not trivial. We need to form a path through the points that rougly traces the circumference. 
# 
# **To calculate the wrist circumference, we will use the function below (nearest neighbor path connecting all points)**

# In[ ]:


# Function to sort the array to generate nearest neighbor path
def nearest_neighbour_sort(df):
    df['Id'] = list(range(df.shape[0]))
    ids = df.Id.values[1:]
    xy = np.array([df.X.values, df.Y.values]).T[1:]
    path = [0,]
    while len(ids) > 0:
        last_x, last_y = df.X[path[-1]], df.Y[path[-1]]
        dist = ((xy - np.array([last_x, last_y]))**2).sum(-1)
        nearest_index = dist.argmin()
        path.append(ids[nearest_index])
        ids = np.delete(ids, nearest_index, axis=0)
        xy = np.delete(xy, nearest_index, axis=0)
    path.append(0)
    return path

wrist_points = []
for wrist in points:
    wrist_points_ = pd.DataFrame(np.array(list(set([tuple(item) for item in wrist if ((np.abs(item[2]-920) < 5) and item[1] > 300)]))), columns = ['X','Y','Z'])
    wrist_points_ = wrist_points_.loc[nearest_neighbour_sort(wrist_points_),].reset_index(drop = True)
    wrist_points_['distance'] = np.concatenate(([0.0],
                                                np.cumsum(np.sqrt((wrist_points_.X[1:].values - wrist_points_.X[:-1].values)**2+
                                                                  (wrist_points_.Y[1:].values - wrist_points_.Y[:-1].values)**2))))
    wrist_points.append(wrist_points_)

# Lineplot showing the list circumference
f, (ax1, ax2) = plt.subplots(1, 2, sharex='all', sharey='all')
ax1.plot(wrist_points[0]['X'], wrist_points[0]['Y'])
ax1.set_title('BMI: 16 Wrist Size')
ax2.plot(wrist_points[1]['X'], wrist_points[1]['Y'])
ax2.set_title('BMI: 37 Wrist Size')
f.show()


# In[ ]:


print("Calculated circumference for the wrist of the person with BMI = 16 is {0:8.2f} mm".format(max(wrist_points[0].distance)))


# In[ ]:


print("Calculated circumference for the wrist of the person with BMI = 37 is {0:8.2f} mm".format(max(wrist_points[1].distance)))


# In[ ]:


for gender in ['Male','Female']:
    subset = df_ansur2_all[df_ansur2_all['Gender'] == gender]
    
    # Draw the density plot
    sns.distplot(subset['wristcircumference'], hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3}, 
                  label = gender)
    
# Plot formatting
plt.legend(prop={'size': 12}, title = 'Gender')
plt.title('Density Plot comparing Wrist Circumference')
plt.xlabel('Wrist Circumference (mm)')
plt.ylabel('Density')
plt.show()


# As expected, the wrist size of the person with very low BMI is on the lower side of men's wrist sizes, while the wrist size of the person with high BMI is in the higher side of mens' wrist-size ranges observed before. 
# 
# ## Summary
# 
# In this note book we accompolished the following: 
# 
# 1. Explored the data from **ANSUR-II Anthropometric Data** and made some interesting observations, such as the ear protrusion is one of the parameters that varies most from person to person. 
# 
# 3. We studied on the **wrist circumference measurements** from ANSUR-II data, and saw it compared well with data used for sizing Apple Watch bands. 
# 
# 4. Imported and visualize 3-D full body models (in the form of STL files) 
#    
# 5. **Extracted wrist-circumferene from the imported STL files**, and confirmed it aligned well with our expectations for the persons with different BMIs. 
