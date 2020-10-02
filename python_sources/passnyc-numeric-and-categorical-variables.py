#!/usr/bin/env python
# coding: utf-8

# <h1 class='font-effect-3d' style='color:#37c9e1; font-family:Orbitron;'> &#x1F310; &nbsp; Code Library, Styling, and Links</h1>
# `GITHUB` Version: &nbsp; &#x1F4D8; &nbsp; [kaggle_passnyc2.ipynb](https://github.com/OlgaBelitskaya/kaggle_notebooks/blob/master/kaggle_passnyc2.ipynb)
# 
# `R` Version: &nbsp; &#x1F4D8; &nbsp; [kaggle_passnyc2_R.ipynb](https://github.com/OlgaBelitskaya/kaggle_notebooks/blob/master/kaggle_passnyc2_R.ipynb)
# 
# The previous notebooks: 
# 
# &#x1F4D8; &nbsp; [PASSNYC. Data Exploration](https://www.kaggle.com/olgabelitskaya/passnyc-data-exploration); &nbsp; [PASSNYC. Data Exploration R](https://www.kaggle.com/olgabelitskaya/passnyc-data-exploration-r)
# 
# Useful `LINKS`: 
# 
# &#x1F4E1; &nbsp; [School Quality Reports. Educator Guide](http://schools.nyc.gov/NR/rdonlyres/967E0EE1-7E5D-4E47-BC21-573FEEE23AE2/0/201516EducatorGuideHS9252017.pdf) & [New York City Department of Education](https://www.schools.nyc.gov)
# 
# &#x1F4E1; &nbsp; [NYC OpenData](https://opendata.cityofnewyork.us/)
# 
# &#x1F4E1; &nbsp; [Pandas Visualization](https://pandas.pydata.org/pandas-docs/stable/visualization.html) & [Pandas Styling](https://pandas.pydata.org/pandas-docs/stable/style.html)

# In[ ]:


get_ipython().run_cell_magic('html', '', "<style> \n@import url('https://fonts.googleapis.com/css?family=Orbitron|Roboto&effect=3d');\nbody {background-color: gainsboro;} \nh3 {color:#818286; font-family:Roboto;}\nspan {color:black; text-shadow:4px 4px 4px #aaa;}\ndiv.output_prompt,div.output_area pre {color:slategray;}\ndiv.input_prompt,div.output_subarea {color:#37c9e1;}      \ndiv.output_stderr pre {background-color:gainsboro;}  \ndiv.output_stderr {background-color:slategrey;}       \n</style>")


# In[ ]:


import numpy as np,pandas as pd,geopandas as gpd
import pylab as plt,seaborn as sns
from matplotlib import cm
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
style_dict={'background-color':'slategray','color':'#37c9e1',
            'border-color':'white','font-family':'Roboto'}
plt.style.use('seaborn-whitegrid')
path='../input/data-science-for-good/'
path2='../input/ny-school-districts/'


# <h1 class='font-effect-3d' style='color:#37c9e1; font-family:Orbitron;'>  &#x1F310; &nbsp; Loading & Preprocessing the Data <h1>

# In[ ]:


school_explorer=pd.read_csv(path+'2016 School Explorer.csv')
d5_shsat=pd.read_csv(path+'D5 SHSAT Registrations and Testers.csv')
school_explorer.shape,d5_shsat.shape


# In[ ]:


drop_list=['Adjusted Grade','New?','Other Location Code in LCGMS']
school_explorer=school_explorer.drop(drop_list,axis=1)
school_explorer.loc[[427,1023,712,908],'School Name']=['P.S. 212 D12','P.S. 212 D30','P.S. 253 D21','P.S. 253 D27']
school_explorer['School Income Estimate']=school_explorer['School Income Estimate'].astype('str') 
for s in [",","$"," "]:
    school_explorer['School Income Estimate']=    school_explorer['School Income Estimate'].str.replace(s,"")
school_explorer['School Income Estimate']=school_explorer['School Income Estimate'].str.replace("nan","0")
school_explorer['School Income Estimate']=school_explorer['School Income Estimate'].astype(float)
school_explorer['School Income Estimate'].replace(0,np.NaN,inplace=True)
percent_list=['Percent ELL','Percent Asian','Percent Black',
              'Percent Hispanic','Percent Black / Hispanic',
              'Percent White','Student Attendance Rate',
              'Percent of Students Chronically Absent',
              'Rigorous Instruction %','Collaborative Teachers %',
              'Supportive Environment %','Effective School Leadership %',
              'Strong Family-Community Ties %','Trust %']
target_list=['Average ELA Proficiency','Average Math Proficiency']
economic_list=['Economic Need Index','School Income Estimate']
rating_list=['Rigorous Instruction Rating','Collaborative Teachers Rating',
             'Supportive Environment Rating','Effective School Leadership Rating',
             'Strong Family-Community Ties Rating','Trust Rating',
             'Student Achievement Rating']
for el in percent_list:
    school_explorer[el]=school_explorer[el].astype('str')
    school_explorer[el]=school_explorer[el].str.replace("%","")
    school_explorer[el]=school_explorer[el].str.replace("nan","0")
    school_explorer[el]=school_explorer[el].astype(float)
    school_explorer[el].replace(0,np.NaN,inplace=True)
    school_explorer[el]=school_explorer[el].interpolate()
for el in target_list+economic_list:
    school_explorer[el]=school_explorer[el].interpolate()
for el in rating_list:
    moda_value=school_explorer[el].value_counts().idxmax()
    school_explorer[el]=school_explorer[el].fillna(moda_value)   
print('Number of Missing Values: ',sum(school_explorer.isna().sum()))


# In[ ]:


districts=gpd.GeoDataFrame.from_file(path2+"nysd.shp")
# http://prj2epsg.org
districts.crs=('+init=EPSG:2263')
districts=districts.to_crs(('+init=EPSG:4326'))
districts['coords']=districts['geometry'].apply(lambda x:x.representative_point().coords[:])
districts['coords']=[coords[0] for coords in districts['coords']]
fig,ax=plt.subplots(1,figsize=(12,10))
districts.plot(ax=ax,cmap=cm.hsv,alpha=.8,edgecolor='slategray')
for idx,row in districts.iterrows():
    plt.annotate(s=row['SchoolDist'],xy=row['coords'],
                 horizontalalignment='center',fontsize=12)
plt.title('NYC School Districts');


# <h1 class='font-effect-3d' style='color:#37c9e1; font-family:Orbitron;'> &#x1F310; &nbsp; Classification of Variables </h1>

# In[ ]:


school_explorer.describe(include=[np.number]).T.head(23).style.set_properties(**style_dict)


# In[ ]:


school_explorer.describe(include=[np.object]).T.style.set_properties(**style_dict)


# In[ ]:


numeric_list1=school_explorer.describe(include=[np.number]).columns[:23]
numeric_list2=school_explorer.describe(include=[np.number]).columns[23:]
object_list=school_explorer.describe(include=[np.object]).columns


# Of course, the variables `SED Code`, `District`, `Zip` are categorical. 
# 
# Just categories are denoted by numeric values.
# 
# Let's convert string values of other categorical features into numeric.

# In[ ]:


print('District: \n',set(school_explorer['District']),'\n')
print('City: \n',set(school_explorer['City']),'\n')
print('Grades: \n',set(school_explorer['Grades']),'\n')
print('Community School?: \n',
      set(school_explorer['Community School?']),'\n')
print('Rigorous Instruction Rating: \n',
      set(school_explorer['Rigorous Instruction Rating']),'\n')
print('Collaborative Teachers Rating: \n',
      set(school_explorer['Collaborative Teachers Rating']),'\n')
print('Supportive Environment Rating: \n',
      set(school_explorer['Supportive Environment Rating']),'\n')
print('Effective School Leadership Rating: \n',
      set(school_explorer['Effective School Leadership Rating']),'\n')
print('Strong Family-Community Ties Rating: \n',
      set(school_explorer['Strong Family-Community Ties Rating']),'\n')
print('Trust Rating: \n',
      set(school_explorer['Trust Rating']),'\n')
print('Student Achievement Rating: \n',
      set(school_explorer['Student Achievement Rating']),'\n')


# In[ ]:


category_list=['District','Community School?','City','Grades']             
for feature in category_list:
    feature_cat=pd.factorize(school_explorer[feature])
    school_explorer[feature]=feature_cat[0]  
for feature in rating_list:
    feature_pairs=dict(zip(['Not Meeting Target','Meeting Target', 
                            'Approaching Target','Exceeding Target'],
                            ['0','2','1','3']))
    school_explorer[feature].replace(feature_pairs,inplace=True)
    school_explorer[feature]=school_explorer[feature].astype(int)


# In[ ]:


category_list=list(category_list+rating_list)
numeric_list=list(numeric_list1[5:21].append(numeric_list2))
print('Categorical features: \n',category_list,'\n', 
      'Numeric features: \n',numeric_list)


# In[ ]:


print('District: \n',set(school_explorer['District']),'\n')
print('City: \n',set(school_explorer['City']),'\n')
print('Grades: \n',set(school_explorer['Grades']),'\n')
print('Community School?: \n',
      set(school_explorer['Community School?']),'\n')
print('Rigorous Instruction Rating: \n',
      set(school_explorer['Rigorous Instruction Rating']),'\n')
print('Collaborative Teachers Rating: \n',
      set(school_explorer['Collaborative Teachers Rating']),'\n')
print('Supportive Environment Rating: \n',
      set(school_explorer['Supportive Environment Rating']),'\n')
print('Effective School Leadership Rating: \n',
      set(school_explorer['Effective School Leadership Rating']),'\n')
print('Strong Family-Community Ties Rating: \n',
      set(school_explorer['Strong Family-Community Ties Rating']),'\n')
print('Trust Rating: \n',
      set(school_explorer['Trust Rating']),'\n')
print('Student Achievement Rating: \n',
      set(school_explorer['Student Achievement Rating']),'\n')


# <h1 class='font-effect-3d' style='color:#37c9e1; font-family:Orbitron;'>  &#x1F310; &nbsp; Clustering </h1>
# ### By education results and economic indicators

# In[ ]:


X=school_explorer[target_list+economic_list]; nk=10
clf=KMeans(n_clusters=nk,random_state=23)
cluster_labels=clf.fit_predict(np.array(X))
school_explorer['Education Clusters']=cluster_labels
X.groupby(school_explorer['Education Clusters']).mean().sort_values('Average Math Proficiency').style.set_properties(**style_dict)


# Let's arrange these clusters in accordance with educational achievements.

# In[ ]:


indices=list(X.groupby(school_explorer['Education Clusters']).mean()             .sort_values('Average Math Proficiency').index)
feature_pairs=dict(zip(indices,range(10,20)))
school_explorer['Education Clusters'].replace(feature_pairs,inplace=True)
feature_pairs=dict(zip(range(10,20),range(0,10)))
school_explorer['Education Clusters'].replace(feature_pairs,inplace=True)
X.groupby(school_explorer['Education Clusters']).mean().sort_values('Average Math Proficiency').style.set_properties(**style_dict)


# The location of the results with good quality by districts and cluster types of schools can be assessed using visualization.

# In[ ]:


fig,ax=plt.subplots(1,figsize=(12,7))
plt.suptitle('Education Clusters by Districts')
sns.countplot(x="District",hue="Education Clusters", 
              data=school_explorer,ax=ax,palette='bwr')
ax.legend(loc='center left',bbox_to_anchor=(1,.5))
ax.set_xticklabels(range(1,33));


# In[ ]:


fig,ax=plt.subplots(1,figsize=(12,10))
school_explorer.plot(kind="scatter",x="Longitude",y="Latitude",
                     s=30,c="Education Clusters",ax=ax,
                     title='Map of Education Clusters by Schools',
                     cmap=cm.hsv,colorbar=True,alpha=.8)
districts.plot(ax=ax,color='none',edgecolor='slategray');


# In[ ]:


fig,ax=plt.subplots(1,figsize=(7,7))
plt.suptitle('Education Clusters by Community and Non-Community Schools')
sns.boxplot(x="Community School?",y="Education Clusters",
            data=school_explorer,ax=ax,palette='bwr')
ax.set_xticklabels(['YES','NO']);


# ### By education results and social environment

# In[ ]:


cluster_list2=['Average ELA Proficiency','Average Math Proficiency',
               'Student Attendance Rate',
               'Percent of Students Chronically Absent',                     
               'Rigorous Instruction %','Collaborative Teachers %',
               'Supportive Environment %','Effective School Leadership %',
               'Strong Family-Community Ties %','Trust %']
X=school_explorer[cluster_list2]; nk=10
clf=KMeans(n_clusters=nk,random_state=23)
cluster_labels=clf.fit_predict(np.array(X))
school_explorer['Education Clusters']=cluster_labels
X.groupby(school_explorer['Education Clusters']).mean().sort_values('Average Math Proficiency').T.style.set_properties(**style_dict)


# In[ ]:


indices=list(X.groupby(school_explorer['Education Clusters']).mean()             .sort_values('Average Math Proficiency').index)
feature_pairs=dict(zip(indices,range(10,20)))
school_explorer['Education Clusters'].replace(feature_pairs,inplace=True)
feature_pairs=dict(zip(range(10,20),range(0,10)))
school_explorer['Education Clusters'].replace(feature_pairs,inplace=True)
X.groupby(school_explorer['Education Clusters']).mean().sort_values('Average Math Proficiency').T.style.set_properties(**style_dict)


# In[ ]:


fig,ax=plt.subplots(1,figsize=(12,7))
plt.suptitle('Education Clusters by Districts')
sns.countplot(x="District",hue="Education Clusters", 
              data=school_explorer,ax=ax,palette='bwr')
ax.legend(loc='center left',bbox_to_anchor=(1,.5))
ax.set_xticklabels(range(1,33));


# In[ ]:


fig,ax=plt.subplots(1,figsize=(12,10))
school_explorer.plot(kind="scatter",x="Longitude",y="Latitude",
                     s=30,c="Education Clusters",ax=ax,
                     title='Map of Education Clusters by Schools',
                     cmap=cm.hsv,colorbar=True,alpha=.8)
districts.plot(ax=ax,color='none',edgecolor='slategray');


# In[ ]:


fig,ax=plt.subplots(1,figsize=(7,7))
plt.suptitle('Education Clusters by Community and Non-Community Schools')
sns.boxplot(x="Community School?",y="Education Clusters",
            data=school_explorer,ax=ax,palette='bwr')
ax.set_xticklabels(['YES','NO']);


# The distribution by clusters has certain differences for two cases, but the problem regions are clearly the same.

# <h1 class='font-effect-3d' style='color:#37c9e1; font-family:Orbitron;'>  &#x1F310; &nbsp; Manifold Learning </h1>
# Let's divide schools into educational clusters using all numeric variables.

# In[ ]:


X=school_explorer[numeric_list+category_list+target_list]
nk=10; clf=KMeans(n_clusters=nk,random_state=23)
cluster_labels=clf.fit_predict(np.array(X))
school_explorer['Education Clusters']=cluster_labels
indices=list(X.groupby(school_explorer['Education Clusters']).mean()             .sort_values('Average Math Proficiency').index)
feature_pairs=dict(zip(indices,range(10,20)))
school_explorer['Education Clusters'].replace(feature_pairs,inplace=True)
feature_pairs=dict(zip(range(10,20),range(0,10)))
school_explorer['Education Clusters'].replace(feature_pairs,inplace=True)


# In[ ]:


fig,ax=plt.subplots(1,figsize=(12,7))
plt.suptitle('Education Clusters by Districts')
sns.countplot(x="District",hue="Education Clusters", 
              data=school_explorer,ax=ax,palette='bwr')
ax.legend(loc='center left',bbox_to_anchor=(1,.5))
ax.set_xticklabels(range(1,33));


# In[ ]:


fig,ax=plt.subplots(1,figsize=(12,10))
school_explorer.plot(kind="scatter",x="Longitude",y="Latitude",
                     s=30,c="Education Clusters",ax=ax,
                     title='Map of Education Clusters by Schools',
                     cmap=cm.hsv,colorbar=True,alpha=.8)
districts.plot(ax=ax,color='none',edgecolor='slategray');


# Now we can check whether the multidimensional data form a certain structure or don't. 
# 
# For this purpose, a probability algorithm and a transformation to two-dimensional space were used. 
# 
# The color on the chart still marks educational clusters.

# In[ ]:


X=school_explorer[numeric_list+category_list+target_list]
tsne=TSNE(n_components=2,random_state=23)
X_embedded=tsne.fit_transform(np.array(X))
plt.figure(figsize=(7,7))
plt.scatter(X_embedded[:,1],X_embedded[:,0],cmap=cm.hsv,
            c=school_explorer['Education Clusters'])
plt.title('The Data Structure');


# It seems like this clustering is done correctly. 
# 
# And the 7-9 clusters form a separate structure.

# <h1 class='font-effect-3d' style='color:#37c9e1; font-family:Orbitron;'>  &#x1F310; &nbsp; Feature Correlation </h1>

# Many numeric variables demonstrate a strong correlation.

# In[ ]:


corr_matrix=school_explorer[numeric_list1[5:23]].corr()
fig,ax=plt.subplots(1,figsize=(10,8))
sns.heatmap(corr_matrix,ax=ax,cmap="bwr")
ax.set_xticklabels(corr_matrix.columns.values,size=15)
ax.set_yticklabels(corr_matrix.columns.values,size=15);


# Education results (ELA & Math) are correlated really strongly so they can be combined into one indicator.

# In[ ]:


pearson=school_explorer[numeric_list+category_list+target_list].corr(method='pearson')
corr_with_math_results=pearson.iloc[-1]
corr_with_math_results[abs(corr_with_math_results).argsort()[::-1]][:20].to_frame().style.background_gradient(cmap='bwr').set_properties(**{'color':'black','font-family':'Roboto','font-size':'120%'})


# In[ ]:


pearson=school_explorer[numeric_list+category_list+target_list].corr(method='pearson')
corr_with_ela_results=pearson.iloc[-2]
corr_with_ela_results[abs(corr_with_ela_results).argsort()[::-1]][:20].to_frame().style.background_gradient(cmap='bwr').set_properties(**{'color':'black','font-family':'Roboto','font-size':'120%'})


# <h1 class='font-effect-3d' style='color:#37c9e1; font-family:Orbitron;'> &#x1F310; &nbsp; Let's Go Ahead </h1>

# To be continued...
# 
# &#x1F4D8; &nbsp; [PASSNYC. Comparing All Districts with 5th District](https://www.kaggle.com/olgabelitskaya/passnyc-comparing-all-districts-with-5th-district/notebook)
