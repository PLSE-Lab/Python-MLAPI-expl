#!/usr/bin/env python
# coding: utf-8

# <h1 class='font-effect-3d' style='color:#37c9e1; font-family:Orbitron;'> &#x1F310; &nbsp; Code Library, Styling, and Links</h1>
# `GITHUB` Version: &nbsp; &#x1F4D8; &nbsp; [kaggle_passnyc3.ipynb](https://github.com/OlgaBelitskaya/kaggle_notebooks/blob/master/kaggle_passnyc2.ipynb)
# 
# The previous notebooks: 
# 
# &#x1F4D8; &nbsp; [PASSNYC. Data Exploration](https://www.kaggle.com/olgabelitskaya/passnyc-data-exploration); &nbsp; [PASSNYC. Data Exploration R](https://www.kaggle.com/olgabelitskaya/passnyc-data-exploration-r)
# 
# &#x1F4D8; &nbsp;  [PASSNYC. Numeric and Categorical Variables](https://www.kaggle.com/olgabelitskaya/passnyc-numeric-and-categorical-variables); &nbsp; [PASSNYC. Numeric and Categorical Variables R](https://www.kaggle.com/olgabelitskaya/passnyc-numeric-and-categorical-variables-r)
# 
# Useful `LINKS`: 
# 
# &#x1F4E1; &nbsp; [School Quality Reports. Educator Guide](http://schools.nyc.gov/NR/rdonlyres/967E0EE1-7E5D-4E47-BC21-573FEEE23AE2/0/201516EducatorGuideHS9252017.pdf) & [New York City Department of Education](https://www.schools.nyc.gov)
# 
# &#x1F4E1; &nbsp; [NYC OpenData](https://opendata.cityofnewyork.us/)
# 
# &#x1F4E1; &nbsp; [Pandas Visualization](https://pandas.pydata.org/pandas-docs/stable/visualization.html) & [Pandas Styling](https://pandas.pydata.org/pandas-docs/stable/style.html)

# In[ ]:


get_ipython().run_cell_magic('html', '', "<style> \n@import url('https://fonts.googleapis.com/css?family=Orbitron|Roboto&effect=3d');\nbody {background-color: gainsboro;} \nh3 {color:#818286; font-family:Roboto;}\nspan {color:black; text-shadow:4px 4px 4px #aaa;}\ndiv.output_prompt,div.output_area pre {color:slategray;}\ndiv.input_prompt,div.output_subarea {color:#37c9e1;}      \ndiv.output_stderr pre {background-color:gainsboro;}  \ndiv.output_stderr {background-color:slategrey;}              \n</style>")


# In[ ]:


import numpy as np,pandas as pd,geopandas as gpd
import pylab as plt,seaborn as sns
from matplotlib import cm
import matplotlib.colors as mcolors
from descartes import PolygonPatch
from sklearn.preprocessing import minmax_scale
from IPython.display import display
style_dict={'background-color':'slategray','color':'#37c9e1',
            'border-color':'white','font-family':'Roboto'}
plt.style.use('seaborn-whitegrid')
path='../input/data-science-for-good/'
path2='../input/ap-college-board-ny-school-level-results/'
path3='../input/new-york-city-sat-results/'
path4='../input/ny-school-districts/'


# <h1 class='font-effect-3d' style='color:#37c9e1; font-family:Orbitron;'> &#x1F310; &nbsp; Data Loading and Preprocessing</h1>

# In[ ]:


school_explorer=pd.read_csv(path+'2016 School Explorer.csv')
d5_shsat=pd.read_csv(path+'D5 SHSAT Registrations and Testers.csv')
sat_2010=pd.read_csv(path2+'2010-sat-college-board-school-level-results.csv')
ap_2010=pd.read_csv(path2+'2010-ap-college-board-school-level-results.csv')
sat_2012=pd.read_csv(path3+'2012-sat-results.csv')
school_explorer.shape,d5_shsat.shape,sat_2010.shape,ap_2010.shape,sat_2012.shape


# In[ ]:


display(sat_2010.head(3).style.set_properties(**style_dict))
display(ap_2010.head(3).style.set_properties(**style_dict))
sat_2012.head(3).style.set_properties(**style_dict)


# In[ ]:


drop_list=['Adjusted Grade','New?','Other Location Code in LCGMS']
school_explorer=school_explorer.drop(drop_list, axis=1)
# replacing the same values
school_explorer.loc[[427,1023,712,908],'School Name']=['P.S. 212 D12','P.S. 212 D30','P.S. 253 D21','P.S. 253 D27']
# transformation from string to numeric values 
school_explorer['School Income Estimate']=school_explorer['School Income Estimate'].astype('object') 
for s in [",","$"," "]:
    school_explorer['School Income Estimate']=    school_explorer['School Income Estimate'].str.replace(s, "")
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
# transformation to numeric variables and fillna missing values
for el in percent_list:
    school_explorer[el]=school_explorer[el].astype('object')
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
# preprocessing categorical features      
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
category_list=list(category_list+rating_list)
numeric_list=list(school_explorer.columns[[4,5]+list(range(13,24))+[25,27,29,31,33]+list(range(38,158))])     
print('Number of Missing Values: ',sum(school_explorer.isna().sum()))


# In[ ]:


print('Categorical features: \n',category_list,'\n', 
      'Numeric features: \n',numeric_list)    


# In[ ]:


sat_list=['DBN','Number of students who registered for the SHSAT',
          'Number of students who took the SHSAT']
d5_shsat_2016=d5_shsat[sat_list][d5_shsat['Year of SHST']==2016].groupby(['DBN'],as_index=False).agg(np.sum)
d5_shsat_2016['Took SHSAT %']=d5_shsat_2016['Number of students who took the SHSAT']/ d5_shsat_2016['Number of students who registered for the SHSAT']
d5_shsat_2016['Took SHSAT %']=d5_shsat_2016['Took SHSAT %'].fillna(0).apply(lambda x: round(x,3))
d5_shsat_2016.rename(columns={'DBN':'Location Code'},inplace=True)
d5_shsat_2016=pd.merge(school_explorer[['Location Code']+                         numeric_list+category_list+target_list],
                         d5_shsat_2016,on='Location Code')
d5_shsat_2016.shape


# In[ ]:


geo_districts=gpd.GeoDataFrame.from_file(path4+"nysd.shp") # EPSG:2263
geo_districts=geo_districts.to_crs(epsg=4326).sort_values('SchoolDist')
geo_districts=geo_districts.reset_index(drop=True)
districts=school_explorer[numeric_list+target_list].groupby(school_explorer['District']).mean().sort_index()
districts=districts.append(districts.loc[9]).sort_index()
districts=districts.reset_index(drop=True)
districts=pd.concat([geo_districts,districts],axis=1)
districts.shape


# In[ ]:


sat_2010.rename(columns={'DBN':'Location Code'},inplace=True)
ap_2010.rename(columns={'DBN':'Location Code'},inplace=True)
sat_2012.rename(columns={'DBN':'Location Code'},inplace=True)
res_2010=pd.merge(ap_2010, sat_2010,on='Location Code').dropna()
res_2010=res_2010.drop(['SchoolName'],axis=1)
re_dict={'AP Test Takers ':'AP Test Takers 2010',
         'Number of Test Takers':'Number of Test Takers 2010',
         'Critical Reading Mean':'Critical Reading Mean 2010',
         'Mathematics Mean':'Mathematics Mean 2010',
         'Writing Mean':'Writing Mean 2010'} 
res_2010.rename(columns=re_dict,inplace=True)
res_2010['AP Exam Ratio 2010']=res_2010['Number of Exams with scores 3 4 or 5']/res_2010['Total Exams Taken']
res_2010=res_2010.drop(['Total Exams Taken'],axis=1)
res_2010=res_2010.drop(['Number of Exams with scores 3 4 or 5'],axis=1)
res_2010_2012=pd.merge(res_2010, sat_2012, on='Location Code').dropna()
res_2010_2012=res_2010_2012.drop(['School Name'],axis=1)
re_dict={'Num of SAT Test Takers':'Num of SAT Test Takers 2012',
         'SAT Critical Reading Avg. Score':'SAT Critical Reading Avg. Score 2012',
         'SAT Math Avg. Score':'SAT Math Avg. Score 2012',
         'SAT Writing Avg. Score':'SAT Writing Avg. Score 2012'}
res_2010_2012.rename(columns=re_dict,inplace=True)
tend_list=['SAT Critical Reading Avg. Score 2012',
           'SAT Math Avg. Score 2012',
           'SAT Writing Avg. Score 2012']
for s in tend_list:
    res_2010_2012[s]=res_2010_2012[s].astype(float)  
res_2010_2012_2016=pd.merge(school_explorer[['Location Code','District','City',
                          'Longitude','Latitude',
                          'Average ELA Proficiency',
                          'Average Math Proficiency']],
         res_2010_2012,on='Location Code').dropna() 
re_dict={'Average ELA Proficiency':'Average ELA Proficiency 2016',
         'Average Math Proficiency':'Average Math Proficiency 2016'}
res_2010_2012_2016.rename(columns=re_dict,inplace=True)
norm_res_2010_2012_2016=pd.DataFrame(minmax_scale(res_2010_2012_2016                          .iloc[:,[5,6,9,10,11,12,15,16,17]]),
             columns=list(res_2010_2012_2016\
                          .columns[[5,6,9,10,11,12,15,16,17]]))
res_2010.shape,res_2010_2012.shape,res_2010_2012_2016.shape


# In[ ]:


res_2010_2012_2016.head(3).T.style.set_properties(**style_dict)


# <h1 class='font-effect-3d' style='color:#37c9e1; font-family:Orbitron;'>&#x1F310; &nbsp; Enucational Indicators</h1>
# #### Average Math & ELA Proficiency

# In[ ]:


fig,ax=plt.subplots(1,figsize=(12,7))
avg_maths=districts['Average Math Proficiency'].drop([10])
avg_ela=districts['Average ELA Proficiency'].drop([10])
ax.scatter(range(1,33),avg_maths,marker='*',
           s=100,label='Math',c='#3636ff')
ax.scatter(range(1,33),avg_ela,marker='*',
           s=100,label='ELA',c='#ff3636')
plt.vlines(range(1,33),avg_ela,avg_maths,linestyle="dotted")
ax.legend(); ax.set_xticks(list(range(1,33)))
ax.tick_params('x',rotation=0)
plt.title('Average Math & ELA Proficiency by Districts');


# In[ ]:


fig,ax=plt.subplots(1,figsize=(12,7))
avg_maths2=d5_shsat_2016['Average Math Proficiency']
avg_ela2=d5_shsat_2016['Average ELA Proficiency']
took=d5_shsat_2016['Took SHSAT %']
ax.scatter(range(1,22),avg_maths2,marker='*',s=took*500,
           label='Math',c='#ff3636')
ax.scatter(range(1,22),avg_ela2,marker='*',s=took*500,
           label='ELA',c='#3636ff')
plt.vlines(range(1,22),avg_ela2,avg_maths2,linestyle="dotted")
plt.vlines(np.arange(1.,21.2,.2),avg_ela[4],avg_maths[4],
           colors='orchid',linestyle="dotted",
           label='Average Math & ELA in D5')
ax.scatter([8,14,17],avg_maths2[[7,13,16]],label='Leaders',s=500,
           facecolors='none',edgecolors=['#fd0e35','#0066ff','#fc74fd'])
ax.legend(loc=2)
ax.set_xticks(range(1,22))
ax.set_xticklabels(list(d5_shsat_2016['Location Code']),rotation=90)
plt.title('Average Math & ELA Proficiency in General '+          'and by Schools in the 5th District');


# On this graph, we can see two absolute leaders in educational results. The third place was marked also.
# 
# The size of markers on the graph corresponds to the ratio of students who took the SHSAT to those who registered for the SHSAT.
# 
# These leaders have very good results in this ratio too.

# #### Math & ELA 4s Ratio by Grades

# In[ ]:


ela4s_ratio=pd.DataFrame({
    'Grade 3':districts.iloc[:,23].drop([10])/\
    districts.iloc[:,22].drop([10]),
    'Grade 4':districts.iloc[:,43].drop([10])/\
    districts.iloc[:,42].drop([10]),
    'Grade 5':districts.iloc[:,63].drop([10])/\
    districts.iloc[:,62].drop([10]),
    'Grade 6':districts.iloc[:,83].drop([10])/\
    districts.iloc[:,82].drop([10]),
    'Grade 7':districts.iloc[:,103].drop([10])/\
    districts.iloc[:,102].drop([10]),
    'Grade 8':districts.iloc[:,123].drop([10])/\
    districts.iloc[:,122].drop([10])})
math4s_ratio=pd.DataFrame({
    'Grade 3':districts.iloc[:,33].drop([10])/\
    districts.iloc[:,32].drop([10]),
    'Grade 4':districts.iloc[:,53].drop([10])/\
    districts.iloc[:,52].drop([10]),
    'Grade 5':districts.iloc[:,73].drop([10])/\
    districts.iloc[:,72].drop([10]),
    'Grade 6':districts.iloc[:,93].drop([10])/\
    districts.iloc[:,92].drop([10]),
    'Grade 7':districts.iloc[:,113].drop([10])/\
    districts.iloc[:,112].drop([10]),
    'Grade 8':districts.iloc[:,133].drop([10])/\
    districts.iloc[:,132].drop([10])})


# In[ ]:


list(districts.columns[[22,23]]),[ela4s_ratio.shape,math4s_ratio.shape]


# In[ ]:


cmap=cm.get_cmap('Spectral',6)
spectral_cmap=[]
for i in range(cmap.N):
    rgb=cmap(i)[:3]
    spectral_cmap.append(mcolors.rgb2hex(rgb))   
grade_list=['Grade 3','Grade 4','Grade 5',
            'Grade 6','Grade 7','Grade 8']
fig,ax=plt.subplots(nrows=2,figsize=(12,12))
for i in range(6):
    ax[1].plot(range(1,33),ela4s_ratio.iloc[:,i],'-o',
               label=grade_list[i],c=spectral_cmap[i])
    ax[0].plot(range(1,33),math4s_ratio.iloc[:,i],'-o',
               label=grade_list[i],c=spectral_cmap[i]) 
for i in range(2):
    ax[i].legend() 
    ax[i].set_xticks(list(range(1,33)))
    ax[i].tick_params('x',rotation=0) 
ax[0].set_title('Math 4s Ratio by Districts')
ax[1].set_title('ELA 4s Ratio by Districts');


# In[ ]:


list(d5_shsat_2016.iloc[:,[20,40,60,80,100,120]].columns)


# In[ ]:


ela4s_ratio2=pd.DataFrame({
    'Grade 3':d5_shsat_2016.iloc[:,20]/d5_shsat_2016.iloc[:,19],
    'Grade 4':d5_shsat_2016.iloc[:,40]/d5_shsat_2016.iloc[:,39],
    'Grade 5':d5_shsat_2016.iloc[:,60]/d5_shsat_2016.iloc[:,59],
    'Grade 6':d5_shsat_2016.iloc[:,80]/d5_shsat_2016.iloc[:,79],
    'Grade 7':d5_shsat_2016.iloc[:,100]/d5_shsat_2016.iloc[:,99],
    'Grade 8':d5_shsat_2016.iloc[:,120]/d5_shsat_2016.iloc[:,119]})
math4s_ratio2=pd.DataFrame({
    'Grade 3':d5_shsat_2016.iloc[:,30]/d5_shsat_2016.iloc[:,29],
    'Grade 4':d5_shsat_2016.iloc[:,50]/d5_shsat_2016.iloc[:,49],
    'Grade 5':d5_shsat_2016.iloc[:,70]/d5_shsat_2016.iloc[:,69],
    'Grade 6':d5_shsat_2016.iloc[:,90]/d5_shsat_2016.iloc[:,89],
    'Grade 7':d5_shsat_2016.iloc[:,110]/d5_shsat_2016.iloc[:,109],
    'Grade 8':d5_shsat_2016.iloc[:,130]/d5_shsat_2016.iloc[:,129]})
ela4s_ratio2=ela4s_ratio2.fillna(0)
math4s_ratio2=math4s_ratio2.fillna(0)


# In[ ]:


ela4s_ratio2.shape,math4s_ratio2.shape


# In[ ]:


fig,ax=plt.subplots(nrows=2,figsize=(12,12))
for i in range(6):
    ax[1].plot(range(1,22),ela4s_ratio2.iloc[:,i],'-o',
               label=grade_list[i],c=spectral_cmap[i])
    ax[0].plot(range(1,22),math4s_ratio2.iloc[:,i],'-o',
               label=grade_list[i],c=spectral_cmap[i])   
for i in range(2):
    ax[i].legend(); ax[i].set_xticks(range(1,22))
    ax[i].set_xticklabels(list(d5_shsat_2016['Location Code']),
                          rotation=90)
    ax[i].vlines(8,0,1,linestyle="dotted",color='#fd0e35') 
    ax[i].vlines(14,0,1,linestyle="dotted",color='#0066ff')
    ax[i].vlines(17,0,1,linestyle="dotted",color='#fc74fd')    
ax[0].set_title('Math 4s by Schools in the 5th District') 
ax[1].set_title('ELA 4s by Schools in the 5th District');


# During assessing the ratio of the number of students with excellent scores to the number of tested students, the same leaders are identified among schools.

# #### Discipline

# In[ ]:


fig,ax=plt.subplots(1,figsize=(12,7))
ax.scatter(range(1,33),districts['Student Attendance Rate'].drop([10]),
           marker='*',s=100,label='Student Attendance Rateh',c='#3636ff')
y=100-districts['Percent of Students Chronically Absent'].drop([10])
ax.scatter(range(1,33),y,marker='*',s=100,c='#ff3636',
           label='Percent of Students Not Chronically Absent')
plt.vlines(range(1,33),y,districts['Student Attendance Rate'].drop([10]),
           linestyle="dotted")
ax.legend(); ax.set_xticks(list(range(1,33)))
ax.tick_params('x',rotation=0)
plt.title('Student Attendance Rate & Percent of '+          'Students Not Chronically Absent by Districts');


# In[ ]:


fig,ax=plt.subplots(1,figsize=(12,8))
ax.scatter(range(1,22),d5_shsat_2016['Student Attendance Rate'],
           marker='*',s=took*500,label='Student Attendance Rate',c='#3636ff')
y=100-d5_shsat_2016['Percent of Students Chronically Absent']
ax.scatter(range(1,22),y,c='#ff3636',marker='*',s=took*500,
           label='Percent of Students Not Chronically Absent')
y=100-d5_shsat_2016['Percent of Students Chronically Absent']
plt.vlines(range(1,22),y,d5_shsat_2016['Student Attendance Rate'],
           linestyle="dotted")
ax.scatter([8,14,17],d5_shsat_2016['Student Attendance Rate'][[7,13,16]],
           label='Leaders',s=500,facecolors='none',
           edgecolors=['#fd0e35','#0066ff','#fc74fd'])
ax.legend(loc=3); ax.set_xticks(range(1,22))
ax.set_xticklabels(list(d5_shsat_2016['Location Code']),rotation=90)
plt.title('Student Attendance Rate & Percent of Students Not '+          'Chronically Absent by Schools in the 5th District');


# Three leaders among schools also have a good level of attendance.
# 
# The outlier point looks like a mistake in the dataset.

# #### 4S Results of the Economically Disadvantaged Category among All 4S Results

# In[ ]:


ela4s_ratio_ed=pd.DataFrame({
    'Grade 3':districts.iloc[:,31].drop([10])/\
    districts.iloc[:,23].drop([10]),
    'Grade 4':districts.iloc[:,51].drop([10])/\
    districts.iloc[:,43].drop([10]),
    'Grade 5':districts.iloc[:,71].drop([10])/\
    districts.iloc[:,63].drop([10]),
    'Grade 6':districts.iloc[:,91].drop([10])/\
    districts.iloc[:,83].drop([10]),
    'Grade 7':districts.iloc[:,111].drop([10])/\
    districts.iloc[:,103].drop([10]),
    'Grade 8':districts.iloc[:,131].drop([10])/\
    districts.iloc[:,123].drop([10])})
math4s_ratio_ed=pd.DataFrame({
    'Grade 3':districts.iloc[:,41].drop([10])/\
    districts.iloc[:,33].drop([10]),
    'Grade 4':districts.iloc[:,61].drop([10])/\
    districts.iloc[:,53].drop([10]),
    'Grade 5':districts.iloc[:,81].drop([10])/\
    districts.iloc[:,73].drop([10]),
    'Grade 6':districts.iloc[:,101].drop([10])/\
    districts.iloc[:,93].drop([10]),
    'Grade 7':districts.iloc[:,121].drop([10])/\
    districts.iloc[:,113].drop([10]),
    'Grade 8':districts.iloc[:,141].drop([10])/\
    districts.iloc[:,133].drop([10])})


# In[ ]:


ela4s_ratio_ed.shape,math4s_ratio_ed.shape


# In[ ]:


fig,ax=plt.subplots(nrows=2,figsize=(12,12))
ela4s_ratio_ed.plot.bar(ax=ax[1],cmap=cm.Spectral)
math4s_ratio_ed.plot.bar(ax=ax[0],cmap=cm.Spectral); 
for i in range(2):
    ax[i].legend(loc=10,bbox_to_anchor=(1.1,.5))
    ax[i].set_xticklabels(list(range(1,33)))
    ax[i].tick_params('x',rotation=0)    
ax[0].set_title('Math 4s Ratio of Economically Disadvantaged by Districts') 
ax[1].set_title('ELA 4s Ratio of Economically Disadvantaged by Districts');


# In[ ]:


ela4s_ratio_ed2=pd.DataFrame({
    'Grade 3':d5_shsat_2016.iloc[:,28]/d5_shsat_2016.iloc[:,20],
    'Grade 4':d5_shsat_2016.iloc[:,48]/d5_shsat_2016.iloc[:,40],
    'Grade 5':d5_shsat_2016.iloc[:,68]/d5_shsat_2016.iloc[:,60],
    'Grade 6':d5_shsat_2016.iloc[:,88]/d5_shsat_2016.iloc[:,80],
    'Grade 7':d5_shsat_2016.iloc[:,108]/d5_shsat_2016.iloc[:,100],
    'Grade 8':d5_shsat_2016.iloc[:,128]/d5_shsat_2016.iloc[:,120]})
math4s_ratio_ed2=pd.DataFrame({
    'Grade 3':d5_shsat_2016.iloc[:,38]/d5_shsat_2016.iloc[:,30],
    'Grade 4':d5_shsat_2016.iloc[:,58]/d5_shsat_2016.iloc[:,50],
    'Grade 5':d5_shsat_2016.iloc[:,78]/d5_shsat_2016.iloc[:,70],
    'Grade 6':d5_shsat_2016.iloc[:,98]/d5_shsat_2016.iloc[:,90],
    'Grade 7':d5_shsat_2016.iloc[:,118]/d5_shsat_2016.iloc[:,110],
    'Grade 8':d5_shsat_2016.iloc[:,138]/d5_shsat_2016.iloc[:,130]})
ela4s_ratio_ed2=ela4s_ratio_ed2.fillna(0)
math4s_ratio_ed2=math4s_ratio_ed2.fillna(0)


# In[ ]:


ela4s_ratio_ed2.shape,math4s_ratio_ed2.shape


# In[ ]:


fig,ax=plt.subplots(nrows=2,figsize=(12,12))
math4s_ratio_ed2.plot.bar(ax=ax[0],cmap=cm.Spectral)
ela4s_ratio_ed2.plot.bar(ax=ax[1],cmap=cm.Spectral)
for i in range(2):
    ax[i].legend(loc=10,bbox_to_anchor=(1.1,.5))
    ax[i].set_xticks(range(1,22))
    ax[i].set_xticklabels(list(d5_shsat_2016['Location Code']),rotation=90)
    ax[i].vlines(7,0,1,linestyle="dotted",color='#fd0e35')
    ax[i].vlines(13,0,1,linestyle="dotted",color='#0066ff')
    ax[i].vlines(16,0,1,linestyle="dotted",color='#fc74fd') 
ax[0].set_title('Math 4s Ratio of Economically Disadvantaged'+                ' by Schools in the 5th District') 
ax[1].set_title('ELA 4s Ratio of Economically Disadvantaged'+                ' by Schools in the 5th District');


# Two of three mentioned leaders has high test results of all levels for this category of students. 

# #### 4S Results of the Limited English Proficient Category among All 4S Results

# In[ ]:


ela4s_ratio_le=pd.DataFrame({
    'Grade 3':districts.iloc[:,30].drop([10])/\
    districts.iloc[:,23].drop([10]),
    'Grade 4':districts.iloc[:,50].drop([10])/\
    districts.iloc[:,43].drop([10]),
    'Grade 5':districts.iloc[:,70].drop([10])/\
    districts.iloc[:,63].drop([10]),
    'Grade 6':districts.iloc[:,90].drop([10])/\
    districts.iloc[:,83].drop([10]),
    'Grade 7':districts.iloc[:,110].drop([10])/\
    districts.iloc[:,103].drop([10]),
    'Grade 8':districts.iloc[:,130].drop([10])/\
    districts.iloc[:,123].drop([10])})
math4s_ratio_le = pd.DataFrame({
    'Grade 3':districts.iloc[:,40].drop([10])/\
    districts.iloc[:,33].drop([10]),
    'Grade 4':districts.iloc[:,60].drop([10])/\
    districts.iloc[:,53].drop([10]),
    'Grade 5':districts.iloc[:,80].drop([10])/\
    districts.iloc[:,73].drop([10]),
    'Grade 6':districts.iloc[:,100].drop([10])/\
    districts.iloc[:,93].drop([10]),
    'Grade 7':districts.iloc[:,120].drop([10])/\
    districts.iloc[:,113].drop([10]),
    'Grade 8':districts.iloc[:,140].drop([10])/\
    districts.iloc[:,133].drop([10])})


# In[ ]:


ela4s_ratio_le.shape,math4s_ratio_le.shape


# In[ ]:


fig,ax=plt.subplots(nrows=2,figsize=(12,12))
math4s_ratio_le.plot.bar(ax=ax[0],cmap=cm.Spectral)
ela4s_ratio_le.plot.bar(ax=ax[1],cmap=cm.Spectral)
for i in range(2):
    ax[i].legend(loc=10,bbox_to_anchor=(1.1,.5))
    ax[i].set_xticklabels(list(range(1,33)))
    ax[i].tick_params('x',rotation=0)    
ax[0].set_title('Math 4s Ratio of Limited English Proficient by Districts') 
ax[1].set_title('ELA 4s Ratio of Limited English Proficient by Districts');


# In[ ]:


ela4s_ratio_le2=pd.DataFrame({
    'Grade 3':d5_shsat_2016.iloc[:,27]/d5_shsat_2016.iloc[:,20],
    'Grade 4':d5_shsat_2016.iloc[:,47]/d5_shsat_2016.iloc[:,40],
    'Grade 5':d5_shsat_2016.iloc[:,67]/d5_shsat_2016.iloc[:,60],
    'Grade 6':d5_shsat_2016.iloc[:,87]/d5_shsat_2016.iloc[:,80],
    'Grade 7':d5_shsat_2016.iloc[:,107]/d5_shsat_2016.iloc[:,100],
    'Grade 8':d5_shsat_2016.iloc[:,127]/d5_shsat_2016.iloc[:,120]})
math4s_ratio_le2=pd.DataFrame({
    'Grade 3':d5_shsat_2016.iloc[:,37]/d5_shsat_2016.iloc[:,30],
    'Grade 4':d5_shsat_2016.iloc[:,57]/d5_shsat_2016.iloc[:,50],
    'Grade 5':d5_shsat_2016.iloc[:,77]/d5_shsat_2016.iloc[:,70],
    'Grade 6':d5_shsat_2016.iloc[:,97]/d5_shsat_2016.iloc[:,90],
    'Grade 7':d5_shsat_2016.iloc[:,117]/d5_shsat_2016.iloc[:,110],
    'Grade 8':d5_shsat_2016.iloc[:,137]/d5_shsat_2016.iloc[:,130]})
ela4s_ratio_le2 = ela4s_ratio_le2.fillna(0)
math4s_ratio_le2 = math4s_ratio_le2.fillna(0)


# In[ ]:


ela4s_ratio_le2.shape,math4s_ratio_le2.shape


# In[ ]:


fig,ax=plt.subplots(nrows=2,figsize=(12,12))
math4s_ratio_le2.plot.bar(ax=ax[0],cmap=cm.Spectral)
ela4s_ratio_le2.plot.bar(ax=ax[1],cmap=cm.Spectral)
for i in range(2):
    ax[i].legend(loc=10,bbox_to_anchor=(1.1,.5))
    ax[i].set_xticks(list(range(1,22)))
    ax[i].set_xticklabels(list(d5_shsat_2016['Location Code']),rotation=90)
    ax[i].vlines(7,0,1,linestyle="dotted",color='#fd0e35')
    ax[i].vlines(13,0,1,linestyle="dotted",color='#0066ff')
    ax[i].vlines(16,0,1,linestyle="dotted",color='#fc74fd')  
ax[0].set_title('Math 4s Ratio of Limited English Proficient by Schools in the 5th District') 
ax[1].set_title('ELA 4s Ratio of Limited English Proficient by Schools in the 5th District');


# A huge difference is visible in the results of two types of testing (Math & ELA) for this category. 
# 
# In addition, and the 5th District is characterized by a small number of successful students in this category.
# 
# At the same time, it looks like one of the leaders has a great experience for ELL students. 

# <h1 class='font-effect-3d' style='color:#37c9e1; font-family:Orbitron;'>&#x1F310; &nbsp; Economic Indicators</h1>

# In[ ]:


fig,ax=plt.subplots(1,figsize=(12,12))
school_explorer.plot(kind='scatter',x='Longitude',y='Latitude',ax=ax,
                     s=20,c='Economic Need Index',cmap=cm.bwr)
ax.add_patch(PolygonPatch(districts.geometry[4],fc='none',ec='b',zorder=2 ))
districts.plot(ax=ax,column='Percent of Students Chronically Absent', 
               cmap='Greys',alpha=.3,edgecolor='darkslategray')
plt.title('Average Percentage of Chronically Absent Students '+          'by Districts and Economic Need Index by Schools');


# In[ ]:


fig=plt.figure(figsize=(9,9)) ; ax=fig.gca() 
ax.add_patch(PolygonPatch(districts.geometry[4],
                          fc='none',ec='slategray', 
                          alpha=0.5,zorder=2 ))
d5_shsat_2016.plot(kind='scatter',x='Longitude',y='Latitude',ax=ax,
                   s=30,c='Economic Need Index',cmap=cm.bwr)
ax.scatter(d5_shsat_2016['Longitude'][[7,13,16]],
           d5_shsat_2016['Latitude'][[7,13,16]],s=200,
           facecolors='none',edgecolors=['#fd0e35','#0066ff','#fc74fd'])
ax.text(-73.965,40.8377,
        'For the 5th District Average Economic Need Index = '+\
        str(round(districts['Economic Need Index'][4],4)))
plt.title('Economic Need Index by Schools in the 5th District');


# In[ ]:


def highlight(df):
    if df['Economic Need Index']<.6:
        return ['background-color: #fd0e35']
    elif df['Economic Need Index']>.79:
        return ['background-color: #0066ff']
    else:
        return ['background-color: #fc74fd']
pd.DataFrame(d5_shsat_2016['Economic Need Index'].loc[[7,13,16]]).set_index([d5_shsat_2016['Location Code'].loc[[7,13,16]]]).style.apply(highlight, axis=1)


# Three leaders among schools in the 5th District have the certain difference with another one in the economic situation.

# <h1 class='font-effect-3d' style='color:#37c9e1; font-family:Orbitron;'>&#x1F310; &nbsp; Tendency 2010 -> 2012 -> 2016</h1>
# #### Usage the Same Systems of Measuring

# In[ ]:


fig,ax=plt.subplots(nrows=3,figsize=(12,18))
plt.suptitle('Tendencies in Education Indicators 2010 -> '+             '2012 for Schools in Two Datasets',fontsize=15)
tend_list1=['Critical Reading Mean 2010',
            'SAT Critical Reading Avg. Score 2012']
tend_list2=['Mathematics Mean 2010',
            'SAT Math Avg. Score 2012']
tend_list3=['Writing Mean 2010',
            'SAT Writing Avg. Score 2012']
res_2010_2012[tend_list1].plot(ax=ax[0],color=['#3636ff','#ff3636'])
res_2010_2012[tend_list2].plot(ax=ax[1],color=['#3636ff','#ff3636'])
res_2010_2012[tend_list3].plot(ax=ax[2],color=['#3636ff','#ff3636']);


# Such a graphic comparison allows identifying deterioration or improvement in the indicators of the region in general and in each individual school.

# #### Usage the Different Systems of Measuring

# In[ ]:


fig=plt.figure(figsize=(12,10)); ax=fig.gca()
plt.suptitle('Tendencies in Education Indicators 2010 -> 2012'+             ' -> 2016 for Schools in Three Datasets',fontsize=20)
tend_list=['Critical Reading Mean 2010',
           'Writing Mean 2010','Mathematics Mean 2010',
           'SAT Critical Reading Avg. Score 2012',
           'SAT Writing Avg. Score 2012',
           'SAT Math Avg. Score 2012',
           'Average ELA Proficiency 2016',
           'Average Math Proficiency 2016']
norm_res_2010_2012_2016[tend_list].plot(ax=ax,cmap=cm.Spectral)
ax.legend(loc=10,bbox_to_anchor=(1.2,.5));


# When the indicators and the measuring system were transformed, the comparison can be made by scaling on the segment [0;1]. In this case, it is possible to assess how the success of each particular school has changed with respect to the situation in the region or in the district.
# 
# On the graph, we can see the tendency in 2016 moving down in scaled indicators. It needs to find out is it the real failure or it's just because of changing the measuring system.

# <h1 class='font-effect-3d' style='color:#37c9e1; font-family:Orbitron;'>&#x1F310; &nbsp;  Let's Go Ahead</h1>
# It' s time to move to the next step.
# 
# &#x1F4D8; &nbsp; [PASSNYC. Regression Methods](https://www.kaggle.com/olgabelitskaya/passnyc-regression-methods)
