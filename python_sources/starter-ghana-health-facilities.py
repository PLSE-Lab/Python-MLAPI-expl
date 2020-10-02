#!/usr/bin/env python
# coding: utf-8

# <img width=300 height=300 src="https://github.com/ds4good/ghana-datasets/raw/master/doc/image/sdg3.PNG"></img>

# ## Introduction
# As part our [Citizen Data Science Project](https://ds4good.github.io/ghana-datasets/) in Ghana we are publishing and maintening a currated Open dataset on Ghana for the community to apply to social good causes in support for the SDGs. The SDGs provide a working definition of social good and will serve as a guide in our work to harness the power of data for social good. 
# 
# We are hoping this effort will encourage government agencies to publish more open data and leverage external expertise to prioritise and acelerate progress. You are welcome to participate. [Slack](https://join.slack.com/t/ds4good/shared_invite/enQtNDE4NTkxMzI1NDQ2LWJkNDI0ZDEyODY5NDkyNjg3NWI1NmJkNTg0ZmM1MGI1ODUxNGY3MmNjNWVjNzBhMzNlODc3ZjE2YjVkYjFmM2Y)
# 
# We think that this will foster new collaboration between hospitals in close proximity with each other and international partners. (Social) Enterpreneurs in the medical space could also use this as research data.
# 
# This starter-notebook gets you started with Ghana's Health facilities dataset (source: data.gov.gh).
# 
# ### Call to Action - Domain Experts (Public Health Officials)
# This data, in combination with other datasets could provide a means of understanding **Healthcare access** in Ghana. Highlighting deprived areas for prioritisation. It also forms a good bases for understanding Ghana's Health Infrastructure. To this end we, are calling on **Domain Experts (Public Health directors/leaders)** to provide direction (what questions are worth answering?) and additional open data to continue to improve our understanding of Ghana's health infrastructure and disaster readiness.
# 
# Areas of you can help include, 1) we think that health facility are/should have a hierachical classification(e.g. Korle-bu is highest ranked facility) so that we can begin to measure **Access per Population** and thus propose the most valuable places to have new facilities. 2) **support the publishing of more Open Data for richer understanding and data-driven decisions.** 3) **Provide the questions that are valuable to answer**
# 
# ### Call to Action - Data professionals/enthusiats
# This starter kernel (notebook) provides a quick and dirty understanding of the dataset. Put your skills to use and come up with your own discoveries to make the world a better place. e.g are there areas whose healthcare needs should be prioritised and improved? 2) Which other dataset will be useful.
# 
# ### Observations from Exploratory Analysis
# - A **total of 3756** health facilities were listed in the dataset.
# - Clinics are the most common of the *25 listed types of facilities - **1171 (31%)**
# - Majory (58 %) of facilities are government owned. **(2210 out of 3756)**
# - a total of **171** districts were listed. This implies some districts are missing.
# - There were 12 types of ownerships, but the most common Three are Government, Private, and Quasi-Government.
# - A couple of errors in the dataset [ gps locations, record deduplication for Clinic/clinic...]
# 
# ### Questions Worth Exploring further
# - At the last election Ghana listed 275 districts, in this Health Facilities dataset there are 171 districts, are there any districts missing that require attention?
# - Compute a measure **access per population** that will enable highlighting of deprived areas for, decision makers prioritise where to set-up new facilities or up-grade existing facilities.
# 
# ### Call for data
# We will be glad if below data and more are plublished as open data
# - Ghana health facility hierachy/classification
# - Facility Capacities
# - (Maternal) Mortality by facility
# 
# 
# *- info requires correction.
# 
# By: [easimadi](https://www.linkedin.com/in/easimadi/)  
# Date: 08-Aug-2018  
# Acknowledgement: [data.gov.gh](data.gov.gh), [datanix](https://www.datanix.co.uk/blog), [iipgh](https://www.iipgh.org/)  
# 

# ## Content
# 1. <a href="#load"> Loading and Inspecting the dataset</a>
# 2. <a href="#explore">Exploratory Analysis</a>   
# <a href="#e1"> 2.1 How many health facilities are in Ghana & their types? </a>     
# <a href="#e2"> 2.2 Region based analysis: What type of facilities are in each region?</a>  
# <a href="#e3"> 2.3 Region based analysis: What is the ownership structure in the regions? </a>    
# <a href="#e4"> 2.4 Ownership: What type facilities to the different owners typical own?</a>   
# <a href="#e5"> 2.5 See facilities on a Map</a>    
# <a href="#e6"> 2.6 District based analysis? Are there any districts without health facilities, are there deprived populations? etc Help us find answers </a>  

# ## <a id="load"> 1. Loading & Inspecting the dataset </a>

# In[ ]:


#read and inspect the dataframe
import pandas as pd
df_health_facilities =  pd.read_csv("../input/health-facilities-gh.csv")
df_health_facilities.head()


# In[ ]:


# quick summary of the dataset filtered to show only relevant data.
# freq: the frequency of the top item.
df_health_facilities.describe(include = "all")[["Region","District","FacilityName","Type","Town","Ownership"]].iloc[0:4]


# ### Summary
# So with a single ```df.describe``` command we know the following about the data.
# * Number of health facilities: **3756**
# * Number of Regions reported: **10** (_good news :) there are 10 regions in Ghana_)
# * Number of Districts reported: **171** (_ it seems there are some districts to represented_)
# * Types of health facilities in Ghana: **25**, Clinics are most common types of health facility **(1171)**
# * Unique health facility Owners: **12**, Government owns the most heath facilities in Ghana **(2210 out of 3756)**
# * Rabito Clinic is most common health facility in Ghana.
# 
# Hopefully you understand how to interpret [Pandas Describe](https://www.kaggle.com/learn/pandas) command.
# 

# # <a id="explore"> 2. Exporatory Analysis </a>

# ### <a id="e1"> 2.1 How many health facilities are in Ghana and their types?</a>  
# From previous section we know the were **3756 health facilities** listed and as suspected earlier, **clinics (1171)** are the most common health facility in Ghana.  
# CHPS (pronounced `chips`) compounds which became well-known because previous [administration](http://www.peacefmonline.com/pages/politics/politics/201708/323564.php) gave up 10% of their renumeration to build more, is the 3 most common health facility.  
# The promise was to have 2000 CHPS coumpounds, the current listing is 652.
# 
# 
# The data might need a bit of cleaning can you sport them? - clue: actually there are 1173 clinics...
# 
# CHPS - Community Health Programme and Service

# In[ ]:


df = df_health_facilities.groupby("Type").count()[["Region"]].reset_index()

df.columns = ["Type","count"]
df = df.sort_values(by="count",ascending=False)
df.head(5)


# In[ ]:


from bokeh.plotting import figure, output_notebook, show
from bokeh.models import ColumnDataSource, LabelSet
import math

output_notebook()

#plotting with bokeh... api is quite functional needs a bit of getting used-to. 
#for standard visualisations you are more productive using Tableau, Qlik, PowerBI etc.
source = df
plot = figure(plot_width=800, plot_height=600,x_range=df['Type'][0:20],title="Number Of Health Facilities",tools=["xwheel_zoom","reset","pan"])
plot.vbar(source=ColumnDataSource(source), x="Type",width=0.8, top="count",color="cornflowerblue")

labels = LabelSet(x='Type', y='count', text="count", x_offset=-8,
                  source=ColumnDataSource(source),text_font_size="8pt", text_color="#555555",
                   text_align='left')


plot.add_layout(labels)
plot.xaxis.major_label_orientation = math.pi/2
show(plot)


# ## <a id="e2">2.2 Region based analysis </a>
# 
# ### 2.2.1 What are the types of facilities in each region?

# In[ ]:


df_region_by_type = df_health_facilities[["Region","Type","FacilityName"]].groupby(["Region","Type"], as_index=False).count()
df_region_by_type.columns = ["Region","Type","FacilityCount"]

#shaping the data
df_region_by_type_pivot = df_region_by_type.pivot("Region","Type","FacilityCount").fillna(0)
df_region_by_type_pivot["total"] = df_region_by_type_pivot.sum(axis=1)
df_region_by_type_pivot = df_region_by_type_pivot.sort_values("total",ascending=False)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

plt.figure(figsize=(20,6))

plt.yticks(rotation=0)
plt.title("Regions Vs Type of Facilities")
sns.heatmap(df_region_by_type_pivot,annot=True,fmt="0.0f",cmap="RdGy",linewidth=.02)


# ## 2.2.2 <a id="e3">What is the ownership structure like in the regions?<a>

# In[ ]:


# Shaping the data.
df_region_by_owner = df_health_facilities[["Region","Ownership","FacilityName"]].groupby(["Region","Ownership"], as_index=False).count()
df_region_by_owner.columns = ["Region","Ownership","FacilityCount"]
df_region_by_owner_pivot = df_region_by_owner.pivot("Region","Ownership","FacilityCount").fillna(0)
df_region_by_owner_pivot["total"] = df_region_by_owner_pivot.sum(axis=1)
df_region_by_owner_pivot = df_region_by_owner_pivot.sort_values("total",ascending=False)

plt.figure(figsize=(20,6))
plt.yticks(rotation=0)
plt.title("Regions Vs Ownership of Facilities")
sns.heatmap(df_region_by_owner_pivot,annot=True,fmt="0.0f",cmap="RdGy",linewidth=.02)


# ## 2.2.3 <a id="e4">Do owners have preference for a particular Falicity?<a>

# In[ ]:


df_owner_by_type = df_health_facilities[["Ownership","Type","FacilityName"]].groupby(["Ownership","Type"], as_index=False).count()
df_owner_by_type.columns = ["Ownership","Type","FacilityCount"]
df_owner_by_type_pivot = df_owner_by_type.pivot("Ownership","Type","FacilityCount").fillna(0)
df_owner_by_type_pivot["total"] = df_owner_by_type_pivot.sum(axis=1)
df_owner_by_type_pivot = df_owner_by_type_pivot.sort_values("total",ascending=False)


plt.figure(figsize=(20,6))

plt.yticks(rotation=0)
plt.title("Regions Vs Ownership of Facilities")
sns.heatmap(df_owner_by_type_pivot,annot=True,fmt="0.0f",cmap="RdGy",linewidth=.02)


# In[ ]:


from IPython.core.display import display, HTML

display(HTML("""
<style>
#viz1534357692001 {
    height: 800px;
    width: 1000px;
}
</style>
<h2 id="e5"> Explore Ghana Health Facilites Dataset (<a href="https://public.tableau.com/profile/datanix.ds4good#!/vizhome/ghanahealthinfrastructure/Dashboard1?publish=yes">click to view tableau</a>) </h2>
<div class='tableauPlaceholder' id='viz1534357692001' style='position: relative'><noscript><a href='#'>
<img alt='Dashboard 1 ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;QS&#47;QSTMHFD4J&#47;1_rss.png' style='border: none' />
</a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> 
<param name='embed_code_version' value='3' /> <param name='path' value='shared&#47;QSTMHFD4J' /> <param name='toolbar' value='yes' />
<param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;QS&#47;QSTMHFD4J&#47;1.png' /> 
<param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' />
<param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='filter' value='publish=yes' /></object>
</div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1534357692001');                    var vizElement = divElement.getElementsByTagName('object')[0];                    
vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    
var scriptElement = document.createElement('script');                    
scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    
vizElement.parentNode.insertBefore(scriptElement, vizElement);                
</script>"""))

