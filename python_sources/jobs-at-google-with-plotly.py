#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.graph_objs as go
import plotly.offline as po
po.init_notebook_mode(connected=True)


# In[ ]:


df  = pd.read_csv("../input/job_skills.csv")


# In[ ]:


df.head()


# In[ ]:


df = df.drop("Company",axis=1)


# In[ ]:


df.shape


# # Number of jobs in each Category 

# In[ ]:


freq_category = [list(df["Category"]).count(i) for i in list(set(df["Category"]))]


# In[ ]:


barplot_category = [go.Bar(
    x = list(set(df["Category"])),
    y = freq_category   
)]
layout = go.Layout(
    margin = dict(
        b=140
        )
)
fig_category = go.Figure(data=barplot_category,layout=layout  ) 


# In[ ]:


po.iplot(fig_category, config={'showLink': False})


# In[ ]:


df["Minimum Qualifications"] = df["Minimum Qualifications"].fillna("Not available")


# # Plotting Minimum qualifications based on their demand 

# In[ ]:


# df["Qualfications"] = df["Minimum Qualifications"].apply(lambda x: x.split("\n").strip() if x is not "Not available" else "Not available")


# In[ ]:


# df["Degree"] = df["Minimum Qualifications"].apply(lambda x: x.split("degree")[0].strip() if x is not "Not available" else "")


# In[ ]:


# df["Degree"][3]


# In[ ]:


# freq_qualification = [list(df["Degree"]).count(i) for i in list(set(df["Degree"]))]


# In[ ]:


# barplot_qualification = [go.Bar(
#     x = list(df["Degree"]),
#     y = freq_qualification   
# )]


# In[ ]:


# df["Minimum Qualifications"][0]


# In[ ]:


# po.iplot(barplot_qualification, config={'showLink': False})


# # Cities with most demand

# **Lets first split the Location to city and its corresponding country in different columns**

# In[ ]:


df = df.sort_values("Location")


# In[ ]:


df["Cities"] = df["Location"].apply(lambda x: x.split(",")[0].strip() if len(x.split(","))>1 else "N.A." )


# In[ ]:


print(list(set((df["Cities"]))))


# In[ ]:


freq_cities = [list(df["Cities"]).count(i) for i in sorted(list(set(df["Cities"])))]


# In[ ]:


barplot_cities = [go.Scatter(
    x = sorted(list(set(df["Cities"]))),
    y = freq_cities,
    mode = "lines+markers" 
)]


# In[ ]:


po.iplot(barplot_cities, config={'showLink': False})


# **Unsurprisingly US cities have the highest demand. But what countries other than United States is Google hiring a lot?**

# # Countries with high demand other than United States

# In[ ]:


df["Countries"] = df["Location"].apply(lambda x: x.split(",")[len(x.split(","))-1].strip() if len(x.split(","))>1 else x.split(",")[0].strip())


# In[ ]:


# list((df["Countries"])).remove("United States")
list_of_countries = list(set(df["Countries"]))
list_of_countries.remove("USA")
list_of_countries.remove("United States")
print(list_of_countries)


# In[ ]:


freq_countries = [list(df["Countries"]).count(i) for i in sorted(list_of_countries)]


# In[ ]:


barplot_countries = [go.Scatter(
    x = sorted(list_of_countries),
    y = freq_countries,
    mode = "lines+markers" ,
    hoverinfo = 'y',
)]
layout = go.Layout(
    margin = dict(
    b=130
    ))
fig_countries = go.Figure(data=barplot_countries, layout=layout)


# In[ ]:


po.iplot(fig_countries, config={'showLink': False})


# **Ireland huh. Who would have thought?**

# In[ ]:


it_jobs = ['Data Center & Network',
          'Hardware Engineering',
          'IT & Data Management',
          'Network Engineering',
          'Software Engineering',
          'Technical Infrastructure',
          'User Experience & Design'
         ] 


# In[ ]:


df = df.drop("Location",axis = 1)


# In[ ]:


df.head()


# In[ ]:


countries_with_it_jobs=[]
for i in range(len(df["Category"])):
    if df["Category"][i] in it_jobs:
        countries_with_it_jobs.append(df["Countries"][i])


# In[ ]:


freq_countries_it_jobs = [countries_with_it_jobs.count(i) for i in sorted(list(set(countries_with_it_jobs)))]


# In[ ]:


freq_countries_it_jobs


# In[ ]:


barplot_it_jobs = [go.Scatter(
    x = sorted(list(set(countries_with_it_jobs))),
    y = freq_countries_it_jobs,
    mode = "lines+markers" ,
    hoverinfo = 'y',
)]
layout = go.Layout(
    margin = dict(
    b=130
    ))
fig_countries_it_jobs = go.Figure(data=barplot_it_jobs, layout=layout)


# In[ ]:


po.iplot(fig_countries_it_jobs, config={'showLink': False})


# **Lets again omit United States and see**

# In[ ]:


countries_with_it_jobs1 = sorted(list(set(countries_with_it_jobs)))
countries_with_it_jobs1.remove("United States")
countries_with_it_jobs1


# In[ ]:


freq_countries_it_jobs = [countries_with_it_jobs.count(i) for i in sorted(list(set(countries_with_it_jobs1)))]
freq_countries_it_jobs


# In[ ]:


barplot_it_jobs = [go.Scatter(
    x = sorted(list(set(countries_with_it_jobs1))),
    y = freq_countries_it_jobs,
    mode = "lines+markers" ,
    hoverinfo = 'y',
)]
layout = go.Layout(
    margin = dict(
    b=130
    ))
fig_countries_it_jobs = go.Figure(data=barplot_it_jobs, layout=layout)


# In[ ]:


po.iplot(fig_countries_it_jobs, config={'showLink': False})


# **Taiwan. Nice.**

# **More to come** In the mean time check out this amazing [kernel](https://www.kaggle.com/justjun0321/way-to-google-get-a-job-in-goggle-word-cloud). 
