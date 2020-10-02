#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


import os,sys,re,time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from IPython.display import display, HTML
plt.style.use('ggplot') 
base_dir = "../input/"
file_list = ['hs4_eng.csv', 'country_eng.csv', 'hs9_eng.csv', 
             'ym_latest.csv', 'hs2_eng.csv', 'hs6_eng.csv']

ym_all = pd.read_csv(base_dir + "/ym_latest.csv",
                 dtype={'hs2':'str','month':'str',
                        'Country':'str','hs4':'str','hs6':'str','hs9':'str'} )

y_all = pd.read_csv(base_dir + "/year_latest.csv",
                 dtype={'hs2':'str',
                        'Country':'str','hs4':'str','hs6':'str','hs9':'str'} )

country = pd.read_csv(base_dir + "/country_eng.csv",
                     dtype={'Country':'str'})
country.index=country['Country']

hs2 = pd.read_csv(base_dir + "/hs2_eng.csv",
                     dtype={'hs2':'str'})
hs2.index=hs2['hs2']

hs4 = pd.read_csv(base_dir + "/hs4_eng.csv",
                     dtype={'hs4':'str'})
hs4.index=hs4['hs4']
hs6 = pd.read_csv(base_dir + "/hs6_eng.csv",
                     dtype={'hs6':'str'})
hs6.index=hs6['hs6']
hs9 = pd.read_csv(base_dir + "/hs9_eng.csv",
                     dtype={'hs9':'str'})
hs9.index=hs9['hs9']

x = ym_all[['Year','month']].sort_values(['Year','month']).tail(1)
last_month = x['month'].values[0]
last_ym_year = int(x['Year'].values[0])
x = y_all['Year'].copy()
start_year = x.sort_values().head(1).values[0]
last_year = x.sort_values().tail(1).values[0]


# In[ ]:


target_country = "304" # USA"
country_name = country[country['Country']==target_country]['Country_name'].values[0]
display(HTML('<h1>' + country_name + '</h1>'))
display(last_month + " : " +str(last_ym_year) + " " + str(last_year) + " " +str(start_year))


# In[ ]:


def show_simple_year_sum(y):
    # Just Year sum 
    x_y = y.groupby(["Year"],as_index=False)["VY"].sum()[["Year","VY"]]
    x_y.index= x_y['Year']
    x_y.plot.bar(y=['VY'] ,alpha=0.6, figsize=(12,5))


# In[ ]:


def simple_year_export_import(y):
   x_y_exp =y[y['exp_imp']==1].groupby(["Year"],as_index=False)["VY"].sum()[["Year","VY"]]
   x_y_exp.rename(columns={'VY': 'export'}, inplace=True)
   # re index
   x_y_exp.index = pd.RangeIndex(start=1,stop=len(x_y_exp)+1)
   # import
   x_y_imp =y[y['exp_imp']==2].groupby(["Year"],as_index=False)["VY"].sum()[["Year","VY"]]
   x_y_imp.rename(columns={'VY': 'import'}, inplace=True)
   # re index
   x_y_imp.index = pd.RangeIndex(start=1,stop=len(x_y_exp)+1)
   x_exp_imp = pd.concat([x_y_exp,x_y_imp['import']],axis=1) #axis=1  add columns
   x_exp_imp['Year'] = x_exp_imp['Year'].astype(str)
   x_exp_imp.index = x_exp_imp['Year'] 
   return(x_exp_imp)    


# In[ ]:


def show_country_year_export_import(y,target_country,start_year,last_year):
   y = y[(y['Country']==target_country) & (y['Year']>=start_year) & (y['Year']<=last_year)]
   x_exp_imp = simple_year_export_import(y)
   x_exp_imp.plot.bar(y=['export', 'import'] ,alpha=0.6, figsize=(12,5))
   country_name = country[country['Country']==target_country]['Country_name'].values[0]
   #dir(x_y.plot.bar)
   plt.title(country_name + "\n"+ ' export import ' + str(start_year) + "-" + str(last_year), size=20)


# In[ ]:


show_country_year_export_import(y_all,target_country,start_year,last_year)


# In[ ]:


def simple_year_hs(y,hs):
    x_y =y.groupby([hs],as_index=False)["VY"].sum()
    x_y = eval("pd.merge(x_y, " + hs + ", on='"+ hs + "')")
    x_y = x_y.sort_values(by='VY',ascending=False)[0:20]
    x_y = x_y.sort_values(by='VY')
    return(x_y)


# In[ ]:


def show_year_hs(y,exp_imp,hs):
    #  show hs code 
    if exp_imp == 1:
        ei = "export"
    if exp_imp == 2:
        ei = "import"
    x_y = y[(y['exp_imp']==exp_imp) & (y['exp_imp']==exp_imp)]
    x_y = simple_year_hs(x_y,hs)

    c ="x_y.plot.barh(y=['VY'] ,x=['" + hs + "','" + hs + "_name'],alpha=0.6, figsize=(12,5))"
    eval(c)
    #x_y.plot.barh(y=['VY'] ,x=[hs,'hs2_name'],alpha=0.6, figsize=(12,5))
    
    plt.title(hs + " " + ei, size=16)
    c ="x_y.sort_values(by='VY',ascending=False)[0:5][['" + hs + "','VY','" + hs + "_name']]"
    display(eval(c))
    #display(x_y.sort_values(by='VY',ascending=False)[0:5][['hs2','VY','hs2_name']])
    
    


# In[ ]:


def year_hs_country(y,exp_imp,target_year,hs,hs_code):
    if exp_imp == 1:
        ei = "export"
    if exp_imp == 2:
        ei = "import"     
    x_y = y[(y[hs]==hs_code) & (y["exp_imp"]==exp_imp)  &             (y["Year"]==target_year)].groupby(["Country"])["VY"].sum()
    x_y = x_y.to_frame()
    x_y["Country"]=x_y.index
    x_y = pd.merge(x_y,country,on='Country')
    x_sum = x_y["VY"].sum()
    x_y["percent"] = 100*(x_y["VY"]/x_sum)
    #hs_name = hs6[hs6[]==hs_code]
    c = hs + "["+hs+"['"+hs+"']==hs_code]['" + hs + "_name'].values[0]"
    hs_name = eval(c)
    print( ei + "  " +  hs_name)
    display(x_y.sort_values("VY",ascending=False).head())
    


# In[ ]:


y_last = y_all[y_all["Year"]==last_year]
y_country = y_all[y_all["Country"]==target_country]
y_country_last = y_last[y_last['Country']==target_country]


# In[ ]:


# hs2 export japan->usa  from last_year stats
show_year_hs(y_country_last,1,"hs2")


# In[ ]:


# hs2 87( car ) country ranking  
year_hs_country(y_all,1,last_year,"hs2","87")


# In[ ]:


x_y =simple_year_export_import(y_country[y_country["hs2"]=="87"])
x_y.plot.bar(y=['export', 'import'] ,alpha=0.6, figsize=(12,5))

#87  car export import 


# In[ ]:


x_y =simple_year_export_import(y_country[y_country["hs2"]=="84"])
x_y.plot.bar(y=['export', 'import'] ,alpha=0.6, figsize=(12,5))

#84  boiler turbines export import


# In[ ]:


x_y =simple_year_export_import(y_country[y_country["hs2"]=="85"])
x_y.plot.bar(y=['export', 'import'] ,alpha=0.6, figsize=(12,5))

#85 Electrical machinery and equipment and parts 


# In[ ]:


x_y =simple_year_export_import(y_country[y_country["hs2"]=="90"])
x_y.plot.bar(y=['export', 'import'] ,alpha=0.6, figsize=(12,5))
#	90		Optical photographic cinematographic


# In[ ]:


# hs2 90 optical hs4 break down 

show_year_hs(y_country_last[y_country_last['hs2']=="90"],2,"hs4")


# In[ ]:


x_y =simple_year_export_import(y_country[y_country["hs4"]=="9018"])
x_y.plot.bar(y=['export', 'import'] ,alpha=0.6, figsize=(12,5))
#	9018		Instruments_and_appliances_used_in_medical__su..


# In[ ]:


# hs4 9018  hs6 break down (import)
show_year_hs(y_country_last[y_country_last['hs4']=="9018"],2,"hs6")


# In[ ]:


#901890
x_y =simple_year_export_import(y_country[y_country["hs6"]=="901890"])
x_y.plot.bar(y=['export', 'import'] ,alpha=0.6, figsize=(12,5))
#	901890		Other_instruments_and_appliances


# In[ ]:


# hs4 901890  hs6 break down (import)
show_year_hs(y_country_last[y_country_last['hs6']=="901890"],2,"hs9")


# In[ ]:


x_y =simple_year_export_import(y_country[y_country["hs4"]=="9021"])
x_y.plot.bar(y=['export', 'import'] ,alpha=0.6, figsize=(12,5))
#	9021		Orthopaedic_appliances__including_crutches__su...


# In[ ]:


# hs4 9021  hs6 break down (import)
show_year_hs(y_country_last[y_country_last['hs4']=="9021"],2,"hs6")


# In[ ]:


x_y =simple_year_export_import(y_country[y_country["hs6"]=="902139"])
x_y.plot.bar(y=['export', 'import'] ,alpha=0.6, figsize=(12,5))
#	902139		Others.


# In[ ]:


# last year hs2 import
show_year_hs(y_country_last,2,"hs2")


# In[ ]:


year_hs_country(y_all,2,last_year,"hs2","84")


# In[ ]:


show_year_hs(y_country_last,1,"hs4")


# In[ ]:


year_hs_country(y_all,1,last_year,"hs4","8443")


# In[ ]:


x_y =simple_year_export_import(y_country[y_country["hs4"]=="8443"])
x_y.plot.bar(y=['export', 'import'] ,alpha=0.6, figsize=(12,5))
#8443		Printing_machinery_used_for_printing_by_means_...

# hscode definition changed 2007 ?


# In[ ]:


show_year_hs(y_country_last,2,"hs4")


# In[ ]:


year_hs_country(y_all,2,last_year,"hs4","8411")


# In[ ]:


year_hs_country(y_all,2,last_year,"hs4","8802")


# In[ ]:


year_hs_country(y_all,2,last_year,"hs4","1005")


# In[ ]:


year_hs_country(y_all,2,last_year,"hs4","8542")


# 

# In[ ]:


year_hs_country(y_all,2,last_year,"hs4","9018")


# In[ ]:


show_year_hs(y_country_last,1,"hs6")


# In[ ]:


year_hs_country(y_all,1,last_year,"hs6","870323")


# In[ ]:


year_hs_country(y_all,1,last_year,"hs6","870324")


# In[ ]:


year_hs_country(y_all,1,last_year,"hs6","000000")


# In[ ]:


year_hs_country(y_all,1,last_year,"hs6","880330")


# In[ ]:


#880330 y_country
x_y =simple_year_export_import(y_country[y_country["hs6"]=="880330"])
x_y.plot.bar(y=['export', 'import'] ,alpha=0.6, figsize=(12,5))


# In[ ]:


year_hs_country(y_all,1,last_year,"hs6","870840")


# In[ ]:


show_year_hs(y_country_last,2,"hs6")


# In[ ]:


year_hs_country(y_all,2,last_year,"hs6","880240")


# In[ ]:


year_hs_country(y_all,2,last_year,"hs6","100590")


# In[ ]:


year_hs_country(y_all,2,last_year,"hs6","841191")


# In[ ]:


year_hs_country(y_all,2,last_year,"hs6","300490")


# In[ ]:


year_hs_country(y_all,2,last_year,"hs6","300210")


# In[ ]:


x_y =simple_year_export_import(y_country[y_country["hs6"]=="300210"])
x_y.plot.bar(y=['export', 'import'] ,alpha=0.6, figsize=(12,5))
#300210 Antisera__other_blood_fractions_and_immunological_products_


# In[ ]:


def show_year_month_export_import(ym):
   # use ym 
   x_ym_exp =ym[ym['exp_imp']==1].groupby(["Year","month"],as_index=False)["Value"].    sum()[["Year","month","Value"]]
   x_ym_exp.rename(columns={'Value': 'export'}, inplace=True)
   # re index
   x_ym_exp.index = pd.RangeIndex(start=1,stop=len(x_ym_exp)+1)
   # import
   x_ym_imp =ym[ym['exp_imp']==2].groupby(["Year","month"],as_index=False)["Value"].sum()[["Year","month","Value"]]
   x_ym_imp.rename(columns={'Value': 'import'}, inplace=True)
   # re index
   x_ym_imp.index = pd.RangeIndex(start=1,stop=len(x_ym_exp)+1)

   x_exp_imp = pd.concat([x_ym_exp,x_ym_imp['import']],axis=1) #axis=1  add columns
   x_exp_imp['Year'] = x_exp_imp['Year'].astype(str)
   x_exp_imp.index = x_exp_imp[['Year','month']] 

   x_exp_imp.plot.bar(y=['export', 'import'] ,alpha=0.6, figsize=(12,5))
   #dir(x_y.plot.bar)
   plt.title( ' export import ' , size=20)


# In[ ]:


ym=ym_all[ym_all["Country"]==target_country]
show_year_month_export_import(ym)


# In[ ]:




