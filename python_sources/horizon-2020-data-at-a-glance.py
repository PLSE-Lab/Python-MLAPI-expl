#!/usr/bin/env python
# coding: utf-8

# # What Countries Benefit from Horizon 2020?
# Looking through data published by CORDIS, I have found that the EU contributes not only to companies and organizations located in the EU-member countries but to companies and organizations all over the world.
# 
# Because of Brexit divorce, it was interesting to me,  where the UK stands amid other EU-member countries.
# 
# It turns out that for projects started in the period from 2014 to 2019 within the Horizon 2020 framework, the total EU contribution to non-EU countries is comparable to the EU contribution to Italy, Spain, and the Netherlands. 
# 
# Switzerland, Norway, and Israel hold leadership amid non-EU countries. The total EU contribution to each of these countries, within the Horizon 2020 framework, is comparable to Austria, Demark, Finland, Greece, and Ireland, respectively.

# In[ ]:


# Install csvvalidator to validate datasets
get_ipython().system('pip install csvvalidator')


# In[ ]:


import os
import numpy as np
import pandas as pd
from bokeh.io import output_notebook, push_notebook, show
from bokeh.models import ColumnDataSource, HoverTool, LabelSet, CategoricalColorMapper
from bokeh.layouts import row
from bokeh.plotting import figure
from bokeh.palettes import Spectral4, Spectral11

from ipywidgets import interact, IntSlider, Checkbox

from h2020 import *

output_notebook()
print(os.listdir('../input/eu-research-projects-under-horizon-2020'))


# In[ ]:


# Members of the EU
eu_list = ['Austria','Belgium','Bulgaria','Croatia','Cyprus','Czechia','Denmark',
           'Estonia','Finland','France','Germany','Greece','Hungary','Ireland',
           'Italy','Latvia','Lithuania','Luxembourg','Malta','Netherlands','Poland',
           'Portugal','Romania','Slovakia','Slovenia','Spain','Sweden','United Kingdom']


# Load and validate data...

# #### Countries

# In[ ]:


countries = H2020Countries('../input/eu-research-projects-under-horizon-2020/cordisref-countries.csv')
countries.validate()
countries.print_validation_summary()
df_c = countries.read()
df_c = df_c[df_c.language == 'en'].dropna(subset=['euCode']).set_index('euCode').sort_index()
df_c['is_in_eu'] = df_c.name.isin(eu_list)


# #### Projects

# In[ ]:


projects = H2020Projects('../input/eu-research-projects-under-horizon-2020/cordis-h2020projects.csv') 
projects.validate()
projects.print_validation_summary()
df_p = projects.read()


# In[ ]:


df_p_year = df_p.dropna(subset=['startDate']).copy()
df_p_year['startYear'] = df_p_year.startDate.dt.year
df_p_year = df_p_year[['startYear']]
df_p_year.sort_index(inplace=True)


# #### Organizations

# In[ ]:


org = H2020Organizations('../input/eu-research-projects-under-horizon-2020/cordis-h2020organizations.csv') 
org.validate()
org.print_validation_summary()
df_org = org.read()


# #### Programs

# In[ ]:


prog = H2020Programmes('../input/eu-research-projects-under-horizon-2020/cordisref-H2020programmes.csv')
prog.validate()
prog.print_validation_summary()
df_pg = prog.read()
df_pg.dropna(how='all', inplace=True)
df_pg = df_pg[df_pg.language=='en']
df_pg.drop(['language'], axis=1, inplace=True)
df_pg['shortTitle'].fillna(df_pg['title'], inplace=True)


# In[ ]:


org_columns = ['projectRcn','projectID','ecContribution', 'country']
df_org_c = pd.merge(df_org[org_columns], df_c[['name','is_in_eu']], left_on='country',right_index=True)
df_org_p = pd.merge(df_org_c, df_p_year, left_on='projectRcn', right_index=True)

df_org_p_eu = df_org_p.copy()

df_org_p_eu['_code'] = df_org_p_eu.apply(lambda x: x.country if x.is_in_eu else 'OTH', axis=1)
df_org_p_eu['_country'] = df_org_p_eu.apply(lambda x: x['name'] if x.is_in_eu else 'Other non-EU countries', axis=1)
df_org_p_eu['_legend'] = df_org_p_eu.apply(lambda x: 'EU Member' if x.is_in_eu else 'Other non-EU countries', axis=1)
# Mark the UK with another color on the plot
df_org_p_eu.loc[df_org_p_eu['country']=='UK','_legend']='UK'


df_cn_yr_eu = (df_org_p_eu.loc[:,['projectID','ecContribution','_country','_code','startYear','_legend']]
                                            .groupby(['startYear', '_country','_code','_legend'])
                                            .agg({'projectID':'size','ecContribution':'sum'}))


# In[ ]:


df_org_p_oth = df_org_p.copy()

# Calculate cut off sum for countries to display
max_cont_oth = df_org_p_oth[~df_org_p_oth.is_in_eu].groupby(['country']).ecContribution.sum().max()

df_org_p_oth['_code'] = df_org_p_oth['country']
df_org_p_oth['_country'] = df_org_p_oth['name']
df_org_p_oth['_legend'] = df_org_p_oth.apply(lambda x: 'EU Member' if x.is_in_eu else 'non-EU country', axis=1)

df_cn_yr_oth = (df_org_p_oth.loc[:,['projectID','ecContribution','_code','_country','startYear','_legend']]
                                            .groupby(['startYear','_code', '_country','_legend'])
                                            .agg({'projectID':'size','ecContribution':'sum'}))
# Generate a list of countries to display
df_cn_tot = df_cn_yr_oth.groupby(['_code']).ecContribution.sum()
code_list = df_cn_tot.loc[(df_cn_tot<=max_cont_oth+1) & (df_cn_tot>0)].index

# Select only countries that are in the list
df_cn_yr_oth = df_cn_yr_oth[(df_cn_yr_oth.index.get_level_values(1).isin(code_list))]


# In[ ]:


def plot_contribution_prj_num_cn(df_cn_yr, year=2019):
    max_yr = df_cn_yr.index.max()[0]
    min_yr = df_cn_yr.index.min()[0]
    
    # Data for totals
    df_cn = df_cn_yr.loc[slice(year),:].groupby(['_country','_code','_legend']).agg({'projectID':'sum','ecContribution':'sum'})
    df_cn.reset_index(['_code','_legend'],inplace=True)
    # Data for the most recent yr
    df_cn_1yr = df_cn_yr.loc[year,:]
    df_cn_1yr.reset_index(['_code','_legend'],inplace=True)
    
    # Make the ColumnDataSource: source1
    source1 = ColumnDataSource(data={
        'x'           : df_cn.projectID,
        'y'           : df_cn.ecContribution/1000000,
        'country'     : df_cn.index.values,
        'country_code': df_cn._code,
        'legend'      : df_cn._legend})

    # Save the minimum and maximum values of the projectID column: xmin1, xmax1
    xmin1, xmax1 = min(source1.data['x']), max(source1.data['x'])

    # Save the minimum and maximum values of the ecContribution column: ymin1, ymax1
    ymin1, ymax1 = min(source1.data['y']), max(source1.data['y'])

    # Make the ColumnDataSource: source2
    source2 = ColumnDataSource(data={
        'x'           : df_cn_1yr.projectID,
        'y'           : df_cn_1yr.ecContribution/1000000,
        'country'     : df_cn_1yr.index.values,
        'country_code': df_cn_1yr._code,
        'legend'      : df_cn_1yr._legend})

    # Save the minimum and maximum values of the projectID column: xmin2, xmax2
    xmin2, xmax2 = min(source2.data['x']), max(source2.data['x'])

    # Save the minimum and maximum values of the ecContribution column: ymin2, ymax2
    ymin2, ymax2 = min(source2.data['y']), max(source2.data['y'])
    # Create the figure: p1
    p1 = figure(y_axis_label='EC Contribution (mlns euros)', x_axis_label='Number of Projects',
                plot_height=400, plot_width=710, x_range = (xmin1-2, xmax1*1.1), y_range = (ymin1-2,ymax1*1.1))
    p1.title.text = 'EC Contribution and Number of Projects started in period from {} to {} grouped by countries'.format(min_yr, year)

    # p2 = figure(y_axis_label='EC Contribution (mlns euros)', x_axis_label='Number of Projects',
    #             plot_height=400, plot_width=700, x_range = (xmin2-2, xmax2*1.1), y_range = (ymin2-2,ymax2*1.1))
    p2 = figure(x_axis_label='Number of Projects',
                plot_height=400, plot_width=700, x_range = (xmin2-2, xmax2*1.1), y_range = (ymin2-2,ymax2*1.1))
    p2.title.text = 'EC Contribution and Number of Projects started in {} grouped by countries'.format(year)
    # Create a HoverTool: hover
    hover = HoverTool(tooltips=[('Country', '@country'),
                                ('EC Contribution (mlns euros)', '@y{1.11}'),
                                ('Number of Projects', '@x{int}')])

    # Add the HoverTool to the plots p1 and p2
    p1.add_tools(hover)
    p2.add_tools(hover)

    legend = np.unique(df_cn_yr.index.get_level_values(3))
    color_mapper = CategoricalColorMapper(factors=legend, palette=Spectral4)

    # Add a circle glyph to the figures p1 and p2
    p1.circle(x='x', y='y', source=source1, size=10, fill_alpha=0.8, color=dict(field='legend', transform=color_mapper), legend_field='legend')
    p2.circle(x='x', y='y', source=source2, size=10, fill_alpha=0.8, color=dict(field='legend', transform=color_mapper), legend_field='legend')

    # Create labelsets for the figures p1 and p2
    labels1 = LabelSet(x='x', y='y', text='country_code', level='glyph',
                  x_offset=5, y_offset=5, source=source1, render_mode='canvas', 
                  text_font_size='8pt', text_alpha=0.6)
    labels2 = LabelSet(x='x', y='y', text='country_code', level='glyph',
                  x_offset=5, y_offset=5, source=source2, render_mode='canvas', 
                  text_font_size='8pt', text_alpha=0.6)
    
    p1.legend.location = 'top_left'    
    p1.legend.background_fill_alpha = 0.3
    p1.xgrid.visible = False 
    p1.ygrid.visible = False
    p2.legend.location = 'top_left'
    p2.legend.background_fill_alpha = 0.3
    p2.xgrid.visible = False 
    p2.ygrid.visible = False
    

    # Add labels1 and labels2 to the figures p1 and p2
    p1.add_layout(labels1 )
    p2.add_layout(labels2)

    # Create row layout from p1 and p2
    layout = row(p1, p2)

    def update_plot(yr=year, chk1=True, chk2=True):
        df_cn = (df_cn_yr.loc[slice(yr),:].groupby(['_country','_code','_legend'])
                                         .agg({'projectID':'sum','ecContribution':'sum'}))
        df_cn.reset_index(['_code','_legend'], inplace = True)
        df_cn_1yr = df_cn_yr.loc[yr,:]
        df_cn_1yr.reset_index(['_code','_legend'], inplace = True)
 
        p1.title.text = 'EC Contribution and Number of Projects started in the period from {} to {} grouped by countries'.format(min_yr, yr)
        p2.title.text = 'EC Contribution and Number of Projects started in {} grouped by countries'.format(yr)
        new_data1 = {
                     'x'           : df_cn.projectID,
                     'y'           : df_cn.ecContribution/1000000,
                     'country'     : df_cn.index.values,
                     'country_code': df_cn._code,
                     'legend'      : df_cn._legend}
        source1.data = new_data1
        new_data2 = {
                     'x'           : df_cn_1yr.projectID,
                     'y'           : df_cn_1yr.ecContribution/1000000,
                     'country'     : df_cn_1yr.index.values,
                     'country_code': df_cn_1yr._code,
                     'legend'      : df_cn_1yr._legend}
        source2.data = new_data2
        if chk1:
            # set x_range and y_range for p1
            p1.x_range.start = min(new_data1['x'])-2
            p1.x_range.end = max(new_data1['x'])*1.1
            p1.y_range.start = min(new_data1['y'])-2
            p1.y_range.end = max(new_data1['y'])*1.1
        if chk2:
            # set x_range and y_range for p2
            p2.x_range.start = min(new_data2['x'])-2
            p2.x_range.end = max(new_data2['x'])*1.1
            p2.y_range.start = min(new_data2['y'])-2
            p2.y_range.end = max(new_data2['y'])*1.1
  
        push_notebook()

    show(layout, notebook_handle=True)
    interact(update_plot, 
             yr=IntSlider(value=year, min=2014, max=max_yr, step=1,  description='Year:'),
             chk1=Checkbox(value=True, description='Auto set range (left plot)',disabled=False,indent=True),
             chk2=Checkbox(value=True, description='Auto set range (right plot)',disabled=False,indent=True))


# ### Comparison of EC Contribution between EU-Member Countries and Total EC Contribution to Non-EU Countries
# Execute the cell below to interact with the plot

# In[ ]:


plot_contribution_prj_num_cn(df_cn_yr_eu)


# ### Comparison of EC Contribution between Non-EU Countries and Some EU-Member Countries
# Execute the cell below to interact with the plot

# In[ ]:


plot_contribution_prj_num_cn(df_cn_yr_oth)


# # Where does the money go?
# There are about 300 programs under Horizon 2020, which have a hierarchical structure. I decided to group programs by some level of this hierarchy. 
# 
# It is worth to mention that a project can be linked to several programs, and it is hard to verify which amount of money goes to which program. In that case, the whole amount goes to each program group.
# 
# In general, research, breakthrough technologies, health, transport, energy, small and medium-sized enterprises get more support from the EU.

# In[ ]:


df_t = df_p[['id','programme']].copy()
df_t.set_index('id',inplace=True)
df_t = df_t['programme'].str.split(';', expand=True)
df_t = (pd.melt(df_t.reset_index(),id_vars=['id'],value_name='programme')
               .dropna(subset=['programme']).drop('variable',axis=1)
               .set_index('id').sort_index())
df_t['programme_l'] = df_t.programme.str.split('.')
df_t.programme.value_counts()
df_t['group'] = (df_t.programme_l.apply(lambda x: x[0]+'.'+x[1]+'.'+x[2]+'.' if len(x)>=4 else
                                                  x[0]+'.'+x[1]+'.' if len(x)==3 else
                                                  x[0]+'.' if len(x)==2 else
                                                  x[0]))
# We don't need `progaramme_l` anymore
df_t.drop('programme_l', axis=1, inplace=True)


# Load and validate data...

# #### Programs

# In[ ]:


prog = H2020Programmes('../input/eu-research-projects-under-horizon-2020/cordisref-H2020programmes.csv', encoding='utf-8-sig')
prog.validate()
prog.print_validation_summary()
df_pg = prog.read()
df_pg.dropna(how='all', inplace=True)
df_pg = df_pg[df_pg.language=='en']
df_pg.drop(['language'], axis=1, inplace=True)
df_pg['shortTitle'].fillna(df_pg['title'], inplace=True)
df_pg.set_index('code', inplace=True)


# In[ ]:


df_t = pd.merge(df_t['group'], df_pg['shortTitle'], left_on='group', right_index=True)
df_t = df_t.rename(columns={'shortTitle': 'g_name'})


# In[ ]:


df_pg_yr = pd.merge(df_t[['g_name']], df_org_p[['startYear', 'projectID', 'ecContribution','country']], 
                      left_index=True, right_on='projectID')

# Project can belong to several programs in the same group, that's why we can have duplicated rows
df_pg_yr.drop_duplicates(subset=['g_name', 'startYear','projectID','ecContribution'], inplace=True)
df_pg_yr = pd.merge(df_pg_yr, df_c['is_in_eu'], left_on='country', right_index=True)

df_pg_yr_all = df_pg_yr.groupby(['startYear','g_name', 'projectID']).agg({'ecContribution': 'sum'})
df_pg_yr_all.reset_index('projectID', inplace=True)

df_pg_yr_all = df_pg_yr_all.groupby(['startYear','g_name']).agg({'projectID': 'size', 'ecContribution': 'sum'})

topg_list_all = (list(df_pg_yr_all.groupby('g_name')
                              .agg({'ecContribution': 'sum'})
                              .sort_values('ecContribution', ascending=False)
                              .reset_index()
                              .loc[slice(9),'g_name']))
df_pg_yr_all.reset_index('g_name', inplace=True)
df_pg_yr_all['_legend'] = df_pg_yr_all.apply(lambda x: x['g_name'] if x['g_name'] in topg_list_all else 'Other', axis=1)
df_pg_yr_all.set_index(['g_name'], append=True, inplace=True)


# In[ ]:


df_pg_yr_oth = df_pg_yr[~df_pg_yr.is_in_eu].groupby(['startYear','g_name', 'projectID']).agg({'ecContribution': 'sum'})
df_pg_yr_oth.reset_index('projectID', inplace=True)

df_pg_yr_oth = df_pg_yr_oth.groupby(['startYear','g_name']).agg({'projectID': 'size', 'ecContribution': 'sum'})

topg_list_oth = (list(df_pg_yr_oth.groupby('g_name')
                              .agg({'ecContribution': 'sum'})
                              .sort_values('ecContribution', ascending=False)
                              .reset_index()
                              .loc[slice(9),'g_name']))
df_pg_yr_oth.reset_index('g_name', inplace=True)
df_pg_yr_oth['_legend'] = df_pg_yr_oth.apply(lambda x: x['g_name'] if x['g_name'] in topg_list_oth else 'Other', axis=1)
df_pg_yr_oth.set_index(['g_name'], append=True, inplace=True)


# In[ ]:


def plot_contribution_prj_num_pg(df_pg_yr, year=2019):
    max_yr = df_pg_yr.index.max()[0]
    min_yr = df_pg_yr.index.min()[0]
    
    # Data for totals
    df_pgl = df_pg_yr.loc[slice(year),:].groupby(['g_name','_legend']).agg({'projectID':'sum','ecContribution':'sum'})
    df_pgl.reset_index(['_legend'],inplace=True)
    
    # Data for max_yr
    df_pgl_1yr = df_pg_yr.loc[year,:].copy()
    
    # Set `year` column to use it in hover tool
    df_pgl['year'] = year
    df_pgl_1yr['year'] = year
    
    # Data to draw a line
    df_pgl_all = df_pg_yr.groupby(level=1).cumsum()
 
    # Make the ColumnDataSource: source1
    source1 = ColumnDataSource(data={
        'x'           : df_pgl.projectID,
        'y'           : df_pgl.ecContribution/1000000,
        'sname'       : df_pgl.index.values,
        'legend'      : df_pgl._legend,
        'year'        : df_pgl.year})

    # Save the minimum and maximum values of the projectID column: xmin1, xmax1
    xmin1, xmax1 = min(source1.data['x']), max(source1.data['x'])

    # Save the minimum and maximum values of the ecContribution column: ymin1, ymax1
    ymin1, ymax1 = min(source1.data['y']), max(source1.data['y'])

    # Make the ColumnDataSource: source2
    source2 = ColumnDataSource(data={
        'x'           : df_pgl_1yr.projectID,
        'y'           : df_pgl_1yr.ecContribution/1000000,
        'sname'       : df_pgl_1yr.index.values,
        'legend'      : df_pgl_1yr._legend,
        'year'        : df_pgl_1yr.year})

    # Save the minimum and maximum values of the projectID column: xmin2, xmax2
    # xmin2, xmax2 = min(source2.data['x']), max(source2.data['x'])
    xmin2, xmax2 = min(df_pg_yr['projectID']), max(df_pg_yr['projectID'])

    # Save the minimum and maximum values of the ecContribution column: ymin2, ymax2
    # ymin2, ymax2 = min(source2.data['y']), max(source2.data['y'])
    ymin2, ymax2 = min(df_pg_yr['ecContribution'])/1000000, max(df_pg_yr['ecContribution']/1000000)
    # Create the figure: p1
    p1 = figure(y_axis_label='EC Contribution (mlns euros)', x_axis_label='Number of Projects',
                plot_height=500, plot_width=760, x_range = (xmin1-1, xmax1*1.05), y_range = (ymin1-1,ymax1*1.8))
    p1.title.text = 'EC Contribution and Number of Projects started in period from {} to {} grouped by programs'.format(min_yr, year)

    # p2 = figure(y_axis_label='EC Contribution (mlns euros)', x_axis_label='Number of Projects',
    #             plot_height=500, plot_width=600, x_range = (xmin2-1, xmax2*1.05), y_range = (ymin2-1,ymax2*1.05))
    p2 = figure(x_axis_label='Number of Projects',
                plot_height=500, plot_width=600, x_range = (xmin2-1, xmax2*1.05), y_range = (ymin2-1,ymax2*1.05))
    p2.title.text = 'EC Contribution and Number of Projects started in {} grouped by programs'.format(year)
    # Create a HoverTool: hover
    hover = HoverTool(tooltips=[('Year', '@year'),
                                ('Program group', '@sname'),
                                ('EC Contribution (mlns euros)', '@y{1.11}'),
                                ('Number of Projects', '@x{int}')], names=['circle1','circle2','scatter1','scatter2'])

    # Add the HoverTool to the plots p1 and p2
    p1.add_tools(hover)
    p2.add_tools(hover)

    legend = np.unique(df_pg_yr._legend)
    color_mapper = CategoricalColorMapper(factors=legend, palette=Spectral11)
    
   
    # Draw lines
    ls1=[]
    ls2=[]
    g_list = list(legend)
    g_list.remove('Other')
    for idx, gr in enumerate(g_list):
        df_pgl_gr = df_pgl_all.loc[(slice(max_yr),[gr]),:]
        ls1.append(ColumnDataSource(data={
                                         'x'         : df_pgl_gr.projectID,
                                         'y'         : df_pgl_gr.ecContribution/1000000,
                                         'year'      : df_pgl_gr.index.get_level_values(0),
                                         'sname'     : df_pgl_gr.index.get_level_values(1)}))   
        p1.line(x='x', y='y', source=ls1[idx], line_alpha=0.5, line_color='lightgrey',
               line_width=1)
        p1.scatter(x='x', y='y', source=ls1[idx], color='lightgrey', fill_alpha=0.1,
                  line_alpha=0.3, name='scatter1')
        df_pgl_1yr_gr = df_pg_yr.loc[(slice(max_yr),[gr]),:]
        ls2.append(ColumnDataSource(data={
                                         'x'         : df_pgl_1yr_gr.projectID,
                                         'y'         : df_pgl_1yr_gr.ecContribution/1000000,
                                         'year'      : df_pgl_1yr_gr.index.get_level_values(0),
                                         'sname'     : df_pgl_1yr_gr.index.get_level_values(1)}))    
        p2.line(x='x', y='y', source=ls2[idx], line_alpha=0.5, line_color='lightgrey',
               line_width=1)
        p2.scatter(x='x', y='y', source=ls2[idx], color='lightgrey', fill_alpha=0.1,
                  line_alpha=0.3, name='scatter2')

    # Add a circle glyph to the figures p1 and p2
    p1.circle(x='x', y='y', source=source1, size=10, fill_alpha=0.8, color=dict(field='legend', transform=color_mapper), 
              legend_field='legend', line_color='grey', line_alpha=0.3, name='circle1')
    
    p2.circle(x='x', y='y', source=source2, size=10, fill_alpha=0.8, color=dict(field='legend', transform=color_mapper), 
              line_color='grey', line_alpha=0.3, name='circle2')

    p1.legend.location = 'top_left'    
    p1.legend.background_fill_alpha = 0.3
    p1.legend.label_text_font_size = '8pt'
    p1.legend.padding = 3
    p1.xgrid.visible = False 
    p1.ygrid.visible = False
    p2.xgrid.visible = False 
    p2.ygrid.visible = False

    # Create row layout from p1 and p2
    layout = row(p1, p2)

    def update_plot(yr=year, chk1=True, chk2=True):
        df_pgl = (df_pg_yr.loc[slice(yr),:].groupby(['g_name','_legend'])
                                           .agg({'projectID':'sum','ecContribution':'sum'}))
        df_pgl.reset_index(['_legend'], inplace = True)
        df_pgl_1yr = df_pg_yr.loc[yr,:].copy()
        
        # Set `year` for a hover tool
        df_pgl['year'] = yr
        df_pgl_1yr['year'] = yr
        
 
        p1.title.text = 'EC Contribution and Number of Projects started in the period from {} to {} grouped by programs'.format(min_yr, yr)
        p2.title.text = 'EC Contribution and Number of Projects started in {} grouped by programs'.format(yr)
        new_data1 = {
                     'x'           : df_pgl.projectID,
                     'y'           : df_pgl.ecContribution/1000000,
                     'sname'       : df_pgl.index.values,
                     'legend'      : df_pgl._legend,
                     'year'        : df_pgl.year}
        source1.data = new_data1
        new_data2 = {
                     'x'           : df_pgl_1yr.projectID,
                     'y'           : df_pgl_1yr.ecContribution/1000000,
                     'sname'       : df_pgl_1yr.index.values,
                     'legend'      : df_pgl_1yr._legend,
                     'year'        : df_pgl_1yr.year}
        source2.data = new_data2
        if chk1:
            # set x_range and y_range for p1
            p1.x_range.start = min(new_data1['x'])-1
            p1.x_range.end = max(new_data1['x'])*1.05
            p1.y_range.start = min(new_data1['y'])-1
            p1.y_range.end = max(new_data1['y'])*1.8
        if chk2:
            # set x_range and y_range for p2
            p2.x_range.start = min(new_data2['x'])-1
            p2.x_range.end = max(new_data2['x'])*1.05
            p2.y_range.start = min(new_data2['y'])-1
            p2.y_range.end = max(new_data2['y'])*1.05
  
        push_notebook()

    show(layout, notebook_handle=True)
    interact(update_plot, 
             yr=IntSlider(value=year, min=2014, max=max_yr, step=1,  description='Year:'),
             chk1=Checkbox(value=True, description='Auto set range (left plot)',disabled=False,indent=True),
             chk2=Checkbox(value=False, description='Auto set range (right plot)',disabled=False,indent=True))


# ### Comparison of EC Contribution between Programs for All Countries
# Execute the cell below to interact with the plot

# In[ ]:


plot_contribution_prj_num_pg(df_pg_yr_all)


# ### Comparison of EC Contribution between Programs for non-EU Countries
# Execute the cell below to interact with the plot

# In[ ]:


plot_contribution_prj_num_pg(df_pg_yr_oth)


# # Further explorations
# I think it would be interesting to dive deeper into the programs' hierarchy. What programs stand behind FET and LEIT, and ERC? It is also interesting to explore programs by countries.
# 
# If you find the kernel interesting or found some mistakes, please, let me know.
# 
# You can find more information about the Horizon 2020 framework program [here](https://data.europa.eu/euodp/en/data/dataset/cordisH2020projects)
# 
#  Thank you for reading!
