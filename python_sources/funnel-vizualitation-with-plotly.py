#!/usr/bin/env python
# coding: utf-8

# <p>Hi there!</p>
# <p>Today i want to play with the incredible library Plotly to make some interesting funnels.</p>
# <p>Im gonna use the dataSets from Ecommerce websites.</p>
# <p>My goal this weekend was develop some graphic funnels and extract some insights.</p>
# 
# <ul>
# 	<li><a href="#C1">1. Data preparation</a></li>
# 	<li><a href="#C2">2. Basic conversion funnel</a></li>
# 	<li><a href="#C3">3. Preparing and mergind data for the segmented funnels</a></li>
# 	<li><a href="#C4">4. First segmented funnel - By gender</a></li>
# 	<li><a href="#C5">5. Preparing data for the second segmented funnel</a></li>
# 	<li><a href="#C6">6. Second segmented funnel - By gender and device</a></li>
# </ul>

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import plotly
import plotly.plotly as py
from plotly import graph_objs as go
from __future__ import division
import plotly.figure_factory as ff
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
get_ipython().run_line_magic('matplotlib', 'inline')

dfHome = pd.read_csv('../input/home_page_table.csv')
dfSearch = pd.read_csv('../input/search_page_table.csv')
dfPaymentC1 = pd.read_csv('../input/payment_confirmation_table.csv')
dfPaymentP2 = pd.read_csv('../input/payment_page_table.csv')
UserTable = pd.read_csv('../input/user_table.csv')



# At the first moment i want to explore the data and see how many not NaN values have each data set. I want to recall that each data set has only one step of the conversion funnel and they are connected by the user id

# In[ ]:


UserTable.head()


# In[ ]:


dfHome.head()


# <strong><h2 id="C1">Data preparation</h2></strong>

# In[ ]:


data_table = [['Phases', 'Values'],
               ['dfHome', dfHome['user_id'].count()],
               ['dfSearch', dfSearch['user_id'].count()],
               ['dfPaymentP2', dfPaymentP2['user_id'].count()],
               ['dfPaymentC1', dfPaymentC1['user_id'].count()],
               ['UserTable', UserTable['user_id'].count()]]


# In[ ]:


data_table


# In[ ]:


table = ff.create_table(data_table)
iplot(table)


# Now i have a table where i can see the different steps of the funnel

# <strong><h2 id="C2">Basic conversion funnel</h2></strong>

# <p>Now i'm gonna make the first funnel. This funnel is gonna be basic.<br />In this first step i just want to show you how the differents phases changes the number of retention of the ussers.</p>
# <p>The phases are the same as from the dable</p>

# In[ ]:


# chart stages data
values = [90400, 45200, 6030, 452]
phases = ['Home', 'Search', 'Payment', 'Confirm']

# color of each funnel section
colors = ['rgb(0, 102, 204)', 'rgb(51, 153, 255)', 'rgb(0, 102, 204)', 'rgb(204, 255, 255)']

n_phase = len(phases)
plot_width = 700

# height of a section and difference between sections 
section_h = 100
section_d = 10

# multiplication factor to calculate the width of other sections
unit_width = plot_width / max(values)

# width of each funnel section relative to the plot width
phase_w = [int(value * unit_width) for value in values]

# plot height based on the number of sections and the gap in between them
height = section_h * n_phase + section_d * (n_phase - 1)

# list containing all the plot shapes
shapes = []

# list containing the Y-axis location for each section's name and value text
label_y = []

for i in range(n_phase):
        if (i == n_phase-1):
                points = [phase_w[i] / 2, height, phase_w[i] / 2, height - section_h]
        else:
                points = [phase_w[i] / 2, height, phase_w[i+1] / 2, height - section_h]

        path = 'M {0} {1} L {2} {3} L -{2} {3} L -{0} {1} Z'.format(*points)

        shape = {
                'type': 'path',
                'path': path,
                'fillcolor': colors[i],
                'line': {
                    'width': 1,
                    'color': colors[i]
                }
        }
        shapes.append(shape)
        
        # Y-axis location for this section's details (text)
        label_y.append(height - (section_h / 2))

        height = height - (section_h + section_d)
        
# For phase names
label_trace = go.Scatter(
    x=[-350]*n_phase,
    y=label_y,
    mode='text',
    text=phases,
    textfont=dict(
        color='rgb(255, 255, 255)',
        size=15
    )
)
 
# For phase values
value_trace = go.Scatter(
    x=[350]*n_phase,
    y=label_y,
    mode='text',
    text=values,
    textfont=dict(
        color='rgb(200,200,200)',
        size=15
    )
)

data = [label_trace, value_trace]
 
layout = go.Layout(
    title="<b>Conversion Funnel</b>",
    titlefont=dict(
        size=40,
        color='rgb(255, 255, 255)'
    ),
    shapes=shapes,
    height=560,
    width=800,
    showlegend=False,
    paper_bgcolor='rgba(44,58,71,1)',
    plot_bgcolor='rgba(44,58,71,1)',
    xaxis=dict(
        showticklabels=False,
        zeroline=False,
    ),
    yaxis=dict(
        showticklabels=False,
        zeroline=False
    )
)
 
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# <strong><h2 id="C3">Preparing and mergind data for the segmented funnels</h2></strong>

# <p>As we can see this funnel represent the same thing as we see in the table from above but with more visual impact.</p>
# <p>Also, the ecommerce should focus on the drop between the Search page and the Payment page.<br />They should use some strategy in order to hold back the customer and try to not let him go</p>
# <p>The thing is that this is yet a little bit useless, i want to make a funnel with the variation per gender and device.</p>
# <p>Lets make in the first place the segmentation by gender</p>
# 
# <p>To make that.<br />First of all i want to change the name of the columns adding the steps.<br />Then in the second place merge the different dataSets where we can have the steps.<br />In third place count the values of each step in order to make the famous table of the phases.</p>
# 
# 

# In[ ]:


UserTable.head()


# In[ ]:


dfHome = dfHome.rename(columns={'page':'Step One'})
dfSearch = dfSearch.rename(columns={'page':'Step Two'})
dfPaymentP2 = dfPaymentP2.rename(columns={'page':'Step Three'})
dfPaymentC1 = dfPaymentC1.rename(columns={'page':'Step Four'})


# In[ ]:


dfPaymentC1.head()


# In[ ]:


dfT = UserTable.merge(dfHome, how ='outer',on='user_id').merge(dfSearch, how ='outer', on='user_id').merge(dfPaymentP2, how ='outer', on='user_id').merge(dfPaymentC1, how ='outer', on='user_id')


# In[ ]:


dfT.head()


# In[ ]:


Step_One_Male = (dfT['sex'] == 'Male') & (dfT['Step One'] == 'home_page')
Step_One_Female = (dfT['sex'] == 'Female') & (dfT['Step One'] == 'home_page')

Step_Two_Male = (dfT['sex'] == 'Male') & (dfT['Step Two'] == 'search_page')
Step_Two_Female = (dfT['sex'] == 'Female') & (dfT['Step Two'] == 'search_page')

Step_Three_Male = (dfT['sex'] == 'Male') & (dfT['Step Three'] == 'payment_page')
Step_Three_Female = (dfT['sex'] == 'Female') & (dfT['Step Three'] == 'payment_page')

Step_Four_Male = (dfT['sex'] == 'Male') & (dfT['Step Four'] == 'payment_confirmation_page')
Step_Four_Female = (dfT['sex'] == 'Female') & (dfT['Step Four'] == 'payment_confirmation_page')

data_table2 = [['Phases', 'Man', 'Woman'],
               ['Home', Step_One_Male.sum(), Step_One_Female.sum()],
               ['Search', Step_Two_Male.sum(), Step_Two_Female.sum()],
               ['Payment', Step_Three_Male.sum(), Step_Three_Female.sum()],
               ['Confirmation', Step_Four_Male.sum(), Step_Four_Female.sum()]]


# In[ ]:


data_table2


# In[ ]:


table = ff.create_table(data_table2)
iplot(table)


# To create the funnel i need a new dataFrame with the values per phase, and that values has to be integer not object

# In[ ]:


df = pd.DataFrame(np.array([['Home', 45325, 45075], ['Search', 22524, 22676], ['Payment', 2930, 3100], ['Confirmation', 211, 241]]),
                            columns=['','Man', 'Woman'])
df = df.set_index('')

df['Man'] = df['Man'].astype('int')
df['Woman'] = df['Woman'].astype('int')


# In[ ]:


df.dtypes


# <strong><h2 id="C4">First segmented funnel - By gender</h2></strong>

# Once the data is ready, lets run the funnel!

# In[ ]:


total = [sum(row[1]) for row in df.iterrows()]


n_phase, n_seg = df.shape

plot_width = 600
unit_width = plot_width / total[0]
 
phase_w = [int(value * unit_width) for value in total]
 
# height of a section and difference between sections 
section_h = 100
section_d = 10

# shapes of the plot
shapes = []
 
# plot traces data
data = []
 
# height of the phase labels
label_y = []

height = section_h * n_phase + section_d * (n_phase-1)

# rows of the DataFrame
df_rows = list(df.iterrows())

# iteration over all the phases
for i in range(n_phase):
    # phase name
    row_name = df.index[i]
    
    # width of each segment (smaller rectangles) will be calculated
    # according to their contribution in the total users of phase
    seg_unit_width = phase_w[i] / total[i]
    seg_w = [int(df_rows[i][1][j] * seg_unit_width) for j in range(n_seg)]
    
    # starting point of segment (the rectangle shape) on the X-axis
    xl = -1 * (phase_w[i] / 2)
    
    # iteration over all the segments
    for j in range(n_seg):
        # name of the segment
        seg_name = df.columns[j]
        
        # corner points of a segment used in the SVG path
        points = [xl, height, xl + seg_w[j], height, xl + seg_w[j], height - section_h, xl, height - section_h]
        path = 'M {0} {1} L {2} {3} L {4} {5} L {6} {7} Z'.format(*points)
        
        shape = {
                'type': 'path',
                'path': path,
                'fillcolor': colors[j],
                'line': {
                    'width': 1,
                    'color': colors[j]
                }
        }
        shapes.append(shape)
        
        # to support hover on shapes
        hover_trace = go.Scatter(
            x=[xl + (seg_w[j] / 2)],
            y=[height - (section_h / 2)],
            mode='markers',
            marker=dict(
                size=min(seg_w[j]/2, (section_h / 2)),
                color='rgba(255,255,255,1)'
            ),
            text="Segment : %s" % (seg_name),
            name="Value : %d" % (df[seg_name][row_name])
        )
        data.append(hover_trace)
        
        xl = xl + seg_w[j]

    label_y.append(height - (section_h / 2))

    height = height - (section_h + section_d)
    
    # For phase names
label_trace = go.Scatter(
    x=[-350]*n_phase,
    y=label_y,
    mode='text',
    text=df.index.tolist(),
    textfont=dict(
        color='rgb(200,200,200)',
        size=15
    )
)

data.append(label_trace)
 
# For phase values (total)
value_trace = go.Scatter(
    x=[350]*n_phase,
    y=label_y,
    mode='text',
    text=total,
    textfont=dict(
        color='rgb(200,200,200)',
        size=15
    )
)

data.append(value_trace)

layout = go.Layout(
    title="<b>Segmented Funnel Chart</b>",
    titlefont=dict(
        size=20,
        color='rgb(230,230,230)'
    ),
    hovermode='closest',
    shapes=shapes,
    showlegend=False,
    paper_bgcolor='rgba(44,58,71,1)',
    plot_bgcolor='rgba(44,58,71,1)',
    xaxis=dict(
        showticklabels=False,
        zeroline=False,
    ),
    yaxis=dict(
        showticklabels=False,
        zeroline=False
    )
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)


# <strong><h2 id="C5">Preparing data for the second segmented funnel</h2></strong>

# Wow! This funnel looks nice, but the difference between genders is very small, so... lets make something awesome and lets make a funnel segmented per devices and gender at the same time. Sounds good? Lets prepare the data again.
# <p>As we can see, the ecommerce should focus on the drop between the Search page and the Payment page.<br />They should use some strategy in order to hold back the customer and try to not let him go</p>

# In[ ]:


Step_One_Desktop_Male =(dfT['sex'] == 'Male') &  (dfT['device'] == 'Desktop') & (dfT['Step One'] == 'home_page')
Step_One_Mobile_Male =(dfT['sex'] == 'Male') & (dfT['device'] == 'Mobile') & (dfT['Step One'] == 'home_page')
Step_One_Desktop_Female =(dfT['sex'] == 'Female') &  (dfT['device'] == 'Desktop') & (dfT['Step One'] == 'home_page')
Step_One_Mobile_Female =(dfT['sex'] == 'Female') & (dfT['device'] == 'Mobile') & (dfT['Step One'] == 'home_page')

Step_Two_Desktop_Male =(dfT['sex'] == 'Male') & (dfT['device'] == 'Desktop') & (dfT['Step Two'] == 'search_page')
Step_Two_Mobile_Male =(dfT['sex'] == 'Male') & (dfT['device'] == 'Mobile') & (dfT['Step Two'] == 'search_page')
Step_Two_Desktop_Female =(dfT['sex'] == 'Female') & (dfT['device'] == 'Desktop') & (dfT['Step Two'] == 'search_page')
Step_Two_Mobile_Female =(dfT['sex'] == 'Female') & (dfT['device'] == 'Mobile') & (dfT['Step Two'] == 'search_page')

Step_Three_Desktop_Male =(dfT['sex'] == 'Male') & (dfT['device'] == 'Desktop') & (dfT['Step Three'] == 'payment_page')
Step_Three_Mobile_Male =(dfT['sex'] == 'Male') & (dfT['device'] == 'Mobile') & (dfT['Step Three'] == 'payment_page')
Step_Three_Desktop_Female =(dfT['sex'] == 'Female') & (dfT['device'] == 'Desktop') & (dfT['Step Three'] == 'payment_page')
Step_Three_Mobile_Female =(dfT['sex'] == 'Female') & (dfT['device'] == 'Mobile') & (dfT['Step Three'] == 'payment_page')

Step_Four_Desktop_Male =(dfT['sex'] == 'Male') & (dfT['device'] == 'Desktop') & (dfT['Step Four'] == 'payment_confirmation_page')
Step_Four_Mobile_Male =(dfT['sex'] == 'Male') & (dfT['device'] == 'Mobile') & (dfT['Step Four'] == 'payment_confirmation_page')
Step_Four_Desktop_Female =(dfT['sex'] == 'Female') & (dfT['device'] == 'Desktop') & (dfT['Step Four'] == 'payment_confirmation_page')
Step_Four_Mobile_Female =(dfT['sex'] == 'Female') & (dfT['device'] == 'Mobile') & (dfT['Step Four'] == 'payment_confirmation_page')

data_tableTop = [['Phases', 'Desktop Male', 'Mobile Male', 'Desktop Female', 'Mobile Female'],
               ['Home', Step_One_Desktop_Male.sum(), Step_One_Mobile_Male.sum(), Step_One_Desktop_Female.sum(), Step_One_Mobile_Female.sum()],
               ['Search',Step_Two_Desktop_Male.sum(), Step_Two_Mobile_Male.sum(), Step_Two_Desktop_Female.sum(), Step_Two_Mobile_Female.sum()],
               ['Payment', Step_Three_Desktop_Male.sum(), Step_Three_Mobile_Male.sum(), Step_Three_Desktop_Female.sum(), Step_Three_Mobile_Female.sum()],
               ['Confirmation', Step_Four_Desktop_Male.sum(), Step_Four_Mobile_Male.sum(), Step_Four_Desktop_Female.sum(), Step_Four_Mobile_Female.sum()]]


# In[ ]:


data_tableTop


# In[ ]:


table = ff.create_table(data_tableTop)
iplot(table)


# In[ ]:


dfFull = pd.DataFrame(np.array([['Home', 30203, 15122, 29997, 15078], ['Search', 15009, 7515, 15091, 7585], ['Payment', 1480, 1450, 1530, 1570], ['Confirmation', 76, 135, 74, 167]]),
                            columns=['','Desktop Male', 'Mobile Male', 'Desktop Female', 'Mobile Female'])

dfFull = dfFull.set_index('')


# In[ ]:


dfFull['Desktop Male'] = dfFull['Desktop Male'].astype('int')
dfFull['Mobile Male'] = dfFull['Mobile Male'].astype('int')
dfFull['Desktop Female'] = dfFull['Desktop Female'].astype('int')
dfFull['Mobile Female'] = dfFull['Mobile Female'].astype('int')


# In[ ]:


dfFull


# <strong><h2 id="C6">Second segmented funnel - By gender and device</h2></strong>

# <p>Yeah! Right now we can extract some insights. As we can see with the visual impact even if more people enter by desktop, the biggest part of this people drops. <br />But, the people who enter by mobile is almost the half in comparasion with the desktop people, but they dont drop as much as the desktop people.</p>
# <p>Thats quite interesting, another good point is that more women stays until the end.</p>
# <p>Again, the ecommerce should focus in the drop between search and payment</p>
# <p>Sadly, in the funnel we can't see the conversion page with visual impact, lets make a pie chart to see how its looks like</p>
# 

# In[ ]:


colors = ['rgb(63,92,128)', 'rgb(90,131,182)', 'rgb(255,255,255)', 'rgb(127,127,127)']
total = [sum(row[1]) for row in dfFull.iterrows()]
n_phase, n_seg = dfFull.shape

plot_width = 800
unit_width = plot_width / total[0]
 
phase_w = [int(value * unit_width) for value in total]
 
# height of a section and difference between sections 
section_h = 100
section_d = 10

# shapes of the plot
shapes = []
 
# plot traces data
data = []
 
# height of the phase labels
label_y = []

height = section_h * n_phase + section_d * (n_phase-1)

# rows of the DataFrame
df_rows = list(dfFull.iterrows())

# iteration over all the phases
for i in range(n_phase):
    # phase name
    row_name = dfFull.index[i]
    
    # width of each segment (smaller rectangles) will be calculated
    # according to their contribution in the total users of phase
    seg_unit_width = phase_w[i] / total[i]
    seg_w = [int(df_rows[i][1][j] * seg_unit_width) for j in range(n_seg)]
    
    # starting point of segment (the rectangle shape) on the X-axis
    xl = -1 * (phase_w[i] / 2)
    
    # iteration over all the segments
    for j in range(n_seg):
        # name of the segment
        seg_name = dfFull.columns[j]
        
        # corner points of a segment used in the SVG path
        points = [xl, height, xl + seg_w[j], height, xl + seg_w[j], height - section_h, xl, height - section_h]
        path = 'M {0} {1} L {2} {3} L {4} {5} L {6} {7} Z'.format(*points)
        
        shape = {
                'type': 'path',
                'path': path,
                'fillcolor': colors[j],
                'line': {
                    'width': 1,
                    'color': colors[j]
                }
        }
        shapes.append(shape)
        
        # to support hover on shapes
        hover_trace = go.Scatter(
            x=[xl + (seg_w[j] / 2)],
            y=[height - (section_h / 2)],
            mode='markers',
            marker=dict(
                size=min(seg_w[j]/2, (section_h / 2)),
                color='rgba(255,255,255,1)'
            ),
            text="Segment : %s" % (seg_name),
            name="Value : %d" % (dfFull[seg_name][row_name])
        )
        data.append(hover_trace)
        
        xl = xl + seg_w[j]

    label_y.append(height - (section_h / 2))

    height = height - (section_h + section_d)
    
    # For phase names
label_trace = go.Scatter(
    x=[-600]*n_phase,
    y=label_y,
    mode='text',
    text=dfFull.index.tolist(),
    textfont=dict(
        color='rgb(200,200,200)',
        size=15
    )
)

data.append(label_trace)
 
# For phase values (total)
value_trace = go.Scatter(
    x=[600]*n_phase,
    y=label_y,
    mode='text',
    text=total,
    textfont=dict(
        color='rgb(200,200,200)',
        size=15
    )
)

data.append(value_trace)

layout = go.Layout(
    title="<b>Segmented Funnel Chart</b>",
    titlefont=dict(
        size=20,
        color='rgb(230,230,230)'
    ),
    hovermode='closest',
    shapes=shapes,
    showlegend=False,
    paper_bgcolor='rgba(44,58,71,1)',
    plot_bgcolor='rgba(44,58,71,1)',
    xaxis=dict(
        showticklabels=False,
        zeroline=False,
    ),
    yaxis=dict(
        showticklabels=False,
        zeroline=False
    )
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)


# In[ ]:


labels = ['Desktop Male', 'Mobile Male', 'Desktop Female', 'Mobile Female']
values = [76, 135, 74, 167]
colors = ['rgb(63,92,128)', 'rgb(90,131,182)', 'rgb(255,255,255)']

trace = go.Pie(labels=labels, values=values,
               textfont=dict(size=25),
               marker=dict(colors=colors, 
                           line=dict(color='#000000', width=1)))

iplot([trace], filename='styled_pie_chart')


# So, here we can see how Mobile predominates over Desktop.
# 
# And more womens buy finally the product.
# 
# <p>And i want to recall again the the drop between the setp 2 and step 3 is not normal, they should make some CRO and AB Test strategy</p>
# 
# 

# In[ ]:





# In[ ]:





# In[ ]:




