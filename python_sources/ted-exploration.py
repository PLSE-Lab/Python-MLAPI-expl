#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Setup

import numpy as np
import pandas as pd

metadata = pd.read_csv('../input/ted-talks/ted_main.csv')
transcripts = pd.read_csv('../input/ted-talks/transcripts.csv')

metadata.describe()


# Ideas: (possible target) "Reaction ratio" - the fraction of viewers that clicked a rating for the video
# 
# Time dependence: In the beginning TED may have had stricter requirements for talks, meaning that the first talks may be by speakers of higher renown, drawing more views.
#  
#  Are rating numbers correlated with time since the start of TED? (Do the first videos provoke a stronger response in viewers?)
#  
#  How does the fraction of highly rated/viewed videos vary for each year?
#  
# Targets:
# 
#  Views - Why do people view a talk? If views of other talks by famous speakers are not uniform, then the title/summary of the talk must have the biggest impact, provided that users are shown the talk at all. The latter is probably dependent on the tags of the talk, defining its target audience, so the impacts of the talk (rates/comments) are based on the target audience. HOWEVER: In the beginning there were few talks, meaning that talks were much more accessible to everyone browsing the website. The problem is we do not know how the comment/rate numbers changed with time.
#  
#  Does the number of related talks affect views? Count all related talks, then only the related talks coming after the talk in question that lead back to it.
#  
#  Comments - How does a talk provoke comments? Is it the same as rating? Should be a smaller fraction (only some users that rate will comment, but probably most that comment will rate), check this!

# In[ ]:


# Get timedeltas from the end point
from datetime import datetime

end_date = datetime(year=2017, month=9, day=21)
metadata['days_online'] = end_date - metadata['published_date'].map(datetime.fromtimestamp)
metadata['days_online'] = metadata['days_online'].map(lambda x: x.days)
metadata[['days_online']].describe()


# There seem to be entries with dates after the stated end date of the dataset. We'll get them in line.

# In[ ]:


end_date = datetime(year=2017, month=9, day=23)
metadata['days_online'] = end_date - metadata['published_date'].map(datetime.fromtimestamp)
metadata['days_online'] = metadata['days_online'].map(lambda x: x.days)
metadata[['days_online']].describe()


# In[ ]:


metadata['views'].hist()


# In[ ]:


metadata['comments'].hist()


# The views and comments need some normalizing and, as a target, feel like something that should be subject to diminishing returns. We'll put them in log scale to make this visible.

# In[ ]:


metadata['log_views'] = np.log2(metadata['views'])
metadata['log_comments'] = np.log2(metadata['comments'])
metadata['log_views'].hist()


# In[ ]:


metadata['log_comments'].hist()


# The distributions became almost perfectly normal. After some research it turns out that human comments online actually follow a log-normal distribution, so the transformation is the right one.
# 
# Is there a correlation between views/comments and days online? (Sounds logical)

# In[ ]:


metadata[['log_views', 'log_comments', 'days_online', 'languages', 'duration']].corr()


# There is a moderate correlation between views and comments, as expected. The same is true for the number of languages available, also expected (more languages means a larger audience). Longer talks seem to have less translations (again expected, translators would preffer the shorter/easier tasks). It's interesting that there are more comments for older talks, but less views: the audience has turned from engaging actively (commenting) to simply observing (viewing) over time.
# 
# Lets see how those correlations look like.

# In[ ]:


import matplotlib.pyplot as plt

# made a little helper
def scatter_trend(*, x, y, color=None):
    #plt.ylim(min(y), max(y))
    plt.scatter(x, y, c=color)
    # calc the trendline
    z = np.polyfit(x, y, 2)
    p = np.poly1d(z)
    # ensure the x is ordered for the line plot
    sorted_data = sorted(list(zip(x, p(x))))
    new_x = [i[0] for i in sorted_data]
    new_p = [i[1] for i in sorted_data]
    plt.plot(new_x, new_p, "r--")
    plt.show()

scatter_trend(x=metadata['log_views'], y=metadata['log_comments'], color=metadata['days_online'])


# It seems that if we look at the days_online axis things will be clearer?

# In[ ]:


scatter_trend(x=metadata['log_views'], color=metadata['log_comments'], y=metadata['days_online'])


# Still not good enough, let's go 3D.
# 
# Note: The ipyvolume library that I found for 3D visualizations installs on Kaggle, but the visuals do not appear. If it is installed locally they appear.

# In[ ]:


get_ipython().system('pip install ipyvolume')


# In[ ]:


import ipyvolume as ipv

def plot3D(*, x, y, z):
    def scale(s):
        return s.apply(lambda i: i/s.max())
    _x, _y, _z = (scale(metadata[x]), scale(metadata[y]), scale(metadata[z]))
    fig = ipv.figure()
    scatter = ipv.quickscatter(_x, _y, _z, marker='sphere', color='green')
    ipv.pylab.xlim(_x.min(), _x.max())
    ipv.pylab.ylim(_y.min(), _y.max())
    ipv.pylab.zlim(_z.min(), _z.max())
    ipv.pylab.xlabel(x)
    ipv.pylab.ylabel(y)
    ipv.pylab.zlabel(z)
    ipv.show()
    
plot3D(x='log_views', y='log_comments', z='days_online')


# In[ ]:


plot3D(x='log_views', y='log_comments', z='languages')


# The languages dependence looks nearly linear in the log scale.
# 
# Some ideas for modelling:
# 
# Targets:
# 
#     views (measure of how interesting the talk looks): modelled on tags, title, desc, time online, related talks, languages
#     total rates (measure of emotional influence of talk): modelled on time online, views, transcript, speaker occupation
#     comments (measure of response provoked by talk): modelled the same as total rates
# 
# We can also predict:
# 
#     the annual (or per view) rating rate
#     the annual (or per view) comment rate
# 
# What cannot be done with this dataset:
# 
#     get the sentiment of the audience towards the speaker prior to the viewing
#     use rates or comments as predictors, as that would leak information on the final response of the audience: rates and comments happen after a talk is viewed
# 
# Limitations of the data:
# 
#     No true way to measure stage presence or quality of delivery of the talk, which would be very important for the response of the audience
# 
# How to overcome the limitations:
# 
#     a speaker's occupation is a very low-resolution way to get an idea of the type of stand/delivery the speaker will make, and final response will still depend on who the audience are. Maybe an evaluation of the pairing between the speaker occupation and the tags (domain of the talk) can give a predictor for delivery?
# 
# Modelling preparations:
# 
#     one-hot encode tags
#     get days_online
#     get title and description sentiment
#     get number of related talks
#     define occupation and tags match level
# 
# Type of model: fully-connected neural network
# 
