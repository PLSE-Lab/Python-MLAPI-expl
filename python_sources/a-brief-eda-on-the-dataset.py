#!/usr/bin/env python
# coding: utf-8

# # A brief analysis on the plight of the disabled community in India by State.
# 
# There are more than 26.8 million people or 2.2% of the population currently who have disabilities in India (Census 2011) which itself is said to be a very conservative estimate.  There is a lot of stigma associated with the disabled community and a very high inequality in terms of social as well as monetary status between the disabled community and the entire population.
# 
# In this analysis, two main parameters are considered for comparison. And the comparison is done by state. The two parameters are literacy levels, and the % of the population in the workforce.
# 
# ### Methodology for the project: 
# First the data required is collected from official government sources. We use the 2011 census as the primary source of data. The data is processed using SQL and formed into a single well structured CSV table. 
# Next the data is visualized using Tableau to form insights into the data.
# 
# Finally, using Python, we analyse the data and use  Jupyter notebooks  to draw conclusions and form the final report where the table formed during data processing will by analysed using Pandas, the  Tableau worksheets will be imported and embedded and the remaining analysis is conducted.
# 
# ### Source of data: 
# The data is scraped/obtained from directly or indirectly from  the data provided
# by The Ministry of Statistics and Programme Implementation and The Office of the Registrar General.
# 
# In the following section, the dataset prepared using SQL will be uploaded from the local source.
# 
# ### Importing the dataset and required libraries:
# 
# 

# In[ ]:


# importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


# reading the file into a dataframe and displaying its first  5 rows:
df=pd.read_csv('/kaggle/input/disabled-community-dataset/disabled_community_dataset.csv')
df.head()


# Given above is the first 5 rows of the dataset prepared. 
# The columns and what they represent is as :
# 1. State
# 2. number_disabled : It gives the total number of people in the region that are disabled.
# 3. total_population: It gives the total number of people in the region.
# 4. percent_disabled: It gives the total percentage of the people disabled in athe given region.
# 5. literacy_rate_disabled : It represents the literacy rate of the disabled community in the region.
# 6. literacy_rate_general : It shows the total literacy rate of the population in the state.
# 7. workforce_rate_disabled : It tells us the total percent of all the disabled people that are part of the workforce in the given region.(inclusive all ages).
# 8.  workforce_rate_general : It shows the total percent of all the people that are part of the workforce in the given region(inclusive of all ages).

# ### Exploratory data analysis:
# As a first step, the summary statistics is viewed and given below.

# In[ ]:


df.describe()


# Looking at the summary statistics, it is very clear that there is a marked difference between the literacy rate of the disabled people and the population in total. 
# A similar (albeit lower) difference is seen between the workforce percentages.
# Note that the data does not include Telangana due to the fact that Telangana was formed after 2011. In its stead we have placed the average values for India as a total which can be a little misleading and will skew the data very slightly. 
# 
# Next, the visualization done in tableau is given below as an interactive embedded view.

# In[ ]:


get_ipython().run_cell_magic('html', '', "<div class='tableauPlaceholder' id='viz1590415570636' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Di&#47;DisabledcommunityIndia&#47;Dashboard1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='DisabledcommunityIndia&#47;Dashboard1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Di&#47;DisabledcommunityIndia&#47;Dashboard1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1590415570636');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.minWidth='420px';vizElement.style.maxWidth='650px';vizElement.style.width='100%';vizElement.style.minHeight='587px';vizElement.style.maxHeight='887px';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.minWidth='420px';vizElement.style.maxWidth='650px';vizElement.style.width='100%';vizElement.style.minHeight='587px';vizElement.style.maxHeight='887px';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else { vizElement.style.width='100%';vizElement.style.height='727px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# The states with higher values of literacy can be seen with deeper shade of red.
# One interesting point to note is that there approximately a consistent difference of 20% in the difference between the literacy rates of the disabled people and the general population for every state which is extremely high.
# We can see that Kerala has the highest literate population among the states with a literacy rate of 70% closely followed by Goa, and seem to be following the trend of the general population's literacy rate. We can see how the literacy rate of the disabled community varies with that of the total population below.
# 
# 

# In[ ]:


sns.set_style('whitegrid')
sns.regplot(x=df['literacy_rate_disabled'],y=df['literacy_rate_general'],color='purple');
sns.set(rc={'figure.figsize':(20,8)})


# There is a clear upward trend as expected.
# 
# For the final part of the analysis we will take a look at the relationship between the literacy rate and the percentage of the population in the workforce.
# 
# Given below is the relation between the literacy rate and the percent of the population in the workforce for both the disabled community and the general population. Hover over the points to get more information.

# In[ ]:


get_ipython().run_cell_magic('html', '', "<div class='tableauPlaceholder' id='viz1590419786438' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;wo&#47;workforcevsliteracy&#47;Dashboard2&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='workforcevsliteracy&#47;Dashboard2' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;wo&#47;workforcevsliteracy&#47;Dashboard2&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1590419786438');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else { vizElement.style.width='100%';vizElement.style.height='727px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# Surprisingly, there is a clear downward trend between the literacy rate of the disabled population and the workforce rate which is interesting as we expect a higher working population as the literacy rate increases. However, here the opposite is observed. The higher the literacy rate the lower the percentage of the population in the workforce, and even more oddly this trend seems to be only applicable for the disabled community, no such trend can be seen in the general population.
# Let us check whether how strong this relationship using an appropriate correlation coefficient.
# 
# We will be using Pearson's correlation coefficient for our test.
# 
# 

# In[ ]:


#Importing library for finding the correlation coefficient:
from scipy.stats.stats import pearsonr
#Finding the correlation coefficient 'r' and p value 'p':
correlation=pearsonr(df['literacy_rate_disabled'],df['workforce_rate_disabled'])
print('Correlation coefficient r = ',correlation[0])
print('p value = ', correlation[1])


# A correlation coefficient of -0.56 indicates the values are moderately correlated and is generally can be interpreted as a moderate negative relationship. The p-value roughly indicates the probability of an uncorrelated system producing datasets that have a Pearson correlation at least as extreme as the one computed from these datasets. A p value of 0.0003 indicates that it is extremely unlikely that there is no relation between the two columns.

# A possible reason for this trend is that states with higher literacy rates tend to have a more wealthy population which could lead to lesser need for disabled people to work. Another point to be noted, all work is counted in the census and literacy is not a prerequisite for being a working member. A large proportion of the population works in the informal sector and the agricultural sectors which often times do not need people0 to be literate to be employable.

# ## Conclusion
# A few important points can be inferred from this analysis. A major one being that there is a massive gap in terms of literacy between the disabled population and the general population which is quite concerning. A similar problem is seen in the workforce statistics. The workforce participation rate may not look that grim however, an overlooked point is the fact that a huge proportion of the labour workforce is severely underpaid and literate members of the community have a much higher income which indicates that there is a high chance of a very large income gap between the disabled population and the general population. This is indeed a problem especially considering that more than 26 million people are disabled. 
