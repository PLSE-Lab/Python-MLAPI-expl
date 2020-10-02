#!/usr/bin/env python
# coding: utf-8

# # Airplane DataSet Analysis

# This NoteBook is a Little Effort to test and practice the Data Science Skills and Data Visualiztion Skills that I have learned as a Begginer.

# First of All Importing and Setting the Inline Grpahs for the Jupyter Notebook to Show.

# In[ ]:


import pandas as pandasInstance
import numpy as numpyInstance
import seaborn as seabornInstance
import matplotlib.pyplot as matplotlibInstance
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf


# Now setting the offline and inline setting that are necessary for the Visulaizations.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
init_notebook_mode(connected=True)
cf.go_offline()


# Now Importing the DataSet and Saving it into a DataFrame Variable for the Future Process.

# In[ ]:


airplanesRecord = pandasInstance.read_csv('../input/DelayedFlights.csv')


# Now Checking the Header of the Imported Files, inorder to get the idea about data present.

# In[ ]:


airplanesRecord.head()


# Checking the Information of the Data, inorder to find information of each coulmn.

# In[ ]:


airplanesRecord.info()


# Now as We Have all the basic information Required to Procede, so we can start analysis.

# Let's Get Information by each month Year.

# In[ ]:


groupedByYear = airplanesRecord.groupby(by='Year')


# In[ ]:


groupedByYearCount = groupedByYear.count()


# In[ ]:


groupedByYearCount


# Let's Get Information about each month by the means of grouping.

# In[ ]:


groupedByMonth = airplanesRecord.groupby(by='Month').count()


# In[ ]:


groupedByMonth


# If we want to see the number of flight's per month.

# In[ ]:


matplotlibInstance.figure(figsize=(12,10))
groupedByMonth['FlightNum'].iplot(title='Flights to Month Comparison',color='red')


# Now we can also see that which plane with TailNum has how many flights in the each month.

# Making a pivot table with Tail Number as Index and month as column.

# In[ ]:


tailNumberMonth = airplanesRecord.groupby(by=['TailNum','Month']).count()['FlightNum'].unstack()


# In[ ]:


matplotlibInstance.figure(figsize=(25,20))
matplotlibInstance.tight_layout()
seabornInstance.heatmap(tailNumberMonth)


# Now if we want to see how many flights have been delayed due to weather we can see that too.

# In[ ]:


seabornInstance.jointplot(x='Month',y='WeatherDelay',data=groupedByMonth.reset_index(),kind='kde',color='red')


# In[ ]:


matplotlibInstance.figure(figsize=(25,20))
matplotlibInstance.tight_layout()
seabornInstance.countplot(x='TailNum',data=airplanesRecord)


# The Reason Behind such type of graph is that the Data is Very Larger so it is Difficult to Handle such amount of data.

# Now if we want to check the top ten planes with tail number in record.

# In[ ]:


airplanesRecord['TailNum'].value_counts().head(10)


# Now if we want to check that on which day which amount of flights have been arrived or record is kept.

# In[ ]:


groupedByDay = airplanesRecord.groupby(by=['DayofMonth']).count()


# In[ ]:


matplotlibInstance.figure(figsize=(12,10))
groupedByDay['FlightNum'].iplot(color='red')


# The City Abreviation of the City Which has Occured most in the Data.

# In[ ]:


airplanesRecord['Dest'].value_counts().head(1)


# In[ ]:




