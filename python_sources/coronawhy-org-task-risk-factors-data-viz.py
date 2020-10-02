#!/usr/bin/env python
# coding: utf-8

# 

# This work was made possible with the support of:
# 
# [Manga Solutions](https://www.mangasolutions.com/), [NASA Jet Propulsion Laboratory](https://www.jpl.nasa.gov/), [The Gordon and Betty Moore Foundation](https://www.moore.org/), [xViz](https://xViz.com/), [Slack](https://www.slack.com/), [Trello](https://www.trello.com/) 

# # CoronaWhy Task-Risk
# 
# The Task-Risk team within the [CoronaWhy](http://www.coronawhy.org) community have started producing some sample results on: understanding COVID-19 risk factors. [More info in this Notebook](https://www.kaggle.com/arturkiulian/coronawhy-org-task-risk-factors).
# 

# In[ ]:


from IPython.display import IFrame
IFrame('https://app.powerbi.com/view?r=eyJrIjoiY2E5YjFkZjItN2Q2ZS00MGI5LWFiMWQtZmY0OWRiZTlkNDVmIiwidCI6ImRjMWYwNGY1LWMxZTUtNDQyOS1hODEyLTU3OTNiZTQ1YmY5ZCIsImMiOjEwfQ%3D%3D', width=800, height=500)


# Here's the link to view it in Full Screen:
# 
# https://app.powerbi.com/view?r=eyJrIjoiY2E5YjFkZjItN2Q2ZS00MGI5LWFiMWQtZmY0OWRiZTlkNDVmIiwidCI6ImRjMWYwNGY1LWMxZTUtNDQyOS1hODEyLTU3OTNiZTQ1YmY5ZCIsImMiOjEwfQ%3D%3D
# 
# This data visualisation has multiple pages - use the page navigation control (bottom center) e.g. **< 1 of 9 >**. Pro-tip: click between the **<** and **>** for a menu of the pages. The first page is for detailed analysis of the results, the later pages are summary statistics to help explain the task-risk methods including a page of Keyword/ngram word clouds.
# 
# Like all Power BI reports, the visuals are highly interactive. Select a bar or row in almost any visual to cross-filter the other visuals on the page. Ctrl-click to multi-select (across visuals and/or within a visual).
# 
# **Suggested method of use:**
# First use the slicers and visuals at the top of the page to narrow the list of papers down to your topic of interest. Then select a paper of interest from the central table - that will filter the details table below. To review the details table in full screen, use the **Focus mode** button in the top-right corner of the details table.
# 
# This visualisation helps to explore the results from the CoronaWhy Task-Risk team datasets.  [More info in this Notebook](https://www.kaggle.com/arturkiulian/coronawhy-org-task-risk-factors). The key attributes focused on are the Risk Factor and Keywords/Ngrams used to search the papers for that Risk Factor.  This was a quick early effort at visualisation, which should evolve over time in response to feedback.
# 
# The secondary data source is the metadata file and associated json files from the [CORD-19 Research Challenge](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/).  
# 
# > The input data is presented in some quick visuals to assist exploration and understanding of the data.
# 
# A static export of the visuals is [available here in PowerPoint format](https://drive.google.com/open?id=1pX3NmeFe50nXcPgRvHpMx1rjczAXew7h).
