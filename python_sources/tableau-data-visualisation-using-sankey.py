#!/usr/bin/env python
# coding: utf-8

# 

# ## If you like this kernel Greatly Appreciate  to UPVOTE.Thank you
# 
# 
# https://public.tableau.com/profile/pavan.kumar.sanagapati
# 
# #  Tableau Data Visualisation using  Sankey
# 
# Sankey diagrams are a specific type of flow diagram, in which the width of the arrows is shown proportionally to the flow quantity.
# 
# The illustration shows a Sankey diagram which represents all the primary energy flows into a factory. The widths of the bands are directly proportional to energy production, utilization and losses. The primary energy sources are gas, electricity and coal/oil and represent energy inputs at the left hand side of the Sankey diagram. They can also visualize the energy accounts, material flow accounts on a regional or national level, and also the breakdown of cost of item or services.
# 
# Sankey diagrams put a visual emphasis on the major transfers or flows within a system. They are helpful in locating dominant contributions to an overall flow. Often, Sankey diagrams show conserved quantities within defined system boundaries.
# 
# ## Case Study - African Mobile Data Profitability  Analysis
# 
# #### Source - https://www.superdatascience.com/case-study-014-tableau-african-mobile-profitability-analysis/
# 
# You are a Data Consultant brought in by African Mobile, a mobile company that aggressively expanded across the entire continent of Africa beginning in 2013. Business has been humming along for the past four years, and they would like you to conduct a profitability analysis of their business. This will involve looking at not only profitability, but also Salesperson performance. African Mobile has requested the following:
# 1. An interactive dashboard showing Profit by City, tied to Profit by Segment and a Profit Trend (Use a map, a bar chart, and a line chart). African Mobile should be able
# to select a City on the map and see the other charts adjust.
# 2. An interactive scatterplot showing the relationship between Profit and Sales by either Region, Country, or City, depending on their choice (Use a scatter and a parameter).
# 3. An interactive Salesperson analysis showing Contracts Sold by each person, as well as a second chart showing difference from a selected Salesperson (Use two bar charts
# and a parameter).
# 4. A two-way matrix, over time, tracking Salesperson quarterly performance against benchmarks (in parentheses) for Contracts Sold (10) and Close Rate (30%) (Use the
# Pages functionality).
# 
# ### Challenge
# 5. African Mobile has long experience with Tableau and is not easily impressed. Cap your workbook off with a Sankey Chart showing how Sales flow between Segment
# and Region to really blow them away.

# # Sankey Chart
# 
# 
# This is Tableau data visualisation representation of the African Mobile data obtained from the above source to display profitability analysis of mobile users across segment and region to clear articulate the profitability and areas where the sales team need to focus for improvement.

# In[ ]:


#Import section

from IPython.display import IFrame
IFrame('https://public.tableau.com/views/AfricanMobileSalesData-SankeyChart/Dashboard1?:embed=y&:display_count=yes', width=800, height=925)


# In[ ]:




