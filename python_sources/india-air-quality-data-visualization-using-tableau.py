#!/usr/bin/env python
# coding: utf-8

# ### Objective:
# 
# * I have tried to do data stroy telling,Data visualization and Dashboarding using Tableau.My intent behind doing this visualization is to explore my visualization skills with tableau also help other by sharing my work which can help them to give ideas about visualization.
# 
# ### Brief About Dataset:
# 
# * This data is combined (across the years and states) and largely clean version of the Historical Daily Ambient Air Quality Data released by the Ministry of Environment and Forests and Central Pollution Control Board of India under the National Data Sharing and Accessibility Policy (NDSAP).
# 
# 1.	Stn_code (Dimension): Stores Particular Station Code for Sates or cities.
# 2.	Sampling_Date (Dimension): Stores particular month and air quality report code.
# 3.	State (Dimension): Stores Name of States.
# 4.	Location (Dimension): Stores Name of Cities.
# 5.	Agency (Dimension): Local Authority name which collected report.
# 6.	 So2 (Measure): Level of Sulphur dioxide.
# 7.	No2 (Measure): Level of Nitrogen dioxide.
# 8.	Rspm (Measure): Level of Respirable Suspended Particulate Matter.
# 9.	Spm (Measure): Level of Suspended Particulate Matter.
# 10.	 location_monitoring_station (Dimension): Place from Air quality measured.
# 11.	 pm2_5 (Dimension): Level between 2 and 5.
# 12.	Date (Dimension): Stores dates monthly from Year 1987 to 2000.
# 13.	Type (Dimension): Stores zone in which air quality measures like industrial, residential.
# 
# 
# ### Data Cleannig:
# 
# * Since dataset is ready to use it doesn't require any pre-processing before doing visualization although I have done some basic cleaning with dataset for example renaming column name Area to Area Type and correcting spelling of some cities and states and for doing this changes I have followed basic steps in Microsfot Excel.
# 
# 
# ### Basic Knowledge About Air Quality Measurements:
# 
# * (So2)-Sulfur dioxide,(No2)-Nitrogen dioxide Are One of Most Dengerous Outdoor Air Pollutants.Outdoor NO2 exposure may increase the risk of respiratory tract infections through the pollutant's interaction with the immune system.Sulfur dioxide (SO2) contributes to respiratory symptoms in both healthy patients and those with underlying pulmonary disease. 
# 
# * Respirable Suspended particulate(RSPM) matters are produced from combustion processes, vehicles and industrial sources.
# 
# * Respirable Suspended particulate(RSPM) matters are produced from combustion processes, vehicles and industrial sources.

# # Data Visualization

# ## Initial Years So2 And No2 Level In India Air

# In[ ]:


from IPython.display import IFrame


# ### Between Year 1987 - 2000 in year 1995 was noted highest level of no2 and so2 level.

# In[ ]:


IFrame('https://public.tableau.com/views/Final_Exam_AssignmentStudent_Id_0748396_15935618099320/Yearly_Avg_So2_No2_90s_decade?::embed=y&:showVizHome=no',width=1000, height=775)


# ## After Year 2000 So2 And No2 Level In India Air

# ### In year 2001 Level of so2 and no2 were highest in India Air 

# In[ ]:


IFrame('https://public.tableau.com/views/Final_Exam_AssignmentStudent_Id_0748396_15935618099320/Yearly_Avg_So2_No2_After_2000?:embed=y&:showVizHome=no',width=1000, height=775)


# ## Growing Avg Spm Levels Across Indian States 

# ### Among All India State Dehli is having highest AVG Spm level

# In[ ]:


IFrame('https://public.tableau.com/views/Final_Exam_AssignmentStudent_Id_0748396_15935618099320/Avg_Spm_State_Wise?:embed=y&:showVizHome=no',width=1000, height=775)


# ## Top 20 Worst Air Quality Citites By Their Avg Rspm Level

# ### Gaziabad City of Uttar Pradesh State Has Highest AVG rspm level

# In[ ]:


IFrame('https://public.tableau.com/views/Final_Exam_AssignmentStudent_Id_0748396_15935618099320/Top_20_Avg_Rspm_City_Wise?:language=en&:embed=y&:showVizHome=no',width=1000, height=500)


# ## Types Of Areas Which have Bad Air Quality By So2 And No2

# ### Residential Area have High Avg So2 level where Industrial,Rural Area have high No2 level

# In[ ]:


IFrame('https://public.tableau.com/views/Final_Exam_AssignmentStudent_Id_0748396_15935618099320/Area_Wise_So2_No2?:language=en&:language=en&:embed=y&:showVizHome=no',width=1000, height=775)


# ## India Capital Is Having Badest Air Qualitly (Animation Chart Press Play button on right side play bar)

# In[ ]:


IFrame('https://public.tableau.com/views/Final_Exam_AssignmentStudent_Id_0748396_15935618099320/Avg_So2_No_2_Over_Year?:language=en&:language=en&:embed=y&:showVizHome=no',width=1000, height=775)


# # Dashboard

# ### Comparing Avg So2 No2 level before and after year 2000 in India Air Quality

# In[ ]:


IFrame('https://public.tableau.com/views/Final_Exam_AssignmentStudent_Id_0748396/Before_After_Comparison?:language=en&:language=en&:embed=y&:showVizHome=no',width=1000, height=775)


# # Data Story Telling

# In[ ]:


IFrame('https://public.tableau.com/views/Final_Exam_AssignmentStudent_Id_0748396/India_Air_Quality?:language=en&:language=en&:embed=y&:showVizHome=no',width=1100, height=500)

