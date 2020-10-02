#!/usr/bin/env python
# coding: utf-8

# # Covid-19 in India Analysis
# These datasets are not as recent as the situation is in India now.

# Here I am importing the different libraries I will be using and giving them a name. 

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# These are the datasets I will be using for my code and the name for them. They contain information from India's Covid-19 situation, and you will see the information I analyzed and pulled out from my code.

# In[ ]:


data = pd.read_csv("../input/covid19-in-india/AgeGroupDetails.csv")
data2 = pd.read_csv("../input/covid19-in-india/IndividualDetails.csv")


# I used "data.columns" in order to print out all of the columns in the dataset to see which ones I needed, so I could make my graphs using them.

# In[ ]:


print(data.columns)


# I printed out the information I needed from the columns "AgeGroup" and "Percentage" in order to make my graphs.

# In[ ]:


print(data[['AgeGroup','Percentage']])


# The first line, specifying the age groups, contains the x values, and the second line, the number percentages for each corresponding age group, contains the y values I pulled from the output of my previous cell.  

# In[ ]:


AgeGroups=['0-9','10-19','20-29','30-39','40-49','50-59','60-69','70-79','80+','unknown']
Percentage=[3.18,3.9,24.86,21.1,16.18,11.13,12.86,4.05,1.45,1.30]


# Here I had to create a y position using "np.arrange", so that I could graph my values. If I had just written "plt.bar(AgeGroups,Percentages)", it would not graph it because it does not honor the string values.

# In[ ]:


ypos = np.arange(len(AgeGroups))
print(ypos)


# Here I used "plt.xticks" in order to label the bars with the corresponding name. I titled and labeled the other parts of the graph, and I used "plt.bar" to graph the values and "plt.show()" to display the graph.

# In[ ]:


plt.xticks(ypos,AgeGroups)
plt.ylabel("Percentages")
plt.xlabel("Age Groups")
plt.title("Covid-19 Cases in India by Age Group")
plt.bar(ypos,Percentage)
plt.show()


# I used the same values to create this graph because I wanted it to be visualized in a different way.

# In[ ]:


plt.xticks(ypos,AgeGroups)
plt.ylabel("Percentages")
plt.xlabel("Age Groups")
plt.title("Covid-19 Cases in India by Age Group")
plt.plot(ypos,Percentage)
plt.show()


# I printed out the second dataset to see the column names using "data2", and I also used "data2.shape" to know how many rows and columns there are.

# In[ ]:


print(data2.shape)
data2


# I used "pd.value_counts" in order to see the number of each case status.

# In[ ]:


print(pd.value_counts(data2['current_status']))


# The first line is the x values and the second line is the y values I pulled from the output of my previous cell.

# In[ ]:


case_status=['Deceased','Recovered','Active'] 
number_of_cases=[46,181,27662]


# Here, like my previous graphs, I had to create a y position using "np.arrange", so that I could graph my values. If I had just written "plt.bar(case_status,number_of_cases)", it would not graph it because it does not honor the string values.

# In[ ]:


yposition = np.arange(len(case_status))
print(yposition)


# Here I used "plt.xticks" in order to label the bars with the corresponding name. I titled and labeled the positions of the graph, and I used "plt.bar" to graph the values and "plt.show()" to present the graph.

# In[ ]:


plt.xticks(yposition,case_status)
plt.ylabel("Number of Cases")
plt.xlabel("Case Status")
plt.title("The Status of Covid-19 Cases in India")
plt.bar(yposition,number_of_cases)
plt.show()


# My code from here to the next graph is the same as the code above I just made it only the "Deceased" and "Recovered" case numbers, so the values would be more visible.

# In[ ]:


case_status2=['Deceased','Recovered'] 
number_of_cases2=[46,181]


# In[ ]:


yposition2 = np.arange(len(case_status2))
print(yposition2)


# In[ ]:


plt.xticks(yposition2,case_status2)
plt.ylabel("Number of Cases")
plt.xlabel("Case Status")
plt.title("The Status of Covid-19 Cases in India")
plt.bar(yposition2,number_of_cases2)
plt.show()


# # Learning Reflection
#    I have learned lots of new coding skills over the course of me working on my East Final Project. I learned how to import a dataset on a Kaggle notebook. After importing the dataset, I learned how to analyze it. I was able to display the dataset to see all the different rows, columns, and values. I learned how to pull the data out in order to perform a specific task. 
#    
#    I learned how to use matplotlib, pandas, and numpy to graph a dataset. I was able to interpret the data I had previously pulled out. I learned how to use numpy to fix the issues with my coding. With this data, I was able to make bar and line plot graphs out of it. I learned how to label, title, and display the graphs. 
#    
#    This coding experience has taught me a lot about the topic. I learned that coding takes a lot of time to understand. There are many instances where you will try to get a result you want, but it will constantly say "error". For me, this learning process was all about trial and error and understanding the values, variables, functions, arguements, and strings I am typing. However, when it clicks and the code works, it is one of the best feelings in the world as you get to see the process it took and how you learned to perform a specific task you wanted to be done.
