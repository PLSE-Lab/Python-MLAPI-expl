#!/usr/bin/env python
# coding: utf-8

# # Intermediate Seaborn
# 
# ### I will use seaborn library and dataset for this project.

# In[ ]:


#First of all , I will import seaborn librariy and load tips dataset.

import seaborn as sns
tips = sns.load_dataset('tips')
df = tips.copy()
df.head()


# > **Dataset Story**
# > 
# > total_bill: (tip and tax not include)
# > 
# > tip
# > 
# > sex: gender of the bill payer (0=male, 1=female)
# > 
# > smoker: is there any smoker in the group? (0=No, 1=Yes)
# > 
# > day: (3=Thur, 4=Fri, 5=Sat, 6=Sun)
# > 
# > time: (0=Lunch, 1=Dinner)
# > 
# > size: how many people are in the group?

# In[ ]:


# some statistic indicators

df.describe().T


# **Let's look closely at the variables.**

# In[ ]:


df["sex"].value_counts()


# In[ ]:


df["smoker"].value_counts()


# In[ ]:


df["day"].value_counts()


# In[ ]:


df["time"].value_counts()


# **Now we know the data more closely, we will make some graphs so that we can explore the relationship between the variables.**

# # Boxplot Graphs

# **We produce the Boxplot chart so that we can see the:**
# 
# - Minimum Value
# - Median
# - Value Distribution
# - Maksimum Value
# - Outliers

# In[ ]:


sns.boxplot(x=df["total_bill"]);


# ![](https://miro.medium.com/max/552/1*2c21SkzJMf3frPXPAR_gZA.png)

# source: https://towardsdatascience.com/understanding-boxplots-5e2df7bcbd51

# In[ ]:


# We can also change the direction of the graph.

sns.boxplot(x=df["total_bill"],orient = "v");


# ## Crosswise Variables

# **Now we gonna cross the variable for explore same information. We start ask questions to dataset**

# In[ ]:


# Which days restaurant make more money?

sns.boxplot(x="day", y="total_bill",data = df);


# In[ ]:


df["day"].value_counts()


# In[ ]:


# Which days survers make more money?

sns.boxplot(x="day", y="tip",data = df);


# In[ ]:


# And which part of the day restaurant make more money?

sns.boxplot(x="time", y="total_bill",data = df);


# In[ ]:


# What is relation between 'group size' and 'total bill' ?

sns.boxplot(x="size", y="total_bill",data = df);


# In[ ]:


# And Which gender pays the bill by days?

sns.boxplot(x="day", y="total_bill",hue = "sex", data = df);


# # Violin Graphs

# **We produce the Violin graph so that we can see the:**
# 
# - Minimum Value
# - Median
# - More Sensitive Data Distribution.
# - Maksimum Value
# - Interquartile Range

# <img src="//images.ctfassets.net/fi0zmnwlsnja/sdfgtcRp16wTNOcRceGQm/5bfcb73d2261d49ff20dd7857e0152b1/Screen_Shot_2019-03-01_at_11.36.10_AM.png" width="400px">

# source: https://mode.com/blog/violin-plot-examples/

# In[ ]:


sns.catplot(y="total_bill", kind = "violin",data= df);


# In[ ]:


sns.boxplot(x=df["total_bill"]);


# **You can select best graph depend your project. Violin or Boxplot ? **

# ## Crosswise Variables

# In[ ]:


# Let's cross 'day' and 'total_bill'

sns.catplot(x="day", y="total_bill", kind = "violin",data= df);


# In[ ]:


sns.boxplot(x="day", y="total_bill",data = df);


# In[ ]:


# And now cross 3 variable:  'day','total bill' and 'sex'

sns.catplot(x="day", y="total_bill", hue="sex",kind = "violin",data= df);


# # Correlation Charts

# > What is Correlation? Correlation is a statistical technique that can show whether and how strongly pairs of variables are related. For example, height and weight are related; taller people tend to be heavier than shorter people.
# > 
# > We continue to work with the same data set.

# <img src="https://lytongblog.files.wordpress.com/2018/12/Scatter-Plots-and-Correlation-Examples.png?w=863 863w, https://lytongblog.files.wordpress.com/2018/12/Scatter-Plots-and-Correlation-Examples.png?w=150 150w, https://lytongblog.files.wordpress.com/2018/12/Scatter-Plots-and-Correlation-Examples.png?w=300 300w, https://lytongblog.files.wordpress.com/2018/12/Scatter-Plots-and-Correlation-Examples.png?w=768 768w, https://lytongblog.files.wordpress.com/2018/12/Scatter-Plots-and-Correlation-Examples.png?w=1024 1024w, https://lytongblog.files.wordpress.com/2018/12/Scatter-Plots-and-Correlation-Examples.png 1197w" width="700px">

# source : https://lytongblog.wordpress.com/2018/12/21/correlation-between-two-variables/

# In[ ]:


sns.scatterplot(x="total_bill", y="tip",data=df);


# ## Crosswise Variables

# In[ ]:


# Now we cross 3 variables: 'tip', 'time' and 'total_bill'

sns.scatterplot(x="total_bill", y="tip", hue = "time", data =df);


# In[ ]:


# Cross 'tip', 'day' and 'total_bill'

sns.scatterplot(x="total_bill", y="tip", hue = "day", style ="time", data =df);


# In[ ]:


# Now we cross 4 variables: 'tip', 'time','total_bill' and 'day'

sns.scatterplot(x="total_bill", y="tip", hue = "time", style ="day", data =df);


# In[ ]:


# And explore new vizulation methods.

sns.scatterplot(x="total_bill", y="tip",hue = "size", size = "size", data =df);


# # Showing Linear Relationship

# > What is Linear Relationship?
# > 
# > A linear relationship (or linear association) is a statistical term used to describe a straight-line relationship between a variable and a constant. Linear relationships can be expressed either in a graphical format where the variable and the constant are connected via a straight line or in a mathematical format where the independent variable is multiplied by the slope coefficient, added by a constant, which determines the dependent variable.

# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


sns.lmplot(x="total_bill", y="tip",data=df);


# In[ ]:


sns.lmplot(x="total_bill", y="tip",hue ="smoker" , data=df);


# In[ ]:


sns.lmplot(x="total_bill", y="tip",hue ="smoker", col = "time" , data=df);


# In[ ]:


sns.lmplot(x="total_bill", y="tip",hue ="smoker", col = "time" ,row = "sex", data=df);


# ### Please don't forget vote it :)

# In[ ]:




