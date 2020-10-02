#!/usr/bin/env python
# coding: utf-8

# # VIOLIN PLOT

#     Violin plot is used to visualize the distribution of variable which has multiple groups.It is similar to Box plot but gives more information about density. We can use such plots when we have large amount of data and when its difficult to track individual record. Violinplot is part of seaborn library.

# # When to use?

# The input data for violin format should be as mentined below.

# In[ ]:


import pandas as pd
Group=['X','X','Y','Y','Z','Z']
Data=[12,23,34,23,56,76]
df=pd.DataFrame({"Group":Group,"Data":Data})
df


# Or Data should be like below

# In[ ]:


Group1=[1,2,3,4]
Group2=[5,6,7,8]
Group3=[9,1,3,4]
df=pd.DataFrame({"Group1":Group1,"Group2":Group2,"Group3":Group3})
df


# Plotting distribution of data based on single variable , Here we are plotting data sepal_length of iris dataset on y axis.

# In[ ]:


import seaborn as sns
df = sns.load_dataset('iris')
sns.violinplot( y=df["sepal_length"] )
print(df.head())


# Same plot which we have drawn based on Y axis has been plotted on X axis. So basic idea is how to mold the format of the violin plot in vertical orhorizontal format by altering the X and Y axis data.

# In[ ]:


import seaborn as sns
df = sns.load_dataset('iris')
sns.violinplot( x=df["sepal_length"] )
print(df.head())


# How to Plot violin plot based on two parameters?
# 
# Just look below part of code, where we have considered the iris dataset and we are plotting the violin plot of species and sepal_lenth, which are specified on x and y axis. We can change colour of the violin plots which we will see n next part of notebook.

# In[ ]:


import seaborn as sns
df = sns.load_dataset('iris')
sns.violinplot( x=df["species"], y=df["sepal_length"] )
print(df.head())


# How to plot the Violin plots without mentioning x and y axis?
# 
# Yes, You can plot the violin plot without mentioning x and y while ploting , the alternative is to use df.iloc/df.ix/df.loc to select which rows to select and columns to select.
# Here we have used data=df.iloc[:,0:2] which means select all rows as we have not sliced rows but 0:2 means select 0th and 1st column of dataframe which is sepal_lenth and sepal_width.
# So this will plot sepal_lenth , sepal_width on x axis and consider distribution of data of all rows.

# In[ ]:


import seaborn as sns
df = sns.load_dataset('iris')
sns.violinplot(data=df.iloc[:,0:2])
print(df.head())


# How to use hue in Violin plot?
# 
# Now we are considering tips dataset. Here we are plotting days on x axis while total_bill on y axis. Here we are mentioning hue as smoker , and we know smoker has only 2 values as yes and no, which will plot the distibution of total_bills for each days for two distinct values of smoker [Yes,No]. 
# Palette in seaborn represent the colour palette . There are six variations of the default theme, called deep, muted, pastel, bright, dark, and colorblind.

# In[ ]:


import seaborn as sns
df = sns.load_dataset('tips')
sns.violinplot(x="day", y="total_bill", hue="smoker", data=df, palette="Pastel1")
print(df.head())


# How to plot Violin plot in vertical/Horizontal format?
# 
# We have already seen how to plot the violin in horizontal/vertical format. Just switch the x and y axis.

# In[ ]:


import seaborn as sns
df = sns.load_dataset('iris')
sns.violinplot( y=df["species"], x=df["sepal_length"] )
print(df.head())


# How to change linewidth of violin plot?
# 
# Here is the solution, Just include parameter linewidth and specify the size/ you can use width as well to change width of violin plot

# In[ ]:


import seaborn as sns
df = sns.load_dataset('iris')
sns.violinplot( x=df["species"], y=df["sepal_length"], linewidth=5)


# In[ ]:


sns.violinplot( x=df["species"], y=df["sepal_length"], width=0.3)


# How to use sequential colour palettes?
# 
# Sequential colour palettes are nothing but the variation of brightness/darness in single colour. Here we are using Blues , which reperesent all variation of colour Blue.

# In[ ]:


import seaborn as sns
df = sns.load_dataset('iris')
sns.violinplot( x=df["species"], y=df["sepal_length"], palette="Blues")


# How to use consistent colour for violin?
# 
# Yes, You can use same colour for all violins in your plots using "color" parameter rather than using "palette".Here we are using skyblue color to represent single colour to all violins in plots.

# In[ ]:


import seaborn as sns
df = sns.load_dataset('iris')
sns.violinplot( x=df["species"], y=df["sepal_length"], color="skyblue")


# How to create your own colour palette and asign it to violins in your plot?
# 
# 1.Create dictionary of species of iris as key and colour as value [g=Green, b=Blue, m=Magenta]
# 
# 2.Use self created palette(Dictionary) as value in violin plot for palette parameter..

# In[ ]:


import seaborn as sns
df = sns.load_dataset('iris')
my_pal = {"versicolor": "g", "setosa": "b", "virginica":"m"}
sns.violinplot( x=df["species"], y=df["sepal_length"], palette=my_pal)


# Aternative way to create own pallette dictionary:

# In[ ]:


import seaborn as sns
df = sns.load_dataset('iris')
my_pal = {species: "r" if species == "versicolor" else "b" for species in df.species.unique()}
sns.violinplot( x=df["species"], y=df["sepal_length"], palette=my_pal)


# How to change order of violins in violin plots?
# 
# You can also change order of violins from your plot using order parameter. Check below example for more clarity. After applying order parameter you can see the x axis are as per the order we have specified.

# In[ ]:


import seaborn as sns
df = sns.load_dataset('iris')
sns.violinplot(x='species', y='sepal_length', data=df, order=[ "versicolor", "virginica", "setosa"])


# You can also create order as specified below. Here we are grouping the data based on species which are on x axis and we are finding out median of sepal_lenth which is on y axis, and as we are using -1 in steps which specifies that we want the order in desending order.

# In[ ]:


import seaborn as sns
df = sns.load_dataset('iris')
my_order = df.groupby(by=["species"])["sepal_length"].median().iloc[::-1].index
sns.violinplot(x='species', y='sepal_length', data=df, order=my_order)


# Please upvote notebook if you found it helpful.

# In[ ]:




