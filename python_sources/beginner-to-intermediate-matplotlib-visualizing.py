#!/usr/bin/env python
# coding: utf-8

# # Notebook Description
#     This notebook is long and it is highly recommended to fork and run the cells.
#     
#   This notebook is written to analyze the IRIS dataset with visualizations by using only Matplotlib. Seaborn is better than matplotlib at making statistical plots. But still seaborn is built over matplotlib and matplotlib is used for supporting it too. So visualizing data without using seaborn will give a better base for matplotlib.
#   
#   This notebook contains Exploratory Data Analysis with Visualization, Simple model building and its visualization and Additional features in Matplotlib. The latter part of the notebook is about varying the plotting parameters for better visualizations.
#   
#   For complete beginners, there are collapsed cells after almost all the code cells. If you want description for the code, you can expand the cells.    

# # Importing Modules and loading Data

# Only two modules will be imported in this notebook, pandas and matplotlib.pyplot!

# In[ ]:


import pandas as pd                       
import matplotlib.pyplot as plt           
data=pd.read_csv('../input/iris/Iris.csv') 


#      Code Descriptions:
#      * pandas is a library for data manipulation and analysis.
#      * pd.read_csv() reads the csv file into a pandas dataframe object.

# # Data descriptions

# In[ ]:


print(data.info()) #information on datatype, memory usage, null or non null values
print(data.describe()) #descriptive statistics
print(data.head())     #First few parts of the data


#     Code Descriptions
#     * data.info() - information on datatype, total number of non-null values of each columns/features and total memory usage
#     * data.describe() - descriptive statistics
#     * data.head() - first 5 rows of the data

# Here we note that 
# - All the entries in 6 columns and 150 rows donot have any null values.
# - There is no significant outliers in any of the columns (from descriptive statistics).
# 
# So the data is tidy and doesn't require cleaning!
# 
# If we want to predict the species of the flower, the target variable/column would be 'Species'.

# In[ ]:


data.Species.value_counts()


#     Code Descriptions
#     * data.Species returns a pandas series containing the Species column of 'data'
#     * value_counts() method of a series returns the count for unique values in the series. Now this command returns the number of distinct values of the target variable.

# We find that the 'Species' column has three unique species and the dataset has equal number of flowers in all species. 
# 
# We can also represent it visually using a pie chart and bar chart. 

# In[ ]:


Target_counts = data.Species.value_counts()
Target_labels = data.Species.unique()
plt.pie(Target_counts,explode = [0.1,0.1,0.1],labels = Target_labels)


#     Code Descriptions
#     * data.Species.unique() - returns the unique values of the 'Species' column.
#     * plt.pie() - plots the pie chart from the given data, explode arguments specifices how much a pie must be exploded from the rest. In this case we have equal explotions for the three. You can try varying them.
#     * 'labels' keyword sets the labels of the cooresponding sector. 

# In[ ]:


plt.bar(Target_labels,Target_counts)
plt.ylabel('Counts')
plt.title('Distribution of the Target variable')


#     Code Descriptions
#     * plt.bar() plots a bar chart
#     * plt.ylabel() sets the y axis label
#     * plt.title sets plot title

# Dataset has six column, one is target and of the five other columns, 'Id' cannot be a factor affecting the Species. The column was included for better data collection and unique identification of the records/rows. Both these purposes would not be required in the analysis, so id should be dropped from the dataset.

# In[ ]:


data.drop('Id',axis=1,inplace=True)

print(data.head())


#     Code Descriptions
#     * data.drop() - drops rows or labels from the data. In this case columns are dropped (axis=1).
#     * here inplace=True will alter the dataset 'data'. It is equivalent to data=data.drop(...) 
#     * inplace=False or default would return the result but would not alter the dataset

# **Classifications based on the Sepal**

# Now, we have 4 columns/features to classify the species. 
# 
# Length and Width of Sepal and length and width of Pedal. If you got a sepal with its length and width can you predict the species? Lets see how

# In[ ]:


#Trifurcating data according to there species
df1 = data[data.Species=='Iris-setosa']
df2 = data[data.Species=='Iris-versicolor']
df3 = data[data.Species=='Iris-virginica']


#     Code Descriptions
#     * data.Species == 'Iris-setosa' returns a series with boolean values whether the species is setosa or not (True or False).
#     * data[data.Species=='Iris-setosa'] returns those rows of the data which has target species as setosa.

# In[ ]:


#plotting sepal Length vs width for different species in the same plot 
plt.plot(df1.SepalLengthCm,df1.SepalWidthCm,linestyle='none',marker='o',c='red',label='setosa')
plt.plot(df2.SepalLengthCm,df2.SepalWidthCm,linestyle='none',marker='o',c='green',label='versicolor')
plt.plot(df3.SepalLengthCm,df3.SepalWidthCm,linestyle='none',marker='o',c='blue',label='virginica')

#setting title,labels,legend
plt.title('Sepal Width and Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')

plt.legend()

plt.show() 


#     Code Descriptions
#     * plt.plot() is used to plot line or scatter plots.
#     * linestyle, marker and label are intutive keywords.
#     * 'c' representation of colors to be used for each species.
#     * plt.legend() displays the legends representing which plot which row corresponds to which Species

# This code can be shortened as the following

# In[ ]:


#This is a simplified version of above cell
plt.plot(df1.SepalLengthCm,df1.SepalWidthCm,'ro')
plt.plot(df2.SepalLengthCm,df2.SepalWidthCm,'go')
plt.plot(df3.SepalLengthCm,df3.SepalWidthCm,'bo')

#Note that this can be further simplified by using seaborn, 
#but this notebook is for strictly using matplotlib   
plt.title('Sepal Width and Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')

plt.legend(['setosa','versicolor','virginica'])

plt.show()


# From the figure we clearly see that 'setosa' can be clearly distinguished from the other two. But classifying 'versicolor' and 'virginica' would not be easy. We can also see given a pedal will we be able to classify the three classes.

# In[ ]:


plt.plot(df1.PetalLengthCm,df1.PetalWidthCm,'ro')
plt.plot(df2.PetalLengthCm,df2.PetalWidthCm,'go')
plt.plot(df3.PetalLengthCm,df3.PetalWidthCm,'bo')

plt.title('Petal Width and Length')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')

plt.legend(['setosa','versicolor','virginica'])

plt.show()


# > From this figure we can clearly distinguish 'setosa' from other two and also we can distinguish other two to an extend.
# Is it possible to distinguish with just one feature? Lets try plotting boxplot and resolve the issue.

# In[ ]:


plt.boxplot([df1.SepalLengthCm,df2.SepalLengthCm,df3.SepalLengthCm])
plt.title('Sepal Length')
plt.xticks([1,2,3],['Setosa','Versicolor','Virginica'])
plt.xlabel('Species')
plt.ylabel('Sepal Length in cm')


#     Code Descriptions
#     * plt.boxplot() shows the distribution of the series.
#     * plt.xticks() to specify specific points(ticks) in X axis

# Generally Virginica has longer sepals and Setosa has shorter sepals, but this cannot be taken as a factor as there is a significant overlap in the range of values.

# In[ ]:


plt.boxplot([df1.SepalWidthCm,df2.SepalWidthCm,df3.SepalWidthCm])
plt.title('Sepal Width')
plt.xticks([1,2,3],['Setosa','Versicolor','Virginica'])
plt.xlabel('Species')
plt.ylabel('Sepal Width in cm')


# In[ ]:


plt.subplot(121)

plt.boxplot([df1.PetalLengthCm,df2.PetalLengthCm,df3.PetalLengthCm])
plt.title('Petal Length')
plt.xticks([1,2,3],['Setosa','Versicolor','Virginica'])
plt.xlabel('Species')
plt.ylabel('Petal Length in cm')

plt.subplot(122)

plt.boxplot([df1.PetalWidthCm,df2.PetalWidthCm,df3.PetalWidthCm])
plt.title('Petal Width')
plt.xticks([1,2,3],['Setosa','Versicolor','Virginica'])
plt.xlabel('Species')
plt.ylabel('Petal Width in cm')

plt.tight_layout()

plt.show()


#     Codes Descriptions
#     * plt.subplot() - To make subplots. We can plot many axes in a single figure.
#     * plt.subplot(121) - Subplot 1 of 1 x 2 subplots ( 1 row, 2 columns)
#     * plt.subplot(122) - Subplot 2 of 1 x 2 subplots ( 1 row, 2 columns)
#     * plt.tightlayouts() - Organizes the labels, plots of the axes in the plot.

# With the petal length alone we can tell whether the flower is setosa or not. Also with width of the petal we can find whether it is setosa or not. However we cannot distinguish between Versicolor and Viriginica as there is a overlap in the range of values.

# # Building a simple model
# At this point we can try building a simple model to predict whether a given species is setosa or not. For this the column petal length is sufficient, so we will take this and the species column. 
# 
# 
# And before that we have to set the threshold for classification

# In[ ]:


# Classification threshold
threshold=max(df1.PetalLengthCm)
print(threshold)


# Now we can clearly distinguish. If the petal length is less than 1.9 cm, then it is setosa else it is not setosa. 
# 
# **We are supposed to solve a three-class classification problem, but here we can solve two class classsification problem (Setosa or not) for the sake of simplicity. **

# In[ ]:


# we will define a function here to return whether the flower is setosa or not
def predict(length):
    if length<=1.9:
        return True
    else:
        return False    


# In[ ]:


#data is the dataframe containing the complete dataset
twoclass=data[['PetalLengthCm','Species']]


#     Code Description
#     * data[col_list] returns a dataframe containing only the columns from the col_list. col_list is the list of column names required. Here we required PetalLengthCm and Species.
#     * So twoclass is the dataframe for 2 class classification with one feature and one target variable 

# Also we have reduced a 4 dimensional problem(4 features) to  1D (1 feature) problem.

# In[ ]:


# We store whether the Species is setosa or not from the known Species column.
# This is the actual value
actual=twoclass.Species=='Iris-setosa'
#The Species are predicted from petal length
predicted=twoclass.PetalLengthCm.apply(predict)


#     Code Descriptions
#     * actual contains the actual value as in the dataset.
#     * .apply() - attribute of a series object applies the function in the argument to the series. This does the prediction for the respective value.

# Now we have predicted the values and we know the actual value. We will evaluate how well we have predicted the result. 

# In[ ]:


accuracy=sum(actual==predicted)/len(actual)*100
print('accuracy is {}%'.format(accuracy))


# Now we have constructed a model to classify whether the given species is setosa or not with 100% accuracy!
# 
# **kudos!** 
# 
# **We have built a model based on decision tree classification.**
# 
# What we have done here is:
# 
# * We have set the threshold based on 150 data.
# * We classified the 150 data based on the threshold. 
# 
# This is like answering book back questions. You need to predict those data that has not appeared before. Thats why we have to split the entire data set into two and make use of one to train and other to evaluate. This is an introduction to machine learning and i would come with a new notebook 'Machine learning with Iris' very soon.
# 
# You can also try finding threshold of Petal width and classify whether the species is setosa or not.
# 
# We would here visualize the results of the model
# 

# In[ ]:


#Visualizing the result
plt.plot(twoclass.PetalLengthCm,predicted,'bo')
plt.title('Setosa or not')
plt.xlabel('Petal length in cm')
plt.yticks([0,1],['Not setosa','setosa'])
plt.vlines(1.9,0,1,colors='red') 


#     Code Descriptions
#     * plt.vlines(1.9,0,1) - plots a vertical line at x = 1.9 and the line extends from y=0 to y=1. We specify the color as red 

# Here red line is the threshold. We see that the line clearly seperates setosa or not leading to 100% accuracy.

# # Other Visualizations
# Lets visualize the data with some more visualization tools. Note that we will be making use of Univariate analysis here. 

# **Violin Plots**
# Violin plots are similar to box plot except that they have the distribution of data in addition to the range in the form of curve rather than a box.

# In[ ]:


#violin plot
plt.figure(figsize=(15,17))

plt.subplot(221)
plt.violinplot([df1.SepalLengthCm,df2.SepalLengthCm,df3.SepalLengthCm],showmedians=True)
plt.xlabel('Species')
plt.xticks([1,2,3],['Setosa','Versicolor','Virginica'])
plt.ylabel('Sepal Length in cm')
plt.title('Sepal Length')

plt.subplot(222)
plt.violinplot([df1.SepalWidthCm,df2.SepalWidthCm,df3.SepalWidthCm],showmedians=True)
plt.xlabel('Species')
plt.xticks([1,2,3],['Setosa','Versicolor','Virginica'])
plt.ylabel('Sepal Width in cm')
plt.title('Sepal Width')

plt.subplot(223)
plt.violinplot([df1.PetalLengthCm,df2.PetalLengthCm,df3.PetalLengthCm],showmedians=True)
plt.xlabel('Species')
plt.xticks([1,2,3],['Setosa','Versicolor','Virginica'])
plt.ylabel('Petal Length in cm')
plt.title('Petal Length')

plt.subplot(224)
plt.violinplot([df1.PetalWidthCm,df2.PetalWidthCm,df3.PetalWidthCm],showmedians=True)
plt.xlabel('Species')
plt.xticks([1,2,3],['Setosa','Versicolor','Virginica'])
plt.ylabel('Petal Width in cm')
plt.title('Petal Width')

plt.tight_layout()


# **Histogram**
# 
# Distribution of the values in a series.

# In[ ]:


#histogram
plt.figure(figsize=(15,15))
plt.subplot(221)
plt.hist(data.SepalLengthCm,bins=30)
plt.title('Sepal Length')

plt.subplot(222)
plt.hist(data.SepalWidthCm,bins=30)
plt.title('Sepal Width')

plt.subplot(223)
plt.hist(data.PetalLengthCm,bins=30)
plt.title('Petal Length')

plt.subplot(224)
plt.hist(data.PetalWidthCm,bins=30)
plt.title('Petal Width')


# There seems to be two clusters in both Petal Length and Width. We have already drawn a box plot and we know that one of the cluster corresponds to 'Setosa'. This can be better visualized when the data is plotted seperately for each species.

# **Species wise Histograms**

# In[ ]:


#histogram
plt.figure(figsize=(15,15))
plt.subplot(221)
plt.hist(df1.SepalLengthCm,bins=30)
plt.hist(df2.SepalLengthCm,bins=30)
plt.hist(df3.SepalLengthCm,bins=30)
plt.legend(['Setosa','Versicolor','Virginica'])
plt.title('Sepal Length')

plt.subplot(222)
plt.hist(df1.SepalWidthCm,bins=30)
plt.hist(df2.SepalWidthCm,bins=30)
plt.hist(df3.SepalWidthCm,bins=30)
plt.legend(['Setosa','Versicolor','Virginica'])
plt.title('Sepal Width')

plt.subplot(223)
plt.hist(df1.PetalLengthCm,bins=30)
plt.hist(df2.PetalLengthCm,bins=30)
plt.hist(df3.PetalLengthCm,bins=30)
plt.legend(['Setosa','Versicolor','Virginica'])
plt.title('Petal Length')

plt.subplot(224)
plt.hist(df1.PetalWidthCm,bins=30)
plt.hist(df2.PetalWidthCm,bins=30)
plt.hist(df3.PetalWidthCm,bins=30)
plt.legend(['Setosa','Versicolor','Virginica'])
plt.title('Petal Width')


# # Additional features in Matplotlib

# You are half way! We are going to redo the same plots above to enhance our visualizations!
# 
# We will add attributes for the plots to make them more pleasing visually

# We will first apply style sheets that will customize the style of the plot. Once the style is changed every figure plotted after it will have the same style until the style is again changed. Try removing the '#' and rerun the notebook, also try different built-in styles.

# In[ ]:


#plt.style.use('fivethirtyeight')


# In[ ]:


#plt.style.use('seaborn')


# In[ ]:


#plt.style.use('default')


# Here is the comprehensive list of  built-in style sheets:  
# https://matplotlib.org/3.1.0/gallery/style_sheets/style_sheets_reference.html

# **Saving the Figures**
# 
# We haven't saved any figure yet till now. But we will not leave one from now on...
# 
# plt.savefig(*filename*) is used for saving a figure lets save them.

# **Pie-chart:**

# In[ ]:


Target_counts = data.Species.value_counts()
Target_labels = data.Species.unique()
plt.pie(Target_counts,explode = [0.2,0,0],labels = Target_labels, 
        startangle = 90, autopct = '%1.2f%%', shadow = True, colors = ['red','green','orange'])
plt.axis('equal')
plt.title('Distribution of Species in the Data')
plt.legend(loc='upper right')
plt.savefig('pie.jpg')


#     Code Descriptions
# 
#     * 'startangle' keyword helps rotating the chart in the counter-clockwise direction. Compare this pie chart with the previous where the pie chart ('iris-setosa' sector) begins at 0 deg.
# 
#     * 'shadow' will plot shadow for the pies giving a good effect to the plot.
# 
#     * 'autocpt' argument will add the percentage contribution from each group on their respective pies.
# 
#     It is always recommended to use explosion for one pie alone. Here 'Iris - virginica' is highlighted and this is alone exploded
# 
#     * 'colors' are used to set the colors for each pie. Choosing colors in data visualiztion is one of the most required skills. Read more about colors in the following links:
#     https://medium.com/@Elijah_Meeks/viz-palette-for-data-visualization-color-8e678d996077
#     https://www.dataquest.io/blog/what-to-consider-when-choosing-colors-for-data-visualization/
# 
#     * plt.axis('equal') will constraint the axis in a square.
# 
#     * plt.legend(loc='upper right') plots the legend and locates the legend at the upper right corner. These sort of positioning plays a vital role in your presentations. 

# **Bar Chart**
# 
# We can try horizontal plot here. but,it is highly advisable to use vertical bar plots for few variables and horizontal bar plot for data with many variables.So we restrict to vertical bar plot. Horizontal bar plots are similar but they are constructed using the function plt.barh()  

# In[ ]:


plt.bar(Target_labels,Target_counts,
        width=0.5,color=['Yellow','Green','Blue'],edgecolor='red',linewidth=[2,0,0])
plt.title('Distribution of the Target variable')
plt.savefig('bar.jpg')


#     Code Description
#     * 'width' - specifies the width of the bars 
#     * 'color' - specifies the color of each bars
#     *  edgecolor - specifies the edge color
#     * linewidth=[2,0,0] -specifies line width fot the edges in the bar.

# Here all the species have equal distributions, so there is no point in hightlighting one! But we highlight 'setosa' to show how to highlight for this is a visualization tutorial!

# **Scatter Plot**
# 
# Scatter plot between Petal Length and width will be drawn here . We will vary the size using 's' keyword.

# In[ ]:


plt.plot(df1.PetalLengthCm,df1.PetalWidthCm,linestyle='none',c= 'r',marker='o',ms=5,mec='yellow')
plt.plot(df2.PetalLengthCm,df2.PetalWidthCm,linestyle='none',c='g',marker='d',ms=3,alpha=0.5)
plt.plot(df3.PetalLengthCm,df3.PetalWidthCm,linestyle='none',c='b',marker='^',ms=3,alpha=0.5)

plt.title('Petal Width and Length')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')

plt.legend(['setosa','versicolor','virginica'])

plt.savefig('scatter.jpg')


#     Code Description
#     * 'c' - color
#     * ms - marker size
#     * mec - marker edge color
#     * alpha - transparency with range(0,1). alpha=1 is fully opaque. This useful incase if the plots are overlapped 

# **Box plots**
# 
# We will enhance the box plots we have constructed earlier here. We use Petal length alone to construct boxplot. Here only add a notch to the box plot. There also options for altering the percentile for the whisker length, but this is not generally used unless you are adviced to increase or decrease outliers or for any other special cases.

# In[ ]:


plt.boxplot([df1.PetalLengthCm,df2.PetalLengthCm,df3.PetalLengthCm],notch=True)
plt.title('Petal Length')
plt.xticks([1,2,3],['Setosa','Versicolor','Virginica'])
plt.xlabel('Species')
plt.ylabel('Petal Length in cm')
plt.savefig('boxplot.jpg')


# **The Model Visualization**
#    Here we will make the visualization of our model results more pleasing. We will add shading instead of line.
# 

# In[ ]:


plt.plot(twoclass.PetalLengthCm,predicted,'bo')
plt.title('Setosa or not')
plt.xlabel('Petal length in cm')
plt.yticks([0,1],['Not setosa','setosa'])
plt.axvspan(0,2.0,facecolor='green',alpha=0.3)


#     Code Description
#     * No more code description as this seems intuitive 

# If you are looking for setosa, you can acces the green highlighted area! 

# # Other links
# 
# **Matplotlib+ Seaborn + Pandas: An Ideal Amalgamation for Statistical Data Visualisation**:
#     This is a good article on data visualization. It explains the components of a figure, basics to matplotlib, matplotlib codes for reproducing seaborn functions, how to optimize seaborn functions and groups the seaborn functions based on input.
# https://towardsdatascience.com/matplotlib-seaborn-pandas-an-ideal-amalgamation-for-statistical-data-visualisation-f619c8e8baa3
# 
# **Matplotlib cheatsheet from datacamp**:
#    As you get deeper into visualization, you may need a collection of important functions in matplotlib. This would satisfy your need then.  
# https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Python_Matplotlib_Cheat_Sheet.pdf
# 
# **Matplotlib gallery**:
#     Matplotlib Documentation has a good collection of codes for elegant plots for visualization. The problem with these codes are very long and you can have a glance at the range of visualizations available and choose the one that will satisfy your requirement. I will strongly advice to use seaborn instead of matplotlib for statistical plot. Seaborn is actually built on matplotlib and almost can make statistical plots in a line or two. 
# https://matplotlib.org/gallery/index.html

#     ******************This is the end of the notebook! Thanks for reading!******************
# 
# Any suggestions, feed back and comments are welcomed!
