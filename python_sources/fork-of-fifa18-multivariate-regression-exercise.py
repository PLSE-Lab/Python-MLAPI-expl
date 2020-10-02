#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# (c) Sydney Sedibe, 2018

import warnings 
warnings.filterwarnings('ignore')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

file_list = os.listdir("../input")
print(file_list)


# **This dataset contains information about Players from the EA Sports game, FIFA 18. We will take this data, clean it, analyse it, and gain insights from it. First let's load the data.**

# In[ ]:


odf = pd.read_csv("../input/" + file_list[1], low_memory=False) # odf = original dataframe with complete dataset
odf.head()


# **Now, let's inspect it...**

# In[ ]:


def showDetails(df):
    print("-------------------------------------------------------------------------------------------------------------------")
    print('{:>35}'.format("Shape of dataframe:") + '{:>12}'.format(str(df.shape)))
    containsNulls = "Yes" if df.isnull().any().any() else "No"
    print("Does dataframe contain null values: " + containsNulls)
    null_columns = df.columns[df.isnull().any()]
    print("Number of columns with null values: " + str(df[null_columns].isnull().any().sum()))
    null_rows = df[df.isnull().any(axis=1)][null_columns]
    print("Number of records with null values: " + str(len(null_rows)))
    print('{:>35}'.format("Percentage of null records:") + '{:>6.2f}'.format(len(null_rows) / len(df) * 100) + "%")
    print("-------------------------------------------------------------------------------------------------------------------")

showDetails(odf)


# **Our dataframe contains 17 981 records, with 75 columns. There are 27 columns and 2 235 records with null values, and these null records account for 12.43% of the total number of records .**

# **Let's inspect these null records to see if we can find the source of the missing data...**

# In[ ]:


nv_df = odf[odf.isnull().any(axis=1)] # nv_df ==> null value dataframe
showDetails(nv_df)
nv_df.head()


# **A quick inspection of the null value records in our dataframe show that those null value records are for goalkeepers, mostly. This is because goalkeepers are not assigned a score for playing other positions on the field.**

# **Let's clean the Wage and value columns so that it only contains digits and no other characters.**

# In[ ]:


def toFloat(string):
    """Function to convert Wage and Value strings to floats"""
    string = string.strip(" ")
    if string[-1] == 'M':
        return float(string[1:-1]) * 1000000
    elif string[-1] == 'K':
        return float(string[1:-1]) * 1000
    else:
        return float(string[1:])


# In[ ]:


odf["NumericalWage"] = [toFloat(x) for x in odf["Wage"]]
odf["NumericalValue"] = [toFloat(x) for x in odf["Value"]]


# Now the 'Wage' and 'Value' columns have been converted to floats. Unfortunately, there's a problem. Some of the values are zero.

# In[ ]:


print("There are " + str(len(odf[odf["NumericalWage"] == 0])) + " rows with a wage value of 0 in the NumericalWage column")
print("There are " + str(len(odf[odf["NumericalValue"] == 0])) + " rows with a player-value of 0 in the NumericalValue column")


# For those values, we will replace them with the respective average values of their columns.

# In[ ]:


def replaceZeroValues(df, column):
    subset = df[ df[column] != 0 ][column]
    nonzero_mean = subset.mean()
    print("The nonzero_mean for " + column + " is " + str(nonzero_mean))
    df.loc[ df[column] == 0, column ] = nonzero_mean


# In[ ]:


replaceZeroValues(odf, "NumericalWage")
replaceZeroValues(odf, "NumericalValue")


# And now...

# In[ ]:


print("There are " + str(len(odf[odf["NumericalWage"] == 0])) + " rows with a wage value of 0 in the NumericalWage column")
print("There are " + str(len(odf[odf["NumericalValue"] == 0])) + " rows with a player-value of 0 in the NumericalValue column")


# In[ ]:


odf["NumericalValue"].iloc[164]


# **Now let's plot overall versus value...**

# In[ ]:


# First, let's define a function to quickly scatter-plot two columns and draw a trendline that we can reuse
def myplot(x, y):
    """Draws a scatter plot of columns x and y"""
    ax = plt.scatter(x, y)
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    m, c = z
    formula = ("y = " + str(m) + " x " + str(c)) if (c < 0) else ("y = " + str(m) + " x + " + str(c))
    print(formula)
    plt.title(x.name + " vs " + y.name)
    plt.xlabel(x.name)
    plt.ylabel(y.name)
    plt.plot(x, p(x), "red")
    plt.show()


# In[ ]:


myplot(odf["Overall"], odf["NumericalValue"])


# **The scatterplot is not linearly distributed. Let's log the player value column to try and get it to be more linear.**

# In[ ]:


odf[odf['NumericalValue'] == 0] = odf['NumericalValue'].mean()


# In[ ]:


odf["LogValue"] = np.log(odf['NumericalValue'].astype('float64'))
logValueMax = odf["LogValue"].min()
logValueMax


# In[ ]:


myplot(odf["Overall"], odf["LogValue"])


# Now the scatterplot is more linear. We can see there's a strong positive linear relationship between Overall player score and Market Value

# It seems some of the rows in the "Sprint speed" column have string values instead of int (e.g 73+7). Let's identify those rows.

# In[ ]:


stringRows = [x for x in odf["Sprint speed"] if ("+" in x) or ("-" in x)]
print("There are " + str(len(stringRows)) + " rows with special characters in the 'Sprint speed' column.")
stringRows


# Let's clean the column up and drop the extra values...

# In[ ]:


def removeExtraChars(string):
    sc = "" #special character: either '+' or '-'
    if "+" in string:
        sc = "+"
    elif "-" in string:
        sc = "-"
    else:
        return string
    return string[:string.find(sc)]


# In[ ]:


odf["SprintSpeed"] = odf["Sprint speed"].apply(removeExtraChars)
total = 0
[total+1 for rowValue in odf["SprintSpeed"] if "+" in rowValue or "-" in rowValue]
print(str(total) + " rows in the new SprintSpeed column contain +/-")


# In[ ]:


odf['SprintSpeed'] = odf['SprintSpeed'].astype('float')
odf.info()


# Now that we have a new cleaned column, let's plot it...

# In[ ]:


myplot(odf["SprintSpeed"], odf["LogValue"])

