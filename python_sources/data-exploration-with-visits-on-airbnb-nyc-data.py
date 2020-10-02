#!/usr/bin/env python
# coding: utf-8

# <font size="4">**Introduction**</font>
# <space> </space>
# Airbnb has been one of the most disruptive companies in the tourism & hospitality industry in the past decade. Airbnb NYC dataset offers us a chance to analyze and learn more about one of the most popular tourism destinations in the whole world. There are over a thousand economic entities (in the toursim industry) who could learn so much about the market by just regularly forking insights from such open datasets. As we will see, despite not having any bookings count or time stamp data for calculation of traditional ROI models, the data set is still useful in learning more about Visits, Pricing, Boroughs etc.
# 
# I have previously lived in New York City and that gives me an incentive to analyze the data especially comparing different bouroughs and neighbourhoods which paints a more robust picture in my imagination.

# <font size="4">**Objective**</font>
# <space> </space>
# Build a Visitation based approach and try to derive more insights than general exploratory analysis. Such an approach could answer deeper business questions for hotels, toursim offices, economic bodies etc.
# 

# <font size="4">**Approach**</font>
# <space> </space>
# 1. Load the Data set
# 2. Data Wrangling
# 3. Data Verification
# 4. Data Analysis & Visualisation
# 
# Because of limited data we cannot go into too much detail but we can atleast highlight some trends and try to build secondary insights that help support them.

# <font size="4">**Loading Libraries & Dataset**</font>

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# <p style="color:##808080;"> <font size="3"> 
# #Below is a default code in Kaggle Kernel. Use Data = pd.read_csv('AB_NYC_2019.csv') if doing it in a text editor like Jupyter
# </p>
# <space> </space>
# <p style="color:#808080;"> #Note: The python file must be in the same folder as data file for it to read the data set without file path </font>  </p>

# In[ ]:


Data = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")


# In[ ]:


Data.shape #Checking the number of rows and columns


# In[ ]:


Data.head(5) #Quick overview of the table and display first 5 rows of the dataset


# <p style="color:##808080;"> <font size="3"> 
# #We can also see the last 5 rows with the tail function. Remember the count starts from 0 so the last row is 48,894 ! 
# </font> </p>

# In[ ]:


Data.tail(5)


# <font size="4">**Data Wrangling**</font>
# <space> </space>

# <p style="color:##808080;"> <font size="3"> 
# #Let's drop the columns we are not going to use for analysis. id,host_name, last review are not useful at the moment.  
# </p>
# <space> </space>
# <p style="color:##808080;">
# #host_name will also be dropped and should be redacted for privacy reasons
# </font> </p>

# In[ ]:


Data.drop(['id','name','host_name','last_review'], axis=1, inplace=True)


# <p style="color:##808080;"> <font size="3"> 
# #We are now checking if there are any null values left in the remaining columns  
# </font> </p>

# In[ ]:


Data.isnull().sum()


# <p style="color:##808080;"> <font size="3"> 
# #Only reviews_per_month has null values. We can replace them with 0 with the fillna function  
# </font> </p>

# In[ ]:


Data.fillna({'reviews_per_month':0}, inplace=True)
Data.reviews_per_month.isnull().sum() #Checking the changes made
Data.isnull().sum() #There are no null values left


# <font size="4">**Data Verification**</font>

# <p style="color:##808080;"> <font size="3"> 
# #We will now try to run unique value functions to confirm that our dataset does not have any additional errors.  
# </font> </p>
# 
# <space> </space>
# 
# <p style="color:##808080;"> <font size="3"> 
# #There are only four boroughs in New York city & Airbnb only offers three types of accomodation categories
# </font> </p>

# In[ ]:


Data.neighbourhood_group.unique()


# In[ ]:


Data.room_type.unique()


# <font size="4">**Data Analysis & Visualizations**</font>

# <p style="color:##808080;"> <font size="3">
# #Before we highlight any trends we must check how many listings are available by borough
#     </font> </p>
# <Space> </space> 
# <p style="color:##808080;"> <font size="3"> 
# #we will use the value counts function to count the listings in each borough and also calculate it by percentage 
#     </font> </p>
# 

# In[ ]:


top_borough=Data.neighbourhood_group.value_counts()
top_borough1=pd.concat([Data.neighbourhood_group.value_counts() , 
                        Data.neighbourhood_group.value_counts(normalize=True).mul(100)],axis=1, keys=('listings','percentage'))
print (top_borough1)


# <p style="color:##808080;"> <font size="3"> 
# #We are using the seaborn library to do the data visualizations and Matplotlib style commands to change labels & chart title
# </font> </p>

# In[ ]:


sns.set(rc={'figure.figsize':(12,8)})                     #Setting figure size for future visualizations
sns.set_palette("pastel")                                 #Set the palette to the "pastel" default palette
V1 = sns.countplot(x='neighbourhood_group', data=Data)    #Using Seaborn to create a countplot directly 
V1.set_xlabel('Borough')                                  #Changing Labels
V1.set_ylabel('Listings')
V1.set_xticklabels(V1.get_xticklabels(), rotation=45)     #Rotating Labels slightly


# <p style="color:##808080;"> <font size="3"> 
# #There are a lot of customizations in seaborn. Running a similar analysis with top 10 Neighbourhoods of New York we can cutomize many chart attributes
# </font> </p>

# In[ ]:


V2 = sns.countplot(y='neighbourhood',                                            #Create a Horizontal Plot
                   data=Data,                                                    
                   order=Data.neighbourhood.value_counts().iloc[:10].index,      #We want to view the top 10 Neighbourhoods
                   edgecolor=(0,0,0),                                            #This cutomization gives us black borders around our plot bars
                   linewidth=2)
V2.set_title('Listings by Top NYC Neighbourhood')                                #Set Title
V2.set_xlabel('Neighbourhood')                                  
V2.set_ylabel('Listings')


# <p style="color:##808080;"> <font size="3">
# #For Visitation based insights we can use the number_of_reviews column as Visits as only visitors who have stayed in an Airbnb would leave a review of the place </font>
# </p> 

# <p style="color:##808080;">
# <font size="5">**number_of_reviews = Visits**</font> </p>

# In[ ]:


Data = Data.rename(columns = {"number_of_reviews" : "Visits"})   #Renaming the column to Visits
Data.head(2)                                                     #Checking the change that was made


# In[ ]:


Listings_by_borough = pd.DataFrame(Data.neighbourhood_group.value_counts().reset_index().values, columns=['Borough', 'Listings']) #Creating a new table directly into a dataframe
Listings_by_borough = Listings_by_borough.sort_index(axis=0, ascending=True)                                                      #sorting the data
Listings_by_borough ['% Listings']=  (Listings_by_borough['Listings'] / Listings_by_borough['Listings'].sum())*100                #Adding a % Listings column
Listings_by_borough                                                                                                               #Printing the table


# In[ ]:


V10 = sns.barplot(x='Borough', y = '% Listings',                                           
                   data=Listings_by_borough,                                                         
                   edgecolor=(0,0,0),                                            
                   linewidth=2)
V10.set_title('% Listings by Borough')
V10.set_xlabel('Borough')                                  
V10.set_ylabel('% Listings')


# <p style="color:##808080;"> <font size="3">
# #We will use Groupby function and numpy sum function to get Visits by Borough and then add another column % Visits to also analyze % Visits by Borough 
#     </font>
# </p>

# In[ ]:


visits_by_borough = Data.groupby(['neighbourhood_group'])['Visits'].agg(np.sum).reset_index()            #Using Groupby to get 'by Borough' and numpy sum function to get 'Total Vists'
visits_by_borough.columns = ['Borough', 'Visits']                                                        #Renaming the columns
visits_by_borough = visits_by_borough.sort_values('Visits', ascending=False)                             #Sorting Visit Values in descending order
visits_by_borough ['% Visits']=  (visits_by_borough['Visits'] / visits_by_borough['Visits'].sum())*100   #Creating a new column called % Visits

visits_by_borough                                                                                        #Printing the table


# In[ ]:


V3 = sns.barplot(x='Borough', y = '% Visits',                                           
                   data=visits_by_borough,                                                         
                   edgecolor=(0,0,0),                                            
                   linewidth=2)
V3.set_title('% Visits by Borough')
V3.set_xlabel('Borough')                                  
V3.set_ylabel('% Visits')


# <p style="color:##808080;"> <font size="3">
# #We will continue to use seaborn to visualize data and also pick up a few tricks for automation for future reports incase we have to re-run the analysis.
#     </font>
# </p>

# <p style="color:##808080;">
# <font size="3">
# #Seaborn has a function called **estimator** which can help us do basic calculations like sum, mean through numpy as well as complex functions like % of the numeric variable. </p>
#     <space> </space>
# <p style="color:##808080;">
# <font size="3">#This eliminates the need to code data transformations through groupby and numpy sum functions before data visualization
#     </font>
# </p>

# In[ ]:


V4 = sns.barplot(
    x='neighbourhood_group', y='Visits', 
    estimator=np.sum,                          # "sum" function from numpy as estimator , you can also use lambda x: sum(x==0)*100.0/len(x) for a percentage function
    data=Data,                                 # Raw dataset fed directly to Seaborn
    edgecolor=(0,0,0), 
    linewidth=2,
    ci=None)                                   #Removes error bars

V4.set_title('Visits by Borough')
V4.set_xlabel('Borough')                                  
V4.set_ylabel('Visits')


# In[ ]:


V9=sns.barplot(x='neighbourhood',
               y='Visits',
estimator=np.sum,
data=Data,
ci=None,           
order=Data.neighbourhood.value_counts().iloc[:10].index)

V9.set_title(' Total Visits by Neighbourhood')
V9.set_xlabel('Neighbourhood')                                  
V9.set_ylabel('Visits')
V9.set_xticklabels(V9.get_xticklabels(), rotation=45);


# In[ ]:


sns.set(style="whitegrid")                                     #Setting a new style
V6 = sns.barplot(
    x='neighbourhood', y='Visits', 
    estimator=np.mean,                                         # "mean" function from numpy as estimator
    data=Data,                                                 # Raw dataset fed directly to Seaborn
    ci=None,                               
    order=Data.neighbourhood.value_counts().iloc[:10].index)   #Top 10 Neighbourhoods only #Another Order function to get specific values order=Data['neighbourhood'].value_counts().index.tolist()[0:10]

V6.set_title('Avg. Visits by Neighbourhood')
V6.set_xlabel('Neighbourhood')                                  
V6.set_ylabel('Visits')
V6.set_xticklabels(V6.get_xticklabels(), rotation=45)


# In[ ]:


V7 = sns.barplot(x='room_type',
                 y='Visits',
                 estimator=np.sum,                                         
                 data=Data,
                 ci=None,
                 order=Data.room_type.value_counts().index)   

V7.set_title('Visits by Roomtype')                                
V7.set_xlabel('Room Type')
V7.set_ylabel('Visits')


# In[ ]:


rt = Data.groupby(['room_type'])               #Generate a table to look at the numbers, grouped by room_type
vrt = rt['Visits'].agg(np.sum).reset_index()   #aggregating the data with numpy sum function
vrt


# <p style="color:##808080;">
# <font size="3">
#     Price Analysis & (Price+Visits) Double Axis Analysis
#     </font>
# </p>

# <p style="color:##808080;">
# <font size="3">#While we can do a very detailed Price Analysis and create price segments, our goal right now is to garner some insights with respect to visits and if we can identify any trends when we compare them grouped by other attributes
#     </font>
# </p>

# In[ ]:


price_bin=Data.price.value_counts(bins=[0,25,50,100,150,200,250,300,350,400,450,500,1000,2000,5000,10000])  #Using binning function to see listings fall in what price range
price_bin


# In[ ]:


V8=price_bin.plot(kind ='bar')
V8.set_title('Listings by Price Range')
V8.set_ylabel('Listings')
V8.set_xlabel('Price Range')
V8.set_xticklabels(V8.get_xticklabels(), rotation=45)


# In[ ]:


Price_by_NG =Data.groupby(                                          #Groupby Borough
   ['neighbourhood_group'], as_index=False                                
).agg(
    {
         'Visits':sum,
         'price':'mean'
    }
)

Price_by_NG = np.round(Price_by_NG, decimals=2)                     #Function to generate avg_price with only upto two decimals
Price_by_NG = Price_by_NG.rename(columns = {"price" : "Avg_Price"}) #Switching the column name to avg_price
Price_by_NG = Price_by_NG.sort_values('Visits',ascending=False)     #Sorting values by descending for Visits
Price_by_NG


# <p style="color:##808080;">
# <font size="3">#While Seaborn can easily generate a cat plot that accomodates three variables in the chart (with Hue) and we can use the col function to further add a category variable and generate more charts, Two axis plots are more helpful in qucikly understanding the comparision between three variables. Let's look at both below -
#     </font>
# </p>

# In[ ]:


sns.catplot(x='Avg_Price' , y='Visits', hue='neighbourhood_group', data=Price_by_NG, height=6, aspect=2);


# In[ ]:


Price_by_NG1 =Data.groupby(                                          #Groupby Borough
   ['neighbourhood_group', 'room_type'], as_index=False                                
).agg(
    {
         'Visits':sum,
         'price':'mean'
    }
)

Price_by_NG1 = np.round(Price_by_NG1, decimals=2)                     #Function to generate avg_price with only upto two decimals
Price_by_NG1 = Price_by_NG1.rename(columns = {"price" : "Avg_Price"}) #Switching the column name to avg_price
Price_by_NG1 = Price_by_NG1.sort_values('Visits',ascending=False)     #Sorting values by descending for Visits
Price_by_NG1


# In[ ]:


sns.relplot(x='Avg_Price' , y='Visits', hue='neighbourhood_group',col='room_type', data=Price_by_NG1);


# In[ ]:


Price_by_NG2 =Data.groupby(                                          
   ['neighbourhood','room_type'], as_index=False                                
).agg(
    {
         'Visits':sum,
         'price':'mean'
    }
)

Price_by_NG2 = np.round(Price_by_NG2, decimals=2)                     
Price_by_NG2 = Price_by_NG2.rename(columns = {"price" : "Avg_Price"}) 
Price_by_NG2 = Price_by_NG2.sort_values('Visits',ascending=False)
Price_by_NG2 = Price_by_NG2.head(10)
sns.catplot(x='Avg_Price' , y='Visits', hue='neighbourhood', col='room_type',aspect=2, data=Price_by_NG2 );
Price_by_NG2


# In[ ]:


Price_by_N =Data.groupby(
   ['neighbourhood'], as_index=False
).agg(
    {
         'Visits':sum,
         'price':'mean'
    }
)

Price_by_N = np.round(Price_by_N, decimals=2)
Price_by_N = Price_by_N.sort_values('Visits',ascending=False)

Price_by_N = Price_by_N.head(10)

Price_by_N


# In[ ]:


fig,ax = plt.subplots()                                                             # create figure and axis objects with subplots()
ax.plot(Price_by_N.neighbourhood, Price_by_N.Visits, color="green", marker="o")     # make a plot
ax.set_xlabel("Neighbourhood",fontsize=14)                                          # set x-axis label
ax.set_ylabel("Visits",color="green",fontsize=14)                                   # set y-axis label
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

ax2=ax.twinx()                                                                      # twin object for two different y-axis on the sample plot
ax2.plot(Price_by_N.neighbourhood, Price_by_N.price,color="blue",marker="s")        # make a plot with different y-axis using second axis object
ax2.set_ylabel("Avg_Price",color="blue",fontsize=14)
plt.show()
# save the plot as a file
#fig.savefig('two_different_y_axis_for_single_python_plot_with_twinx.jpg',
            #format='jpeg',
            #dpi=100,
            #bbox_inches='tight')'''


# In[ ]:


fig,ax = plt.subplots()
ax.plot(Price_by_NG.neighbourhood_group, Price_by_NG.Visits, color="green", marker="o")
ax.set_xlabel("Borough",fontsize=14)
ax.set_ylabel("Visits",color="green",fontsize=14)

ax2=ax.twinx()
ax2.plot(Price_by_NG.neighbourhood_group, Price_by_NG.Avg_Price,color="blue",marker="s")
ax2.set_ylabel("Avg_Price",color="blue",fontsize=14)
plt.show()


# <p style="color:##808080;">
# <font size="3">#Analyzing data from the double axis graphs we can safely conclude that Price is not the only reason for a person choosing a specific neighbourhood or Borough
#     </font>
# </p>

# <font size="4">**Conclusion**</font>

# <p style="color:##808080;">
# <font size="3">#Many of these data visualizations will form the basis of our insights deck. There is definitely room for more improvement but it also shows the extent of how far data insights projects can go. Our main goal was to create possible insights with Visits as when we compare to Listings it is a completely different story. We also saw how contrasting visits with price by Borough for example is a much more robust direction than the one with just listings. This is story of tourism and conversion, where visits matter more than availablity analysis as we move towards an ROI model and also understanding what affects a purchase decision of a customer on the Airbnb platform.
#     </font>
# </p>
