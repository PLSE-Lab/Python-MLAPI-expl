#!/usr/bin/env python
# coding: utf-8

# In[71]:


# Have used plotty so please make sure to install below commands 
#!pip install plotly
#!pip install cufflinks


# # Case Study : Car Price Prediction using Linear Regression 

# ### Problem Statement
# 
# >A Chinese automobile company Geely Auto aspires to enter the US market by setting up their manufacturing unit there and producing cars locally to give competition to their US and European counterparts.<br><br>
# They have contracted an automobile consulting company to understand the factors on which the pricing of cars depends. Specifically, they want to understand the factors affecting the pricing of cars in the American market, since those may be very different from the Chinese market. <br>
# The company wants to know:<br>
# 1. Which variables are significant in predicting the price of a car
# 2. How well those variables describe the price of a car<br><br>
# Based on various market surveys, the consulting firm has gathered a large dataset of different types of cars across the Americal market.  <br>

# ### Business Goal 
# 
# >Is is required to model the price of cars with the available independent variables. It will be used by the management to understand how exactly the prices vary with the independent variables. They can accordingly manipulate the design of the cars, the business strategy etc. to meet certain price levels. Further, the model will be a good way for management to understand the pricing dynamics of a new market. 

# ----

# ### Loading Moduels & Libraries 

# In[72]:


import pandas as pd
import numpy as np

import cufflinks as cf
import plotly as py
import plotly.graph_objs as go
import ipywidgets as widgets
from scipy import special 
import matplotlib.pyplot as plt
import seaborn as sns
import math
from IPython.display import Markdown, display ,HTML
import statsmodels.api as sm # import API
from matplotlib.pyplot import xticks


sns.set(style="whitegrid")

pd.set_option('display.max_columns', 100)

py.offline.init_notebook_mode(connected=True) # plotting in offilne mode 
cf.set_config_file(offline=False, world_readable=True, theme='ggplot')

pd.set_option('display.max_colwidth', -1) # make sure data and columns are displayed correctly withput purge
pd.options.display.float_format = '{:20,.2f}'.format # display float value with correct precision 

import warnings
warnings.filterwarnings('ignore')


# ### Meta Data Helper Utilities 

# In[73]:


def log(string):
    display(Markdown("> <span style='color:blue'>"+string+"</span>"))

def header(string):
    display(Markdown("------ "))
    display(Markdown("### "+string))
    
def header_red(string):
    display(Markdown("> <span style='color:red'>"+string))   
    
def color_red(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for 65 %
    null values , black otherwise.
    """
    color = 'red' if val > 65 else 'black'
    return 'color: %s' % color  

def get_variable_type(element) :
    """
     Check is columns are of Contineous or Categorical variable.
     Assumption is that if 
                 unique count < 20 then categorical 
                 unique count >= 20 and dtype = [int64 or float64] then contineous
     """
    if element==0:
        return "Not Known"
    elif element < 20 and element!=0 :
        return "Categorical"
    elif element >= 20 and element!=0 :
        return "Contineous" 
    
def get_meta_data(dataframe) :
    """
     Method to get Meta-Data about any dataframe passed 
    """
    metadata_matrix = pd.DataFrame({
                    'Datatype' : dataframe.dtypes.astype(str), # data types of columns
                    'Non_Null_Count': dataframe.count(axis = 0).astype(int), # total elements in columns
                    'Null_Count': dataframe.isnull().sum().astype(int), # total null values in columns
                    'Null_Percentage': dataframe.isnull().sum()/len(dataframe) * 100, # percentage of null values
                    'Unique_Values_Count': dataframe.nunique().astype(int) # number of unique values
                     })
    
    metadata_matrix = predict_variable_type(metadata_matrix)
    return metadata_matrix
        
def display_columns_with_1_unique_value(dataframe):
    unique_values_count_1 = dataframe[dataframe["Unique_Values_Count"] == 1]
    drop_value_col = unique_values_count_1.index.tolist()
    lenght = len(drop_value_col)
    header("Columns with only one unique value : "+str(lenght))
    if lenght == 0 :
        header_red("No columns with only one unique values.")  
    else :    
        log("Columns with only one unique value :")
        for index,item in enumerate(drop_value_col) :
            print(index,".",item)
            
def predict_variable_type(metadata_matrix):
    metadata_matrix["Variable_Type"] = metadata_matrix["Unique_Values_Count"].apply(get_variable_type).astype(str)
    metadata_matrix["frequency"] = metadata_matrix["Null_Count"] - metadata_matrix["Null_Count"]
    metadata_matrix["frequency"].astype(int)
    return metadata_matrix 


def list_potential_categorical_type(dataframe,main) :
    header("Stats for potential Categorical datatype columns")
    metadata_matrix_categorical = dataframe[dataframe["Variable_Type"] == "Categorical"]
    # TO DO *** Add check to skip below if there is no Categorical values 
    length = len(metadata_matrix_categorical)
    if length == 0 :
        header_red("No Categorical columns in given dataset.")  
    else :    
        metadata_matrix_categorical = metadata_matrix_categorical.filter(["Datatype","Unique_Values_Count"])
        metadata_matrix_categorical.sort_values(["Unique_Values_Count"], axis=0,ascending=False, inplace=True)
        col_to_check = metadata_matrix_categorical.index.tolist()
        name_list = []
        values_list = []
        for name in col_to_check :
            name_list.append(name)
            values_list.append(main[name].unique())
        temp = pd.DataFrame({"index":name_list,"Unique_Values":values_list})
        metadata_matrix_categorical = metadata_matrix_categorical.reset_index()
        metadata_matrix_categorical = pd.merge(metadata_matrix_categorical,temp,how='inner',on='index')
        display(metadata_matrix_categorical.set_index("index")) 
        
def get_potential_categorical_type(dataframe,main,unique_count) :
    metadata_matrix_categorical = dataframe[dataframe["Variable_Type"] == "Categorical"]
    metadata_matrix_categorical = dataframe[dataframe["Unique_Values_Count"] == unique_count]
    length = len(metadata_matrix_categorical)
    if length == 0 :
        header_red("No Categorical columns in given dataset.")  
    else :    
        metadata_matrix_categorical = metadata_matrix_categorical.filter(["Datatype","Unique_Values_Count"])
        metadata_matrix_categorical.sort_values(["Unique_Values_Count"], axis=0,ascending=False, inplace=True)
        col_to_check = metadata_matrix_categorical.index.tolist()
        name_list = []
        values_list = []
        for name in col_to_check :
            name_list.append(name)
            values_list.append(main[name].unique())
        temp = pd.DataFrame({"index":name_list,"Unique_Values":values_list})
        metadata_matrix_categorical = metadata_matrix_categorical.reset_index()
        metadata_matrix_categorical = pd.merge(metadata_matrix_categorical,temp,how='inner',on='index')
        display(metadata_matrix_categorical.set_index("index")) 
        
def plot_data_type_pie_chat(dataframe) : 
    header("Stats for Datatype Percentage Distribution")
    dataframe_group = dataframe.groupby("Datatype").frequency.count().reset_index()
    dataframe_group.sort_values(["Datatype"], axis=0,ascending=False, inplace=True)
    trace = go.Pie(labels=dataframe_group["Datatype"].tolist(), values=dataframe_group["frequency"].tolist())
    layout = go.Layout(title="Datatype Percentage Distribution")
    fig = go.Figure(data=[trace], layout=layout)    
    py.offline.iplot(fig)
    
def pairplot(x_axis,y_axis) :
    sns.pairplot(car_df,x_vars=x_axis,y_vars=y_axis,height=4,aspect=1,kind="scatter")
    plt.show()

def heatmap(x,y,dataframe):
    plt.figure(figsize=(x,y))
    sns.heatmap(dataframe.corr(),cmap="OrRd",annot=True)
    plt.show()
        
def plot_box_chart(dataframe) :
    data = []
    for index, column_name in enumerate(dataframe) :
        data.append(
        go.Box(y=dataframe.iloc[:, index],name=column_name))   
        
    layout = go.Layout(yaxis=dict(title="Frequency",zeroline=False),boxmode='group')
    fig = go.Figure(data=data, layout=layout)    
    py.offline.iplot(fig)    
    
def bar_count_plot(dataframe,col_name) :
    plt.figure(figsize=(16,8))
    plt.title(col_name + 'Histogram')
    sns.countplot(car_df[col_name], palette=("plasma"))
    xticks(rotation = 90)
    plt.show()
    
def color_red(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for value 
    greater than 10 , black otherwise.
    """
    color = 'red' if val > 5 else 'black'
    return 'color: %s' % color

def color_code_vif_values(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for 10
    , black otherwise.
    """
    if val > 10 : color = 'red' 
    elif val > 5 and val <= 10 : color = 'blue'
    elif val > 0 and val <= 5 : color = 'darkgreen'
    else : color = 'black'
    return 'color: %s' % color

def drop_col(dataframe,col_to_drop) :
    dataframe.drop([col_to_drop],axis=1,inplace=True)
    return dataframe


# # Loading dataset for car price analysis

# In[74]:


encoding_latin="latin"
car_df = pd.read_csv("../input/CarPrice_Assignment.csv",low_memory = False,encoding = encoding_latin)
car_df.head()


# In[75]:


car_df.info()


# > <span style='color:blue'><b>Analysis<b> - Dataset is clean and no NaN susbtitution is required</span>

# ## Meta data analysis for loaded dataset

# In[76]:


metadata_matrix_dataframe = get_meta_data(car_df)

# 1. Columns List with only 1 unique values
display_columns_with_1_unique_value(metadata_matrix_dataframe)

# 2. Display Data Type percenatges 
plot_data_type_pie_chat(metadata_matrix_dataframe)

# 3. Potential Categorical Variable columns 
list_potential_categorical_type(metadata_matrix_dataframe,car_df)


# # Data Cleanup / Rectification of existing values
# > 1. Looking at the above dataset we need **not to replace any nan/null values** as our data is cleaned. 
# 2. But if we look at categorical values we see <b>drivewheel</b> columns which has values rwd, fwd, 4wd. Out of these fwd & 4wd represnt same values and hence 4wd needs to be susbtitued with fwd. 
# 3. Create out new columns with as **CarManufacturer** using **CarName** columns.
# 4. Drop column - car_ID as it will not add any value.

# In[77]:


car_df["CarCompany"] = car_df["CarName"].apply(lambda carName : carName.split(" ")[0].title())
car_df = car_df.replace(to_replace ="4wd", value ="fwd") 

# Dropping car name as it will not add any values for our price prediction 
car_df.drop(["CarName"],axis=1,inplace=True)
car_df.drop(["car_ID"],axis=1,inplace=True)


# In[78]:


car_df["CarCompany"].unique()


# **Looking at above data we found that few company names are same but are misspelt.** <br>
# `1. 'Maxda', 'Mazda' --> Mazda`<br>
# `2. 'Porsche','Porcshce' --> Porsche`<br>
# `3. 'Toyota', 'Toyouta' --> Toyota`<br>
# `4. 'Vokswagen', 'Volkswagen' --> Volkswagen`<br>
# `5. 'Vw','Volvo' --> Volvo`<br>

# In[79]:


car_df = car_df.replace(to_replace ="Maxda", value ="Mazda") 
car_df = car_df.replace(to_replace ="Porcshce", value ="Porsche") 
car_df = car_df.replace(to_replace ="Toyouta", value ="Toyota") 
car_df = car_df.replace(to_replace ="Vokswagen", value ="Volkswagen") 
car_df = car_df.replace(to_replace ="Vw", value ="Volvo") 


# In[80]:


car_df["CarCompany"].unique()


# In[81]:


car_df.head()


# # Reading and Understanding the Data
# > **Dependent Variable**
# - Visualization of  Price
# 
# > **Independent Variable**
# - Visualising Numeric Variables
# - Visualising Categorical Variables
# 

# ### Dependent Variable

# In[82]:


plot_box_chart(pd.DataFrame(car_df["price"]))
log("Analysis : Price field has median around 10K with most expensive car values at 45k and cheapest car is 5k")


# ### Independent Variable

# #### 1. Visualising Numeric Variables
# - Analyizing trends by looking pairplot of all the **Independent variables Vs Dependent variable**.

# In[83]:


car_df_describe = car_df.describe()
display(car_df_describe)


# #### Plotting Pair Plot for better visualizations 

# In[84]:


y_vars=['price']
x_vars=['wheelbase','curbweight','boreratio']
pairplot(x_vars,y_vars)
log("Analysis : Wheelbase and Curbweight are positively correlated but gets spread out at higer values.")


# In[85]:


x_vars=['carlength','carwidth', 'carheight']
pairplot(x_vars,y_vars)
log("Analysis : Carlength & Carwidth are more correlated compared to carheight which is more spreadout but positive.")


# In[86]:


x_vars=['enginesize','horsepower','stroke']
pairplot(x_vars,y_vars)
log("Analysis : Enginesize & Horsepower are postively correlated but Stroke is more spread out(might not be related).")


# In[87]:


x_vars=['compressionratio','peakrpm',"symboling"]
pairplot(x_vars,y_vars)
log("Analysis : Compressionratio and Peakrpm is not correlated.")


# In[88]:


x_vars=['citympg', 'highwaympg']
pairplot(x_vars,y_vars)

log("Analysis : Citympg & Highwaympg is **negative** correlated, cheaper cars have better milage compare to expensive ones.")


# #### Plotting heatmap for numeric variables

# In[89]:


heatmap(20,12,car_df)


# > > <span style='color:blue'> Analysis : Looking at the above heat map we can see that above inferenses drawn between price and other features holds true.<br> Positive Relation with Price : wheelbase,carlenght,carwidth,curbweight,enginesize,boreratio,horesepower
# </span>

# #### Checking Multicolinearity b/w Independent variable

# In[90]:


independent_col_list = ['wheelbase', 'carlength', 'carwidth', 'carheight','curbweight', 
                        'enginesize','boreratio','horsepower','citympg', 'highwaympg']

heatmap(14,10,car_df.filter(independent_col_list))


# > <span style='color:blue'>Analysis : citympg and highwaympg have highest dependent (0.97) on each other and we have to keep track of them to avoid issues with muticoliniarity.</span>

# ### Group independent varible for correlation analysis

# **1. Looking at correlation between Car Dimensions Specific Variable i.e. weight , height etc**

# In[91]:


dimension_col_list = ['wheelbase', 'carlength', 'carwidth', 'carheight','curbweight']
heatmap(12,8,car_df.filter(dimension_col_list))


# > <span style='color:blue'>Analysis : Wheelbase , carlength , carwidth and carweight [ 0.80 - 0.88 ] are higly correlated and we have to select one out of them. Carheight is not correlated and will not affect model buildingin negative way</span>

# **2. Looking at correlation between Car Performance Specifc Varibale**

# In[92]:


performance_col_list = ['enginesize','boreratio','horsepower']
heatmap(12,8,car_df.filter(performance_col_list))


# > <span style='color:blue'>Analysis : Horsepower and enginesize are highly correlated and we need to select one from them. Boreratio is not related as will be included in model building</span>

# #### 2. Visualising Categorical Variables

# In[138]:


bar_count_plot(car_df,"CarCompany")
log("Analysis : Looking at above graph Toyota seems to be really popular among car company followed Nissan and Mazda.")


# In[94]:


# internal function written by me for better visualizatoin and understanding
metadata_matrix_dataframe = get_meta_data(car_df)
list_potential_categorical_type(metadata_matrix_dataframe,car_df)


# **Looking at the above table , we can see below columns have two unique values**
# >1. fueltype
# 2. aspiration
# 3. doornumber
# 4. drivewheel
# 5. enginelocation 
# 

# In[95]:


plt.figure(figsize=(20, 12))
plt.subplot(2,3,1)
sns.boxplot(x = 'fueltype', y = 'price', data = car_df, palette=("plasma"))
plt.subplot(2,3,2)
sns.boxplot(x = 'aspiration', y = 'price', data = car_df, palette=("plasma"))
plt.subplot(2,3,3)
sns.boxplot(x = 'doornumber', y = 'price', data = car_df ,palette=("plasma"))
plt.subplot(2,3,4)
sns.boxplot(x = 'drivewheel', y = 'price', data = car_df , palette=("plasma"))
plt.subplot(2,3,5)
sns.boxplot(x = 'enginelocation', y = 'price', data = car_df , palette=("plasma"))
plt.show()


# ><span style='color:blue'>Analysis : <br>Average price of diesel car are more expensive than gas but gas have more expensive car range.<br>
# >Cars with turbo aspiration have generally expensive.<br>
# >Most expensive cars have two doors.<br>
# >Rear engine cars are more expensive than front engine location <br></span>

# #### Looking at the above table , we can see other varible and checkig any correlatoin between them and price

# In[96]:


plt.figure(figsize=(14,6))
sns.boxplot(x='carbody',y='price',data = car_df, palette=("plasma"))
plt.show()
log("Analysis : Hardtop is a clear winner and is the preffered choice among other.")

plt.figure(figsize=(14,6))
sns.boxplot(x='fuelsystem',y='price',data = car_df, palette=("plasma"))
plt.show()
log("Analysis : MPFI is the most common one among cars.")

plt.figure(figsize=(14,6))
sns.boxplot(x='enginetype',y='price',data = car_df,palette=("plasma"))
plt.show()
log("Analysis : ohcv is the most common engine type.")

plt.figure(figsize=(14,6))
sns.boxplot(x='cylindernumber',y='price',data = car_df,palette=("plasma"))
plt.show()
log("Analysis : Expensive cars have Eight cylinder , four cylinder are the cheapest one.")


# In[97]:


plt.figure(figsize=(14,8))
sns.barplot(x = 'cylindernumber', y = 'price', hue = 'fueltype',data = car_df,palette=("plasma"))
plt.show()
log("Analysis : Not all cars comes in both gas and diesel variants. Cars with cylinder four,six and five have both variants.")


# In[98]:


plt.figure(figsize=(20,6))
plt.subplot(1,2,1)
sns.barplot(x = 'aspiration', y = 'price', hue = 'fueltype',data = car_df,palette=("plasma"))

plt.subplot(1,2,2)
sns.barplot(x = 'enginelocation', y = 'price', hue = 'fueltype',data = car_df,palette=("plasma"))
plt.show()

log("Analysis : Can with rear engine dosent have any diesel models. Turo mode is more prevalent in diesel.")


# In[99]:


plt.figure(figsize=(20,5))
sns.barplot(x = 'symboling', y = 'price',data = car_df,palette=("plasma"))
plt.show()
log("Analysis : Top categories for car symboling is -1 and 3. ")


# # Data Preparation
# 
# > 1. You can see that your dataset has many columns with values only two values.
# 2. But in order to fit a regression line, we would need numerical values and not string. Hence, we need to convert them to 1s and 0s accordingly.
# 3. Convert higer categorical variables using **Dummy Variables**

# In[100]:


metadata_matrix_dataframe = get_meta_data(car_df)
get_potential_categorical_type(metadata_matrix_dataframe,car_df,2)


# **Looking at the above table ,we can have to covert below columns in 1's and 0's**
# >1. fueltype   {"gas": 1, "diesel": 0}
# 2. aspiration {"std": 1, "turbo": 0}
# 3. doornumber {"two": 1, "four": 0}
# 4. drivewheel {"rwd": 1, "fwd": 0}
# 5. enginelocation {"front": 1, "rear": 0}

# In[101]:


def binary_dummy_replace(x) :
     return x.map({"gas":1,"diesel":0,
                   "std":1,"turbo":0,
                   "two":1, "four":0,
                   "rwd": 1, "fwd": 0,
                   "front": 1, "rear": 0})


# In[102]:


col_to_replace =  ["fueltype","aspiration","doornumber","drivewheel","enginelocation"]
car_df[col_to_replace] = car_df[col_to_replace].apply(binary_dummy_replace)

car_df.head(2)


# ### Dummy Variables

# In[103]:


def create_dummy_variable(dataframe,column_name):
    dummy_values = pd.get_dummies(dataframe[column_name],drop_first=True)
    dataframe = pd.concat([dataframe,dummy_values],axis=1)
    dataframe.drop([column_name],axis=1,inplace=True)
    return dataframe


# In[104]:


metadata_matrix_dataframe = get_meta_data(car_df)
list_potential_categorical_type(metadata_matrix_dataframe,car_df)


# #### Creating dummy variables

# In[105]:


car_df = create_dummy_variable(car_df,"carbody")
car_df = create_dummy_variable(car_df,"cylindernumber")
car_df = create_dummy_variable(car_df,"enginetype")
car_df = create_dummy_variable(car_df,"fuelsystem")
car_df = create_dummy_variable(car_df,"CarCompany")
car_df.head(2)


# In[106]:


car_df_describe = car_df.describe()
display(car_df_describe)


# ## Deleting Features 

# #### Below are the features which are not related to price because they are  - 
# 1. Dependent on other variable (muticolinearity)
# 2. No visual variance with respect to price.
# 
# `But as we are using RFE method below features will not be deleted manually and RFE will automatically identify and help us in eliminating the same.` 

# In[107]:


#drop_col(car_df,"CarCompany")
#drop_col(car_df,"compressionratio")
#drop_col(car_df,"peakrpm")
#drop_col(car_df,"stroke")

#drop_col(car_df,"wheelbase")
#drop_col(car_df,"curbweight")
#drop_col(car_df,"carwidth")

#drop_col(car_df,"citympg")

car_df.head(2)


# # Preparing Train and Test data

# In[108]:


from sklearn.model_selection import train_test_split

# We specify this so that the train and test data set always have the same rows, respectively
np.random.seed(0)
car_df_train , car_df_test = train_test_split(car_df,train_size=0.7,test_size=0.3,random_state=100)


# In[109]:


car_df_train.shape


# In[110]:


car_df_test.shape


# ## Rescaling the Train dataset
# > It is extremely important to rescale the variables so that they have a comparable scale. If we don't have comparable scales, then some of the coefficients as obtained by fitting the regression model might be very large or very small as compared to the other coefficients. This might become very annoying at the time of model evaluation. So it is advised to use standardization or normalization so that the units of the coefficients obtained are all on the same scale. As you know, there are two common ways of rescaling:
# 1. Min-Max scaling
# 2. Standardisation (mean-0, sigma-1)<br><br>
# This time, we will use **MinMax scaling**.

# In[111]:


from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()


# In[112]:


car_df_train.columns


# In[113]:


col_to_scale = ['wheelbase','carlength','carwidth','carheight','curbweight',
                'boreratio',"enginesize",'horsepower','price','citympg',
                'highwaympg','symboling','stroke','compressionratio','peakrpm']

car_df_train[col_to_scale] = scale.fit_transform(car_df_train[col_to_scale])
car_df_train.head(2)


# # Model Building Approach
# > We will be using mixed approach to find the relevent features. 
# 1. Identify features using RFE (Recurcive Feature Elimination)
# 2. Manual approach to find correct fit

# ### Dividing into x_train and y_train sets for the model building

# In[114]:


y_train = car_df_train.pop('price')
x_train = car_df_train


# ### REF (Recurcive Feature Elimination)
# > Usin RFE gives us an automated way of selectig important attributes which can influence dependent variable. We will be using mixed apporach here and as a first step we will simply use features that are returned by RFE as a starting model. On the contrary after visual analyis we can see a lot of field have abosolutly no relation with Price and can be removed before we even start building model. But as per the advice from TA , I have not removed any field manually and my model is completely rely on RFE output.Below is one of the many models that I tried with various combinations and selected the best one for submittion.  

# In[115]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


# In[116]:


regression = LinearRegression()
regression.fit(x_train,y_train)

rfe = RFE(regression,10)
rfe = rfe.fit(x_train,y_train)

list(zip(x_train.columns,rfe.support_,rfe.ranking_))


# In[117]:


col = x_train.columns[rfe.support_]


# In[118]:


print("Columns selected by RFE : ", col)


# In[119]:


model_count = 0

def statsmodel_summary(y_var,x_var) :
    global model_count
    model_count = model_count + 1
    text = "MODEL - " + str(model_count)
    header(text)
    x_var_const = sm.add_constant(x_var) # adding constant
    lm = sm.OLS(y_var,x_var_const).fit() # calculating the fit
    print(lm.summary()) # print summary for analysis
    display_vif(x_var_const.drop(['const'],axis=1))
    return x_var_const , lm
    
def display_vif(x) :
    # Calculate the VIFs for the new model
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    vif = pd.DataFrame()
    X = x
    vif['Features'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.set_index("Features")
    vif = vif.sort_values(by = "VIF", ascending = False)
    df = pd.DataFrame(vif.VIF).style.applymap(color_code_vif_values)
    display(df)
  


# # Model building and appropriate features selection

# In[120]:


x_new = x_train[col]
x_new,lm_new = statsmodel_summary(y_train,x_new)


# [](http://)> <span style='color:blue'> 
#     Analyis - Delete feature with highest p value <br>
#     Next Step - [ P-value > 0.05 ] value for feature **twelve** is highest and needs to be deleted to create stable model. <br> 
#     Model Attribute - Adj. R-squared:0.906 
# </span>

# In[121]:


x_new,lm_new = statsmodel_summary(y_train,drop_col(x_new,'twelve'))


# > > <span style='color:blue'> 
#     Analyis - [ P-value > 0.05 ] all feature are in acceptable range.<br>
#     Next Step - Deleting features **curbweight** where VIF is very high. <br> 
#     Model Attribute** - Adj. R-squared:0.906 
# </span>    

# In[122]:


x_new,lm_new = statsmodel_summary(y_train,drop_col(x_new,"curbweight"))


# * > <span style='color:blue'> 
#     Analyis - No considerable change in P- value but boreratio have p-value 0.77 which is greater than p > 0.05<br>
#     Next Step - Deleting features boreratio.  <br> 
#     Model Attribute - Adj. R-squared:0.895
# </span>    

# In[123]:


x_new,lm_new = statsmodel_summary(y_train,drop_col(x_new,"boreratio"))


# [](http://)> <span style='color:blue'> 
#     Analyis - P-value chnaged for few features <br>
#     Next Step - Deleting features Porsche where p > 0.05.  <br> 
#     Model Attribute - Adj. R-squared:0.893 (reduced by minor percentage)
# </span>    

# In[124]:


x_new,lm_new = statsmodel_summary(y_train,drop_col(x_new,"Porsche"))


# * > <span style='color:blue'> 
#     Analyis - no change in P-value and all features are in range<br>
#     Next Step - Deleting features carwidth where VIF > 5.  <br> 
#     Model Attribute - Adj. R-squared:0.892 (reduced by minor percentage)
# </span>    

# In[125]:


x_new,lm_new = statsmodel_summary(y_train,drop_col(x_new,"carwidth"))


# [](http://)> <span style='color:blue'> 
#     Analyis - Considerable change in P-value for **three** <br>
#     Next Step - Deleting features **three** where p > 0.05.  <br> 
#     Model Attribute - Adj. R-squared:0.828 (reduced by minor percentage)
# </span>   

# In[126]:


x_new,lm_new = statsmodel_summary(y_train,drop_col(x_new,"three"))


# * > <span style='color:blue'> 
#     Analyis - All features have p-value < 0.05 <br>
#     Next Step - Final model created and will be used against test data<br> 
#     Model Attribute - Adj. R-squared:0.828
# </span>  

# In[127]:


x_new.head(2)


# ### Recidual Analysis

# In[128]:


y_train_price = lm_new.predict(x_new)
fig = plt.figure(figsize=(9,6))
sns.distplot((y_train - y_train_price), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)   
plt.show()


# [](http://)> <span style='color:blue'> 
#     Analyis - Error terms are distributed around zero which signifies that model prediction is not by chance. <br>
# </span>  

# # Making Predictions

# In[129]:


car_df_test.head(2)


# In[130]:


car_df_test[col_to_scale] = scale.transform(car_df_test[col_to_scale])
car_df_test.head(2)


# In[131]:


y_test = car_df_test.pop('price')
x_test = car_df_test


# In[132]:


final_features = list(x_new.columns)
final_features.remove('const')
print(final_features)


# ### Now let's use our model to make predictions.
# 

# In[133]:


# Creating X_test_new dataframe by dropping variables from X_test
x_test_new = x_test.filter(final_features)

# Adding a constant variable 
x_test_new = sm.add_constant(x_test_new)

# Making predictions
y_pred = lm_new.predict(x_test_new)


# ## Model Evaluation

# ### Calculate the R-squared score on the test dataset

# In[134]:


from sklearn.metrics import r2_score 
r2 = r2_score(y_test, y_pred)
log("R-square calculated for test data is : "+str(r2))


# In[135]:


x_test.head()


# In[136]:


# Plotting y_test and y_pred to understand the spread.
fig = plt.figure(figsize=(9,6))
plt.scatter(y_test,y_pred)
fig.suptitle('Price [ Test Vs Predicted ]', fontsize=20)              # Plot heading 
plt.xlabel('Test', fontsize=18)                          # X-label
plt.ylabel('Predicted', fontsize=18)                          # Y-label
plt.show()


# **<span style='color:blue'> We can see that the equation of our best fitted line is:**<br>
# 
# price = 0.2024 - ( 0.2872 * enginelocation ) + (1.1880 * enginesize ) + (0.2516 * rotor) + (0.1988 * Bmw)
# </span>                   

# # Final Model Summary 

# In[ ]:





# In[137]:


print(lm_new.summary())


# ### Infrences Conclusion

# > <span style='color:blue'> R-sqaured and Adjusted R-squared (extent of fit) - 0.833 and 0.828 - 83% variance explained.</span>
# 
# > <span style='color:blue'>F-stats and Prob(F-stats) (overall model fit) - 172.3 and 1.27e-52(approx. 0.0) - Model fit is significant and explained 83% variance is just not by chance.</span>
# 
# > <span style='color:blue'>p-values - p-values for all the coefficients seem to be less than the significance level of 0.05. - meaning that all the predictors are statistically significant.</span>
# 
# > <span style='color:blue'>AIC and BIC Values - We can see that there is a diffrence between AIC ( 280.0 ) and BIC ( 265.2 ) fields and BIC has lesser value due to feature penalty and are in range.</span>
# 
# > <span style='color:blue'>Conclusion : As per final model attributes which are best suited for predicting **Price** are - <br>  1.enginelocation <br>  2.enginesize<br>  3.rotor<br>  4.Bmw<br>
# </span>

# In[ ]:




